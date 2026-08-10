"""
predict.py — Too Small to Handle.

Two small detectors behind a cheap router.

    wide image (aspect > 1.6)  -> LIGHTS: crop a horizontal band, cut it into 3
                                  square tiles, detect in each at 768, merge
    tall image                 -> SIGNS : one pass over the whole frame at 640

Why the tiling: a traffic light here is ~12 px wide in a 2704 px frame. Resizing
the whole frame to 640 turns it into 2.8 px and its colour is gone. Cropping the
0.30-0.90 height band (which holds 96% of lights and no sky or road) and slicing
it into 3 squares of 912 px lets us detect at 768, where the same light is
10.1 px -- visible to a stride-8 head, at the same total pixel cost as one wide
pass.

Runtime is ONNX Runtime on CPU with static shapes. No torch, no ultralytics.

    pip install -r requirements.txt
    python predict.py --test eval_data
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent

# ---- geometry / routing -----------------------------------------------------
BAND = (0.30, 0.90)      # fraction of image height kept for the lights branch
ROUTER_MIN_AR = 1.6      # width/height above this -> lights branch
ROUTER_MAX_AR = 1.4      # below this -> signs branch; between the two -> both

# ---- thresholds (tuned on val; see WRITEUP.md) ------------------------------
# Confidence is NOT a cut-off in the metric: every box submitted counts as a
# detection, so a low threshold buys recall at a steep precision cost. 0.40 was
# the measured optimum for lights (F1 0.554 at 0.05 -> 0.793 at 0.40).
LIGHT_CONF = {0: 0.40, 1: 0.40, 2: 0.40}     # red, yellow, green
SIGN_CONF = 0.40
NMS_IOU = 0.50

LIGHT_CLASS_OUT = (8, 9, 10)                 # model 0,1,2 -> competition ids

_LIGHTS = None
_SIGNS = None


# =============================================================================
# ONNX session wrapper
# =============================================================================
class Detector:
    """One ONNX detector. Handles letterboxing, the forward pass and decoding.

    Ultralytics exports raw output shaped (1, 4+nc, N): 4 box values in
    xywh (centre form, input-tensor pixels) followed by one score per class.
    No NMS is baked in, so we do it here.
    """

    def __init__(self, path: Path, imgsz: int, threads: int):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so,
                                         providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.imgsz = imgsz

    def _letterbox(self, img):
        """Resize keeping aspect, pad to a square imgsz. Returns the tensor plus
        the scale and padding needed to map boxes back to source pixels.

        The lights tiles are already square, so the padding path is skipped
        entirely for them -- allocating and filling a 768x768 canvas three times
        per frame was measurable overhead for zero effect.
        """
        h, w = img.shape[:2]
        s = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        if nw == self.imgsz and nh == self.imgsz:
            canvas, dx, dy = resized, 0, 0          # square in, square out
        else:
            canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
            dx, dy = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
            canvas[dy:dy + nh, dx:dx + nw] = resized

        # One pass: uint8 HWC BGR -> float32 CHW RGB scaled to 0..1. Doing the
        # divide as a multiply into a preallocated buffer avoids a second copy.
        blob = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None],
                                    dtype=np.float32)
        blob *= (1.0 / 255.0)
        return blob, s, dx, dy

    def __call__(self, img, conf_by_class, offset=(0, 0)):
        """Detect in one image. Returns [(cls, score, x1,y1,x2,y2), ...] in the
        coordinates of `img`, shifted by `offset` (used for tiles)."""
        blob, s, dx, dy = self._letterbox(img)
        out = self.sess.run(None, {self.iname: blob})[0]           # (1, 4+nc, N)
        p = out[0].T                                               # (N, 4+nc)
        boxes_xywh, scores = p[:, :4], p[:, 4:]

        cls = scores.argmax(1)
        best = scores[np.arange(len(scores)), cls]

        # Per-class threshold: rare classes can justify a different cut-off.
        thr = np.array([conf_by_class.get(int(c), 1.0) for c in cls]) \
            if isinstance(conf_by_class, dict) \
            else np.full(len(cls), float(conf_by_class))
        keep = best >= thr
        if not keep.any():
            return []
        boxes_xywh, best, cls = boxes_xywh[keep], best[keep], cls[keep]

        # xywh (centre) in tensor pixels -> xyxy in source pixels
        xy, wh = boxes_xywh[:, :2], boxes_xywh[:, 2:]
        x1y1 = (xy - wh / 2 - np.array([dx, dy])) / s
        x2y2 = (xy + wh / 2 - np.array([dx, dy])) / s
        boxes = np.concatenate([x1y1, x2y2], 1)
        boxes[:, [0, 2]] += offset[0]
        boxes[:, [1, 3]] += offset[1]

        return [(int(c), float(sc), *map(float, b))
                for c, sc, b in zip(cls, best, boxes)]


# =============================================================================
# tiling — must match the training-time geometry exactly
# =============================================================================
def tile_geometry(img_w: int, img_h: int, band=BAND, n_tiles=None):
    """Pixel boxes of the tiles for one image. Square tiles of side = band
    height, laid across the full width with even overlap.

    n_tiles uses ceil, not round: rounding can leave an uncovered stripe down
    the middle of the frame (a 2704-wide crop with a 1140 band would give 2
    tiles covering 2280 px and silently miss 424 px).
    """
    y0, y1 = band
    cy0, cy1 = int(round(y0 * img_h)), int(round(y1 * img_h))
    ch, cw = max(1, cy1 - cy0), img_w
    if n_tiles is None:
        n_tiles = max(1, math.ceil(cw / ch))
    if n_tiles == 1:
        return [(0, cy0, cw, cy1)]
    ts = min(ch, cw)
    stride = (cw - ts) / (n_tiles - 1)
    return [(int(round(i * stride)), cy0,
             min(int(round(i * stride)) + ts, cw), cy1) for i in range(n_tiles)]


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float):
    """Greedy NMS, used to merge duplicates from the overlapping tile seams."""
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        b, rest = boxes[i], boxes[order[1:]]
        xx1 = np.maximum(b[0], rest[:, 0])
        yy1 = np.maximum(b[1], rest[:, 1])
        xx2 = np.minimum(b[2], rest[:, 2])
        yy2 = np.minimum(b[3], rest[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        area_r = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        iou = inter / np.maximum(area_b + area_r - inter, 1e-9)
        order = order[1:][iou < thr]
    return keep


# =============================================================================
# the required API
# =============================================================================
def _pick(w: Path, stem: str) -> Path:
    """Prefer the statically quantized int8 graph, fall back to fp32.

    Static (calibrated) quantization is the one that helps a conv net: it cut
    the lights branch from 176 ms to 148 ms at 4 threads. The *dynamic*
    quantization exported during training made it SLOWER (239 ms) and is not
    shipped -- dynamic quant suits transformers, not convolutions.
    """
    qs = w / f"{stem}_qs.onnx"
    return qs if qs.exists() else w / f"{stem}.onnx"


def load_model(weights=None):
    """Runs once, before the clock starts. Builds both ONNX sessions."""
    global _LIGHTS, _SIGNS
    w = Path(weights) if weights else HERE / "weights"
    threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or min(8, os.cpu_count() or 4)
    _LIGHTS = Detector(_pick(w, "lights"), 768, threads)
    _SIGNS = Detector(_pick(w, "signs"), 640, threads)
    return _LIGHTS, _SIGNS


def predict(image_bgr: np.ndarray):
    """One full-resolution BGR frame in; boxes out as
    (class_id, x1, y1, x2, y2, confidence) in original-image pixels."""
    if _LIGHTS is None:
        load_model()

    H, W = image_bgr.shape[:2]
    ar = W / max(H, 1)
    out = []

    # Lights and signs never co-occur in the training data (0 of 2595 images),
    # so the aspect ratio picks the branch. That is a property of this
    # collection though, not a law -- anything in the ambiguous middle runs both
    # branches rather than silently dropping half the classes.
    do_lights = ar > ROUTER_MIN_AR
    do_signs = ar < ROUTER_MAX_AR
    if not do_lights and not do_signs:
        do_lights = do_signs = True

    if do_lights:
        dets = []
        for (x0, y0, x1, y1) in tile_geometry(W, H):
            crop = image_bgr[y0:y1, x0:x1]
            if crop.size:
                dets += _LIGHTS(crop, LIGHT_CONF, offset=(x0, y0))
        if dets:
            b = np.array([d[2:] for d in dets], dtype=np.float32)
            s = np.array([d[1] for d in dets], dtype=np.float32)
            c = np.array([d[0] for d in dets])
            for k in np.unique(c):                    # merge tile seams per class
                m = c == k
                for i in nms(b[m], s[m], NMS_IOU):
                    x1, y1, x2, y2 = b[m][i]
                    out.append((LIGHT_CLASS_OUT[int(k)],
                                float(x1), float(y1), float(x2), float(y2),
                                float(s[m][i])))

    if do_signs:
        # NMS is NOT optional here. The exported ONNX graph has no NMS baked in,
        # so one sign comes back as ~17 near-identical boxes. Every box counts as
        # a detection in this metric, so omitting this cost mAP50_signs
        # 0.904 -> 0.487.
        dets = _SIGNS(image_bgr, SIGN_CONF)
        if dets:
            b = np.array([d[2:] for d in dets], dtype=np.float32)
            s = np.array([d[1] for d in dets], dtype=np.float32)
            c = np.array([d[0] for d in dets])
            for k in np.unique(c):
                m = c == k
                for i in nms(b[m], s[m], NMS_IOU):
                    x1, y1, x2, y2 = b[m][i]
                    out.append((int(k), float(x1), float(y1), float(x2),
                                float(y2), float(s[m][i])))

    return out


# =============================================================================
# CLI:  python predict.py --test eval_data
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="folder of test images")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    load_model(args.weights)

    root = Path(args.test)
    paths = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"{len(paths)} images from {root}")

    rows, times = [], []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))            # disk read is not timed
        if img is None:
            rows.append((p.stem, ""))
            continue
        t0 = time.perf_counter()
        dets = predict(img)
        times.append((time.perf_counter() - t0) * 1000)

        parts = []
        for c, x1, y1, x2, y2, s in dets:
            parts += [str(int(c)), f"{s:.4f}", f"{x1:.1f}", f"{y1:.1f}",
                      f"{x2:.1f}", f"{y2:.1f}"]
        rows.append((p.stem, " ".join(parts)))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(paths)} p50={np.median(times):.0f}ms", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "PredictionString"])
        w.writerows(rows)

    t = np.array(times)
    print(f"\nwrote {args.out}  ({len(rows)} rows)")
    print(f"latency: p50 {np.median(t):.1f} ms | p95 {np.percentile(t,95):.1f} ms "
          f"| max {t.max():.1f} ms")


if __name__ == "__main__":
    main()
