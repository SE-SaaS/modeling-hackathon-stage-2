"""
shared_code.py — the engine shared by every experiment in this problem.

Each experiment folder holds only a config.yaml plus a thin train.py that calls
into here. Everything real lives in this file:

  Section 1  config loading (shared_config.yaml merged with the experiment's)
  Section 2  dataset construction  (the part that actually matters)
  Section 3  training + export via Ultralytics
  Section 4  the competition metric, reimplemented so we can rank experiments
  Section 5  Modal plumbing (image, resources, remote path repointing)

--------------------------------------------------------------------------
Why the dataset section is the interesting one
--------------------------------------------------------------------------
The dataset holds two problems that do not belong in the same model:

  lights (8,9,10)  median 12 px wide, 5442 instances, only in 2704x1520 dashcam
                   frames, clustered low in the frame (median cy ~0.64/0.72)
  signs  (0..7)    median 458 px wide, 685 instances, only in portrait phone
                   photos, ~70-140 instances per class

They never co-occur: 0 of 2595 images contain both. See
data_code/01_dataset_analysis.txt.

So we build TWO datasets and train TWO models.

The signs dataset is trivial: keep sign images, keep classes 0..7.

The lights dataset is where the 12-pixel problem is solved. Resizing a full
2704x1520 frame to 640 turns a 12 px light into 2.8 px -- below the stride-8
grid, and its colour is gone. Instead we:

  1. crop a horizontal band [y0,y1] of the frame (default 0.30..0.90, which
     contains 96% of train / 98% of val light boxes and drops the sky and road)
  2. cut that band into N square tiles across its width, N = round(w/h) of the
     crop, with even overlap so nothing falls between tiles
  3. train on those tiles at a normal square imgsz

A 2704x1520 frame -> band 2704x912 -> 3 tiles of 912x912. At imgsz 640 a 12 px
light is 12 * 640/912 = 8.4 px, which a stride-8 head can actually see. The
three tiles cost 3*640^2 = 1.23 MPix, the same as one 1920x640 pass, but every
tensor is square -- Ultralytics only accepts an int imgsz for training, and
mosaic/scale augmentation behaves normally.

Tiling is defined in normalized terms, so it does not care that image sizes
vary. Inference must reproduce the identical crop+tile geometry; predict.py
imports tile_geometry() from here so there is exactly one definition of it.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

PKG_DIR = Path(__file__).resolve().parent

SIGN_IDS = tuple(range(8))
LIGHT_IDS = (8, 9, 10)
CLASS_NAMES = {
    0: "Bus stop", 1: "Crossroad", 2: "No entry", 3: "No parking",
    4: "No stopping", 5: "Speed limit", 6: "Yield", 7: "direction of road",
    8: "red light", 9: "yellow light", 10: "green light",
}

# Packages installed into the Modal container. Kept in one place so every
# experiment shares a single cached image layer.
_MODAL_PIP = [
    "ultralytics==8.4.11",
    "opencv-python-headless==4.10.0.84",
    "numpy<2",
    "pyyaml",
    "onnx==1.17.0",
    "onnxruntime==1.19.2",
    "onnxslim",
]


# =============================================================================
# Section 1 — config
# =============================================================================
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; `override` wins. Lists are replaced, not merged."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(experiment_dir, config_name: str = "config.yaml",
                verbose: bool = False) -> dict:
    """shared_config.yaml merged with <experiment_dir>/config.yaml."""
    experiment_dir = Path(experiment_dir)
    with open(PKG_DIR / "shared_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    exp_path = experiment_dir / config_name
    if exp_path.exists():
        with open(exp_path, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    if verbose:
        print(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


# =============================================================================
# Section 2 — dataset construction
# =============================================================================
def tile_geometry(img_w: int, img_h: int, band: tuple[float, float],
                  n_tiles: int | None = None):
    """Crop+tile geometry for one image. THE single definition of it — training
    data and inference both come through here, so they cannot drift apart.

    Returns (tiles, crop_y0_px, tile_size_px) where tiles is a list of
    (x0, y0, x1, y1) pixel boxes in ORIGINAL image coordinates.

    The band is a fraction of image height. Tiles are squares of side equal to
    the band height, laid across the full width with even overlap, so a light
    near a seam still lands whole inside at least one tile.
    """
    y0, y1 = band
    cy0 = int(round(y0 * img_h))
    cy1 = int(round(y1 * img_h))
    ch = max(1, cy1 - cy0)
    cw = img_w

    if n_tiles is None:
        # Square tiles of side = band height, as many as it takes to COVER the
        # width. Must be ceil, not round: round() on a 2704-wide / 1140-tall
        # crop gives 2 tiles of 1140 = 2280 px and silently leaves a 424 px
        # blind stripe down the middle of every frame.
        n_tiles = max(1, math.ceil(cw / ch))

    ts = min(ch, cw) if n_tiles == 1 else ch
    ts = min(ts, cw)

    if n_tiles == 1:
        xs = [0]
        ts = cw          # single tile spans the whole band (may be non-square)
    else:
        # Even spacing; last tile ends exactly at the right edge.
        stride = (cw - ts) / (n_tiles - 1)
        xs = [int(round(i * stride)) for i in range(n_tiles)]

    tiles = [(x, cy0, min(x + ts, cw), cy1) for x in xs]
    return tiles, cy0, ts


def _yolo_to_xyxy(cx, cy, w, h, W, H):
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def _xyxy_to_yolo(x1, y1, x2, y2, W, H):
    return (((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, (x2 - x1) / W, (y2 - y1) / H)


def _read_labels(path: Path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                rows.append((int(p[0]), *map(float, p[1:])))
    return rows


def _build_one_lights_image(args):
    """Worker: crop+tile ONE source image, write tiles and their labels.

    A box is kept for a tile when its centre is inside the tile; the box is then
    clipped to the tile. If clipping removes more than (1-min_keep) of its area
    the box is dropped instead, so we never train on a sliver of a light.
    """
    import cv2

    (img_path, lbl_path, out_img_dir, out_lbl_dir, band, n_tiles,
     min_keep, jpg_quality) = args

    img = cv2.imread(str(img_path))
    if img is None:
        return 0, 0
    H, W = img.shape[:2]
    rows = [r for r in _read_labels(Path(lbl_path)) if r[0] in LIGHT_IDS]

    tiles, _, _ = tile_geometry(W, H, band, n_tiles)
    n_written, n_boxes = 0, 0

    for ti, (tx0, ty0, tx1, ty1) in enumerate(tiles):
        tw, th = tx1 - tx0, ty1 - ty0
        if tw <= 1 or th <= 1:
            continue

        keep = []
        for cls, cx, cy, w, h in rows:
            bx1, by1, bx2, by2 = _yolo_to_xyxy(cx, cy, w, h, W, H)
            ccx, ccy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if not (tx0 <= ccx < tx1 and ty0 <= ccy < ty1):
                continue
            area = max(1e-6, (bx2 - bx1) * (by2 - by1))
            kx1, ky1 = max(bx1, tx0), max(by1, ty0)
            kx2, ky2 = min(bx2, tx1), min(by2, ty1)
            if (kx2 - kx1) <= 0 or (ky2 - ky1) <= 0:
                continue
            if ((kx2 - kx1) * (ky2 - ky1)) / area < min_keep:
                continue
            ncx, ncy, nw, nh = _xyxy_to_yolo(kx1 - tx0, ky1 - ty0,
                                             kx2 - tx0, ky2 - ty0, tw, th)
            # classes 8,9,10 -> 0,1,2 for the 3-class lights model
            keep.append((LIGHT_IDS.index(cls), ncx, ncy, nw, nh))

        stem = f"{Path(img_path).stem}_t{ti}"
        crop = img[ty0:ty1, tx0:tx1]
        cv2.imwrite(str(Path(out_img_dir) / f"{stem}.jpg"), crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        Path(out_lbl_dir, f"{stem}.txt").write_text(
            "".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}\n" for c, a, b, w, h in keep))
        n_written += 1
        n_boxes += len(keep)

    return n_written, n_boxes


def _build_one_signs_image(args):
    """Worker: signs need no geometry work — link/copy the image, filter labels
    to classes 0..7. Images are left at full resolution; Ultralytics resizes."""
    img_path, lbl_path, out_img_dir, out_lbl_dir = args
    rows = [r for r in _read_labels(Path(lbl_path)) if r[0] in SIGN_IDS]
    if not rows:
        return 0, 0
    dst = Path(out_img_dir) / Path(img_path).name
    if not dst.exists():
        try:
            os.link(img_path, dst)          # hard link: no copy, no extra disk
        except OSError:
            shutil.copy2(img_path, dst)
    Path(out_lbl_dir, Path(img_path).stem + ".txt").write_text(
        "".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}\n" for c, a, b, w, h in rows))
    return 1, len(rows)


def _is_lights_image(W: int, H: int, min_ar: float) -> bool:
    """Router used for BOTH dataset building and inference: landscape frames are
    dashcam frames (lights), everything else is a phone photo of a sign.

    Verified on all 2595 labelled images: ar>1.6 -> 5442 lights / 0 signs;
    ar<=1.6 -> 0 lights / 685 signs. It is a property of this collection, so
    inference keeps a fallback rather than trusting it blindly.
    """
    return (W / max(H, 1)) > min_ar


def prepare_dataset(cfg: dict, dest_root, source_root=None, force: bool = False) -> Path:
    """Build the YOLO dataset this experiment trains on. Returns its data.yaml.

    Skips the work if the destination already carries a .done stamp matching the
    current settings, so re-running an experiment does not rebuild it.
    """
    import cv2  # noqa: F401  (imported early so a missing wheel fails fast)

    d = cfg["dataset"]
    branch = d["branch"]
    src = Path(source_root or cfg["paths"]["data_root"])
    dest = Path(dest_root)

    band = tuple(d.get("band", [0.30, 0.90]))
    n_tiles = d.get("n_tiles")            # None -> derived from the crop aspect
    min_keep = float(d.get("min_box_keep", 0.6))
    jpg_q = int(d.get("jpg_quality", 92))
    min_ar = float(d.get("router_min_ar", 1.6))
    yellow_rep = int(d.get("yellow_oversample", 1))

    stamp_key = json.dumps(dict(branch=branch, band=band, n_tiles=n_tiles,
                                min_keep=min_keep, jpg_q=jpg_q, min_ar=min_ar,
                                yellow_rep=yellow_rep), sort_keys=True)
    stamp = dest / ".done"
    if stamp.exists() and stamp.read_text() == stamp_key and not force:
        print(f"[dataset] reusing {dest}")
        return dest / "data.yaml"
    if dest.exists() and force:
        shutil.rmtree(dest)

    names = ({0: "red light", 1: "yellow light", 2: "green light"}
             if branch == "lights" else {i: CLASS_NAMES[i] for i in SIGN_IDS})

    from PIL import Image
    stats = {}
    for split in ("train", "val"):
        s_img, s_lbl = src / split / "images", src / split / "labels"
        o_img, o_lbl = dest / split / "images", dest / split / "labels"
        o_img.mkdir(parents=True, exist_ok=True)
        o_lbl.mkdir(parents=True, exist_ok=True)

        jobs = []
        for p in sorted(s_img.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            with Image.open(p) as im:
                W, H = im.size
            lp = s_lbl / (p.stem + ".txt")
            rows = _read_labels(lp)
            is_light_img = _is_lights_image(W, H, min_ar)

            if branch == "lights":
                if not is_light_img or not any(r[0] in LIGHT_IDS for r in rows):
                    continue
                jobs.append((str(p), str(lp), str(o_img), str(o_lbl),
                             band, n_tiles, min_keep, jpg_q))
            else:
                if is_light_img or not any(r[0] in SIGN_IDS for r in rows):
                    continue
                jobs.append((str(p), str(lp), str(o_img), str(o_lbl)))

        worker = _build_one_lights_image if branch == "lights" else _build_one_signs_image
        n_img = n_box = 0
        with ProcessPoolExecutor(max_workers=min(16, (os.cpu_count() or 4))) as ex:
            for a, b in ex.map(worker, jobs, chunksize=8):
                n_img += a
                n_box += b
        stats[split] = dict(source_images=len(jobs), written_images=n_img, boxes=n_box)
        print(f"[dataset] {branch}/{split}: {len(jobs)} src -> {n_img} imgs, {n_box} boxes")

    # Yellow lights appear 99 times in train and carry a third of the lights
    # score. Duplicating the tiles that contain one is the cheapest way to stop
    # the loss from ignoring them; it costs epochs, not architecture.
    if branch == "lights" and yellow_rep > 1:
        n = _oversample_class(dest / "train", cls_id=1, repeats=yellow_rep)
        print(f"[dataset] yellow oversample x{yellow_rep}: +{n} tiles")

    data_yaml = dest / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(
        {"path": str(dest), "train": "train/images", "val": "val/images",
         "nc": len(names), "names": names}, sort_keys=False))
    (dest / "build_stats.json").write_text(json.dumps(
        {"key": json.loads(stamp_key), "splits": stats}, indent=2))
    stamp.write_text(stamp_key)
    return data_yaml


def _oversample_class(split_dir: Path, cls_id: int, repeats: int) -> int:
    """Duplicate every tile containing class `cls_id`, (repeats-1) extra copies."""
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    added = 0
    for lp in sorted(lbl_dir.glob("*.txt")):
        rows = _read_labels(lp)
        if not any(r[0] == cls_id for r in rows):
            continue
        ip = img_dir / (lp.stem + ".jpg")
        if not ip.exists():
            continue
        for k in range(1, repeats):
            dst_i, dst_l = img_dir / f"{lp.stem}_os{k}.jpg", lbl_dir / f"{lp.stem}_os{k}.txt"
            if dst_i.exists():
                continue
            try:
                os.link(ip, dst_i)
            except OSError:
                shutil.copy2(ip, dst_i)
            shutil.copy2(lp, dst_l)
            added += 1
    return added


# =============================================================================
# Section 3 — training + export
# =============================================================================
def run_experiment(cfg: dict, exp_dir, out_dir=None, persist_fn=None,
                   fresh: bool = False) -> dict:
    """Build dataset -> train -> validate -> export ONNX (+ optional INT8).

    persist_fn is Modal's volume.commit, called at the end of each stage so a
    long run's outputs survive even if the container is interrupted.
    """
    from ultralytics import YOLO

    out_dir = Path(out_dir or Path(exp_dir) / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["experiment"]["name"]

    ds_root = Path(cfg["paths"].get("dataset_cache", out_dir / "dataset"))
    data_yaml = prepare_dataset(cfg, ds_root)
    if persist_fn:
        persist_fn()

    t = dict(cfg["train"])
    load_from = t.pop("load_from", None)

    # Resume a run the container was killed part-way through: Ultralytics picks
    # the epoch, optimizer state and LR schedule back up from last.pt, so a
    # crashed 150-epoch run costs only the epochs it had not reached.
    last_pt = out_dir / "train" / "weights" / "last.pt"
    if fresh and (out_dir / "train").exists():
        # --force: throw the previous attempt away rather than resuming into it.
        print(f"[train] fresh run requested, clearing {out_dir / 'train'}")
        shutil.rmtree(out_dir / "train", ignore_errors=True)
    if last_pt.exists() and not fresh:
        print(f"[train] resuming from {last_pt}")
        model = YOLO(str(last_pt))
        results = model.train(resume=True)
    else:
        model = YOLO(t.pop("model"))
        # A .yaml architecture (e.g. the P2 variant) starts from random weights.
        # load_from warm-starts every layer whose shape matches a released
        # checkpoint; the layers unique to the variant stay random.
        if load_from:
            model = model.load(load_from)
        results = model.train(
            data=str(data_yaml),
            project=str(out_dir),
            name="train",
            exist_ok=True,
            **t,
        )
    if persist_fn:
        persist_fn()

    best = Path(results.save_dir) / "weights" / "best.pt"
    summary = {"experiment": name, "best_weights": str(best)}

    # Validate at the training imgsz, on the same tiled val set.
    m = YOLO(str(best))
    val = m.val(data=str(data_yaml), imgsz=t["imgsz"], split="val", verbose=False)
    summary["val"] = {
        "map50": float(val.box.map50), "map": float(val.box.map),
        "per_class_map50": {str(k): float(v) for k, v in
                            zip(val.box.ap_class_index.tolist(), val.box.ap50.tolist())},
    }

    # Export. Static shapes only: the private machine runs ONNX Runtime on CPU
    # and dynamic axes cost real latency there.
    ex = cfg.get("export", {})
    if ex.get("onnx", True):
        onnx_path = m.export(format="onnx", imgsz=t["imgsz"], opset=ex.get("opset", 12),
                             simplify=True, dynamic=False, batch=ex.get("batch", 1))
        summary["onnx"] = str(onnx_path)
        summary["onnx_mb"] = round(Path(onnx_path).stat().st_size / 1e6, 2)
        if ex.get("int8", False):
            try:
                summary["onnx_int8"] = quantize_int8(onnx_path)
            except Exception as e:                       # never fail the run on this
                summary["onnx_int8_error"] = repr(e)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if persist_fn:
        persist_fn()
    return summary


def quantize_int8(onnx_path) -> str:
    """Dynamic INT8 quantization. Post-training, no calibration set, no
    retraining — the cheap 2-3x CPU win. Verify accuracy before shipping it;
    tiny objects are exactly where INT8 hurts most."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    onnx_path = Path(onnx_path)
    out = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    quantize_dynamic(str(onnx_path), str(out), weight_type=QuantType.QUInt8)
    return str(out)


# =============================================================================
# Section 4 — the competition metric
# =============================================================================
# Score = 0.5 * F1_lights + 0.5 * mAP50_signs
#   F1_lights : classes 8,9,10 matched by CENTRE DISTANCE, not IoU. A prediction
#               is correct when its centre falls within max(4, 0.5*diag(gt)) of
#               the true centre AND the class matches. F1 per colour, then a
#               plain mean — so yellow (99 train instances) is a third of it.
#   mAP50     : classes 0..7, standard all-point-interpolation AP at IoU 0.5.
# Reimplemented here so an experiment can be ranked on the real objective rather
# than on Ultralytics' mAP, which weights these two halves completely differently.

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def _ap_all_point(rec: np.ndarray, prec: np.ndarray) -> float:
    """All-point interpolated AP (area under the monotonic precision envelope)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def f1_lights(preds: dict, gts: dict) -> dict:
    """preds/gts: {image_id: [(cls, conf, x1,y1,x2,y2), ...]} in original pixels.
    Ground truth entries may omit conf. Returns per-class F1 and their mean."""
    out = {}
    for cls in LIGHT_IDS:
        tp = fp = fn = 0
        for img_id in set(gts) | set(preds):
            g = [b for b in gts.get(img_id, []) if b[0] == cls]
            p = sorted([b for b in preds.get(img_id, []) if b[0] == cls],
                       key=lambda r: -r[1])
            used = [False] * len(g)
            for _, _, x1, y1, x2, y2 in p:
                pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
                best, best_d = -1, None
                for gi, gb in enumerate(g):
                    if used[gi]:
                        continue
                    gx1, gy1, gx2, gy2 = gb[-4:]
                    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                    thr = max(4.0, 0.5 * float(np.hypot(gx2 - gx1, gy2 - gy1)))
                    dist = float(np.hypot(pcx - gcx, pcy - gcy))
                    if dist <= thr and (best_d is None or dist < best_d):
                        best, best_d = gi, dist
                if best >= 0:
                    used[best] = True
                    tp += 1
                else:
                    fp += 1
            fn += used.count(False)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[cls] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    out["mean"] = float(np.mean([out[c] for c in LIGHT_IDS]))
    return out


def map50_signs(preds: dict, gts: dict) -> dict:
    """Standard AP@0.5 per sign class, plain mean over classes present in gt."""
    out = {}
    for cls in SIGN_IDS:
        rows, n_gt = [], 0
        gt_by_img = {}
        for img_id, boxes in gts.items():
            g = np.array([b[-4:] for b in boxes if b[0] == cls], dtype=float).reshape(-1, 4)
            gt_by_img[img_id] = [g, np.zeros(len(g), dtype=bool)]
            n_gt += len(g)
        if n_gt == 0:
            continue
        for img_id, boxes in preds.items():
            for b in boxes:
                if b[0] == cls:
                    rows.append((b[1], img_id, b[2], b[3], b[4], b[5]))
        rows.sort(key=lambda r: -r[0])

        tp = np.zeros(len(rows))
        fp = np.zeros(len(rows))
        for i, (_, img_id, x1, y1, x2, y2) in enumerate(rows):
            g, used = gt_by_img.get(img_id, [np.zeros((0, 4)), np.zeros(0, bool)])
            if len(g) == 0:
                fp[i] = 1
                continue
            ious = _iou_matrix(np.array([[x1, y1, x2, y2]], float), g)[0]
            j = int(np.argmax(ious))
            if ious[j] >= 0.5 and not used[j]:
                used[j] = True
                tp[i] = 1
            else:
                fp[i] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / n_gt
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        out[cls] = _ap_all_point(rec, prec)
    out["mean"] = float(np.mean([v for k, v in out.items() if k != "mean"])) if out else 0.0
    return out


def competition_score(preds: dict, gts: dict) -> dict:
    """The full objective: 0.5 * F1_lights + 0.5 * mAP50_signs."""
    fl = f1_lights(preds, gts)
    ms = map50_signs(preds, gts)
    return {"score": 0.5 * fl["mean"] + 0.5 * ms["mean"],
            "f1_lights": fl, "map50_signs": ms}


# =============================================================================
# Section 5 — Modal
# =============================================================================
def modal_image(python_version: str = "3.11", extra_pip=None):
    """Container image: pip deps + the whole training_scripts/ tree, minus run
    artifacts. One shared base layer keeps every parallel run's image cached."""
    import modal
    img = (modal.Image.debian_slim(python_version=python_version)
           .apt_install("libgl1", "libglib2.0-0")      # opencv runtime deps
           .pip_install(*_MODAL_PIP)
           .env({"YOLO_CONFIG_DIR": "/tmp/ultralytics"}))
    if extra_pip:
        img = img.pip_install(*extra_pip)
    return img.add_local_dir(
        str(PKG_DIR), remote_path="/root/training_scripts",
        ignore=["**/results/**", "**/dataset/**", "**/__pycache__/**"],
    )


def modal_resources(cfg: dict) -> dict:
    m = cfg["modal"]
    return dict(gpu=m["gpu"], cpu=m["cpu_cores"],
                memory=int(m["memory_gb"]) * 1024,
                timeout=int(m["timeout_seconds"]))


def remote_cfg(cfg: dict) -> dict:
    """Copy of cfg with paths repointed at the mounted Modal volumes."""
    rc = copy.deepcopy(cfg)
    m = cfg["modal"]
    rc["paths"]["data_root"] = m["remote_data_root"]
    rc["paths"]["dataset_cache"] = str(
        Path(m["runs_mount"]) / "_datasets" / cfg["dataset"]["cache_key"])
    return rc


# =============================================================================
# Section 6 — run orchestration + automatic result sync
# =============================================================================
# Everything here runs LOCALLY, inside @app.local_entrypoint(), so it can write
# to the local disk. The goal is that `modal run .../train.py` is the only
# command ever needed: it trains, then pulls the results down by itself, and it
# does the right thing when re-run after a crash instead of starting over.
#
# State lives in two places and is derived, never assumed:
#   remote : /runs/<exp>/summary.json exists      -> training finished
#            /runs/<exp>/train/weights/last.pt    -> training started
#   local  : <exp>/results/.sync_state.json       -> what was pulled, and when
#
# Re-running resolves to one of four outcomes:
#   local complete   -> do nothing (unless --force)
#   remote complete  -> skip training, just download
#   remote partial   -> resume training from last.pt, then download
#   nothing anywhere -> train from scratch, then download
#
# The download runs in a finally: block, so a crashed run still yields its
# logs, partial weights and whatever plots exist. That is usually what you need
# to work out why it crashed.

# Fetched by default. Anything not matched is skipped, which keeps the per-epoch
# checkpoints (epoch0.pt, epoch10.pt, ...) off the local disk — they are large
# and best.pt/last.pt already cover every real use.
_FETCH_KEEP_SUFFIXES = (".json", ".csv", ".yaml", ".txt", ".png", ".jpg", ".onnx")
_FETCH_KEEP_NAMES = ("best.pt", "last.pt")


def _should_fetch(rel_path: str, weights_only: bool = False) -> bool:
    name = rel_path.rsplit("/", 1)[-1]
    if name.endswith(".pt"):
        # best/last only; the periodic epoch checkpoints stay remote.
        return name in _FETCH_KEEP_NAMES
    if name.endswith(".onnx"):
        return True                       # a weight too — always wanted
    if weights_only:
        return False
    return name.endswith(_FETCH_KEEP_SUFFIXES)


def _is_file_entry(entry) -> bool:
    """FileEntryType across modal versions without importing a private path."""
    t = getattr(entry, "type", None)
    return getattr(t, "name", str(t)).upper().endswith("FILE")


def fetch_run(volume, exp_name: str, local_dir, weights_only: bool = False,
              verbose: bool = True) -> dict:
    """Download /runs/<exp_name> into local_dir. Idempotent: a file already
    present with the same byte size is skipped, so re-running costs nothing and
    a half-finished download resumes cleanly.

    Writes to a .part file and renames, so an interrupted transfer can never
    leave a truncated file that a later run then mistakes for complete.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    got = skipped = failed = 0
    total_bytes = 0

    try:
        entries = list(volume.listdir(exp_name, recursive=True))
    except Exception as e:
        if verbose:
            print(f"[sync] nothing to fetch for {exp_name}: {e}")
        return dict(downloaded=0, skipped=0, failed=0, bytes=0, present=False)

    for entry in entries:
        if not _is_file_entry(entry):
            continue
        rel = entry.path[len(exp_name):].lstrip("/")
        if not rel or not _should_fetch(rel, weights_only):
            continue

        dst = local_dir / rel
        size = int(getattr(entry, "size", 0) or 0)
        if dst.exists() and size and dst.stat().st_size == size:
            skipped += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        try:
            with open(tmp, "wb") as f:
                for chunk in volume.read_file(entry.path):
                    f.write(chunk)
            tmp.replace(dst)
            got += 1
            total_bytes += dst.stat().st_size
            if verbose:
                print(f"[sync]   + {rel} ({dst.stat().st_size/1e6:.1f} MB)")
        except Exception as e:
            failed += 1
            tmp.unlink(missing_ok=True)
            if verbose:
                print(f"[sync]   ! {rel}: {e}")

    if verbose:
        print(f"[sync] {exp_name}: {got} new, {skipped} current, {failed} failed "
              f"({total_bytes/1e6:.1f} MB)")
    return dict(downloaded=got, skipped=skipped, failed=failed,
                bytes=total_bytes, present=True)


def remote_status(volume, exp_name: str) -> str:
    """'complete' | 'partial' | 'absent', read from the volume itself."""
    try:
        entries = list(volume.listdir(exp_name, recursive=True))
    except Exception:
        return "absent"
    paths = {e.path for e in entries}
    if any(p.endswith("summary.json") for p in paths):
        return "complete"
    if any(p.endswith("last.pt") for p in paths):
        return "partial"
    return "partial" if paths else "absent"


def local_status(local_dir) -> str:
    """'complete' | 'partial' | 'absent', from what is actually on disk.

    Deliberately checks for the files themselves rather than trusting the state
    file, so deleting a weight by hand correctly downgrades the status.
    """
    local_dir = Path(local_dir)
    if not local_dir.exists():
        return "absent"
    has_summary = (local_dir / "summary.json").exists()
    has_best = any(local_dir.rglob("best.pt"))
    if has_summary and has_best:
        return "complete"
    return "partial" if any(local_dir.iterdir()) else "absent"


def _write_state(local_dir, **kw):
    p = Path(local_dir) / ".sync_state.json"
    state = {}
    if p.exists():
        try:
            state = json.loads(p.read_text())
        except Exception:
            state = {}
    state.update(kw)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def orchestrate(cfg: dict, exp_name: str, local_results_dir, volume,
                train_fn, force: bool = False, fetch_only: bool = False,
                weights_only: bool = False) -> str:
    """The whole `modal run` flow. Returns the outcome as a string.

    train_fn is the remote handle's .remote() call, passed in so this stays
    independent of how the caller declared its Modal function.
    """
    local_results_dir = Path(local_results_dir)
    lstat = local_status(local_results_dir)
    rstat = remote_status(volume, exp_name)
    print(f"[run] {exp_name}: local={lstat} remote={rstat}"
          + (" (force)" if force else ""))

    if lstat == "complete" and not force and not fetch_only:
        print(f"[run] {exp_name} already complete locally — nothing to do.")
        print(f"[run] re-run with --force to retrain, or --fetch-only to re-pull.")
        fetch_run(volume, exp_name, local_results_dir, weights_only)   # heal gaps
        return "already-complete"

    if fetch_only:
        fetch_run(volume, exp_name, local_results_dir, weights_only)
        _write_state(local_results_dir, last_action="fetch-only",
                     remote_status=rstat)
        return "fetched"

    if rstat == "complete" and not force:
        print(f"[run] {exp_name} finished remotely — downloading, not retraining.")
        fetch_run(volume, exp_name, local_results_dir, weights_only)
        _write_state(local_results_dir, last_action="fetch-remote-complete",
                     remote_status=rstat)
        return "fetched-remote-complete"

    if rstat == "partial" and not force:
        print(f"[run] {exp_name} has a partial remote run — resuming from last.pt.")

    outcome = "trained"
    error = None
    try:
        train_fn(fresh=force)
    except Exception as e:                 # crash: still pull whatever exists
        outcome, error = "crashed", repr(e)
        print(f"[run] {exp_name} FAILED: {e}")
        print(f"[run] pulling partial results anyway so the logs are local...")
    finally:
        # Always fetch. A crashed run's logs and last.pt are the whole point.
        res = fetch_run(volume, exp_name, local_results_dir, weights_only)
        _write_state(local_results_dir, last_action=outcome, error=error,
                     remote_status=remote_status(volume, exp_name),
                     fetched=res)

    final = local_status(local_results_dir)
    print(f"[run] {exp_name}: {outcome}, local now '{final}' -> {local_results_dir}")
    if error:
        print(f"[run] re-run the same command to resume from where it stopped.")
    return outcome


def resolve_pkg_root() -> Path:
    """training_scripts/ in both environments: /root/training_scripts on Modal,
    this file's parent locally. Used by each experiment's train.py."""
    for cand in (Path("/root/training_scripts"), PKG_DIR):
        if (cand / "shared_code.py").exists():
            return cand
    return PKG_DIR


def set_seed(seed: int, deterministic: bool = True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
