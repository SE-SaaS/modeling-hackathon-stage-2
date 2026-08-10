"""
evaluate.py — score a lights model + a signs model on the REAL objective, on
full-resolution images, and time it on CPU.

Everything up to now has been ranked on Ultralytics' tile-level mAP50. That is
the wrong number twice over:
  * it scores 912x912 TILES, not the full frames the organisers score;
  * it matches lights by IoU 0.5, while the competition matches them by CENTRE
    DISTANCE within max(4px, 0.5*diag) — far more forgiving on a 10px box.

So this runs the actual pipeline (route -> crop -> tile -> detect -> map back ->
merge -> NMS) and reports:

    Score = 0.5 * F1_lights + 0.5 * mAP50_signs

plus per-frame latency (p50/p95) measured single-threaded-ish on CPU, since the
deadline is pass/fail and measured at p95.

    python training_scripts/evaluate.py --lights lights_t768_yellow4 --signs signs_640
    python training_scripts/evaluate.py --lights lights_t768_yellow4 --limit 100
    python training_scripts/evaluate.py --sweep-conf          # pick the threshold

Weights are read from <exp>/results/train/weights/best.pt, i.e. whatever
--fetch-only already pulled down.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import shared_code as sc

HERE = Path(__file__).resolve().parent
LIGHT_OUT_IDS = (8, 9, 10)      # model class 0,1,2 -> competition 8,9,10


# -----------------------------------------------------------------------------
# inference
# -----------------------------------------------------------------------------
def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Plain greedy NMS. Used to merge duplicate detections from the overlapping
    tile seams — a light near a seam is seen by two tiles."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = sc._iou_matrix(boxes[i][None, :], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thr]
    return keep


def predict_lights(model, img, cfg, conf: float, iou: float = 0.5):
    """Crop the band, run every tile, map boxes back to ORIGINAL image pixels.

    Tiles come from shared_code.tile_geometry — the same function that built the
    training data, so inference geometry cannot drift from training geometry.
    """
    H, W = img.shape[:2]
    band = tuple(cfg["dataset"].get("band", [0.30, 0.90]))
    tiles, _, _ = sc.tile_geometry(W, H, band, cfg["dataset"].get("n_tiles"))
    imgsz = cfg["train"]["imgsz"]

    crops = [img[y0:y1, x0:x1] for (x0, y0, x1, y1) in tiles]
    results = model.predict(crops, imgsz=imgsz, conf=conf, verbose=False,
                            device="cpu")

    boxes, scores, classes = [], [], []
    for (x0, y0, _, _), r in zip(tiles, results):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        xyxy[:, [0, 2]] += x0            # tile -> original coordinates
        xyxy[:, [1, 3]] += y0
        boxes.append(xyxy)
        scores.append(r.boxes.conf.cpu().numpy())
        classes.append(r.boxes.cls.cpu().numpy().astype(int))

    if not boxes:
        return []
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    classes = np.concatenate(classes)

    # Merge across tile seams, per class.
    out = []
    for c in np.unique(classes):
        m = classes == c
        for i in _nms(boxes[m], scores[m], iou):
            b = boxes[m][i]
            out.append((LIGHT_OUT_IDS[int(c)], float(scores[m][i]),
                        float(b[0]), float(b[1]), float(b[2]), float(b[3])))
    return out


def predict_signs(model, img, cfg, conf: float):
    """Signs are large; one pass over the whole frame, classes already 0..7."""
    r = model.predict(img, imgsz=cfg["train"]["imgsz"], conf=conf,
                      verbose=False, device="cpu")[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    return [(int(c), float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            for b, s, c in zip(xyxy, r.boxes.conf.cpu().numpy(),
                               r.boxes.cls.cpu().numpy().astype(int))]


# -----------------------------------------------------------------------------
def load_ground_truth(data_root: Path, split: str):
    """{image_id: [(cls, x1,y1,x2,y2), ...]} in original pixels."""
    from PIL import Image
    gts, sizes = {}, {}
    img_dir, lbl_dir = data_root / split / "images", data_root / split / "labels"
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(p) as im:
            W, H = im.size
        sizes[p.stem] = (p, W, H)
        boxes = []
        for c, cx, cy, w, h in sc._read_labels(lbl_dir / (p.stem + ".txt")):
            boxes.append((c, *sc._yolo_to_xyxy(cx, cy, w, h, W, H)))
        gts[p.stem] = boxes
    return gts, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lights", default="lights_t768_yellow4")
    ap.add_argument("--signs", default="signs_640")
    ap.add_argument("--data", default=None, help="dataset_release root")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0, help="first N images only")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--sweep-conf", action="store_true",
                    help="re-score at several thresholds (predictions reused)")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads; the organisers give 'a few'")
    args = ap.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)

    lcfg = sc.load_config(HERE / args.lights)
    scfg = sc.load_config(HERE / args.signs)
    data_root = Path(args.data or lcfg["paths"]["data_root"])

    def weights(exp):
        w = HERE / exp / "results" / "train" / "weights" / "best.pt"
        if not w.exists():
            raise SystemExit(f"missing weights: {w}\n"
                             f"fetch with: modal run problem4/training_scripts/"
                             f"{exp}/train.py --fetch-only")
        return str(w)

    print(f"lights: {args.lights}  imgsz={lcfg['train']['imgsz']} "
          f"band={lcfg['dataset'].get('band')}")
    print(f"signs : {args.signs}   imgsz={scfg['train']['imgsz']}")
    lmodel, smodel = YOLO(weights(args.lights)), YOLO(weights(args.signs))

    gts, sizes = load_ground_truth(data_root, args.split)
    ids = sorted(sizes)
    if args.limit:
        ids = ids[:args.limit]
    print(f"{len(ids)} {args.split} images, conf={args.conf}, "
          f"threads={args.threads}\n")

    min_ar = float(lcfg["dataset"].get("router_min_ar", 1.6))
    preds, times, routed = {}, [], {"lights": 0, "signs": 0}

    for k, img_id in enumerate(ids):
        path, W, H = sizes[img_id]
        img = cv2.imread(str(path))            # disk read is NOT timed
        if img is None:
            continue
        t0 = time.perf_counter()
        if sc._is_lights_image(W, H, min_ar):
            out = predict_lights(lmodel, img, lcfg, args.conf)
            routed["lights"] += 1
        else:
            out = predict_signs(smodel, img, scfg, args.conf)
            routed["signs"] += 1
        times.append((time.perf_counter() - t0) * 1000)
        preds[img_id] = out
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(ids)}  p50={np.median(times):.0f}ms", flush=True)

    gts = {k: v for k, v in gts.items() if k in preds}
    res = sc.competition_score(preds, gts)

    print("\n" + "=" * 64)
    print(f"SCORE = {res['score']:.4f}"
          f"   (0.5*F1_lights {res['f1_lights']['mean']:.4f}"
          f" + 0.5*mAP50_signs {res['map50_signs']['mean']:.4f})")
    print("=" * 64)
    print("F1 per light class (centre-distance matching):")
    for c, nm in zip(sc.LIGHT_IDS, ("red", "yellow", "green")):
        print(f"   {c} {nm:<7} {res['f1_lights'][c]:.4f}")
    print("AP50 per sign class:")
    for c in sc.SIGN_IDS:
        if c in res["map50_signs"]:
            print(f"   {c} {sc.CLASS_NAMES[c]:<20} {res['map50_signs'][c]:.4f}")

    t = np.array(times)
    print(f"\nLatency over {len(t)} frames (CPU, {args.threads} threads), "
          f"image load excluded:")
    print(f"   p50 {np.median(t):6.1f} ms | p95 {np.percentile(t,95):6.1f} ms "
          f"| max {t.max():6.1f} ms")
    print(f"   routed: {routed['lights']} lights frames, {routed['signs']} signs")
    print(f"   DEADLINE 150ms p95: "
          f"{'PASS' if np.percentile(t,95) <= 150 else 'FAIL'}")

    if args.sweep_conf:
        # Confidence is not a cut-off in the metric — every submitted box counts
        # as a detection — so the threshold is a real tunable and worth sweeping.
        print("\nconf sweep (re-filtering the same predictions):")
        print(f"   {'conf':>6} {'score':>7} {'F1_l':>7} {'mAP_s':>7}")
        for c in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            p = {k: [b for b in v if b[1] >= c] for k, v in preds.items()}
            r = sc.competition_score(p, gts)
            print(f"   {c:6.2f} {r['score']:7.4f} {r['f1_lights']['mean']:7.4f} "
                  f"{r['map50_signs']['mean']:7.4f}")

    out_path = HERE / f"_eval_{args.lights}__{args.signs}.json"
    out_path.write_text(json.dumps(
        {"lights": args.lights, "signs": args.signs, "conf": args.conf,
         "n_images": len(preds), "score": res["score"],
         "f1_lights": {str(k): v for k, v in res["f1_lights"].items()},
         "map50_signs": {str(k): v for k, v in res["map50_signs"].items()},
         "latency_ms": {"p50": float(np.median(t)),
                        "p95": float(np.percentile(t, 95)),
                        "max": float(t.max())}}, indent=2))
    print(f"\nwrote {out_path.name}")


if __name__ == "__main__":
    main()
