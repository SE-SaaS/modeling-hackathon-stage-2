"""
make_submission.py — write submission.csv for the practice leaderboard.

    python training_scripts/make_submission.py --lights lights_t768_yellow4 \
        --signs signs_640 --conf 0.40 --out submission.csv

--signs may be omitted while no signs model has finished; those frames then get
"none" and the signs half of the score is 0. The lights half still scores.

Per the rules an image with no detections is the word "none", never a blank
cell — Kaggle reads blanks as nulls and rejects the whole file.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import shared_code as sc
from evaluate import predict_lights, predict_signs

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lights", default="lights_t768_yellow4")
    ap.add_argument("--signs", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import cv2
    import torch
    from PIL import Image
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)

    lcfg = sc.load_config(HERE / args.lights)
    data_root = Path(args.data or lcfg["paths"]["data_root"])
    img_dir = data_root / args.split / "images"

    def weights(exp):
        return str(HERE / exp / "results" / "train" / "weights" / "best.pt")

    lmodel = YOLO(weights(args.lights))
    scfg = smodel = None
    if args.signs:
        scfg = sc.load_config(HERE / args.signs)
        smodel = YOLO(weights(args.signs))

    min_ar = float(lcfg["dataset"].get("router_min_ar", 1.6))
    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    rows, t0 = [], time.perf_counter()
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            W, H = im.size
        img = cv2.imread(str(p))
        dets = []
        if img is not None:
            if sc._is_lights_image(W, H, min_ar):
                dets = predict_lights(lmodel, img, lcfg, args.conf)
            elif smodel is not None:
                dets = predict_signs(smodel, img, scfg, args.conf)

        # class_id confidence x1 y1 x2 y2, space separated, pixels of the original
        parts = []
        for c, s, x1, y1, x2, y2 in dets:
            parts += [str(int(c)), f"{s:.4f}", f"{x1:.1f}", f"{y1:.1f}",
                      f"{x2:.1f}", f"{y2:.1f}"]
        rows.append((p.stem, " ".join(parts) if parts else "none"))

        if (i + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1}/{len(paths)}  {el:.0f}s elapsed, "
                  f"eta {el/(i+1)*(len(paths)-i-1):.0f}s", flush=True)

    out = Path(args.out)
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("image_id,PredictionString\n")
        for stem, s in rows:
            f.write(f"{stem},{s}\n")

    n_det = sum(1 for _, s in rows if s != "none")
    print(f"\nwrote {out}  ({len(rows)} rows, {n_det} with detections, "
          f"{len(rows)-n_det} 'none') in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
