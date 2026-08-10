"""
score_csv.py — score a submission CSV against the val labels with the real metric.

Used to confirm that the shipped ONNX/INT8 pipeline scores the same as the
PyTorch pipeline it was derived from. Quantization hurts small objects most, so
this check is not optional before shipping an int8 graph.

    python training_scripts/score_csv.py submission/sub_t4.csv
"""

import sys
from pathlib import Path

from PIL import Image

import shared_code as sc

DATA = Path(r"C:\Users\mamou\Downloads\too-small-to-handle\dataset_release\dataset_release")


def main():
    csv_path = Path(sys.argv[1])
    split = sys.argv[2] if len(sys.argv) > 2 else "val"

    preds = {}
    for line in csv_path.read_text(encoding="utf-8").splitlines()[1:]:
        img_id, _, ps = line.partition(",")
        v = ps.split()
        boxes = []
        for i in range(0, len(v) - 5, 6):
            c, s, x1, y1, x2, y2 = v[i:i + 6]
            boxes.append((int(c), float(s), float(x1), float(y1), float(x2), float(y2)))
        preds[img_id] = boxes

    gts = {}
    img_dir, lbl_dir = DATA / split / "images", DATA / split / "labels"
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"} or p.stem not in preds:
            continue
        with Image.open(p) as im:
            W, H = im.size
        gts[p.stem] = [(c, *sc._yolo_to_xyxy(cx, cy, w, h, W, H))
                       for c, cx, cy, w, h in sc._read_labels(lbl_dir / (p.stem + ".txt"))]

    res = sc.competition_score(preds, gts)
    print(f"{csv_path.name}: {len(preds)} images")
    print(f"  SCORE      {res['score']:.4f}")
    print(f"  F1_lights  {res['f1_lights']['mean']:.4f}   "
          f"red {res['f1_lights'][8]:.4f} yellow {res['f1_lights'][9]:.4f} "
          f"green {res['f1_lights'][10]:.4f}")
    print(f"  mAP50_signs {res['map50_signs']['mean']:.4f}")


if __name__ == "__main__":
    main()
