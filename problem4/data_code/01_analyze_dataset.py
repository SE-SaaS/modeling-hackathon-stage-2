"""
01_analyze_dataset.py — dataset reconnaissance for "Too Small to Handle".

Answers the questions that decide the pipeline geometry:
  1. What are the real image resolutions? (decides crop input size)
  2. How big are objects in PIXELS, per class? (confirms the 12-px problem)
  3. Where do lights sit vertically/horizontally? (decides the ROI crop band)
  4. How many objects per image, class balance, empty images?
  5. What crop band captures ~100% of lights, and what does it buy us?

Writes a plain-text report next to this script: 01_dataset_analysis.txt

Usage:
    python 01_analyze_dataset.py [--data <path to dataset_release>]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_DATA = Path(r"C:\Users\mamou\Downloads\too-small-to-handle\dataset_release\dataset_release")

CLASS_NAMES = {
    0: "Bus stop", 1: "Crossroad", 2: "No entry", 3: "No parking",
    4: "No stopping", 5: "Speed limit", 6: "Yield", 7: "direction of road",
    8: "red light", 9: "yellow light", 10: "green light",
}
SIGN_IDS = list(range(0, 8))
LIGHT_IDS = [8, 9, 10]

# Candidate ROI bands (y_top, y_bottom) in normalized coords, tested for light coverage.
# NOTE: lights sit LOW in these frames (median cy ~0.64 train / ~0.72 val), not high.
CANDIDATE_BANDS = [
    (0.00, 1.00),
    (0.30, 0.90), (0.35, 0.90), (0.40, 0.90), (0.45, 0.90),
    (0.40, 0.85), (0.45, 0.85), (0.50, 0.85), (0.45, 0.80), (0.50, 0.80),
    (0.30, 1.00), (0.40, 1.00), (0.50, 1.00),
    (0.20, 0.95), (0.35, 0.95),
]

# Candidate input tensors (W, H) for the lights branch. Non-square on purpose:
# a vertical crop of a 16:9 frame is very wide and short, so a square tensor wastes
# most of its pixels on padding.
CANDIDATE_TENSORS = [(960, 960), (1280, 1280), (1280, 384), (1600, 448), (1920, 448), (1920, 640)]


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def fmt_stats(name, values, unit="", nd=2):
    """One-line summary: n, min, p1, p25, median, p75, p99, max, mean."""
    if len(values) == 0:
        return f"{name:<22} n=0"
    v = np.asarray(values, dtype=float)
    return (
        f"{name:<22} n={len(v):<6d} "
        f"min={v.min():.{nd}f} p1={pct(v,1):.{nd}f} p25={pct(v,25):.{nd}f} "
        f"med={np.median(v):.{nd}f} p75={pct(v,75):.{nd}f} p99={pct(v,99):.{nd}f} "
        f"max={v.max():.{nd}f} mean={v.mean():.{nd}f}{unit}"
    )


def load_split(split_dir: Path):
    """Return per-object records and per-image records for one split.

    Reads image size from the JPEG header only (no pixel decode) — fast.
    """
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    objects = []   # dicts: cls, cx, cy, w, h (normalized), W, H, wpx, hpx
    images = []    # dicts: stem, W, H, n_obj, n_signs, n_lights

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(img_path) as im:
            W, H = im.size

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        rows = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                rows.append((c, cx, cy, w, h))

        n_signs = sum(1 for r in rows if r[0] in SIGN_IDS)
        n_lights = sum(1 for r in rows if r[0] in LIGHT_IDS)
        images.append(dict(stem=img_path.stem, W=W, H=H, n_obj=len(rows),
                           n_signs=n_signs, n_lights=n_lights,
                           classes={r[0] for r in rows}))

        for c, cx, cy, w, h in rows:
            objects.append(dict(cls=c, cx=cx, cy=cy, w=w, h=h, W=W, H=H,
                                wpx=w * W, hpx=h * H))
    return objects, images


def band_coverage(light_objs, y0, y1, margin=0.01):
    """Fraction of lights whose FULL box (plus margin) fits inside [y0, y1]."""
    if not light_objs:
        return 0.0
    inside = sum(
        1 for o in light_objs
        if (o["cy"] - o["h"] / 2) >= y0 - margin and (o["cy"] + o["h"] / 2) <= y1 + margin
    )
    return inside / len(light_objs)


def report(out, data_root: Path):
    W = out.write

    def hdr(title):
        W("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78 + "\n")

    all_objects, all_images = {}, {}
    for split in ("train", "val"):
        d = data_root / split
        if not d.exists():
            continue
        all_objects[split], all_images[split] = load_split(d)

    hdr("0. SPLIT SUMMARY")
    for split in all_images:
        imgs, objs = all_images[split], all_objects[split]
        empties = sum(1 for i in imgs if i["n_obj"] == 0)
        W(f"{split:<6} images={len(imgs):<6d} objects={len(objs):<6d} "
          f"empty_images={empties} ({empties/max(len(imgs),1):.1%})\n")

    # ---------------------------------------------------------------- 1. resolution
    hdr("1. IMAGE RESOLUTION  (decides how many pixels a 0.0043-wide light really is)")
    for split in all_images:
        imgs = all_images[split]
        sizes = Counter((i["W"], i["H"]) for i in imgs)
        W(f"\n[{split}] {len(sizes)} distinct resolution(s):\n")
        for (w, h), n in sizes.most_common():
            W(f"    {w:>5d} x {h:<5d}  ar={w/h:.3f}   {n:>5d} images ({n/len(imgs):.1%})\n")
        W("  " + fmt_stats("width  (px)", [i["W"] for i in imgs], nd=0) + "\n")
        W("  " + fmt_stats("height (px)", [i["H"] for i in imgs], nd=0) + "\n")

    # ---------------------------------------------------------------- 2. class counts
    hdr("2. CLASS COUNTS")
    W(f"{'id':>3} {'name':<20} {'train obj':>10} {'val obj':>10}\n")
    for c in range(11):
        tr = sum(1 for o in all_objects.get("train", []) if o["cls"] == c)
        va = sum(1 for o in all_objects.get("val", []) if o["cls"] == c)
        W(f"{c:>3} {CLASS_NAMES[c]:<20} {tr:>10d} {va:>10d}\n")

    W("\nImages containing at least one instance of each class "
      "(this, not object count, is how many training scenes you actually have):\n")
    W(f"{'id':>3} {'name':<20} {'train imgs':>11} {'val imgs':>10}\n")
    for c in range(11):
        row = [c, CLASS_NAMES[c]]
        for split in ("train", "val"):
            n = sum(1 for im in all_images.get(split, []) if c in im["classes"])
            row.append(n)
        W(f"{row[0]:>3} {row[1]:<20} {row[2]:>11d} {row[3]:>10d}\n")

    # ---------------------------------------------------------------- 3. object sizes
    hdr("3. OBJECT SIZE IN PIXELS  (the core of the problem)")
    for split in all_objects:
        W(f"\n[{split}] per-class box width in PIXELS of the original image\n")
        for c in range(11):
            wpx = [o["wpx"] for o in all_objects[split] if o["cls"] == c]
            W("  " + fmt_stats(f"{c} {CLASS_NAMES[c]}", wpx, nd=1) + "\n")

        W(f"\n[{split}] per-class box width NORMALIZED (fraction of image width)\n")
        for c in range(11):
            wn = [o["w"] for o in all_objects[split] if o["cls"] == c]
            W("  " + fmt_stats(f"{c} {CLASS_NAMES[c]}", wn, nd=5) + "\n")

        W(f"\n[{split}] family aggregate\n")
        for fam, ids in (("SIGNS 0-7", SIGN_IDS), ("LIGHTS 8-10", LIGHT_IDS)):
            wpx = [o["wpx"] for o in all_objects[split] if o["cls"] in ids]
            hpx = [o["hpx"] for o in all_objects[split] if o["cls"] in ids]
            wn = [o["w"] for o in all_objects[split] if o["cls"] in ids]
            W("  " + fmt_stats(f"{fam} w(px)", wpx, nd=1) + "\n")
            W("  " + fmt_stats(f"{fam} h(px)", hpx, nd=1) + "\n")
            W("  " + fmt_stats(f"{fam} w(norm)", wn, nd=5) + "\n")
            ar = [o["wpx"] / o["hpx"] for o in all_objects[split]
                  if o["cls"] in ids and o["hpx"] > 0]
            W("  " + fmt_stats(f"{fam} aspect w/h", ar, nd=3) + "\n")

    # ------------------------------------------- 4. what survives a naive resize
    hdr("4. WHAT A LIGHT LOOKS LIKE AFTER RESIZE  (letterbox to square, long side = S)")
    W("Effective pixel width of a light box when the FULL frame is resized to SxS.\n")
    W("Detection needs roughly >= 8 px for a stride-8 head, >= 4-6 px with a P2 (stride-4) head.\n\n")
    for split in all_objects:
        lights = [o for o in all_objects[split] if o["cls"] in LIGHT_IDS]
        if not lights:
            continue
        W(f"[{split}] n_lights={len(lights)}\n")
        W(f"  {'input S':>8} {'median px':>10} {'p25':>8} {'p10':>8} {'>=4px':>8} {'>=6px':>8} {'>=8px':>8}\n")
        for S in (416, 640, 960, 1280, 1600, 1920, 2560):
            # letterbox: scale = S / max(W, H); box px width = w_norm * W * scale
            eff = np.array([o["w"] * o["W"] * (S / max(o["W"], o["H"])) for o in lights])
            W(f"  {S:>8d} {np.median(eff):>10.2f} {pct(eff,25):>8.2f} {pct(eff,10):>8.2f} "
              f"{(eff>=4).mean():>7.1%} {(eff>=6).mean():>7.1%} {(eff>=8).mean():>7.1%}\n")
        W("\n")

    # ---------------------------------------------------------------- 5. spatial prior
    hdr("5. SPATIAL DISTRIBUTION  (can we throw away most of the frame?)")
    for split in all_objects:
        for fam, ids in (("LIGHTS 8-10", LIGHT_IDS), ("SIGNS 0-7", SIGN_IDS)):
            objs = [o for o in all_objects[split] if o["cls"] in ids]
            if not objs:
                continue
            W(f"\n[{split}] {fam}\n")
            W("  " + fmt_stats("cy (norm)", [o["cy"] for o in objs], nd=4) + "\n")
            W("  " + fmt_stats("cx (norm)", [o["cx"] for o in objs], nd=4) + "\n")
            W("  " + fmt_stats("box top y", [o["cy"] - o["h"] / 2 for o in objs], nd=4) + "\n")
            W("  " + fmt_stats("box bot y", [o["cy"] + o["h"] / 2 for o in objs], nd=4) + "\n")
            # vertical histogram in 10% bands
            cys = np.array([o["cy"] for o in objs])
            W("  cy histogram (deciles):\n")
            for k in range(10):
                lo, hi = k / 10, (k + 1) / 10
                n = int(((cys >= lo) & (cys < hi)).sum())
                bar = "#" * int(60 * n / max(len(cys), 1))
                W(f"    {lo:.1f}-{hi:.1f} {n:>6d} {n/len(cys):>6.1%} {bar}\n")

    # ---------------------------------------------------------------- 6. ROI band choice
    hdr("6. ROI CROP BAND + INPUT TENSOR FOR THE LIGHTS MODEL")
    W("Crop = full image width x band [y0,y1] of the height, then FIT (aspect-preserving,\n")
    W("letterbox) into the input tensor TWxTH. scale = min(TW/crop_w, TH/crop_h).\n")
    W("Cropping vertically does NOT raise resolution on its own for these wide frames --\n")
    W("width is the limiting dimension. It only helps if the tensor is also wide and short,\n")
    W("so the same TW covers fewer source pixels of height. What actually buys resolution\n")
    W("is a larger TW, and the crop is what makes a large TW affordable.\n")
    W("'cov' = fraction of light boxes fully inside the band.\n")
    W("'px' = median light width in tensor pixels. 'MPix' = tensor area, the cost proxy.\n\n")
    for split in all_objects:
        lights = [o for o in all_objects[split] if o["cls"] in LIGHT_IDS]
        signs = [o for o in all_objects[split] if o["cls"] in SIGN_IDS]
        if not lights:
            continue
        W(f"[{split}] light-box coverage per band\n")
        W(f"  {'band':>14} {'light cov':>10} {'sign cov':>10} {'band height':>12}\n")
        for y0, y1 in CANDIDATE_BANDS:
            W(f"  [{y0:.2f},{y1:.2f}] {band_coverage(lights, y0, y1):>10.2%} "
              f"{band_coverage(signs, y0, y1):>10.2%} {y1-y0:>12.2f}\n")

        W(f"\n[{split}] median light px in the tensor, band x tensor grid "
          f"(only bands with >=95% light coverage are worth using)\n")
        header = "  " + f"{'band':>14}" + "".join(
            f"{f'{tw}x{th}':>12}" for tw, th in CANDIDATE_TENSORS) + "\n"
        W(header)
        for y0, y1 in CANDIDATE_BANDS:
            cov = band_coverage(lights, y0, y1)
            if cov < 0.90:
                continue
            row = f"  [{y0:.2f},{y1:.2f}]"
            for tw, th in CANDIDATE_TENSORS:
                eff = []
                for o in lights:
                    cw, ch = o["W"], (y1 - y0) * o["H"]
                    s = min(tw / cw, th / ch)
                    eff.append(o["w"] * o["W"] * s)
                row += f"{np.median(eff):>12.2f}"
            W(row + f"   (cov {cov:.1%})\n")
        W("  " + " " * 14 + "".join(
            f"{tw*th/1e6:>11.2f}M" for tw, th in CANDIDATE_TENSORS) + "  <- tensor MPix\n")
        W("\n")

    # ---------------------------------------------------------------- 7. co-occurrence
    hdr("7. PER-IMAGE OBJECT COUNTS")
    for split in all_images:
        imgs = all_images[split]
        W(f"\n[{split}]\n")
        W("  " + fmt_stats("objects / image", [i["n_obj"] for i in imgs], nd=2) + "\n")
        W("  " + fmt_stats("lights / image", [i["n_lights"] for i in imgs], nd=2) + "\n")
        W("  " + fmt_stats("signs  / image", [i["n_signs"] for i in imgs], nd=2) + "\n")
        both = sum(1 for i in imgs if i["n_lights"] > 0 and i["n_signs"] > 0)
        only_l = sum(1 for i in imgs if i["n_lights"] > 0 and i["n_signs"] == 0)
        only_s = sum(1 for i in imgs if i["n_lights"] == 0 and i["n_signs"] > 0)
        none = sum(1 for i in imgs if i["n_obj"] == 0)
        W(f"  images with lights+signs = {both}\n")
        W(f"  images with lights only  = {only_l}\n")
        W(f"  images with signs only   = {only_s}\n")
        W(f"  images with nothing      = {none}\n")
        W(f"  max lights in one image  = {max(i['n_lights'] for i in imgs)}\n")
        W(f"  max signs  in one image  = {max(i['n_signs'] for i in imgs)}\n")

    hdr("7b. IMAGE-TYPE SEPARATION  (do lights and signs live in different images?)")
    W("If they never co-occur, a cheap router (aspect ratio / resolution) can send each\n")
    W("image to ONE branch instead of running both. That halves the per-frame cost.\n")
    W("Risk: this is a property of THIS data collection, and the private test set may\n")
    W("not honour it. Treat as a speed optimisation with a safe fallback, not a hard rule.\n")
    for split in all_images:
        imgs = all_images[split]
        W(f"\n[{split}] breakdown by resolution -> which family appears\n")
        W(f"  {'resolution':>14} {'ar':>7} {'images':>8} {'light imgs':>11} "
          f"{'sign imgs':>10} {'both':>6}\n")
        by_res = defaultdict(list)
        for i in imgs:
            by_res[(i["W"], i["H"])].append(i)
        for (w, h), group in sorted(by_res.items(), key=lambda kv: -len(kv[1])):
            nl = sum(1 for i in group if i["n_lights"] > 0)
            ns = sum(1 for i in group if i["n_signs"] > 0)
            nb = sum(1 for i in group if i["n_lights"] > 0 and i["n_signs"] > 0)
            W(f"  {w:>6d}x{h:<7d} {w/h:>7.3f} {len(group):>8d} {nl:>11d} {ns:>10d} {nb:>6d}\n")

        W(f"\n[{split}] router test: landscape (ar > 1.6) -> lights, else -> signs\n")
        land = [i for i in imgs if i["W"] / i["H"] > 1.6]
        port = [i for i in imgs if i["W"] / i["H"] <= 1.6]
        for label, group in (("ar>1.6 ", land), ("ar<=1.6", port)):
            if not group:
                continue
            nl = sum(1 for i in group if i["n_lights"] > 0)
            ns = sum(1 for i in group if i["n_signs"] > 0)
            ol = sum(i["n_lights"] for i in group)
            os_ = sum(i["n_signs"] for i in group)
            W(f"  {label}: {len(group):>5d} imgs | imgs w/ lights={nl:<5d} "
              f"imgs w/ signs={ns:<5d} | light objs={ol:<5d} sign objs={os_}\n")
        W("  -> objects a perfect ar-router would MISS if each image ran only one branch: "
          f"lights={sum(i['n_lights'] for i in port)}, signs={sum(i['n_signs'] for i in land)}\n")

    hdr("8. DERIVED RECOMMENDATIONS (mechanical, read section 4 and 6 to check them)")
    tr_lights = [o for o in all_objects.get("train", []) if o["cls"] in LIGHT_IDS]
    if tr_lights:
        med_px = np.median([o["wpx"] for o in tr_lights])
        W(f"- Median light width in original pixels: {med_px:.1f}\n")
        for S in (640, 960, 1280):
            eff = np.median([o["w"] * o["W"] * (S / max(o["W"], o["H"])) for o in tr_lights])
            W(f"- Full-frame resize to {S}: median light becomes {eff:.2f} px "
              f"({'unusable' if eff < 4 else 'marginal' if eff < 8 else 'ok'})\n")
        best = max(CANDIDATE_BANDS, key=lambda b: (band_coverage(tr_lights, *b), -(b[1] - b[0])))
        W(f"- Tightest band with best light coverage among candidates: {best} "
          f"-> {band_coverage(tr_lights, *best):.2%}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).with_name("01_dataset_analysis.txt"))
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"data root not found: {args.data}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"Dataset analysis for: {args.data}\n")
        report(f, args.data)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
