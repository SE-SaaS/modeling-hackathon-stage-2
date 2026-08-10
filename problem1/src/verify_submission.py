#!/usr/bin/env python3
"""
Pre-upload gate for submission.csv.

Every check is fatal: a hackathon submission that is silently malformed
(short by a row, a stray NaN, renamed columns) scores zero and there is no
second attempt. Exits non-zero if ANY check fails, so it can gate an upload.

    python verify_submission.py --submission artifacts/submission.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

LABELS = ["angry", "happy", "sad", "fearful", "disgust", "neutral"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--submission", default="artifacts/submission.csv")
    p.add_argument("--sample", default="Dataset/sample_submission.csv")
    p.add_argument("--copy_to", default="submission.csv",
                   help="final resting place the user uploads from")
    args = p.parse_args()

    sub_path, sample_path = Path(args.submission), Path(args.sample)
    if not sub_path.is_file():
        print(f"FAIL: no submission at {sub_path}")
        return 1

    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_path)
    fails: list[str] = []

    print("=" * 64)
    print(f"VERIFYING {sub_path}")
    print("=" * 64)

    # 1. row count
    ok = len(sub) == len(sample)
    print(f"[{'OK ' if ok else 'FAIL'}] row count: {len(sub)} (expect {len(sample)})")
    if not ok:
        fails.append("row count")

    # 2. column names identical to the sample, in the same order
    ok = list(sub.columns) == list(sample.columns)
    print(f"[{'OK ' if ok else 'FAIL'}] columns: {list(sub.columns)} "
          f"(expect {list(sample.columns)})")
    if not ok:
        fails.append("columns")

    # 3. every test file_id present exactly once
    want = set(sample["file_id"].astype(str))
    got = sub["file_id"].astype(str)
    missing, extra = want - set(got), set(got) - want
    dupes = got[got.duplicated()].tolist()
    ok = not missing and not extra and not dupes
    print(f"[{'OK ' if ok else 'FAIL'}] file_ids: {len(want)} expected, "
          f"{missing and len(missing) or 0} missing, "
          f"{extra and len(extra) or 0} unexpected, {len(dupes)} duplicated")
    if missing:
        print(f"        missing e.g. {sorted(missing)[:3]}")
    if dupes:
        print(f"        duplicated e.g. {dupes[:3]}")
    if not ok:
        fails.append("file_ids")

    # 4. no NaNs anywhere
    n_nan = int(sub.isna().sum().sum())
    print(f"[{'OK ' if not n_nan else 'FAIL'}] NaNs: {n_nan}")
    if n_nan:
        fails.append("NaNs")

    # 5. labels are integers in 0..5
    lab = sub["label"]
    numeric = pd.to_numeric(lab, errors="coerce")
    bad = int(numeric.isna().sum() + (~numeric.dropna().between(0, 5)).sum())
    # dtype-agnostic: works whether the column parsed as int64 or float64
    vals = numeric.dropna()
    is_int = bool(len(vals)) and bool((vals % 1 == 0).all())
    ok = bad == 0 and is_int
    print(f"[{'OK ' if ok else 'FAIL'}] labels in 0-5 and integral: "
          f"{bad} out-of-range/non-numeric, integral={is_int}")
    if not ok:
        fails.append("label range")

    # 6. row ORDER matches the sample submission exactly
    ok = got.tolist() == sample["file_id"].astype(str).tolist()
    print(f"[{'OK ' if ok else 'WARN'}] row order matches sample_submission: {ok}")

    # distribution — not pass/fail, but a collapsed class is a loud smell
    print("\npredicted class distribution:")
    vc = lab.value_counts().sort_index()
    for cls, n in vc.items():
        name = LABELS[int(cls)] if 0 <= int(cls) < len(LABELS) else "?"
        print(f"  {int(cls)} {name:<8} {n:>5}  ({n / len(sub) * 100:5.1f}%)")
    if len(vc) < len(LABELS):
        print(f"  [WARN] only {len(vc)}/{len(LABELS)} classes predicted - "
              f"model may have collapsed")

    print("\n" + "=" * 64)
    if fails:
        print(f"RESULT: FAILED ({', '.join(fails)}) — DO NOT UPLOAD")
        return 1

    dest = Path(args.copy_to).resolve()
    shutil.copyfile(sub_path, dest)
    print("RESULT: ALL CHECKS PASSED")
    print(f"UPLOAD THIS FILE:\n  {dest}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
