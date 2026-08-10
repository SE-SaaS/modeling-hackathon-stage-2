"""
run.py — extract features, rank models by leave-one-case-out macro F1, and
sanity-check the winner against a permutation null.

    python problem5/run.py --data "C:/.../RF_ultrasound_dataset"
    python problem5/run.py --cache features.csv        # reuse extracted features
    python problem5/run.py --permutation               # add the null test

Feature extraction is the slow part, so it is cached to CSV; model selection
then takes seconds and can be iterated on freely.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import load_cohort            # noqa: E402
from src.features import build_feature_table  # noqa: E402
from src.model import (evaluate_loo, evaluate_repeated_cv,  # noqa: E402
                       model_zoo, permutation_check)

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = r"C:\Users\mamou\Downloads\RF_ultrasound_dataset (2)\RF_ultrasound_dataset"


def get_features(data_root, cache: Path, force: bool = False):
    if cache.exists() and not force:
        df = pd.read_csv(cache, index_col=0)
        y = df.pop("__label__").values
        print(f"loaded cached features: {df.shape[0]} cases x {df.shape[1]} features")
        return df, y, list(df.index)

    print(f"loading cohort from {data_root}")
    t0 = time.time()
    cases = load_cohort(data_root)
    print(f"  {len(cases)} cases in {time.time()-t0:.0f}s")

    print("extracting features...")
    t0 = time.time()
    X, y, ids = build_feature_table(cases)
    print(f"  {X.shape[0]} cases x {X.shape[1]} features in {time.time()-t0:.0f}s")

    out = X.copy()
    out["__label__"] = y
    out.to_csv(cache)
    print(f"  cached -> {cache}")
    return X, y, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--cache", default=str(HERE / "features.csv"))
    ap.add_argument("--force", action="store_true", help="re-extract features")
    ap.add_argument("--k", type=int, default=20, help="features kept inside each fold")
    ap.add_argument("--permutation", action="store_true")
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    X, y, ids = get_features(args.data, Path(args.cache), args.force)
    print(f"\nclass balance: {int((y==1).sum())} progressive / "
          f"{int((y==0).sum())} non-progressive\n")

    rows = []
    for name, pipe in model_zoo(args.k).items():
        t0 = time.time()
        loo = evaluate_loo(pipe, X.values, y, args.k)
        rep = evaluate_repeated_cv(pipe, X.values, y, k_features=args.k,
                                   n_repeats=10)
        rows.append({"model": name, **{k: v for k, v in loo.items()
                                       if k != "confusion"},
                     "rcv_f1_mean": rep["macro_f1_mean"],
                     "rcv_f1_std": rep["macro_f1_std"],
                     "sec": round(time.time() - t0, 1)})
        print(f"  {name:18s} LOO macroF1={loo['macro_f1']:.3f}  "
              f"bal_acc={loo['balanced_acc']:.3f}  auroc={loo['auroc']:.3f}  "
              f"sens={loo['sensitivity']:.2f} spec={loo['specificity']:.2f}  "
              f"| repCV {rep['macro_f1_mean']:.3f}+-{rep['macro_f1_std']:.3f}")

    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print("\n" + "=" * 78)
    print(res.to_string(index=False))
    res.to_csv(HERE / "results_models.csv", index=False)

    best_name = res.iloc[0]["model"]
    best = model_zoo(args.k)[best_name]
    print(f"\nbest: {best_name}")
    detail = evaluate_loo(best, X.values, y, args.k)
    print(f"  confusion [[TN,FP],[FN,TP]] = {detail['confusion']}")
    print(f"  f1 progressive={detail['f1_progressive']:.3f}  "
          f"non-progressive={detail['f1_nonprogressive']:.3f}")

    summary = {"best_model": best_name, "loo": detail,
               "n_cases": int(len(y)), "n_features": int(X.shape[1])}

    if args.permutation:
        # The result only means something if it beats shuffled labels. With 26
        # cases and hundreds of features this is not a formality.
        print(f"\npermutation test ({args.n_perm} shuffles)...")
        perm = permutation_check(best, X.values, y, args.n_perm, args.k)
        print(f"  real={perm['macro_f1']:.3f}  null mean={perm['null_mean']:.3f}  "
              f"null p95={perm['null_p95']:.3f}  p={perm['p_value']:.4f}")
        summary["permutation"] = perm

    (HERE / "results_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote results_models.csv + results_summary.json")


if __name__ == "__main__":
    main()
