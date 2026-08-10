#!/usr/bin/env python3
"""
Verify whether TRAIN and TEST share speakers, before trusting any split strategy.

Runs four checks:

  1. EXACT DUPLICATES   md5 of raw PCM, train vs test. CPU, ~2 min.
  2. CALIBRATION        same-speaker vs different-speaker cosine distributions,
                        derived from the train pseudo-clusters. Gives a threshold
                        grounded in THIS data, not borrowed from a paper.
  3. OVERLAP            for each test clip, max cosine to any train clip.
                        Where that sits relative to (2) is the answer.
  4. TEST STRUCTURE     cluster the test set alone. Under the CREMA-D hypothesis
                        expect ~21 clusters of ~82 clips, and nothing near the
                        1200-clip TESS signature.

Requires speakers.csv from cluster_speakers.py, and reuses its helpers.

Usage
-----
  # on GPU (Modal) — embeds test clips, caches, then reports
  python verify_test_speakers.py --data_root ./Dataset --speakers speakers.csv

  # locally afterwards — reuses both caches, CPU only
  python verify_test_speakers.py --data_root ./Dataset --speakers speakers.csv
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from cluster_speakers import (
    cluster,
    index_audio,
    l2norm,
    embed_all,
)

RNG = np.random.RandomState(0)


# --------------------------------------------------------------------------- #
# 1. exact duplicate detection
# --------------------------------------------------------------------------- #


def pcm_md5(path: Path) -> str:
    data, _ = sf.read(str(path), dtype="int16", always_2d=False)
    return hashlib.md5(np.ascontiguousarray(data).tobytes()).hexdigest()


def check_duplicates(train_paths: dict[str, Path], test_paths: dict[str, Path]) -> None:
    print("\n===== 1. EXACT DUPLICATE CHECK =====")
    train_hash: dict[str, str] = {}
    for fid, p in train_paths.items():
        train_hash.setdefault(pcm_md5(p), fid)

    hits = []
    for fid, p in test_paths.items():
        h = pcm_md5(p)
        if h in train_hash:
            hits.append((fid, train_hash[h]))

    if hits:
        print(f"  !! {len(hits)} test clips are byte-identical to a train clip")
        for t, tr in hits[:10]:
            print(f"     {t}  ==  {tr}")
        print("  The split is broken. Everything below is secondary.")
    else:
        print(f"  clean — no byte-identical clips across {len(test_paths)} test files")

    # also flag duplicates within train, which inflate any CV
    inner = len(train_paths) - len(train_hash)
    if inner:
        print(f"  note: {inner} duplicate clips WITHIN train (inflates all CV numbers)")


# --------------------------------------------------------------------------- #
# 2. calibration from train pseudo-clusters
# --------------------------------------------------------------------------- #


def calibrate(emb: np.ndarray, speaker: np.ndarray, n_pairs: int = 200_000):
    """Sample same-speaker and different-speaker cosine similarities."""
    n = len(emb)
    by_spk: dict[str, np.ndarray] = {
        s: np.where(speaker == s)[0] for s in np.unique(speaker)
    }
    eligible = [s for s, idx in by_spk.items() if len(idx) >= 2]

    same = []
    for _ in range(n_pairs):
        s = eligible[RNG.randint(len(eligible))]
        i, j = RNG.choice(by_spk[s], size=2, replace=False)
        same.append(float(emb[i] @ emb[j]))

    diff = []
    for _ in range(n_pairs):
        i, j = RNG.randint(n), RNG.randint(n)
        if speaker[i] == speaker[j]:
            continue
        diff.append(float(emb[i] @ emb[j]))

    same_a, diff_a = np.array(same), np.array(diff)

    # equal error rate threshold
    grid = np.linspace(-0.2, 1.0, 601)
    fr = np.array([(same_a < t).mean() for t in grid])   # false reject
    fa = np.array([(diff_a >= t).mean() for t in grid])  # false accept
    k = int(np.argmin(np.abs(fr - fa)))
    thr, eer = float(grid[k]), float((fr[k] + fa[k]) / 2)

    print("\n===== 2. CALIBRATION (from train pseudo-speakers) =====")
    print(f"  same-speaker cosine : mean {same_a.mean():.3f}  p05 {np.percentile(same_a, 5):.3f}")
    print(f"  diff-speaker cosine : mean {diff_a.mean():.3f}  p95 {np.percentile(diff_a, 95):.3f}")
    print(f"  EER threshold       : {thr:.3f}   (EER {eer:.3f})")
    if eer > 0.20:
        print("  [warn] EER above 0.20 — clusters are noisy, treat check 3 as indicative only")
    return thr, same_a, diff_a


# --------------------------------------------------------------------------- #
# 3. test -> train overlap
# --------------------------------------------------------------------------- #


def check_overlap(
    test_emb: np.ndarray,
    train_emb: np.ndarray,
    train_speaker: np.ndarray,
    train_corpus: np.ndarray,
    thr: float,
    same_a: np.ndarray,
    chunk: int = 256,
) -> np.ndarray:
    print("\n===== 3. TEST -> TRAIN NEAREST NEIGHBOUR =====")
    best_sim = np.zeros(len(test_emb), dtype=np.float32)
    best_idx = np.zeros(len(test_emb), dtype=np.int64)

    for s in range(0, len(test_emb), chunk):
        sims = test_emb[s : s + chunk] @ train_emb.T
        best_sim[s : s + chunk] = sims.max(axis=1)
        best_idx[s : s + chunk] = sims.argmax(axis=1)

    above = best_sim >= thr
    same_p05 = float(np.percentile(same_a, 5))

    print(f"  max-similarity to train: mean {best_sim.mean():.3f}  "
          f"p50 {np.median(best_sim):.3f}  p95 {np.percentile(best_sim, 95):.3f}")
    print(f"  test clips above EER threshold ({thr:.3f}): "
          f"{int(above.sum())} / {len(test_emb)}  ({100 * above.mean():.1f}%)")
    print(f"  test clips above same-speaker p05 ({same_p05:.3f}): "
          f"{int((best_sim >= same_p05).sum())}")

    print("\n  nearest train clip's corpus (all test clips):")
    for corpus, c in Counter(train_corpus[best_idx]).most_common():
        print(f"    {corpus:<10} {c:>5}  ({100 * c / len(test_emb):.1f}%)")

    if above.mean() > 0.5:
        print("\n  VERDICT: most test clips match a train voice. The 'unseen speakers'")
        print("           claim does not hold — a looser split is defensible and you")
        print("           are currently leaving training data on the table.")
    elif above.mean() > 0.1:
        print("\n  VERDICT: partial overlap. Some test speakers appear in train.")
        print("           Keep the grouped split, but expect the CV gap to be small.")
    else:
        print("\n  VERDICT: test speakers are genuinely unseen. Grouped CV is correct")
        print("           and a random clip split would have been badly optimistic.")

    n_spk_hit = len(set(train_speaker[best_idx[above]])) if above.any() else 0
    print(f"  distinct train pseudo-speakers matched above threshold: {n_spk_hit}")
    return best_sim


# --------------------------------------------------------------------------- #
# 4. test-set internal structure
# --------------------------------------------------------------------------- #


def check_test_structure(test_emb: np.ndarray, n_clusters: int) -> None:
    print(f"\n===== 4. TEST STRUCTURE (clustering into {n_clusters}) =====")
    assign = cluster(test_emb, n_clusters)
    sizes = sorted(Counter(assign).values(), reverse=True)
    print(f"  cluster sizes: max {sizes[0]}  median {int(np.median(sizes))}  min {sizes[-1]}")
    print(f"  sizes: {sizes}")
    med = float(np.median(sizes))
    if sizes[0] > 600:
        print("  -> a >600 clip cluster suggests TESS is present in TEST too.")
        print("     That contradicts the CREMA-D-only hypothesis. Re-check the plan.")
    elif 60 <= med <= 110:
        print("  -> sizes consistent with CREMA-D actors (~82 clips each).")
        print("     Supports: test is held-out CREMA-D only; TESS/RAVDESS are train-only.")
    else:
        print("  -> sizes do not match any expected corpus signature. Sweep --test_clusters.")


# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./Dataset")
    p.add_argument("--train_csv", type=str, default="./Dataset/train.csv")
    p.add_argument("--speakers", type=str, default="speakers.csv")
    p.add_argument("--model_name", type=str, default="microsoft/wavlm-base-plus-sv")
    p.add_argument("--train_cache", type=str, default="xvectors.npy")
    p.add_argument("--test_cache", type=str, default="xvectors_test.npy")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_clusters", type=int, default=21)
    p.add_argument("--skip_dupes", action="store_true")
    p.add_argument("--durations_cache", type=str, default="durations.csv")
    p.add_argument("--io_workers", type=int, default=32)
    p.add_argument("--load_workers", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)
    index = index_audio(root)

    spk = pd.read_csv(args.speakers)
    spk["file_id"] = spk["file_id"].astype(str)
    train_ids = spk["file_id"].tolist()
    train_set = set(train_ids)
    test_ids = sorted(f for f in index if f not in train_set)
    print(f"train {len(train_ids)}   test {len(test_ids)}")

    if not args.skip_dupes:
        check_duplicates(
            {f: index[f] for f in train_ids}, {f: index[f] for f in test_ids}
        )

    train_emb = np.load(args.train_cache)
    if len(train_emb) != len(train_ids):
        raise ValueError("train cache length does not match speakers.csv")

    test_cache = Path(args.test_cache)
    if test_cache.is_file():
        test_emb = np.load(test_cache)
    else:
        test_df = pd.DataFrame(
            {"file_id": test_ids, "path": [str(index[f]) for f in test_ids]}
        )
        test_emb = embed_all(
            test_df, args.model_name, args.batch_size, device,
            io_workers=args.io_workers,
            load_workers=args.load_workers,
            durations_cache=args.durations_cache,
        )
        np.save(test_cache, test_emb)
        print(f"cached test embeddings -> {test_cache}")

    train_emb, test_emb = l2norm(train_emb), l2norm(test_emb)

    thr, same_a, _ = calibrate(train_emb, spk["pseudo_speaker"].values)
    check_overlap(
        test_emb,
        train_emb,
        spk["pseudo_speaker"].values,
        spk["corpus_guess"].values,
        thr,
        same_a,
    )
    check_test_structure(test_emb, args.test_clusters)


if __name__ == "__main__":
    main()
