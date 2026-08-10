#!/usr/bin/env python3
"""
Recover pseudo-speaker groups for the 6-class SER challenge (no speaker IDs given).

Method
------
1. embed every TRAIN clip with WavLM-base-plus-sv (x-vector, 512-d)
2. L2-normalise, then subtract the per-class mean embedding
   -> removes the emotion direction, which otherwise splits one speaker's
      angry/fearful clips into their own label-pure cluster
3. agglomerative clustering, cosine distance, average linkage
4. print a cluster-size histogram and score it against the corpus hypothesis
   (CREMA-D + TESS + RAVDESS union)

Expected structure if the hypothesis holds (TRAIN ONLY, 9,170 clips):
    2 clusters of ~1200   TESS      (2 speakers, 2400 clips)
   24 clusters of ~44     RAVDESS   (24 speakers, 1056 clips)
  ~70 clusters of ~82     CREMA-D   (~70 speakers, 5714 clips)
  ------------------------------
   96 clusters total

Output: speakers.csv with columns file_id,pseudo_speaker,cluster_size,corpus_guess
Merge that into train.csv and ser_wavlm.py will group on it automatically.

Usage
-----
  python cluster_speakers.py --data_root ./Dataset --train_csv ./Dataset/train.csv \
      --out speakers.csv --n_clusters 96
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.cluster import AgglomerativeClustering
from transformers import WavLMForXVector

SAMPLE_RATE = 16_000
LABELS = ["angry", "happy", "sad", "fearful", "disgust", "neutral"]
LABEL2ID = {n: i for i, n in enumerate(LABELS)}


# --------------------------------------------------------------------------- #
# robust audio discovery (fixes the doubly-nested Dataset/train/train layout)
# --------------------------------------------------------------------------- #


def index_audio(data_root: Path) -> dict[str, Path]:
    """Map file stem -> path, searching recursively. Layout-agnostic."""
    index: dict[str, Path] = {}
    dupes = 0
    for wav in data_root.rglob("*.wav"):
        if wav.stem in index:
            dupes += 1
            continue
        index[wav.stem] = wav
    if not index:
        raise FileNotFoundError(f"no .wav files found anywhere under {data_root}")
    if dupes:
        print(f"[warn] {dupes} duplicate stems ignored (kept first match)")
    print(f"indexed {len(index)} wav files under {data_root}")
    return index


def load_train_manifest(train_csv: Path, index: dict[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("file_id") or cols.get("filename") or cols.get("id")
    label_col = cols.get("label")
    if id_col is None or label_col is None:
        raise ValueError(f"{train_csv} needs file_id and label columns; got {list(df.columns)}")

    df = df.rename(columns={id_col: "file_id", label_col: "label"})
    df["file_id"] = df["file_id"].astype(str).str.replace(r"\.wav$", "", regex=True)

    if df["label"].dtype == object:
        df["label"] = df["label"].str.strip().str.lower().map(LABEL2ID)
    df["label"] = df["label"].astype(int)

    missing = [f for f in df["file_id"] if f not in index]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} clips in {train_csv} not found on disk, e.g. {missing[:3]}"
        )
    df["path"] = [str(index[f]) for f in df["file_id"]]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #


def read_audio(path: str) -> tuple[torch.Tensor, int]:
    """Decode to a mono float32 tensor via libsndfile.

    torchaudio >= 2.9 delegates torchaudio.load to torchcodec, which has no
    wheel on every platform we run on. soundfile reads these PCM_16 wavs
    directly and is already a dependency.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.mean(axis=1))), sr


def audio_num_samples(path: str) -> int:
    """Length in samples at SAMPLE_RATE, read from the header only."""
    info = sf.info(str(path))
    return int(info.frames * SAMPLE_RATE / info.samplerate)


def read_lengths(paths, workers: int = 32, cache: str = "durations.csv") -> list[int]:
    """Header-only length reads, threaded, with an on-disk cache.

    This pass is pure I/O latency, not compute. On a Modal Volume every open
    is a network round-trip, so reading 9,170 headers sequentially costs
    ~15 minutes with the GPU sitting idle. Threads hide the latency; the
    cache removes the cost entirely on reruns.
    """
    paths = [str(p) for p in paths]
    cp = Path(cache)

    known: dict[str, int] = {}
    if cp.is_file():
        known = {k: int(v) for k, v in
                 pd.read_csv(cp).set_index("path")["n_samples"].items()}
    missing = [p for p in paths if p not in known]
    if not missing:
        print(f"durations: {len(paths)} loaded from cache {cp}", flush=True)
        return [known[p] for p in paths]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fresh = list(ex.map(audio_num_samples, missing))
    dt = time.time() - t0

    # merge, never clobber: train and test runs share one cache file
    known.update(dict(zip(missing, fresh)))
    cp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": list(known), "n_samples": list(known.values())}).to_csv(
        cp, index=False
    )
    print(f"durations: {len(missing)} headers in {dt:.1f}s "
          f"({len(missing) / max(dt, 1e-9):.0f}/s, {workers} threads); "
          f"cache now {len(known)} -> {cp}", flush=True)
    return [known[p] for p in paths]


def load_wav(path: str) -> torch.Tensor:
    wav, sr = read_audio(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return (wav - wav.mean()) / (wav.std() + 1e-7)


@torch.no_grad()
def embed_all(
    df: pd.DataFrame,
    model_name: str,
    batch_size: int,
    device: torch.device,
    io_workers: int = 32,
    load_workers: int = 8,
    durations_cache: str = "durations.csv",
) -> np.ndarray:
    """Length-bucketed batching so padding stays under ~5%, with a real mask.

    Both file-touching passes are threaded: decoding is I/O-bound and releases
    the GIL inside libsndfile, so a sequential loop leaves the GPU starved.
    """
    model = WavLMForXVector.from_pretrained(model_name).to(device).eval()

    print("reading durations for length bucketing...", flush=True)
    lengths = read_lengths(df["path"].tolist(), workers=io_workers,
                           cache=durations_cache)
    order = np.argsort(lengths)

    embeddings = np.zeros((len(df), model.config.xvector_output_dim), dtype=np.float32)
    paths = df["path"].tolist()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=load_workers) as pool:
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            wavs = list(pool.map(lambda i: load_wav(paths[i]), idx))
            maxlen = max(w.shape[-1] for w in wavs)

            batch = torch.zeros(len(wavs), maxlen)
            mask = torch.zeros(len(wavs), maxlen, dtype=torch.long)
            for j, w in enumerate(wavs):
                batch[j, : w.shape[-1]] = w  # pad region stays exactly zero
                mask[j, : w.shape[-1]] = 1

            out = model(batch.to(device), attention_mask=mask.to(device))
            embeddings[idx] = out.embeddings.float().cpu().numpy()

            if start % (batch_size * 20) == 0:
                done = start + len(idx)
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(order) - done) / max(rate, 1e-9)
                print(f"  embedded {done}/{len(order)}  "
                      f"{rate:.1f} clips/s  eta {eta / 60:.1f} min", flush=True)

    print(f"embedding done in {(time.time() - t0) / 60:.1f} min", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #


def emotion_center(emb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Subtract the per-class mean. Strips the emotion direction from the space."""
    out = emb.copy()
    for c in np.unique(labels):
        m = labels == c
        out[m] -= out[m].mean(axis=0, keepdims=True)
    return out


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def cluster(emb: np.ndarray, n_clusters: int) -> np.ndarray:
    return AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    ).fit_predict(emb)


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #


def guess_corpus(size: int) -> str:
    """Cluster size is a strong corpus signal under the union hypothesis."""
    if size >= 600:
        return "TESS"
    if size <= 70:
        return "RAVDESS"
    return "CREMA-D"


def report(df: pd.DataFrame, assign: np.ndarray) -> pd.DataFrame:
    sizes = Counter(assign)
    df = df.copy()
    df["pseudo_speaker"] = [f"spk_{a:03d}" for a in assign]
    df["cluster_size"] = [sizes[a] for a in assign]
    df["corpus_guess"] = df["cluster_size"].apply(guess_corpus)

    print("\n===== CLUSTER SIZE HISTOGRAM =====")
    buckets = {"<20": 0, "20-70": 0, "71-200": 0, "201-600": 0, ">600": 0}
    for s in sizes.values():
        if s < 20:
            buckets["<20"] += 1
        elif s <= 70:
            buckets["20-70"] += 1
        elif s <= 200:
            buckets["71-200"] += 1
        elif s <= 600:
            buckets["201-600"] += 1
        else:
            buckets[">600"] += 1
    for k, v in buckets.items():
        print(f"  {k:>8} clips : {v:>3} clusters")

    print("\n===== CORPUS HYPOTHESIS CHECK =====")
    expect = {"TESS": (2, 2400), "RAVDESS": (24, 1056), "CREMA-D": (70, 5714)}
    ok = True
    for corpus, (n_exp, clips_exp) in expect.items():
        sub = df[df["corpus_guess"] == corpus]
        n_got = sub["pseudo_speaker"].nunique()
        clips_got = len(sub)
        flag = "OK " if abs(n_got - n_exp) <= max(2, 0.2 * n_exp) else "MISS"
        if flag == "MISS":
            ok = False
        print(
            f"  [{flag}] {corpus:<8} clusters {n_got:>3} (expect ~{n_exp:>3})   "
            f"clips {clips_got:>5} (expect ~{clips_exp})"
        )

    print("\n===== LABEL PURITY (leakage smell test) =====")
    purity = df.groupby("pseudo_speaker")["label"].apply(
        lambda s: s.value_counts(normalize=True).iloc[0]
    )
    print(f"  mean dominant-class share per cluster: {purity.mean():.3f}")
    print(f"  clusters >0.50 single-class: {int((purity > 0.5).sum())}")
    print("  (a real speaker records all 6 emotions -> expect ~0.17-0.25.")
    print("   High values mean emotion still dominates the embedding space.)")

    print("\n" + ("hypothesis CONFIRMED — split on these groups." if ok else
                  "hypothesis NOT matched — try --n_clusters sweep before trusting this."))
    return df


# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./Dataset")
    p.add_argument("--train_csv", type=str, default="./Dataset/train.csv")
    p.add_argument("--out", type=str, default="speakers.csv")
    p.add_argument("--model_name", type=str, default="microsoft/wavlm-base-plus-sv")
    p.add_argument("--n_clusters", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--no_emotion_center", action="store_true")
    p.add_argument("--cache", type=str, default="xvectors.npy")
    p.add_argument("--durations_cache", type=str, default="durations.csv")
    p.add_argument("--io_workers", type=int, default=32,
                   help="threads for header reads (I/O-bound, high is fine)")
    p.add_argument("--load_workers", type=int, default=8,
                   help="threads for decoding audio in the embedding loop")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    index = index_audio(Path(args.data_root))
    df = load_train_manifest(Path(args.train_csv), index)
    print(f"train clips: {len(df)}")

    cache = Path(args.cache)
    if cache.is_file():
        print(f"loading cached embeddings from {cache}")
        emb = np.load(cache)
        if len(emb) != len(df):
            raise ValueError("cached embeddings do not match manifest length")
    else:
        emb = embed_all(
            df, args.model_name, args.batch_size, device,
            io_workers=args.io_workers,
            load_workers=args.load_workers,
            durations_cache=args.durations_cache,
        )
        np.save(cache, emb)
        print(f"cached embeddings -> {cache}")

    emb = l2norm(emb)
    if not args.no_emotion_center:
        emb = l2norm(emotion_center(emb, df["label"].values))

    print(f"clustering into {args.n_clusters} groups...")
    assign = cluster(emb, args.n_clusters)

    out_df = report(df, assign)
    out_df[["file_id", "pseudo_speaker", "cluster_size", "corpus_guess"]].to_csv(
        args.out, index=False
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
