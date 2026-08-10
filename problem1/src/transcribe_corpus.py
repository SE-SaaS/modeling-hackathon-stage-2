#!/usr/bin/env python3
"""
TRACK B STEP 2 — transcribe every clip and assign a corpus from the SCRIPT.

All three candidate corpora read from fixed scripts, so the transcript is a
fingerprint that no embedding can blur and no threshold has to guess at:

  TESS    - every clip is "Say the word ___" (200 target words)
  RAVDESS - exactly 2 sentences
  CREMA-D - exactly 12 sentences

Assignment order is deliberate: the TESS prefix wins outright, everything
else goes to fuzzy argmax over the 14 templates, and anything below the
ratio floor is reported as UNKNOWN rather than forced into a bucket.

    python transcribe_corpus.py --data_root ./Dataset --out artifacts/corpus.csv
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cluster_speakers import index_audio, read_audio, read_lengths

TESS_PREFIX = "say the word"

RAVDESS_LINES = [
    "kids are talking by the door",
    "dogs are sitting by the door",
]

CREMAD_LINES = [
    "its 11 oclock",
    "that is exactly what happened",
    "im on my way to the meeting",
    "i wonder what this is about",
    "the airplane is almost full",
    "maybe tomorrow it will be cold",
    "i would like a new alarm clock",
    "i think i have a doctors appointment",
    "dont forget a jacket",
    "i think ive seen this before",
    "the surface is slick",
    "well stop in a couple of minutes",
]

TEMPLATES = [(t, "RAVDESS") for t in RAVDESS_LINES] + \
            [(t, "CREMA-D") for t in CREMAD_LINES]


def normalise(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def assign_corpus(t: str, floor: float = 0.65) -> tuple[str, float, str]:
    """(corpus, best_ratio, best_template). Rules applied in order."""
    if t.startswith(TESS_PREFIX):
        return "TESS", 1.0, TESS_PREFIX
    best_r, best_c, best_t = 0.0, "UNKNOWN", ""
    for tpl, corp in TEMPLATES:
        r = SequenceMatcher(None, t, tpl).ratio()
        if r > best_r:
            best_r, best_c, best_t = r, corp, tpl
    if best_r < floor:
        return "UNKNOWN", best_r, best_t
    return best_c, best_r, best_t


def transcribe(paths, model_name, batch_size, load_workers, device):
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=model_name,
                   device=0 if device.type == "cuda" else -1)
    texts: list[str] = []
    t0 = time.time()
    chunk = max(batch_size * 16, 256)

    with ThreadPoolExecutor(max_workers=load_workers) as pool:
        for s in range(0, len(paths), chunk):
            part = paths[s : s + chunk]
            audio = list(pool.map(lambda p: read_audio(p)[0].numpy(), part))
            out = asr(audio, batch_size=batch_size)
            texts.extend(normalise(o["text"]) for o in out)
            done = len(texts)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  transcribed {done}/{len(paths)}  {rate:.1f} clips/s  "
                  f"eta {(len(paths) - done) / max(rate, 1e-9) / 60:.1f} min",
                  flush=True)
    return texts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./Dataset")
    p.add_argument("--train_csv", type=str, default="./Dataset/train.csv")
    p.add_argument("--sample_submission", type=str,
                   default="./Dataset/sample_submission.csv")
    p.add_argument("--out", type=str, default="artifacts/corpus.csv")
    p.add_argument("--model_name", type=str, default="openai/whisper-tiny.en")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--load_workers", type=int, default=8)
    p.add_argument("--durations_cache", type=str, default="durations.csv")
    p.add_argument("--floor", type=float, default=0.65)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    index = index_audio(Path(args.data_root))
    train_ids = (pd.read_csv(args.train_csv)["file_id"].astype(str)
                 .str.replace(r"\.wav$", "", regex=True).tolist())
    test_ids = (pd.read_csv(args.sample_submission)["file_id"].astype(str)
                .str.replace(r"\.wav$", "", regex=True).tolist())

    missing = [f for f in train_ids + test_ids if f not in index]
    if missing:
        raise FileNotFoundError(f"{len(missing)} clips not on disk, e.g. {missing[:3]}")

    df = pd.DataFrame({
        "file_id": train_ids + test_ids,
        "split": ["train"] * len(train_ids) + ["test"] * len(test_ids),
    })
    df["path"] = [str(index[f]) for f in df.file_id]
    print(f"train {len(train_ids)}  test {len(test_ids)}  total {len(df)}", flush=True)

    # length-sort so whisper batches are homogeneous; cache makes this free
    lengths = read_lengths(df["path"].tolist(), cache=args.durations_cache)
    order = np.argsort(lengths)
    ordered = df.iloc[order].reset_index(drop=True)

    texts = transcribe(ordered["path"].tolist(), args.model_name,
                       args.batch_size, args.load_workers, device)
    ordered["transcript"] = texts

    rows = [assign_corpus(t, args.floor) for t in texts]
    ordered["corpus"] = [r[0] for r in rows]
    ordered["match_ratio"] = [round(r[1], 4) for r in rows]
    ordered["template"] = [r[2] for r in rows]

    out = ordered[["file_id", "split", "transcript", "corpus",
                   "match_ratio", "template"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} rows)", flush=True)

    tr = out[out.split == "train"]
    te = out[out.split == "test"]

    print("\n" + "=" * 62)
    print("a) CLIPS PER CORPUS - TRAIN   (predicted 2400 / 1056 / 5714)")
    for c, n in tr.corpus.value_counts().items():
        print(f"     {c:<9} {n:>5}")
    print("\nb) CLIPS PER CORPUS - TEST")
    for c, n in te.corpus.value_counts().items():
        print(f"     {c:<9} {n:>5}")
    print("\nc) UNKNOWN counts")
    print(f"     train {int((tr.corpus == 'UNKNOWN').sum())}"
          f"   test {int((te.corpus == 'UNKNOWN').sum())}")
    print("\nd) TEST only")
    n_say = int(te.transcript.str.startswith(TESS_PREFIX).sum())
    n_rav = int((te.corpus == "RAVDESS").sum())
    print(f"     starts with 'say the word' : {n_say}")
    print(f"     matches either RAVDESS line: {n_rav}")
    print("\ne) DISTINCT CREMA-D SENTENCES SEEN IN TEST (expect 12)")
    seen = Counter(te[te.corpus == "CREMA-D"].template)
    print(f"     distinct = {len(seen)}")
    for t, n in seen.most_common():
        print(f"       {n:>5}  {t}")
    print("=" * 62)


if __name__ == "__main__":
    main()