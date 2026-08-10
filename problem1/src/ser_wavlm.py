#!/usr/bin/env python3
"""
Speaker-independent Speech Emotion Recognition (6 classes) with WavLM-Large.

Classes: angry=0, happy=1, sad=2, fearful=3, disgust=4, neutral=5

Pipeline
--------
waveform -> (frozen) CNN feature encoder -> 24 transformer layers
         -> learnable weighted sum over layers
         -> masked attentive statistics pooling (mean + std)
         -> linear classifier -> 6 logits

Key design choices for generalising to UNSEEN SPEAKERS:
  * folds are split by speaker, never by clip (GroupKFold)
  * the CNN feature encoder is frozen (most speaker-specific component)
  * the lowest N transformer layers are frozen
  * speed perturbation warps formants -> synthesises unseen voices
  * SpecAugment (built into HF WavLM) + gaussian noise + mixup

This dataset ships NO speaker ids (filenames are anonymised clip_NNNNN.wav), so
a `speaker` column is MANDATORY and must come from cluster_speakers.py. There is
deliberately no fallback: a missing speaker column raises, because silently
grouping per-clip degrades GroupKFold into a random split and inflates CV.

Usage
-----
  # 0. recover pseudo-speakers first (separate script, needs a GPU)
  python cluster_speakers.py --data_root ./Dataset --n_clusters 96

  # 1. cheap sanity check: frozen embeddings + logistic regression
  python ser_wavlm.py --probe --data_root ./Dataset --speakers speakers.csv

  # 2. train all folds
  python ser_wavlm.py --data_root ./Dataset --speakers speakers.csv \
      --out_dir ./runs/wavlm --folds 5

  # 3. predict the test set with the trained folds
  python ser_wavlm.py --predict --data_root ./Dataset --out_dir ./runs/wavlm

Expected data layout
--------------------
Any layout — audio is discovered recursively by file stem, so both
`Dataset/train/*.wav` and `Dataset/train/train/*.wav` work.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, WavLMModel

from cluster_speakers import (
    index_audio,
    load_train_manifest,
    read_audio,
    read_lengths,
)

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

LABELS = ["angry", "happy", "sad", "fearful", "disgust", "neutral"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

SAMPLE_RATE = 16_000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# manifest construction
# --------------------------------------------------------------------------- #


def attach_speakers(df: pd.DataFrame, speakers_csv: str | None) -> pd.DataFrame:
    """Attach speaker (and corpus) columns. Raises if they cannot be obtained.

    No fallback by design. Grouping every clip into its own speaker turns
    GroupKFold into a random clip-level split, which is exactly the silent
    failure this pipeline exists to avoid.
    """
    if "speaker" in df.columns and df["speaker"].notna().all():
        return df

    if speakers_csv is None:
        raise ValueError(
            "no `speaker` column in train.csv and --speakers was not given.\n"
            "This dataset has no speaker ids in filenames or metadata, so a "
            "speaker grouping MUST be supplied. Run:\n"
            "    python cluster_speakers.py --data_root ./Dataset --n_clusters 96\n"
            "then pass --speakers speakers.csv"
        )

    path = Path(speakers_csv)
    if not path.is_file():
        raise FileNotFoundError(f"--speakers file not found: {path}")

    spk = pd.read_csv(path)
    required = {"file_id", "pseudo_speaker"}
    if not required.issubset(spk.columns):
        raise ValueError(
            f"{path} must contain columns {sorted(required)}; got {list(spk.columns)}"
        )
    spk["file_id"] = spk["file_id"].astype(str)
    keep = ["file_id", "pseudo_speaker"] + (
        ["corpus_guess"] if "corpus_guess" in spk.columns else []
    )
    merged = df.merge(spk[keep], on="file_id", how="left")

    if merged["pseudo_speaker"].isna().any():
        n = int(merged["pseudo_speaker"].isna().sum())
        missing = merged.loc[merged["pseudo_speaker"].isna(), "file_id"].head(3).tolist()
        raise ValueError(
            f"{n} train clips have no speaker in {path}, e.g. {missing}. "
            "Re-run cluster_speakers.py over the full train set."
        )

    merged = merged.rename(columns={"pseudo_speaker": "speaker"})
    if "corpus_guess" in merged.columns:
        merged = merged.rename(columns={"corpus_guess": "corpus"})
    merged["speaker"] = merged["speaker"].astype(str)
    return merged


def attach_corpus(df: pd.DataFrame, corpus_csv: str) -> pd.DataFrame:
    """Attach the `corpus` column produced by transcribe_corpus.py.

    Corpus comes from the transcript, not from an embedding: all three source
    corpora read from fixed scripts, so the sentence identifies the corpus
    exactly. That is why this needs no threshold and no clustering.
    """
    path = Path(corpus_csv)
    if not path.is_file():
        raise FileNotFoundError(f"--corpus file not found: {path}")

    cp = pd.read_csv(path)
    required = {"file_id", "corpus"}
    if not required.issubset(cp.columns):
        raise ValueError(
            f"{path} must contain columns {sorted(required)}; got {list(cp.columns)}"
        )
    cp["file_id"] = cp["file_id"].astype(str)
    merged = df.merge(cp[["file_id", "corpus"]], on="file_id", how="left")

    if merged["corpus"].isna().any():
        n = int(merged["corpus"].isna().sum())
        missing = merged.loc[merged["corpus"].isna(), "file_id"].head(3).tolist()
        raise ValueError(
            f"{n} train clips have no corpus in {path}, e.g. {missing}. "
            "Re-run transcribe_corpus.py over train AND test."
        )
    return merged


def build_train_manifest(
    data_root: Path,
    train_csv: Path,
    speakers_csv: str | None,
    durations_cache: str = "durations.csv",
    io_workers: int = 32,
    corpus_csv: str | None = None,
    require_speakers: bool = True,
    drop_ids: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Recursive audio discovery + labels + a grouping for the fold split.

    `require_speakers` is False only for --fold_strategy corpus_split, which
    never calls GroupKFold, so the speaker guard below has nothing to protect.
    """
    index = index_audio(data_root)
    df = load_train_manifest(train_csv, index)

    if drop_ids:
        before = len(df)
        df = df[~df["file_id"].isin(set(drop_ids))].reset_index(drop=True)
        print(f"dropped {before - len(df)} of {len(drop_ids)} requested clips")

    if corpus_csv is not None:
        df = attach_corpus(df, corpus_csv)
    if require_speakers:
        df = attach_speakers(df, speakers_csv)

    # header-only reads, threaded + cached; feeds LengthBucketSampler
    df["n_samples"] = read_lengths(
        df["path"].tolist(), workers=io_workers, cache=durations_cache
    )
    return df


def build_test_manifest(data_root: Path, sample_submission: Path) -> pd.DataFrame:
    """Test ids come from sample_submission.csv so the output order is exact."""
    if not sample_submission.is_file():
        raise FileNotFoundError(f"sample submission not found: {sample_submission}")

    index = index_audio(data_root)
    sub = pd.read_csv(sample_submission)
    cols = {c.lower(): c for c in sub.columns}
    id_col = cols.get("file_id") or cols.get("filename") or cols.get("id")
    if id_col is None:
        raise ValueError(
            f"{sample_submission} needs a file_id column; got {list(sub.columns)}"
        )

    file_ids = (
        sub[id_col].astype(str).str.replace(r"\.wav$", "", regex=True).tolist()
    )
    missing = [f for f in file_ids if f not in index]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} test clips listed in {sample_submission} are not on "
            f"disk, e.g. {missing[:3]}"
        )
    return pd.DataFrame(
        {
            "file_id": file_ids,
            "path": [str(index[f]) for f in file_ids],
            "label": -1,
        }
    )


# --------------------------------------------------------------------------- #
# fold construction
# --------------------------------------------------------------------------- #


def grouped_folds(df: pd.DataFrame, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Plain speaker-grouped folds over every clip."""
    gkf = GroupKFold(n_splits=n_folds)
    return list(gkf.split(df, df["label"], groups=df["speaker"]))


def corpus_aware_folds(
    df: pd.DataFrame, n_folds: int, holdout_corpus: str = "CREMA-D"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validate only on held-out `holdout_corpus` speakers.

    Under the working hypothesis the test set is held-out CREMA-D actors, with
    TESS and RAVDESS absent at test time. Mirroring that means validating on
    CREMA-D speakers only, while every other corpus stays in train for all
    folds — they are extra training signal, not a validation target.
    """
    if "corpus" not in df.columns:
        raise ValueError(
            "corpus_aware folds need a `corpus` column. Pass a --speakers csv "
            "produced by cluster_speakers.py (it emits corpus_guess)."
        )

    held = np.where((df["corpus"] == holdout_corpus).values)[0]
    if len(held) == 0:
        raise ValueError(
            f"no clips with corpus == {holdout_corpus!r}; "
            f"present values: {sorted(df['corpus'].unique())}"
        )

    sub = df.iloc[held]
    n_spk = sub["speaker"].nunique()
    if n_spk < n_folds:
        raise ValueError(
            f"only {n_spk} {holdout_corpus} speakers for {n_folds} folds"
        )

    gkf = GroupKFold(n_splits=n_folds)
    folds = []
    for _, va_local in gkf.split(sub, sub["label"], groups=sub["speaker"]):
        va_idx = held[va_local]
        tr_idx = np.setdiff1d(np.arange(len(df)), va_idx, assume_unique=False)
        folds.append((tr_idx, va_idx))

    n_extra = int((df["corpus"] != holdout_corpus).sum())
    print(
        f"corpus-aware folds: validating on {holdout_corpus} only "
        f"({len(held)} clips / {n_spk} speakers); "
        f"{n_extra} clips from other corpora stay in train every fold"
    )
    return folds


def corpus_split_folds(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
    val_frac: float = 0.15,
    holdout_corpus: str = "CREMA-D",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified clip-level val drawn from `holdout_corpus` only.

    Every other corpus (TESS, RAVDESS, UNKNOWN) is pinned to train in every
    fold, mirroring the hypothesis that the test set is held-out CREMA-D.

    NOTE: this is a CLIP-level split, not a speaker-level one. No speaker
    grouping is used, so the same CREMA-D actor can appear in both train and
    val and the reported val macro-F1 is therefore an OPTIMISTIC estimate of
    unseen-speaker test performance. It is a training signal, not a leaderboard
    prediction. The folds differ only by random seed, so their val sets
    overlap — they are ensemble members, not an out-of-fold partition.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    if "corpus" not in df.columns:
        raise ValueError(
            "corpus_split folds need a `corpus` column. Pass --corpus "
            "artifacts/corpus.csv produced by transcribe_corpus.py."
        )

    held = np.where((df["corpus"] == holdout_corpus).values)[0]
    if len(held) == 0:
        raise ValueError(
            f"no clips with corpus == {holdout_corpus!r}; "
            f"present values: {sorted(df['corpus'].unique())}"
        )

    sub = df.iloc[held]
    sss = StratifiedShuffleSplit(
        n_splits=n_folds, test_size=val_frac, random_state=seed
    )
    folds = []
    for _, va_local in sss.split(np.zeros(len(sub)), sub["label"].values):
        va_idx = held[va_local]
        tr_idx = np.setdiff1d(np.arange(len(df)), va_idx, assume_unique=False)
        folds.append((tr_idx, va_idx))

    n_pinned = int((df["corpus"] != holdout_corpus).sum())
    print(
        f"corpus_split: val = {int(round(val_frac * 100))}% of {holdout_corpus} "
        f"({len(held)} clips), stratified by label; "
        f"{n_pinned} clips from other corpora pinned to train every fold"
    )
    print("  corpus counts: " + df["corpus"].value_counts().to_dict().__str__())
    return folds


def test_mirror_folds(
    df: pd.DataFrame, n_folds: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold out CREMA-D + RAVDESS speakers. TESS and UNKNOWN always train.

    This is the split that actually mirrors the test set: transcripts showed
    ZERO TESS clips in test, and RAVDESS present, so validation must contain
    held-out speakers from both scripted-sentence corpora and none from TESS.

    Unlike corpus_split_folds this is speaker-level, so a validation actor is
    never seen during training and the reported macro-F1 is an honest estimate
    rather than an optimistic one.
    """
    for col in ("corpus", "speaker"):
        if col not in df.columns:
            raise ValueError(
                f"test_mirror folds need a `{col}` column. Pass "
                f"--speakers artifacts/speakers_v2.csv and "
                f"--corpus artifacts/corpus.csv."
            )

    rng = np.random.RandomState(seed)
    chunks: dict[str, list[np.ndarray]] = {}
    for corpus in ("CREMA-D", "RAVDESS"):
        spk = sorted(df.loc[df.corpus == corpus, "speaker"].unique())
        if len(spk) < n_folds:
            raise ValueError(
                f"only {len(spk)} {corpus} speakers for {n_folds} folds"
            )
        spk = np.array(spk)
        rng.shuffle(spk)
        chunks[corpus] = np.array_split(spk, n_folds)

    folds = []
    for f in range(n_folds):
        held = set(chunks["CREMA-D"][f]) | set(chunks["RAVDESS"][f])
        val = df["speaker"].isin(held).values
        folds.append((np.where(~val)[0], np.where(val)[0]))

    n_pinned = int(df["corpus"].isin(("TESS", "UNKNOWN")).sum())
    print(
        f"test_mirror: holding out {len(chunks['CREMA-D'][0])}~ CREMA-D and "
        f"{len(chunks['RAVDESS'][0])}~ RAVDESS speakers per fold; "
        f"{n_pinned} TESS/UNKNOWN clips pinned to train every fold"
    )
    return folds


def build_folds(df: pd.DataFrame, args) -> list[tuple[np.ndarray, np.ndarray]]:
    if args.fold_strategy == "test_mirror":
        return test_mirror_folds(df, args.folds, args.seed)
    if args.fold_strategy == "corpus_split":
        return corpus_split_folds(
            df, args.folds, args.seed, args.val_frac, args.holdout_corpus
        )
    if args.fold_strategy == "corpus_aware":
        return corpus_aware_folds(df, args.folds, args.holdout_corpus)
    return grouped_folds(df, args.folds)


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #


def speed_perturb(wav: torch.Tensor, factor: float) -> torch.Tensor:
    """Resample-based speed change. Warps formants -> pseudo new speaker."""
    if abs(factor - 1.0) < 1e-6:
        return wav
    return torchaudio.functional.resample(
        wav, orig_freq=SAMPLE_RATE, new_freq=int(SAMPLE_RATE * factor)
    )


def add_noise(wav: torch.Tensor, snr_db: float) -> torch.Tensor:
    sig_power = wav.pow(2).mean().clamp_min(1e-10)
    noise_power = sig_power / (10 ** (snr_db / 10))
    return wav + torch.randn_like(wav) * noise_power.sqrt()


def crop_to_max(wav: torch.Tensor, target: int, train: bool) -> torch.Tensor:
    """Crop over-long clips. Never pads — padding happens in collate."""
    n = wav.shape[-1]
    if n <= target:
        return wav
    start = random.randint(0, n - target) if train else (n - target) // 2
    return wav[..., start : start + target]


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


@dataclass
class AugConfig:
    speed: bool = True
    speeds: tuple[float, ...] = (0.9, 1.0, 1.1)
    noise_prob: float = 0.3
    snr_range: tuple[float, float] = (10.0, 20.0)


class SERDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        max_seconds: float = 5.0,
        train: bool = True,
        aug: AugConfig | None = None,
        force_speed: float | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.max_len = int(max_seconds * SAMPLE_RATE)
        self.train = train
        self.aug = aug or AugConfig()
        self.force_speed = force_speed  # used for test-time augmentation

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, path: str) -> torch.Tensor:
        wav, sr = read_audio(path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        return wav

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav = self._load(row["path"])

        if self.force_speed is not None:
            wav = speed_perturb(wav, self.force_speed)
        elif self.train and self.aug.speed:
            wav = speed_perturb(wav, random.choice(self.aug.speeds))

        if self.train and random.random() < self.aug.noise_prob:
            wav = add_noise(wav, random.uniform(*self.aug.snr_range))

        wav = crop_to_max(wav, self.max_len, self.train)

        # Normalise BEFORE padding so the pad region added in collate is
        # exactly zero, and the mask and the signal agree.
        wav = (wav - wav.mean()) / (wav.std() + 1e-7)

        return {
            "input_values": wav,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


class LengthBucketSampler(torch.utils.data.Sampler):
    """Batch clips of similar duration together so padding stays low.

    Dynamic padding costs scale with E[max duration] over the batch, which
    grows with batch size — plain shuffling is ~19% padding at batch 4 but
    ~30%+ at batch 16. This shuffles, sorts within a pool of
    `pool_factor * batch_size` clips, cuts batches from the sorted pool, then
    shuffles batch order. Padding stays under ~10% at any batch size while
    batch composition stays close to random.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        pool_factor: int = 50,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.pool = max(1, pool_factor) * batch_size
        self.drop_last = drop_last
        self.epoch = 0
        self.seed = seed
        n_full, rem = divmod(len(self.lengths), batch_size)
        self._len = n_full + (0 if (drop_last or rem == 0) else 1)

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.lengths))

        batches = []
        for s in range(0, len(order), self.pool):
            pool = order[s : s + self.pool]
            pool = pool[np.argsort(self.lengths[pool], kind="stable")]
            for b in range(0, len(pool), self.batch_size):
                chunk = pool[b : b + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk.tolist())

        rng.shuffle(batches)
        return iter(batches)


def collate(batch):
    """Dynamic padding to the batch maximum, with a matching attention mask."""
    wavs = [b["input_values"] for b in batch]
    maxlen = max(w.shape[-1] for w in wavs)

    input_values = torch.zeros(len(wavs), maxlen)
    attention_mask = torch.zeros(len(wavs), maxlen, dtype=torch.long)
    for i, w in enumerate(wavs):
        n = w.shape[-1]
        input_values[i, :n] = w
        attention_mask[i, :n] = 1

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "label": torch.stack([b["label"] for b in batch]),
    }


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


class AttentiveStatsPool(nn.Module):
    """Attention-weighted mean and std over the time axis, honouring a mask."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)  mask: (B, T) with True on valid frames
        x = x.transpose(1, 2)  # (B, D, T)
        logits = self.attn(x)
        logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(logits, dim=2)
        mean = (x * w).sum(dim=2)
        var = (x.pow(2) * w).sum(dim=2) - mean.pow(2)
        std = var.clamp_min(1e-8).sqrt()
        return torch.cat([mean, std], dim=1)  # (B, 2D)


class WavLMEmotion(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        num_classes: int = 6,
        freeze_bottom: int = 9,
        dropout: float = 0.3,
        mask_time_prob: float = 0.05,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        # SpecAugment, applied by HF on the CNN output during training only
        config.apply_spec_augment = True
        config.mask_time_prob = mask_time_prob
        config.mask_feature_prob = 0.02

        self.backbone = WavLMModel.from_pretrained(model_name, config=config)

        # the CNN feature encoder is the most speaker/channel specific part
        self.backbone.feature_extractor._freeze_parameters()

        # freeze the lowest transformer layers: generic acoustics, not emotion
        self.freeze_bottom = freeze_bottom
        for i, layer in enumerate(self.backbone.encoder.layers):
            if i < freeze_bottom:
                for p in layer.parameters():
                    p.requires_grad = False

        n_layers = config.num_hidden_layers + 1  # + embedding output
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))

        dim = config.hidden_size
        self.pool = AttentiveStatsPool(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def embed(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.backbone(
            input_values, attention_mask=attention_mask, output_hidden_states=True
        )
        stack = torch.stack(out.hidden_states, dim=0)  # (L, B, T, D)
        w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        fused = (stack * w).sum(dim=0)  # (B, T, D)

        frame_mask = self.backbone._get_feature_vector_attention_mask(
            fused.shape[1], attention_mask
        ).bool()
        return self.pool(fused, frame_mask)

    def forward(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.head(self.embed(input_values, attention_mask))

    def layer_weight_report(self) -> list[float]:
        return torch.softmax(self.layer_weights.detach().cpu(), dim=0).tolist()


# --------------------------------------------------------------------------- #
# optimisation helpers
# --------------------------------------------------------------------------- #


def build_optimizer(
    model: WavLMEmotion,
    encoder_lr: float,
    head_lr: float,
    layer_decay: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Layer-wise LR decay: deeper layers get a larger learning rate."""
    n_layers = len(model.backbone.encoder.layers)
    groups: list[dict] = []

    for i, layer in enumerate(model.backbone.encoder.layers):
        params = [p for p in layer.parameters() if p.requires_grad]
        if not params:
            continue
        scale = layer_decay ** (n_layers - 1 - i)
        groups.append({"params": params, "lr": encoder_lr * scale})

    head_params = (
        list(model.head.parameters())
        + list(model.pool.parameters())
        + [model.layer_weights]
    )
    groups.append({"params": head_params, "lr": head_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Mixup applied to pooled embeddings (cheaper and stabler than on waveforms)."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        iv = batch["input_values"].to(device, non_blocking=True)
        am = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(iv, am)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(batch["label"].numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def train_fold(
    args,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fold: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(args.seed + fold)

    train_ds = SERDataset(train_df, args.max_seconds, train=True)
    val_ds = SERDataset(val_df, args.max_seconds, train=False)

    if args.bucket_batches:
        train_dl = DataLoader(
            train_ds,
            batch_sampler=LengthBucketSampler(
                train_df["n_samples"].values,
                args.batch_size,
                pool_factor=args.bucket_pool_factor,
                seed=args.seed + fold,
            ),
            num_workers=args.workers,
            collate_fn=collate,
            pin_memory=True,
        )
    else:
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=True,
        )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
    )

    model = WavLMEmotion(
        model_name=args.model_name,
        num_classes=len(LABELS),
        freeze_bottom=args.freeze_bottom,
        dropout=args.dropout,
    ).to(device)

    optimizer = build_optimizer(
        model, args.encoder_lr, args.head_lr, args.layer_decay, args.weight_decay
    )
    total_steps = len(train_dl) * args.epochs
    warmup = int(0.1 * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_f1, best_state = -1.0, None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl):
            iv = batch["input_values"].to(device, non_blocking=True)
            am = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                emb = model.embed(iv, am)
                if args.mixup_alpha > 0:
                    emb, y_a, y_b, lam = mixup(emb, y, args.mixup_alpha)
                    logits = model.head(emb)
                    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(
                        logits, y_b
                    )
                else:
                    loss = criterion(model.head(emb), y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

            if step % 50 == 0:
                print(
                    f"  fold {fold} epoch {epoch} step {step}/{len(train_dl)} "
                    f"loss {running / (step + 1):.4f}",
                    flush=True,
                )

        logits, labels = evaluate(model, val_dl, device)
        preds = logits.argmax(1)
        f1 = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        print(f"[fold {fold}] epoch {epoch}: macro-F1 {f1:.4f}  acc {acc:.4f}")
        best_f1 = max(best_f1, f1)

        # `final` writes every epoch, so the file left on disk is the last one.
        # Selecting on val macro-F1 tunes the checkpoint against the very set
        # it is scored on, which inflates the number when val is a clip-level
        # split of the training corpus.
        if args.checkpoint_policy == "final" or f1 >= best_f1:
            state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(state, out_dir / f"fold{fold}.pt")
            best_state = state

    if args.checkpoint_policy == "best" and best_state is not None:
        model.load_state_dict(best_state)

    print(
        f"[fold {fold}] best macro-F1 {best_f1:.4f} "
        f"(checkpoint kept: {args.checkpoint_policy})"
    )
    print(
        "  layer weights (0=CNN output, 24=top): "
        + " ".join(f"{w:.3f}" for w in model.layer_weight_report())
    )

    logits, labels = evaluate(model, val_dl, device)
    del model
    torch.cuda.empty_cache()
    return logits, labels


# --------------------------------------------------------------------------- #
# probe: frozen embeddings + logistic regression (sanity floor)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def run_probe(args, df: pd.DataFrame, device: torch.device) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("Extracting frozen WavLM embeddings (mean-pooled, middle layers)...")
    backbone = WavLMModel.from_pretrained(args.model_name).to(device).eval()

    ds = SERDataset(df, args.max_seconds, train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)

    feats = []
    for batch in dl:
        iv = batch["input_values"].to(device)
        am = batch["attention_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = backbone(iv, attention_mask=am, output_hidden_states=True)
        # layers 6-14 carry most of the emotional signal
        mid = torch.stack(out.hidden_states[6:15], dim=0).mean(dim=0)
        # masked mean over valid frames only
        fm = backbone._get_feature_vector_attention_mask(mid.shape[1], am)
        fm = fm.unsqueeze(-1).to(mid.dtype)
        pooled = (mid * fm).sum(dim=1) / fm.sum(dim=1).clamp_min(1.0)
        feats.append(pooled.float().cpu().numpy())
    X = np.concatenate(feats)
    y = df["label"].values
    groups = df["speaker"].values

    gkf = GroupKFold(n_splits=args.folds)
    scores = []
    for tr, va in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(scaler.transform(X[tr]), y[tr])
        pred = clf.predict(scaler.transform(X[va]))
        scores.append((accuracy_score(y[va], pred), f1_score(y[va], pred, average="macro")))

    acc = float(np.mean([s[0] for s in scores]))
    f1 = float(np.mean([s[1] for s in scores]))
    print(f"\nPROBE (speaker-grouped {args.folds}-fold): acc {acc:.4f}  macro-F1 {f1:.4f}")
    print("Interpretation:")
    print("  ~0.55-0.65  -> healthy floor, go ahead and fine-tune")
    print("  ~0.20       -> labels or audio loading are broken, fix that first")
    print("  >0.90       -> speaker leakage in your split, check the grouping")


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #


@torch.no_grad()
def run_predict(args, device: torch.device) -> None:
    test_df = build_test_manifest(
        Path(args.data_root), Path(args.sample_submission)
    )
    out_dir = Path(args.out_dir)
    ckpts = sorted(out_dir.glob("fold*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no fold checkpoints found in {out_dir}")

    tta_speeds = [float(s) for s in args.tta_speeds.split(",")]
    total = np.zeros((len(test_df), len(LABELS)), dtype=np.float64)

    print(f"ensembling {len(ckpts)} checkpoint(s): {[c.name for c in ckpts]}")

    for ckpt in ckpts:
        model = WavLMEmotion(
            model_name=args.model_name,
            num_classes=len(LABELS),
            freeze_bottom=args.freeze_bottom,
            dropout=args.dropout,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        fold_probs = np.zeros((len(test_df), len(LABELS)), dtype=np.float64)
        for speed in tta_speeds:
            ds = SERDataset(test_df, args.max_seconds, train=False, force_speed=speed)
            dl = DataLoader(
                ds, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate
            )
            logits, _ = evaluate(model, dl, device)
            fold_probs += torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            print(f"  {ckpt.name} speed {speed} done", flush=True)

        # TTA-averaged test probabilities for this fold, kept so a later
        # ensemble can be reweighted without re-running inference
        fold_probs /= len(tta_speeds)
        np.save(out_dir / f"{ckpt.stem}_test_probs.npy", fold_probs.astype(np.float32))
        total += fold_probs

        del model
        torch.cuda.empty_cache()

    preds = total.argmax(1)
    # column names and row order follow sample_submission.csv exactly
    sub = pd.DataFrame({"file_id": test_df["file_id"], "label": preds})
    sub_path = out_dir / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nwrote {sub_path}  ({len(sub)} rows)")
    print(sub["label"].value_counts().sort_index())


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=str, default="./Dataset")
    p.add_argument("--train_csv", type=str, default="./Dataset/train.csv")
    p.add_argument("--sample_submission", type=str, default="./Dataset/sample_submission.csv")
    p.add_argument("--speakers", type=str, default=None,
                   help="speakers.csv from cluster_speakers.py (required for training)")
    p.add_argument("--out_dir", type=str, default="./runs/wavlm")
    p.add_argument("--model_name", type=str, default="microsoft/wavlm-large")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold_strategy", type=str, default="group",
                   choices=["group", "corpus_aware", "corpus_split", "test_mirror"])
    p.add_argument("--corpus", type=str, default=None,
                   help="corpus.csv from transcribe_corpus.py (required for "
                        "--fold_strategy corpus_split)")
    p.add_argument("--val_frac", type=float, default=0.15,
                   help="corpus_split: val fraction taken from --holdout_corpus")
    p.add_argument("--drop_ids", type=str, default="",
                   help="comma-separated file_ids to exclude from training")
    p.add_argument("--checkpoint_policy", type=str, default="best",
                   choices=["best", "final"],
                   help="which epoch's weights to keep as fold{N}.pt")
    p.add_argument("--holdout_corpus", type=str, default="CREMA-D")
    p.add_argument("--only_fold", type=int, default=-1, help="train a single fold")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--bucket_batches", action=argparse.BooleanOptionalAction, default=True,
                   help="group similar-length clips into batches to cut padding waste")
    p.add_argument("--bucket_pool_factor", type=int, default=50)
    p.add_argument("--durations_cache", type=str, default="durations.csv")
    p.add_argument("--io_workers", type=int, default=32)
    p.add_argument("--max_seconds", type=float, default=5.0)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--layer_decay", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--freeze_bottom", type=int, default=9)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--tta_speeds", type=str, default="0.9,1.0,1.1")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe", action="store_true", help="frozen-embedding sanity check")
    p.add_argument("--predict", action="store_true", help="predict test set")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    set_seed(args.seed)

    if args.predict:
        run_predict(args, device)
        return

    # corpus_split never calls GroupKFold, so it needs no speaker grouping;
    # every other strategy still refuses to run without one.
    uses_speakers = args.fold_strategy != "corpus_split"
    if not uses_speakers and args.speakers:
        raise ValueError(
            "--fold_strategy corpus_split does not use --speakers; remove it "
            "so the split cannot silently depend on a clustering."
        )

    drop_ids = tuple(s.strip() for s in args.drop_ids.split(",") if s.strip())

    df = build_train_manifest(
        Path(args.data_root), Path(args.train_csv), args.speakers,
        durations_cache=args.durations_cache, io_workers=args.io_workers,
        corpus_csv=args.corpus, require_speakers=uses_speakers, drop_ids=drop_ids,
    )
    n_spk = df["speaker"].nunique() if "speaker" in df.columns else "n/a"
    print(f"train clips: {len(df)}   speakers: {n_spk}")
    print(df["label"].value_counts().sort_index().to_string())

    if (df["label"] < 0).any():
        raise ValueError("some training clips have no label — check train.csv")

    if args.probe:
        run_probe(args, df, device)
        return

    splits = build_folds(df, args)

    oof_logits = np.zeros((len(df), len(LABELS)), dtype=np.float32)
    oof_mask = np.zeros(len(df), dtype=bool)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        if args.only_fold >= 0 and fold != args.only_fold:
            continue

        if uses_speakers:
            tr_spk = set(df.iloc[tr_idx]["speaker"])
            va_spk = set(df.iloc[va_idx]["speaker"])
            assert not (tr_spk & va_spk), "speaker leak between train and val!"
            spk_note = f" ({len(tr_spk)} / {len(va_spk)} speakers)"
        else:
            spk_note = " (clip-level split, no speaker grouping)"
        print(f"\n=== fold {fold}: {len(tr_idx)} train / {len(va_idx)} val{spk_note} ===")

        logits, labels = train_fold(
            args, df.iloc[tr_idx], df.iloc[va_idx], fold, device
        )
        oof_logits[va_idx] = logits
        oof_mask[va_idx] = True

        # per-fold artifacts so folds can be ensembled later without retraining
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"fold{fold}_val_logits.npy", logits)
        np.save(out_dir / f"fold{fold}_val_labels.npy", labels)
        df.iloc[va_idx][["file_id"]].to_csv(
            out_dir / f"fold{fold}_val_ids.csv", index=False
        )

        preds = logits.argmax(1)
        print(f"\n----- fold {fold} VALIDATION -----")
        print(f"accuracy  {accuracy_score(labels, preds):.4f}")
        print(f"macro-F1  {f1_score(labels, preds, average='macro'):.4f}")
        print(classification_report(labels, preds, target_names=LABELS, digits=3))
        print("confusion matrix (rows = true, cols = pred):")
        cm = confusion_matrix(labels, preds, labels=list(range(len(LABELS))))
        print("           " + " ".join(f"{n[:5]:>6}" for n in LABELS))
        for name, row in zip(LABELS, cm):
            print(f"  {name:<8} " + " ".join(f"{v:>6}" for v in row))

    if oof_mask.any():
        y_true = df["label"].values[oof_mask]
        y_pred = oof_logits[oof_mask].argmax(1)
        # corpus_split folds share val clips, so this pools overlapping val sets
        # rather than forming a true out-of-fold partition.
        header = "POOLED VALIDATION" if not uses_speakers else "OUT-OF-FOLD"
        print(f"\n===== {header} =====")
        print(f"accuracy  {accuracy_score(y_true, y_pred):.4f}")
        print(f"macro-F1  {f1_score(y_true, y_pred, average='macro'):.4f}")
        print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "oof_logits.npy", oof_logits)
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
