#!/usr/bin/env python3
"""
Modal GPU runner for the problem-1 speaker-recovery pass.

There is no local GPU, so pseudo-speaker clustering and the train/test overlap
check run here. Everything is cached in Modal Volumes: the x-vectors come back
as .npy files so every downstream analysis can be re-run locally on CPU.

Volumes
  ser-p1-data   /data    the Dataset directory (upload once)
  ser-p1-out    /out     speakers.csv, xvectors*.npy
  ser-p1-cache  /cache   HuggingFace weights, so reruns skip the download

Order of operations
-------------------
  python -m modal run src/modal_app.py::upload
  python -m modal run src/modal_app.py::cluster --n-clusters 96
  python -m modal run src/modal_app.py::verify
  python -m modal run src/modal_app.py::train

Results are written back into ./artifacts/ automatically.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "ser-problem1"
# Workspace-dependent. On `aaaaaaaaaaaa` only A10G scheduled, and that
# workspace has now exceeded its spend limit and cannot run anything at all.
# On `awshanaqtah` A10G, L40S and A100-40GB are all ungated (probed with
# gpu_check.py), so fine-tuning runs there:
#   MODAL_PROFILE=awshanaqtah SER_GPU=A100 python -m modal run src/modal_app.py::train
# Volumes are per-workspace, so switching profiles means re-uploading.
GPU = os.environ.get("SER_GPU", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch",
        "torchaudio",
        "transformers==5.14.1",
        "scikit-learn",
        "pandas",
        "soundfile",
        "numpy",
    )
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(Path(__file__).parent), "/root/src")
)

app = modal.App(APP_NAME, image=image)

data_vol = modal.Volume.from_name("ser-p1-data", create_if_missing=True)
out_vol = modal.Volume.from_name("ser-p1-out", create_if_missing=True)
cache_vol = modal.Volume.from_name("ser-p1-cache", create_if_missing=True)

VOLUMES = {"/data": data_vol, "/out": out_vol, "/cache": cache_vol}
ARTIFACTS = Path(__file__).parent.parent / "artifacts"


def _run(cmd: list[str]) -> None:
    """Stream a subprocess, failing loudly on a non-zero exit."""
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd="/root/src")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {cmd}")


def _collect(names: list[str]) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for n in names:
        p = Path("/out") / n
        if not p.is_file():
            raise FileNotFoundError(f"expected output missing: {p}")
        out[n] = p.read_bytes()
    return out


# --------------------------------------------------------------------------- #
# remote functions
# --------------------------------------------------------------------------- #


@app.function(gpu=GPU, volumes=VOLUMES, timeout=60 * 60)
def cluster_remote(n_clusters: int = 96, batch_size: int = 16) -> dict[str, bytes]:
    _run(
        [
            sys.executable, "cluster_speakers.py",
            "--data_root", "/data/Dataset",
            "--train_csv", "/data/Dataset/train.csv",
            "--out", "/out/speakers.csv",
            "--cache", "/out/xvectors.npy",
            # on the volume so a rerun never repeats the network round-trips
            "--durations_cache", "/out/durations.csv",
            "--n_clusters", str(n_clusters),
            "--batch_size", str(batch_size),
        ]
    )
    out_vol.commit()
    return _collect(["speakers.csv", "xvectors.npy"])


@app.function(gpu=GPU, volumes=VOLUMES, timeout=60 * 60)
def transcribe_remote(batch_size: int = 32) -> dict[str, bytes]:
    _run(
        [
            sys.executable, "transcribe_corpus.py",
            "--data_root", "/data/Dataset",
            "--train_csv", "/data/Dataset/train.csv",
            "--sample_submission", "/data/Dataset/sample_submission.csv",
            "--out", "/out/corpus.csv",
            "--durations_cache", "/out/durations.csv",
            "--batch_size", str(batch_size),
        ]
    )
    out_vol.commit()
    return _collect(["corpus.csv"])


@app.function(gpu=GPU, volumes=VOLUMES, timeout=60 * 60)
def verify_remote(test_clusters: int = 21, batch_size: int = 16) -> dict[str, bytes]:
    _run(
        [
            sys.executable, "verify_test_speakers.py",
            "--data_root", "/data/Dataset",
            "--train_csv", "/data/Dataset/train.csv",
            "--speakers", "/out/speakers.csv",
            "--train_cache", "/out/xvectors.npy",
            "--test_cache", "/out/xvectors_test.npy",
            "--durations_cache", "/out/durations.csv",
            "--test_clusters", str(test_clusters),
            "--batch_size", str(batch_size),
        ]
    )
    out_vol.commit()
    return _collect(["xvectors_test.npy"])


@app.function(gpu=GPU, volumes=VOLUMES, timeout=60 * 60 * 12, cpu=8.0, memory=32768)
def train_remote(argv: list[str]) -> dict[str, bytes]:
    """Fine-tune WavLM-Large. `argv` is passed straight to ser_wavlm.py.

    The dataset is copied to container-local disk first. SERDataset re-opens
    every wav on every __getitem__, so an 8-epoch x 5-fold run is ~370k reads;
    over the Volume's FUSE mount that would dwarf the GPU time.
    """
    import shutil
    import time

    t0 = time.time()
    shutil.copytree("/data/Dataset", "/tmp/Dataset", dirs_exist_ok=True)
    n = sum(1 for _ in Path("/tmp/Dataset").rglob("*.wav"))
    print(f"staged {n} wavs -> /tmp/Dataset in {time.time() - t0:.1f}s", flush=True)

    _run([sys.executable, "ser_wavlm.py", *argv])
    out_vol.commit()

    run_dir = Path(argv[argv.index("--out_dir") + 1])
    ckpts = sorted(run_dir.glob("fold*.pt"))
    print(f"checkpoints on volume: "
          f"{[(c.name, f'{c.stat().st_size / 1e6:.0f}MB') for c in ckpts]}")

    # only the small artifacts come back; the ~1.2GB .pt files stay on the Volume
    wanted: dict[str, bytes] = {}
    for pat in ("submission.csv", "oof_logits.npy", "config.json",
                "fold*_val_*.npy", "fold*_val_ids.csv", "fold*_test_probs.npy"):
        for p in sorted(run_dir.glob(pat)):
            wanted[p.name] = p.read_bytes()
    return wanted


@app.function(volumes=VOLUMES, timeout=60 * 30, cpu=8.0)
def probe_io() -> str:
    """Measure Volume read behaviour. CPU-only, so this costs ~nothing.

    Threading did not rescue the header pass, so the question is whether the
    cost is per-open latency (threads should help), FUSE serialisation
    (they won't), or something a bulk copy to container-local disk fixes.
    """
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor

    import soundfile as sf

    out, root = [], Path("/data/Dataset")

    t0 = time.time()
    wavs = sorted(root.rglob("*.wav"))
    out.append(f"rglob            : {len(wavs)} entries in {time.time() - t0:.1f}s")

    def bench(paths, label, workers=None):
        t = time.time()
        if workers:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(lambda p: sf.info(str(p)), paths))
        else:
            [sf.info(str(p)) for p in paths]
        dt = time.time() - t
        rate = len(paths) / max(dt, 1e-9)
        out.append(f"{label:<17}: {len(paths)} headers in {dt:5.1f}s -> "
                   f"{rate:7.1f}/s  (all {len(wavs)} = {len(wavs) / rate / 60:.1f} min)")

    bench(wavs[:200], "volume seq")
    bench(wavs[200:400], "volume thr-32", workers=32)
    bench(wavs[400:600], "volume thr-128", workers=128)

    t0 = time.time()
    shutil.copytree("/data/Dataset", "/tmp/Dataset", dirs_exist_ok=True)
    copy_s = time.time() - t0
    out.append(f"bulk copytree    : whole dataset -> /tmp in {copy_s:.1f}s")

    local = sorted(Path("/tmp/Dataset").rglob("*.wav"))
    t0 = time.time()
    [sf.info(str(p)) for p in local[:600]]
    dt = time.time() - t0
    out.append(f"local seq (/tmp) : 600 headers in {dt:5.1f}s -> "
               f"{600 / max(dt, 1e-9):7.1f}/s")
    out.append(f"=> copy-then-read total for all {len(wavs)}: "
               f"{copy_s + len(wavs) / (600 / max(dt, 1e-9)):.1f}s")
    return "\n".join(out)


@app.function(volumes=VOLUMES, timeout=60 * 20)
def inspect_remote() -> str:
    lines = []
    for root in ("/data", "/out"):
        base = Path(root)
        n = sum(1 for _ in base.rglob("*")) if base.exists() else 0
        lines.append(f"{root}: {n} entries")
        for p in sorted(base.glob("*"))[:10]:
            lines.append(f"  {p}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# local entrypoints
# --------------------------------------------------------------------------- #


def _save(files: dict[str, bytes]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        (ARTIFACTS / name).write_bytes(blob)
        print(f"saved {ARTIFACTS / name}  ({len(blob) / 1e6:.1f} MB)")


@app.local_entrypoint()
def upload(data_dir: str = "./Dataset"):
    """Push the Dataset directory into the data volume (one time, ~870 MB)."""
    src = Path(data_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"no such directory: {src.resolve()}")
    n = sum(1 for _ in src.rglob("*.wav"))
    print(f"uploading {src.resolve()} ({n} wav files) -> ser-p1-data:/Dataset")
    with data_vol.batch_upload(force=True) as batch:
        batch.put_directory(str(src), "/Dataset")
    print("upload complete")
    print(inspect_remote.remote())


@app.local_entrypoint()
def cluster(n_clusters: int = 96, batch_size: int = 16):
    """TASK 3a — recover pseudo-speakers, cache x-vectors."""
    _save(cluster_remote.remote(n_clusters=n_clusters, batch_size=batch_size))


@app.local_entrypoint()
def transcribe(batch_size: int = 32):
    """TRACK B STEP 2 — transcribe all 10,897 clips, assign corpus by script."""
    _save(transcribe_remote.remote(batch_size=batch_size))


@app.local_entrypoint()
def verify(test_clusters: int = 21, batch_size: int = 16):
    """TASK 3b — duplicate / calibration / overlap / test-structure report."""
    _save(verify_remote.remote(test_clusters=test_clusters, batch_size=batch_size))


# the 4 clips excluded from training by explicit instruction; all are
# train-split, so the submission is unaffected
DROP_IDS = "clip_07177,clip_09805,clip_08363,clip_10440"


def _common_argv(model_name: str, batch_size: int, max_seconds: float,
                 freeze_bottom: int, dropout: float, workers: int,
                 out_dir: str) -> list[str]:
    return [
        # staged on container-local disk by train_remote
        "--data_root", "/tmp/Dataset",
        "--train_csv", "/tmp/Dataset/train.csv",
        "--sample_submission", "/tmp/Dataset/sample_submission.csv",
        "--out_dir", out_dir,
        # keyed by path, so the /tmp paths need their own cache file
        "--durations_cache", "/out/durations_local.csv",
        "--model_name", model_name,
        "--batch_size", str(batch_size),
        "--max_seconds", str(max_seconds),
        "--freeze_bottom", str(freeze_bottom),
        "--dropout", str(dropout),
        "--workers", str(workers),
    ]


@app.local_entrypoint()
def train(
    corpus: str = "/out/corpus.csv",
    speakers: str = "/out/speakers_v2.csv",
    fold_strategy: str = "test_mirror",
    out_dir: str = "/out/runs/wavlm",
    model_name: str = "microsoft/wavlm-large",
    epochs: int = 8,
    batch_size: int = 16,
    max_seconds: float = 5.0,
    encoder_lr: float = 1e-5,
    head_lr: float = 1e-4,
    layer_decay: float = 0.9,
    weight_decay: float = 0.01,
    freeze_bottom: int = 9,
    dropout: float = 0.3,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    seed: int = 42,
    folds: int = 5,
    val_frac: float = 0.15,
    only_fold: int = 0,
    bucket_pool_factor: int = 8,
    checkpoint_policy: str = "final",
    workers: int = 8,
):
    """TASK 4 — fine-tune WavLM-Large.

    Default splitter is `test_mirror`: CREMA-D + RAVDESS speakers are held out
    for validation and TESS/UNKNOWN are pinned to train, mirroring the corpus
    mix the transcripts found in the test set. It is speaker-level, so a val
    actor is never seen in training. `corpus_split` is the older clip-level
    fallback and is the one strategy that must NOT receive --speakers.

    --only-fold defaults to 0: folds are launched one at a time so each
    commits to the Volume before the next starts.
    """
    argv = _common_argv(model_name, batch_size, max_seconds, freeze_bottom,
                        dropout, workers, out_dir) + [
        "--corpus", corpus,
        "--fold_strategy", fold_strategy,
        "--val_frac", str(val_frac),
        "--drop_ids", DROP_IDS,
        "--checkpoint_policy", checkpoint_policy,
        "--bucket_pool_factor", str(bucket_pool_factor),
        "--folds", str(folds),
        "--only_fold", str(only_fold),
        "--epochs", str(epochs),
        "--encoder_lr", str(encoder_lr),
        "--head_lr", str(head_lr),
        "--layer_decay", str(layer_decay),
        "--weight_decay", str(weight_decay),
        "--label_smoothing", str(label_smoothing),
        "--mixup_alpha", str(mixup_alpha),
        "--seed", str(seed),
    ]
    # corpus_split raises if handed a speaker grouping; every other strategy
    # requires one
    if fold_strategy != "corpus_split":
        argv += ["--speakers", speakers]
    _save(train_remote.remote(argv))


@app.local_entrypoint()
def predict(
    out_dir: str = "/out/runs/wavlm",
    model_name: str = "microsoft/wavlm-large",
    batch_size: int = 16,
    max_seconds: float = 5.0,
    freeze_bottom: int = 9,
    dropout: float = 0.3,
    tta_speeds: str = "0.9,1.0,1.1",
    workers: int = 8,
):
    """Ensemble every fold*.pt in out_dir over the test set -> submission.csv."""
    argv = _common_argv(model_name, batch_size, max_seconds, freeze_bottom,
                        dropout, workers, out_dir) + [
        "--predict",
        "--tta_speeds", tta_speeds,
    ]
    _save(train_remote.remote(argv))


@app.local_entrypoint()
def inspect():
    print(inspect_remote.remote())


@app.local_entrypoint()
def probe():
    """CPU-only I/O diagnosis of the data Volume."""
    print(probe_io.remote())
