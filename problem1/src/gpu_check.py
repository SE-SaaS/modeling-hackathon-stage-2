#!/usr/bin/env python3
"""Minimal probe: can the ACTIVE Modal workspace schedule a given GPU?

Declares one GPU function and nothing else — no volumes, no data, no image
build beyond debian_slim. Modal validates the GPU spec locally at app
creation, so a payment-gated tier fails in seconds and costs nothing. If it
is NOT gated, a container starts and runs nvidia-smi for a few seconds.

    GPU=A100 python -m modal run src/gpu_check.py
"""

import os

import modal

GPU = os.environ.get("GPU", "A10G")

app = modal.App("ser-gpu-check")


@app.function(gpu=GPU, image=modal.Image.debian_slim(), timeout=120)
def ping() -> str:
    import subprocess

    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return (r.stdout or r.stderr).strip()


@app.local_entrypoint()
def main():
    print(f"RESULT {GPU}: OK -> {ping.remote()}")
