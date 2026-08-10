"""
_fetch_all.py — pull every experiment's results down in one command, and print
a status table.

Each experiment's own `modal run .../train.py` already syncs itself when it
finishes. This is for the times that is not enough: a terminal you closed, a
laptop that slept, results spread across several workspaces, or just wanting to
see where all 9 runs stand without reading 9 scrollbacks.

    python problem4/training_scripts/_fetch_all.py                 # status + fetch
    python problem4/training_scripts/_fetch_all.py --status        # status only
    python problem4/training_scripts/_fetch_all.py --weights-only  # skip plots
    python problem4/training_scripts/_fetch_all.py --only lights_t768 signs_640

Reads whichever workspace MODAL_CONFIG_PATH currently points at, so to collect
runs spread over several workspaces, set it and re-run — downloads are
idempotent, so re-running never re-transfers what is already local.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import shared_code as sc  # noqa: E402


def discover() -> list[str]:
    """Experiment folders = any dir here holding a config.yaml."""
    return sorted(p.name for p in HERE.iterdir()
                  if p.is_dir() and (p / "config.yaml").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="report only, download nothing")
    ap.add_argument("--weights-only", action="store_true", help="skip plots/CSVs")
    ap.add_argument("--only", nargs="*", default=None, help="limit to these experiments")
    args = ap.parse_args()

    import modal

    names = args.only or discover()
    if not names:
        print("no experiments found — run _make_experiments.py first")
        return

    cfg = sc.load_config(HERE / names[0])
    vol = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)
    print(f"volume: {cfg['modal']['runs_volume']}\n")

    rows = []
    for name in names:
        local_dir = HERE / name / "results"
        rstat = sc.remote_status(vol, name)
        lstat = sc.local_status(local_dir)

        if not args.status and rstat != "absent" and lstat != "complete":
            print(f"[fetch] {name} (local={lstat} remote={rstat})")
            sc.fetch_run(vol, name, local_dir, args.weights_only)
            lstat = sc.local_status(local_dir)

        # Surface the headline number so the table is worth reading.
        score = ""
        sp = local_dir / "summary.json"
        if sp.exists():
            try:
                s = json.loads(sp.read_text())
                v = s.get("val", {})
                score = f"mAP50={v.get('map50', float('nan')):.4f}"
                if s.get("onnx_mb"):
                    score += f"  onnx={s['onnx_mb']}MB"
            except Exception:
                score = "(unreadable summary.json)"
        rows.append((name, lstat, rstat, score))

    w = max(len(r[0]) for r in rows)
    print(f"\n{'experiment':<{w}}  {'local':<9} {'remote':<9} result")
    print("-" * (w + 32))
    for name, lstat, rstat, score in rows:
        print(f"{name:<{w}}  {lstat:<9} {rstat:<9} {score}")

    done = sum(1 for r in rows if r[1] == "complete")
    print(f"\n{done}/{len(rows)} complete locally")
    missing = [r[0] for r in rows if r[2] == "absent"]
    if missing:
        print(f"never started in this workspace: {', '.join(missing)}")
        print("(check the other workspaces — set MODAL_CONFIG_PATH and re-run)")


if __name__ == "__main__":
    main()
