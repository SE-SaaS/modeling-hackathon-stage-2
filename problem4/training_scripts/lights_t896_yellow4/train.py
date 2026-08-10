"""
train.py — experiment: lights_t896_yellow4
=========================================
Thin entry point. shared_code does the work; this file only names the
experiment and hands off.

Run locally:  python training_scripts/lights_t896_yellow4/train.py
Run on Modal: modal run training_scripts/lights_t896_yellow4/train.py

The Modal path trains AND pulls the results down to this folder's results/ dir
by itself, including when the run crashes. Re-running is safe: it detects what
already exists locally and remotely and resumes rather than starting over.

  modal run .../train.py                  train (or resume), then download
  modal run .../train.py --fetch-only     download only, never train
  modal run .../train.py --force          retrain from scratch, ignore state
  modal run .../train.py --weights-only   download only best.pt/last.pt/onnx
"""

import sys
from pathlib import Path

EXP_NAME = "lights_t896_yellow4"

for _cand in (Path("/root/training_scripts"), Path(__file__).resolve().parent.parent):
    if (_cand / "shared_code.py").exists():
        PKG_ROOT = _cand
        break
else:
    PKG_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc          # noqa: E402

EXP_DIR = PKG_ROOT / EXP_NAME
cfg = sc.load_config(EXP_DIR)


def run_local():
    sc.set_seed(cfg["reproducibility"]["seed"], cfg["reproducibility"]["deterministic"])
    sc.run_experiment(cfg, EXP_DIR)


try:
    import modal

    app = modal.App(f"tiny-{EXP_NAME}".replace("_", "-"))
    _image = sc.modal_image()
    _data_vol = modal.Volume.from_name(cfg["modal"]["data_volume"], create_if_missing=True)
    _runs_vol = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(
        image=_image,
        volumes={cfg["modal"]["data_mount"]: _data_vol,
                 cfg["modal"]["runs_mount"]: _runs_vol},
        **sc.modal_resources(cfg),
    )
    def train_remote(fresh: bool = False):
        rcfg = sc.remote_cfg(cfg)
        out_dir = Path(cfg["modal"]["runs_mount"]) / EXP_NAME
        sc.set_seed(rcfg["reproducibility"]["seed"], rcfg["reproducibility"]["deterministic"])
        try:
            sc.run_experiment(rcfg, EXP_DIR, out_dir=out_dir,
                              persist_fn=_runs_vol.commit, fresh=fresh)
        finally:
            _runs_vol.commit()

    @app.local_entrypoint()
    def modal_main(force: bool = False, fetch_only: bool = False,
                   weights_only: bool = False):
        """Runs on the LOCAL machine: decides train vs resume vs just-download,
        then syncs /runs/lights_t896_yellow4 into this folder's results/ dir. See the module
        docstring for the flags."""
        sc.orchestrate(
            cfg=cfg,
            exp_name=EXP_NAME,
            local_results_dir=Path(__file__).resolve().parent / "results",
            volume=_runs_vol,
            train_fn=train_remote.remote,
            force=force,
            fetch_only=fetch_only,
            weights_only=weights_only,
        )

except ImportError:
    pass


if __name__ == "__main__":
    run_local()
