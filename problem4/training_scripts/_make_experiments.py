"""
_make_experiments.py — generates the experiment folders.

Every experiment is a thin train.py (identical except for EXP_NAME) plus a
config.yaml holding only what that experiment changes. Generating them keeps the
9 train.py files from drifting apart. Re-run it after editing the template;
it never overwrites a config.yaml that already exists.

    python training_scripts/_make_experiments.py
"""

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

TRAIN_PY = '''"""
train.py — experiment: {name}
{underline}
Thin entry point. shared_code does the work; this file only names the
experiment and hands off.

Run locally:  python training_scripts/{name}/train.py
Run on Modal: modal run training_scripts/{name}/train.py

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

EXP_NAME = "{name}"

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

    app = modal.App(f"tiny-{{EXP_NAME}}".replace("_", "-"))
    _image = sc.modal_image()
    _data_vol = modal.Volume.from_name(cfg["modal"]["data_volume"], create_if_missing=True)
    _runs_vol = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(
        image=_image,
        volumes={{cfg["modal"]["data_mount"]: _data_vol,
                 cfg["modal"]["runs_mount"]: _runs_vol}},
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
        then syncs /runs/{name} into this folder's results/ dir. See the module
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
'''

# -----------------------------------------------------------------------------
# The grid.
#
# LIGHTS — the score lives here. Band 0.30-0.90 crops the frame to 2704x912,
# tiled into 3 squares of 912. A 12 px light becomes 12 * imgsz/912 px:
#     imgsz 640 -> 8.4 px      cost 3*640^2 = 1.23 MPix
#     imgsz 768 -> 10.1 px     cost 3*768^2 = 1.77 MPix
#     imgsz 896 -> 11.8 px     cost 3*896^2 = 2.41 MPix
# The size sweep answers accuracy-vs-latency directly. The other three vary one
# thing each against the 768 middle: the head, the yellow imbalance, the band.
#
# SIGNS — big and obvious, but ~70-140 instances per class. Resolution is not
# the problem, data is; so the sweep is augmentation, not input size.
# -----------------------------------------------------------------------------
EXPERIMENTS = {
    # ---------------- lights: input size sweep ----------------
    "lights_t640": dict(
        notes="Lights baseline. 3 tiles of the 0.30-0.90 band @640 -> 8.4px light. "
              "Cheapest of the sweep; the latency floor to beat.",
        dataset=dict(branch="lights", cache_key="lights_b30_90_t3"),
        train=dict(model="yolo11n.pt", imgsz=640, batch=32, epochs=70),
    ),
    "lights_t768": dict(
        notes="Middle of the size sweep @768 -> 10.1px light. The reference the "
              "head/yellow/band variants are compared against.",
        dataset=dict(branch="lights", cache_key="lights_b30_90_t3"),
        train=dict(model="yolo11n.pt", imgsz=768, batch=24, epochs=70),
    ),
    "lights_t896": dict(
        notes="Largest of the sweep @896 -> 11.8px light, near native. Tests "
              "whether accuracy is still climbing when latency runs out.",
        dataset=dict(branch="lights", cache_key="lights_b30_90_t3"),
        train=dict(model="yolo11n.pt", imgsz=896, batch=16, epochs=70),
    ),

    # ---------------- lights: one-variable variants @768 ----------------
    "lights_t768_p2": dict(
        notes="Stride-4 (P2) head instead of stride-8. Ultralytics has no "
              "yolo11-p2, so this is yolov8n-p2, warm-started from yolov8n.pt "
              "(the P2 layers themselves start random). Tests whether a finer "
              "grid beats a bigger input at equal cost.",
        dataset=dict(branch="lights", cache_key="lights_b30_90_t3"),
        train=dict(model="yolov8n-p2.yaml", load_from="yolov8n.pt",
                   imgsz=768, batch=16, epochs=90),
    ),
    "lights_t768_yellow4": dict(
        notes="Yellow is 99 of 5442 light instances and a third of F1_lights. "
              "Duplicates every tile containing one 4x. Same geometry as "
              "lights_t768, so the delta is purely the imbalance fix.",
        dataset=dict(branch="lights", cache_key="lights_b30_90_t3_y4",
                     yellow_oversample=4),
        train=dict(model="yolo11n.pt", imgsz=768, batch=24, epochs=70),
    ),
    "lights_t768_wideband": dict(
        notes="Band 0.20-0.95 instead of 0.30-0.90: 98.2%/98.7% box coverage "
              "instead of 96.0%/97.8%, but a taller crop means smaller lights "
              "at the same imgsz. Tests whether the recall ceiling is worth it.",
        dataset=dict(branch="lights", cache_key="lights_b20_95_t3",
                     band=[0.20, 0.95]),
        train=dict(model="yolo11n.pt", imgsz=768, batch=24, epochs=70),
    ),

    # ---------------- signs ----------------
    "signs_640": dict(
        notes="Signs baseline. Median sign is 458px, so 640 is already generous; "
              "this exists to show that more resolution is not the answer.",
        dataset=dict(branch="signs", cache_key="signs"),
        train=dict(model="yolo11n.pt", imgsz=640, batch=16, epochs=110),
    ),
    "signs_640_aug": dict(
        notes="Same as signs_640 with much heavier augmentation. With ~70-140 "
              "instances per class the bottleneck is data, not capacity.",
        dataset=dict(branch="signs", cache_key="signs"),
        train=dict(model="yolo11n.pt", imgsz=640, batch=16, epochs=140,
                   mosaic=1.0, close_mosaic=15, mixup=0.15, copy_paste=0.3,
                   scale=0.9, degrees=10.0, translate=0.2, shear=3.0,
                   perspective=0.0005, hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
                   erasing=0.2),
    ),
    "signs_960": dict(
        notes="Signs at 960 with the heavy augmentation. Control for the claim "
              "that signs do not need resolution; if this ties signs_640_aug, "
              "ship the 640 and spend the time on lights.",
        dataset=dict(branch="signs", cache_key="signs"),
        train=dict(model="yolo11n.pt", imgsz=960, batch=8, epochs=140,
                   mosaic=1.0, close_mosaic=15, mixup=0.15, copy_paste=0.3,
                   scale=0.9, degrees=10.0, translate=0.2, shear=3.0,
                   perspective=0.0005, hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
                   erasing=0.2),
    ),
}


def main():
    for name, spec in EXPERIMENTS.items():
        d = HERE / name
        d.mkdir(parents=True, exist_ok=True)

        (d / "train.py").write_text(
            TRAIN_PY.format(name=name, underline="=" * (len(name) + 22)),
            encoding="utf-8")

        cfg_path = d / "config.yaml"
        if cfg_path.exists():
            print(f"  kept   {name}/config.yaml (already exists)")
            continue

        cfg = {"experiment": {"name": name, "notes": spec["notes"]},
               "dataset": spec["dataset"],
               "train": spec["train"]}
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"  wrote  {name}/")

    print(f"\n{len(EXPERIMENTS)} experiments in {HERE}")


if __name__ == "__main__":
    main()
