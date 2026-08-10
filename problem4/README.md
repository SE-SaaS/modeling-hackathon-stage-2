# Problem 4 — Too Small to Handle

Find 12-pixel traffic lights and 8 kinds of road sign in one image, on CPU,
inside a hard per-frame deadline.

`Score = 0.5 * F1_lights + 0.5 * mAP50_signs`

**Branch:** `problem4-tiny-detector`

---

## The three facts the whole design rests on

Measured in [data_code/01_dataset_analysis.txt](data_code/01_dataset_analysis.txt):

1. **Lights and signs never share an image.** 0 of 2595. Dashcam frames
   (2704×1520, ar 1.78) hold every light; portrait phone photos (ar ≤ 1.33) hold
   every sign. An aspect-ratio check routes each frame to one branch — so only
   one model runs per frame.
2. **A 12px light does not survive a resize.** Median light is 12.0 px wide.
   Full-frame at 640 → 2.8 px. At 960 → 4.3 px. Both unusable.
3. **Signs are not a resolution problem, they're a data problem.** Median sign
   is 458 px, but each class has only 70–140 instances.

## The approach

Two models, routed by aspect ratio.

**Lights** — crop the band `[0.30, 0.90]` of the frame height (96% of light
boxes, drops sky and road), cut it into 3 square tiles across the width, detect
at a normal square `imgsz`. A 2704×1520 frame → 2704×912 band → 3 tiles of 912.
At `imgsz=768` a 12px light is **10.1 px** — visible to a stride-8 head. Cost is
3×768² , the same as one wide 2304×768 pass, but every tensor is square so
Ultralytics trains it normally (it only accepts an int `imgsz` for training).

**Signs** — plain detector at 640 on the full image. The effort goes into
augmentation, not resolution.

---

## The models

Everything is **Ultralytics YOLO11 nano**, COCO-pretrained. Two separate
instances of it — one per branch — never a shared backbone.

| Property | Lights model | Signs model |
| --- | --- | --- |
| Checkpoint | `yolo11n.pt` | `yolo11n.pt` |
| Params | ~2.6 M | ~2.6 M |
| Classes | 3 (red/yellow/green) | 8 (sign types) |
| Input | 3 tiles @ 640–896 | 1 full frame @ 640 |
| FP32 ONNX | ~10 MB | ~10 MB |
| INT8 ONNX | ~3 MB | ~3 MB |

Combined on disk: **~6 MB INT8 / ~20 MB FP32**, comfortably inside any
plausible size budget. Runtime is **ONNX Runtime on CPU**, static shapes.

One experiment (`lights_t768_p2`) instead uses **`yolov8n-p2.yaml`** — a YOLOv8
nano with an extra stride-4 (P2) detection head. Ultralytics ships no
`yolo11-p2.yaml` (verified against the installed 8.4.11 package: only
`yolov8-p2.yaml` and `yolo26-p2.yaml` exist), so the P2 test has to be a v8. It
is warm-started from `yolov8n.pt`, meaning the backbone transfers but the P2
layers start random — expect it to need more epochs to be judged fairly.

**Why nano and nothing larger.** The deadline is pass/fail with no partial
credit, and the lights branch spends its budget on three forward passes per
frame rather than one. A `yolo11s` at 3×768² would not fit. Capacity is not the
binding constraint here — a nano has plenty for 3 and 8 classes with a few
thousand instances. Input resolution is the constraint, so that is where the
compute goes.

**Pretraining.** COCO includes `traffic light` (class 9) and `stop sign`
(class 11) among its 80 classes, so the backbone arrives already knowing what a
traffic light looks like — from far more instances than our 5442. The rules
permit COCO/ImageNet backbones and ban any outside traffic-light or sign
dataset, so this is the strongest pretraining legally available.

**`yolo26n` is tested as an alternative** (`lights_t768_y26`,
`signs_640_aug_y26`): same COCO pretraining, near-identical size (2.57M vs
2.62M params), but end-to-end / NMS-free. The lights branch runs 3 tiles per
frame and so pays NMS three times, making this a direct latency lever.

**What we are not using, and why:** no RT-DETR (transformer attention is slow on
CPU), no NanoDet/PicoDet (weaker tooling, no ONNX/export path this mature), no
two-stage detector, no separate colour-classifier CNN yet — that stays in
reserve for if yellow F1 comes back broken.

---

## Steps, in order

- [x] **1. Analyse the data** — `python problem4/data_code/01_analyze_dataset.py`
      → writes `01_dataset_analysis.txt`. Resolutions, per-class box sizes in
      pixels, spatial priors, post-resize light visibility, band/tile tradeoffs.
- [x] **2. Build the training engine** — `training_scripts/shared_code.py`
      (dataset construction, training, export, and the competition metric
      reimplemented so experiments are ranked on the real objective).
- [x] **3. Define the experiment grid** — `training_scripts/_make_experiments.py`
      generates 11 experiment folders, each a `config.yaml` + a thin `train.py`.
- [x] **4. Verify the tiling geometry** — round-trip test confirmed box centres
      reconstruct to 0.00 px on train; tile coverage has no gaps at any of the 7
      resolutions in the dataset.
- [x] **5. Make runs self-syncing** — `modal run .../train.py` trains *and*
      downloads its own results, including after a crash. Re-running detects
      local + remote state and resumes instead of restarting. State machine
      tested against a fake volume across all 5 paths.
- [ ] **6. Upload data to Modal + launch the 11 runs** — see
      [modal_commands.txt](modal_commands.txt). All 11 are independent and run in
      parallel. `python training_scripts/_fetch_all.py` gives a status table.
- [ ] **7. Pick the winners** — one lights config, one signs config, judged on
      `competition_score()`, not on Ultralytics' mAP (which weights the two
      halves completely differently).
- [ ] **8. Write `predict.py`** — router → crop/tile → ONNX Runtime → merge
      tiles → NMS → boxes in original-image pixels. Must import
      `tile_geometry()` from `shared_code` so inference and training geometry
      cannot drift.
- [ ] **9. Measure p95 latency on CPU** — the deadline is pass/fail, no partial
      credit. If it clears without INT8, ship FP32.
- [ ] **10. INT8 quantize, re-validate** — keep it only if light F1 holds.
- [ ] **11. Package the zip** — `predict.py`, `requirements.txt`, weights,
      `WRITEUP.md`, named `<team-slug>__<SECRET_CODE>.zip`.

---

## The experiment grid

| # | Experiment | Model | imgsz | What it varies | Light px |
|---|---|---|---|---|---|
| 1 | `lights_t640` | `yolo11n.pt` | 640 | size sweep, cheapest | 8.4 |
| 2 | `lights_t768` | `yolo11n.pt` | 768 | **the reference** | 10.1 |
| 3 | `lights_t896` | `yolo11n.pt` | 896 | size sweep, dearest | 11.8 |
| 4 | `lights_t768_p2` | `yolov8n-p2.yaml` ← `yolov8n.pt` | 768 | stride-4 head | 10.1 |
| 5 | `lights_t768_yellow4` | `yolo11n.pt` | 768 | yellow tiles ×4 | 10.1 |
| 6 | `lights_t768_wideband` | `yolo11n.pt` | 768 | band 0.20–0.95 | 8.1 |
| 7 | `lights_t768_y26` | `yolo26n.pt` | 768 | **NMS-free family** | 10.1 |
| 8 | `signs_640` | `yolo11n.pt` | 640 | baseline | — |
| 9 | `signs_640_aug` | `yolo11n.pt` | 640 | heavy augmentation | — |
| 10 | `signs_960` | `yolo11n.pt` | 960 | heavy aug + resolution | — |
| 11 | `signs_640_aug_y26` | `yolo26n.pt` | 640 | **NMS-free family** | — |

Experiments 4–6 each change exactly one thing against `lights_t768`, so each
result is attributable.

---

## Layout

```
problem4/
  README.md                    this file
  modal_commands.txt           every command, in order
  data_code/
    01_analyze_dataset.py      the analysis
    01_dataset_analysis.txt    its output
  training_scripts/
    shared_code.py             the engine
    shared_config.yaml         defaults, merged under every experiment
    _make_experiments.py       generates the 11 folders
    <experiment>/
      config.yaml              only what this experiment changes
      train.py                 thin entry point (local + Modal)
```

---

## Decisions made, and why

**Two models, not one with two heads.** A shared backbone runs on one input
tensor, which forbids the crop — and the crop is the whole point. Separate
models also train and debug independently.

**Tiles, not a wide tensor.** Ultralytics accepts only an int `imgsz` for
training. Tiling gets the same effective resolution at the same pixel cost with
square tensors throughout.

**No distillation, no pruning.** Pruning yields sparsity ONNX Runtime CPU won't
execute faster — training a smaller width is strictly better. Distillation needs
the teacher trained first, doubling a schedule that's already tight. Both are
last-20% moves; resolution and routing are the first 80%.

**PTQ, not QAT.** Ultralytics has no QAT path, so QAT means hand-rolling
`torch.ao.quantization` around their loop and hoping it exports. Dynamic INT8 is
20 minutes with a known-good FP32 fallback.

## Open risks

- The aspect-ratio router is a property of *this* collection. If the private
  test set mixes lights and signs in one frame, routing on it alone drops half
  the score. `predict.py` must run both branches when the ratio is ambiguous.
- The `[0.30, 0.90]` band caps light recall at ~96–98%. Deliberate: the
  resolution it buys is worth more than the tail.
- INT8 hurts small objects most. Gate it on measured val F1, not on the size win.
