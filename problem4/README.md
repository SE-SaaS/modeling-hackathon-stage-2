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

## Steps, in order

- [x] **1. Analyse the data** — `python problem4/data_code/01_analyze_dataset.py`
      → writes `01_dataset_analysis.txt`. Resolutions, per-class box sizes in
      pixels, spatial priors, post-resize light visibility, band/tile tradeoffs.
- [x] **2. Build the training engine** — `training_scripts/shared_code.py`
      (dataset construction, training, export, and the competition metric
      reimplemented so experiments are ranked on the real objective).
- [x] **3. Define the experiment grid** — `training_scripts/_make_experiments.py`
      generates 9 experiment folders, each a `config.yaml` + a thin `train.py`.
- [x] **4. Verify the tiling geometry** — round-trip test confirmed box centres
      reconstruct to 0.00 px on train; tile coverage has no gaps at any of the 7
      resolutions in the dataset.
- [ ] **5. Upload data to Modal + launch the 9 runs** — see
      [modal_commands.txt](modal_commands.txt). All 9 are independent and run in
      parallel.
- [ ] **6. Pick the winners** — one lights config, one signs config, judged on
      `competition_score()`, not on Ultralytics' mAP (which weights the two
      halves completely differently).
- [ ] **7. Write `predict.py`** — router → crop/tile → ONNX Runtime → merge
      tiles → NMS → boxes in original-image pixels. Must import
      `tile_geometry()` from `shared_code` so inference and training geometry
      cannot drift.
- [ ] **8. Measure p95 latency on CPU** — the deadline is pass/fail, no partial
      credit. If it clears without INT8, ship FP32.
- [ ] **9. INT8 quantize, re-validate** — keep it only if light F1 holds.
- [ ] **10. Package the zip** — `predict.py`, `requirements.txt`, weights,
      `WRITEUP.md`, named `<team-slug>__<SECRET_CODE>.zip`.

---

## The experiment grid

| # | Experiment | What it varies | Light px |
|---|---|---|---|
| 1 | `lights_t640` | imgsz 640 | 8.4 |
| 2 | `lights_t768` | imgsz 768 — the reference | 10.1 |
| 3 | `lights_t896` | imgsz 896 | 11.8 |
| 4 | `lights_t768_p2` | stride-4 P2 head (yolov8n-p2) | 10.1 |
| 5 | `lights_t768_yellow4` | yellow tiles duplicated ×4 | 10.1 |
| 6 | `lights_t768_wideband` | band 0.20–0.95 (98% coverage) | 8.1 |
| 7 | `signs_640` | baseline | — |
| 8 | `signs_640_aug` | heavy augmentation | — |
| 9 | `signs_960` | heavy aug at 960 | — |

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
    _make_experiments.py       generates the 9 folders
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
