# Too Small to Handle — Writeup

**Two small detectors behind a free router.** Final val score **0.812**
(F1_lights 0.721, mAP50_signs 0.904). Kaggle practice board: 0.804.

---

## 1. The observation everything else follows from

Before choosing an architecture we measured the dataset
(`01_analyze_dataset.py`). Three facts came out, and the third one decided the
whole design:

1. A traffic light is **12.0 px wide** (median), i.e. 0.0044 of the image width.
   Letterbox the frame to 640 and it becomes 2.8 px — below a stride-8 grid
   cell, colour already destroyed. Measured: only 8.3% of lights survive at
   ≥8 px at 640; at 1920 it is 54.8%, but that costs 9× the compute.
2. Signs are the opposite problem: median **458 px**, but only 70–140 instances
   per class.
3. **Lights and signs never appear in the same image. Zero times in 2,595.**
   Every light is in a 2704×1520 landscape dashcam frame; every sign is in a
   portrait phone photo (aspect ≤ 1.33).

Fact 3 is worth more than any architecture choice: a `w/h > 1.6` test routes
each frame to exactly one branch, with **0 objects missed** on both splits. Half
the per-frame compute, for one division.

## 2. Architecture

```text
                 ┌─ aspect > 1.6 ──→ LIGHTS: crop band, 3 tiles, 768   → cls 8,9,10
frame ─→ aspect ─┤
                 └─ aspect < 1.4 ──→ SIGNS : whole frame, 640          → cls 0..7
                   (ambiguous → run both, so the router can never silently
                    drop half the classes on unseen data)
```

**Lights — buying resolution by throwing pixels away.** Crop the `[0.30, 0.90]`
height band (holds 96.0% train / 97.8% val of light boxes, discards sky and
road), then slice it across into 3 square tiles:

```text
2704×1520 frame → 2704×912 band → 3 tiles of 912×912 → each detected at 768
```

A 12 px light in a 912 px tile viewed at 768 is **10.1 px** — visible. Cost is
3×768² = 1.77 MPix, identical to one wide 2304×768 pass, but every tensor is
square. That matters practically: Ultralytics accepts only an integer `imgsz`
for training, so a genuinely wide tensor cannot be trained at all. Tiles get the
same effective resolution within the tooling's constraints.

Both branches are **YOLO11n** (2.6M params), COCO-pretrained. COCO includes
`traffic light` and `stop sign` among its 80 classes, so the backbone starts
already knowing the target objects — the strongest pretraining the rules allow
(outside traffic datasets are banned).

## 3. What we tried, and what failed

13 controlled runs, each changing one variable. **Six of the eight ideas failed.**

| Idea | Result | Why it failed |
| --- | --- | --- |
| Input 640 (cheapest) | 0.568 | 8.4 px light — yellow collapses to 0.447 |
| Input 896 (dearest) | 0.648 | +0.010 over 768 for +35% latency. Saturated. |
| **Yellow oversample ×4** | **0.656** ✅ | **shipped** |
| Yellow oversample ×8 | real F1 **0.671** vs 0.793 | Duplicating 77 images 8× memorises them |
| Stride-4 (P2) head | 0.640 | Best localisation (mAP50-95 0.287), but the metric matches lights by *centre distance*, so finer boxes buy nothing — and it costs +20% size |
| Wider band 0.20–0.95 | 0.632 | +2% box coverage costs 2 px of light size. Bad trade. |
| Signs at 960 | 0.957 vs 0.973 | Signs were never resolution-limited |
| Signs heavy aug + yolo26n | 0.943 | Two variables at once — not attributable, but clearly worse |

**The most valuable failure: our own metric was lying to us.** Ultralytics
reports mAP50 on *tiles* at IoU 0.5. The competition scores *full frames* and
matches lights by centre distance. These disagree badly on 12 px boxes, so we
reimplemented the real metric (`evaluate.py`) and re-ranked everything:

| Run | tile mAP50 | **real F1** | would we have shipped it? |
| --- | --- | --- | --- |
| `lights_t896_yellow4` | **0.677** (best) | 0.753 | yes, on mAP50 — wrongly |
| `lights_t768_yellow8` | 0.657 | 0.671 | maybe — wrongly |
| `lights_t768_yellow4` | 0.656 | **0.793** | correct answer |

Three times the proxy pointed at a worse model. Validated end to end: local
0.8123 vs Kaggle 0.80439.

**The single cheapest win was not a model change at all.** Confidence is not a
cut-off in this metric — every submitted box counts as a detection — so the
threshold is a real tunable. Nobody had set it. F1_lights went **0.554 → 0.793**
between conf 0.05 and 0.40, collapsing again by 0.50. That is +0.12 total score
for one constant.

## 4. Speed

The deadline is pass/fail, so this got as much attention as accuracy.

| Configuration | p95 (4 threads, full 496 frames) |
| --- | --- |
| PyTorch/Ultralytics, 80-frame subset | 209 ms ← misleading |
| PyTorch/Ultralytics, full set | 533 ms |
| **ONNX Runtime FP32 (shipped)** | **197.7 ms** (p50 182.7) |
| ONNX Runtime + static INT8 | ~148 ms, but broken — see below |

Four things we got wrong, and the corrections:

1. **We benchmarked on an unrepresentative subset.** The first 80 val images are
   all 2704×1520; the full set contains 5712×4284 frames. p95 more than doubled
   when measured honestly. Always time the whole set.
2. **Dynamic INT8 made it slower** — 239 ms vs 176 ms. `quantize_dynamic` is
   built for transformers; on a conv net it inserts quantize/dequantize pairs
   without reaching integer convolution kernels.
3. **Static INT8 was fast but destroyed the model.** Calibrated per-channel QDQ
   cut p95 to ~148 ms and then returned **1 detection across 496 images**. Tiny
   objects are where int8 fails hardest, and this was a total collapse rather
   than a graceful degradation. We rejected it and shipped FP32. Quantizing
   without re-scoring would have handed in a model that passes the clock and
   detects nothing.
4. **The signs branch needed its own NMS.** The exported ONNX graph has no NMS
   baked in; Ultralytics had been doing it internally. One sign came back as ~17
   near-identical boxes and, since every box counts as a detection,
   mAP50_signs fell 0.904 -> 0.487 until we added it.

**We are over the 150 ms target at 4 threads (197.7 ms) and under it at 8.**
This is the honest state of the submission. The available lever, untaken for
lack of time, is dropping the lights branch to 640 (-30% compute) at a
measured accuracy cost, or QAT instead of PTQ.

We chose PTQ over QAT deliberately: Ultralytics exposes no QAT path, so it would
have meant hand-rolling `torch.ao.quantization` around their training loop with
no guarantee of a clean export — high integration risk for a typical +1–2 mAP.
We also skipped pruning: unstructured sparsity is not executed faster by ONNX
Runtime on CPU, so training a narrower model dominates it.

## 5. Honest limitations

- **The router is an empirical property of this dataset, not a law.** If the
  private set puts a light and a sign in one frame, a strict router would drop
  half the classes. Mitigated: only decisive aspect ratios take one branch;
  anything between 1.4 and 1.6 runs both.
- **The crop caps lights recall at ~96%.** Lights outside the 0.30–0.90 band are
  invisible. Deliberate — the resolution gained is worth more than the tail —
  but it is a hard ceiling, not noise.
- **Yellow is still the weak class** (99 training instances, worth a third of
  F1_lights). Oversampling ×4 helped; ×8 overfit. A dedicated colour classifier
  on light crops is the obvious next step and we ran out of time for it.
- **Latency does not clear 150 ms at 4 threads** (197.7 ms p95); it does at 8.
  This is the biggest open risk in the submission.
- **Per-class confidence thresholds are untuned.** We use 0.40 for all three
  colours; yellow being rare probably wants a different one.

## 6. Reproducing

```bash
pip install -r requirements.txt
python predict.py --test eval_data
```

Training and analysis live in the repo: `data_code/01_analyze_dataset.py`
(the measurements above), `training_scripts/` (13 experiments, one config each),
`evaluate.py` (the real metric + CPU timing), `quantize_static.py`.
