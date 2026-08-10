# Problem 4 — Too Small to Handle

**Branch:** `problem4-tiny-detector` · **Kaggle score: 0.80439**

---

## 1. The task

One dashcam-style image goes in. Find two very different things in it:

| | Traffic lights | Road signs |
| --- | --- | --- |
| Classes | 3 — red, yellow, green | 8 — bus stop, crossroad, no entry, … |
| Typical size | **12 pixels wide** | **458 pixels wide** |
| Training examples | 5,442 | 685 (only 70–140 per class) |

Score is an equal split of the two:

```text
Score = 0.5 * F1_lights  +  0.5 * mAP50_signs
```

Both halves are "higher is better", max 1.0.

Two extra rules that shape everything:
- **CPU only at test time**, hard **150 ms per frame (p95)**. Miss it and you are
  not ranked at all — no partial credit.
- You submit **code, not predictions**. They run your `predict.py` themselves on
  images you never see.

---

## 2. Why this is hard

A traffic light is ~12 px wide in a 2704-px-wide photo. The standard move is to
shrink the image to 640 px before detection. Do that and the light becomes
**2.8 px** — smaller than one grid cell of the detector, and its colour is gone.
The model cannot find what the resize destroyed.

Meanwhile the signs are huge and easy, but there are barely any examples of each.

So it is really two different problems sharing one folder.

---

## 3. What we found in the data

Run `python data_code/01_analyze_dataset.py` → writes
[`01_dataset_analysis.txt`](data_code/01_dataset_analysis.txt).

Three facts, all measured, that decided the whole design:

1. **Lights and signs never appear in the same image.** Not once in 2,595
   images. Wide landscape photos (2704×1520) contain *only* lights. Tall
   portrait phone photos contain *only* signs.
2. **A light really is 12 px** (median). At 640 px input → 2.8 px. Confirmed
   unusable.
3. **Signs are already big enough.** Median 458 px. Their problem is having only
   ~100 examples per class, not resolution.

---

## 4. The solution

Because lights and signs never share an image, we check the image shape and send
it to **one of two separate models** — never both. This halves the work per
frame.

```text
                    ┌─ wide image (ratio > 1.6) ──→  LIGHTS model
image ──→ shape? ──┤
                    └─ tall image               ──→  SIGNS model
```

### The lights model — how we keep the 12 px light

Instead of shrinking the whole photo, we **cut out a horizontal strip and slice
it into 3 squares**:

```text
2704 x 1520 photo
   │  keep only the middle strip (30%-90% of the height) — the sky and road
   │  contain no lights, so throwing them away costs nothing
   ▼
2704 x 912 strip
   │  slice across into 3 overlapping squares
   ▼
[912x912] [912x912] [912x912]   →  detector runs on each at 768 px
```

A 12 px light inside a 912 px tile, viewed at 768 px, is **10.1 px** — big
enough for the detector to see. Same total pixels as one big wide image, but
every piece is square, which is what the training library requires.

The 3 tiles' results are merged back into the original photo's coordinates.

### The signs model

No tricks needed. One pass over the whole image at 640 px.

---

## 5. The two models

Both are the **same off-the-shelf network**: `yolo11n` ("YOLO11 nano"), the
smallest in its family, pre-trained on COCO.

| | Lights model | Signs model |
| --- | --- | --- |
| Network | `yolo11n` | `yolo11n` |
| Size | 2.6M parameters (~5.5 MB) | 2.6M parameters (~5.5 MB) |
| Detects | 3 classes | 8 classes |
| Sees | 3 tiles @ 768 px | 1 whole image @ 640 px |
| Score | **F1 0.793** | **mAP50 0.973** |

**Why two models and not one?** One network can only take one input size. The
lights need the crop-and-tile trick; the signs do not. Sharing a backbone would
force both to use the same input and destroy the lights. Two small models are
also faster than one big one, and can be tuned and debugged separately.

**Why the *smallest* network?** The 150 ms deadline is pass/fail. The lights
branch already spends its budget running 3 tiles instead of 1. Capacity was
never the problem — resolution was — so the compute goes there instead.

**Why COCO pre-training?** COCO happens to include `traffic light` and
`stop sign` among its 80 classes, so the network already knows what a traffic
light looks like before we start. Competition rules allow COCO/ImageNet and ban
any traffic-specific dataset, so this is the best legal starting point.

---

## 6. The experiments — what each one tested

13 runs, each changing **one thing** so the result is attributable. All trained
on Modal (cloud A100 GPUs), in parallel.

### Lights runs

| # | Run | The one thing it changed | mAP50 | **Real F1** | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | `lights_t640` | input 640 px (light = 8.4 px) | 0.568 | — | too small |
| 2 | `lights_t768` | input 768 px (light = 10.1 px) | 0.639 | — | the reference |
| 3 | `lights_t896` | input 896 px (light = 11.8 px) | 0.648 | — | barely helps |
| 4 | `lights_t768_p2` | detector head that looks at finer detail | 0.640 | — | costs more, no gain |
| **5** | **`lights_t768_yellow4`** | **yellow examples repeated ×4** | 0.656 | **0.793** | ✅ **SHIPPED** |
| 6 | `lights_t768_wideband` | taller strip (more coverage, smaller lights) | 0.632 | — | worse trade |
| 7 | `lights_t768_y26` | newer network, no NMS step | not run | — | latency idea |
| 12 | `lights_t896_yellow4` | ×4 yellow **and** 896 px | 0.677 | 0.753 | best mAP50, worse in reality |
| 13 | `lights_t768_yellow8` | yellow repeated ×8 instead of ×4 | 0.657 | 0.671 | overfits |

### Signs runs

| # | Run | The one thing it changed | mAP50 | Verdict |
| --- | --- | --- | --- | --- |
| **8** | **`signs_640`** | plain baseline | **0.973** | ✅ **SHIPPED** |
| 9 | `signs_640_aug` | heavy image augmentation | running | — |
| 10 | `signs_960` | bigger input, 960 px | 0.957 | worse **and** slower |
| 11 | `signs_640_aug_y26` | newer network | not run | — |

> **Two different columns, and they disagree.** `mAP50` is the training
> library's own score, measured on *tiles*. **Real F1** is the actual
> competition metric, measured on *whole images* — the competition matches
> lights by how close the centre is, not by box overlap, which is far more
> forgiving on a 12 px box. Run 12 has the best `mAP50` and run 13 the most
> balanced, yet **both lose to run 5 on the metric that counts**. Picking on
> `mAP50` alone would have shipped a worse model.

---

## 7. What we learned

1. **Resolution is the whole game for lights — and it stops paying at 768 px.**
   640→768 gained a lot; 768→896 gained almost nothing but costs 35% more time.
2. **The confidence threshold was worth +0.12 score and had never been set.**
   Every box you submit counts as a guess, so submitting low-confidence boxes
   floods the score with false alarms. Best value: **0.40**.
3. **Yellow is the bottleneck.** Only 99 training examples but worth a third of
   the lights score. Repeating those images ×4 helped; ×8 overfit and hurt.
4. **Signs did not need anything clever.** The plain baseline won; a bigger
   input made it *worse* (0.957 vs 0.973).
5. **Test on the real metric, always.** Three separate times the training
   library's score pointed at the wrong model.
6. **The rules' submission format is wrong.** Images with no detections are
   documented as the word `none`, but the grader rejects that *and* rejects
   blanks. A dummy class-0 box at confidence 0.0001 satisfies both harmlessly.

---

## 8. Status

- [x] Analyse the data
- [x] Build the training pipeline (Modal, self-syncing, resumable)
- [x] Run 13 experiments (10 finished)
- [x] Build the real-metric evaluator
- [x] Pick the models — `lights_t768_yellow4` + `signs_640`, conf 0.40
- [x] Kaggle practice submission — **0.80439**
- [ ] **`predict.py` on ONNX Runtime** ← the blocker
- [ ] Measure p95 latency, confirm under 150 ms
- [ ] INT8 quantise if needed, re-check accuracy
- [ ] Package the Drive zip

> ### ⚠ The open problem
>
> Accuracy is fine; **speed is not**. Current p95 is **209 ms against a 150 ms
> limit** — measured through PyTorch, which is not the deployment path. The fix
> is ONNX Runtime (typically 3× faster on CPU) and, if that is not enough, INT8
> quantisation or dropping to 2 tiles. Until this passes, the accuracy does not
> matter: over the limit means not ranked.

---

## 9. How to run it

```bash
# analyse the dataset
python problem4/data_code/01_analyze_dataset.py

# train one experiment on Modal (also downloads its own results when done,
# resumes if it crashed, does nothing if already complete)
modal run problem4/training_scripts/lights_t768_yellow4/train.py

# status of every experiment
python problem4/training_scripts/_fetch_all.py

# score a model pair on the REAL metric + measure CPU latency
python problem4/training_scripts/evaluate.py \
    --lights lights_t768_yellow4 --signs signs_640 --sweep-conf

# build submission.csv
python problem4/training_scripts/make_submission.py \
    --lights lights_t768_yellow4 --signs signs_640 --conf 0.40
```

Full command list, including one-time Modal setup:
[`modal_commands.txt`](modal_commands.txt).

### Files

```text
problem4/
  README.md                     this file
  modal_commands.txt            every command, in order
  submission.csv                current Kaggle submission
  data_code/
    01_analyze_dataset.py       dataset analysis
    01_dataset_analysis.txt     its output — the evidence for every decision
  training_scripts/
    shared_code.py              engine: tiling, training, export, real metric
    shared_config.yaml          settings shared by all experiments
    _make_experiments.py        generates the 13 experiment folders
    _fetch_all.py               status table + download results
    evaluate.py                 score on the real metric + time it on CPU
    make_submission.py          write submission.csv
    <experiment>/
      config.yaml               only what this experiment changes
      train.py                  entry point (runs locally or on Modal)
      results/                  downloaded outputs (weights are gitignored)
```

---

## 10. Known risks

- **The shape-based router assumes lights and signs never share an image.** True
  for all 2,595 images we have, but it is a property of how this data was
  collected. If the private test set mixes them, `predict.py` must run both
  models when the shape is ambiguous.
- **The crop caps lights recall at ~96%.** Lights outside the 30–90% strip are
  invisible to us. Deliberate: the resolution gained is worth more than the tail.
- **INT8 quantisation hurts small objects most.** Ship it only if measured F1
  holds.
