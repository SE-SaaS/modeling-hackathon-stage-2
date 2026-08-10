# Problem 5 — 3D RF Ultrasound Chemotherapy Response Prediction

**Final scores are at the bottom of this file → [FINAL TEST SCORES](#final-test-scores)**

---

## Data

26 labelled cases (18 progressive/stable, 8 non-progressive), Analyze 7.5
`.hdr/.img` pairs, `(289, W, n_slices)` int16 with per-case variation in width
(648/656) and slice count (8–15). Tumour masks cover ~0.4–1.9% of each volume.

Two properties measured rather than assumed, both of which change the results if
you get them wrong:

- **Axial axis = 0.** Verified spectrally — mean power-spectrum centroid is
  8.5 MHz along axis 0 (matching the 10 MHz transducer) vs 4.6 MHz along axis 1.
  The Hilbert transform and all spectral features run along this axis.
- **A large DC offset is present** (slice means ≈ −11000 on an int16 range) and
  is removed per A-line, otherwise the DC bin dominates every spectrum.

## Method

No deep learning. With 26 cases a 3D CNN memorises the cohort; the defensible
route (and the one the reference paper takes) is quantitative-ultrasound and
texture features feeding a small regularised classifier.

**122 features per case**, aggregated from slice level by mean/std/min/max:

| Family | What it measures |
| --- | --- |
| First-order | echo brightness and variability inside the mask |
| Speckle (Nakagami) | scatterer organisation — necrosis changes this |
| QUS spectral | slope / intercept / midband fit from **raw RF** — scatterer size and acoustic concentration. The information the envelope discards, and the reason an RF dataset beats a B-mode one. |
| GLCM texture | intra-tumour heterogeneity on the log-envelope |
| Shape | tumour volume and per-slice area |

**Validation: leave-one-case-out.** Every slice of a case is aggregated into one
row before modelling, so slice-level leakage is structurally impossible rather
than merely avoided. Scaling and feature selection are refit inside every fold.
All models use `class_weight='balanced'` because the ranking metric is macro F1
on an 18/8 split.

## Model comparison (leave-one-case-out)

| Model | macro F1 | balanced acc | AUROC | sens | spec |
| --- | --- | --- | --- | --- | --- |
| **logreg_l1_C1** | **0.756** | 0.799 | **0.882** | 0.72 | 0.88 |
| logreg_l2_C1 | 0.745 | 0.764 | 0.792 | 0.78 | 0.75 |
| svm_linear | 0.745 | 0.764 | 0.743 | 0.78 | 0.75 |
| logreg_l2_C0.1 | 0.710 | 0.736 | 0.771 | 0.72 | 0.75 |
| svm_rbf | 0.660 | 0.674 | 0.694 | 0.72 | 0.62 |
| rf_200 | 0.511 | 0.514 | 0.715 | 0.78 | 0.25 |
| gb_shallow | 0.458 | 0.458 | 0.521 | 0.67 | 0.25 |

Linear models beat the tree ensembles decisively — expected at n=26, where
trees overfit the majority class (specificity collapses to 0.25).

## Is it real? Permutation test

300 label shuffles, same LOO procedure:

- real macro F1 **0.756**
- null mean 0.471, null 95th percentile 0.694
- **p = 0.0166**

The result is above chance. With 26 cases and 122 features this check is not a
formality — a naive pipeline can reach 0.69 on shuffled labels.

## The important caveat: how much is just tumour size?

Ablating feature families (best model per set, LOO macro F1):

| Feature set | n features | macro F1 | AUROC |
| --- | --- | --- | --- |
| Shape/size only | 6 | **0.782** | 0.889 |
| All features | 122 | 0.756 | 0.882 |
| Speckle only | 20 | 0.782 | 0.660 |
| QUS only | 24 | 0.710 | 0.826 |
| **All except shape/size** | 116 | **0.627** | 0.729 |
| GLCM texture only | 24 | 0.578 | 0.583 |

**Tumour volume alone matches or beats the full model.** In the fitted L1 model
`shp_voxels` carries a coefficient 4.6× larger than any other feature.

This is the single most important thing to know about this result. It may be
genuine biology — non-progressive tumours being larger is plausible — but it may
equally be a cohort artifact that will not transfer to the organisers' held-out
cases. The texture/QUS signal is real but weaker (0.627 without size, still well
above the 0.471 null).

We report both, and recommend the full model, because size is a legitimate
measurable property of the provided masks rather than a leak. But a reviewer
should know the model is substantially a size classifier.

## Reproducing

```bash
pip install -r requirements.txt
python run.py --data "<path to RF_ultrasound_dataset>" --permutation
```

Outputs `features.csv`, `results_models.csv`, `results_summary.json`.

---

# FINAL TEST SCORES

**Model: L1-regularised logistic regression** (`logreg_l1_C1`), 20 features
selected inside each fold, `class_weight='balanced'`.

**Validation: leave-one-case-out over all 26 labelled cases.** No held-out
organiser test set has been released to us, so these are the honest
cross-validated numbers, not test-set numbers.

| Metric | Score |
| --- | --- |
| **Macro F1 (primary ranking metric)** | **0.756** |
| Balanced accuracy | 0.799 |
| AUROC | 0.882 |
| Accuracy | 0.769 |
| Sensitivity (progressive recall) | 0.722 |
| Specificity (non-progressive recall) | 0.875 |
| F1 progressive | 0.812 |
| F1 non-progressive | 0.700 |

Confusion matrix `[[TN, FP], [FN, TP]] = [[7, 1], [5, 13]]`

**Statistical significance:** permutation test over 300 label shuffles,
**p = 0.0166** (null mean 0.471).

**Known limitation:** tumour size alone achieves 0.782 macro F1; removing all
size features drops the model to 0.627. The model is substantially driven by
tumour volume, which may not transfer to the held-out cohort.
