"""
model.py — case-level cross-validation, metrics, and a small model zoo.

The whole design is dictated by n = 26 cases (18 vs 8):

  * **No deep learning.** A 3D CNN has more parameters than we have voxels of
    signal; it would memorise the cohort and tell us nothing. Small regularised
    classifiers on interpretable features are the defensible choice, and match
    what the reference paper does.
  * **Leave-One-Out at the CASE level.** With 8 minority cases, a 5-fold split
    puts 1-2 of them in each test fold, so fold-to-fold variance swamps any real
    difference. LOO uses every case as a test point exactly once.
  * **Every slice of a case stays together.** Features are already aggregated to
    the case, so leakage is structurally impossible rather than merely avoided.
  * **Scaling and selection go INSIDE the CV loop.** Fitting a scaler or picking
    features on all 26 cases and then cross-validating is the classic way to
    invent a few points of accuracy that vanish on the test set.
  * **Macro F1** is the ranking metric, so class_weight='balanced' everywhere —
    an unweighted model on 18/8 learns to answer "progressive" and scores well
    on accuracy while being useless.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 0


def model_zoo(k_features: int = 20) -> dict[str, Pipeline]:
    """Small, heavily regularised candidates.

    Each is a Pipeline so that scaling and univariate selection are refit inside
    every CV fold. k is small on purpose: with 26 samples, 20 features is
    already generous and anything larger fits noise.
    """
    def pipe(clf, k=k_features):
        return Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("clf", clf),
        ])

    return {
        "logreg_l2_C0.1": pipe(LogisticRegression(C=0.1, class_weight="balanced",
                                                  max_iter=5000)),
        "logreg_l2_C1": pipe(LogisticRegression(C=1.0, class_weight="balanced",
                                                max_iter=5000)),
        "logreg_l1_C1": pipe(LogisticRegression(C=1.0, penalty="l1",
                                                solver="liblinear",
                                                class_weight="balanced")),
        "svm_linear": pipe(SVC(kernel="linear", C=0.5, class_weight="balanced",
                               probability=True, random_state=RANDOM_STATE)),
        "svm_rbf": pipe(SVC(kernel="rbf", C=1.0, gamma="scale",
                            class_weight="balanced", probability=True,
                            random_state=RANDOM_STATE)),
        "rf_200": pipe(RandomForestClassifier(n_estimators=300, max_depth=3,
                                              min_samples_leaf=3,
                                              class_weight="balanced",
                                              random_state=RANDOM_STATE), k=40),
        "gb_shallow": pipe(GradientBoostingClassifier(n_estimators=100,
                                                      max_depth=2,
                                                      learning_rate=0.05,
                                                      random_state=RANDOM_STATE), k=40),
    }


def evaluate_loo(pipeline, X, y, k_features: int | None = None) -> dict:
    """Leave-one-case-out CV. Returns the metric set the organisers listed.

    Predictions are collected across folds and scored once, rather than averaged
    per fold: with a single test case per fold, per-fold F1 is degenerate.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(y)
    pred = np.zeros(n, dtype=int)
    prob = np.zeros(n, dtype=float)

    for tr, te in LeaveOneOut().split(X):
        p = _clone_fit(pipeline, X[tr], y[tr], k_features)
        pred[te] = p.predict(X[te])
        prob[te] = _prob(p, X[te])

    return _metrics(y, pred, prob)


def evaluate_repeated_cv(pipeline, X, y, n_splits: int = 5, n_repeats: int = 20,
                         k_features: int | None = None) -> dict:
    """Repeated stratified k-fold as a second opinion on LOO.

    LOO is nearly unbiased but high variance; repeated stratified CV is the
    opposite. Agreement between the two is a much better sign than either alone,
    and disagreement usually means the result is fragile.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    scores = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=r)
        pred = np.zeros(len(y), dtype=int)
        for tr, te in skf.split(X, y):
            p = _clone_fit(pipeline, X[tr], y[tr], k_features)
            pred[te] = p.predict(X[te])
        scores.append(f1_score(y, pred, average="macro"))
    return {"macro_f1_mean": float(np.mean(scores)),
            "macro_f1_std": float(np.std(scores)),
            "macro_f1_min": float(np.min(scores)),
            "macro_f1_max": float(np.max(scores))}


def _clone_fit(pipeline, Xtr, ytr, k_features):
    from sklearn.base import clone
    p = clone(pipeline)
    if k_features is not None and "select" in p.named_steps:
        p.set_params(select__k=min(k_features, Xtr.shape[1]))
    else:
        sel = p.named_steps.get("select")
        if sel is not None and isinstance(sel.k, int):
            p.set_params(select__k=min(sel.k, Xtr.shape[1]))
    p.fit(Xtr, ytr)
    return p


def _prob(p, X):
    if hasattr(p, "predict_proba"):
        return p.predict_proba(X)[:, 1]
    if hasattr(p, "decision_function"):
        return p.decision_function(X)
    return p.predict(X).astype(float)


def _metrics(y, pred, prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auroc = float(roc_auc_score(y, prob))
    except ValueError:
        auroc = float("nan")
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "auroc": auroc,
        "sensitivity": float(tp / max(tp + fn, 1)),      # progressive recall
        "specificity": float(tn / max(tn + fp, 1)),      # non-progressive recall
        "f1_progressive": float(f1_score(y, pred, pos_label=1)),
        "f1_nonprogressive": float(f1_score(y, pred, pos_label=0)),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def permutation_check(pipeline, X, y, n_perm: int = 200, k_features=None,
                      seed: int = 0) -> dict:
    """Re-run LOO on shuffled labels to get a null distribution.

    With 26 cases and hundreds of features, a macro F1 of 0.7 can happen by
    chance. This is the check that says whether the result means anything, and
    it is the first thing a reviewer should ask for.
    """
    rng = np.random.default_rng(seed)
    real = evaluate_loo(pipeline, X, y, k_features)["macro_f1"]
    null = []
    for _ in range(n_perm):
        null.append(evaluate_loo(pipeline, X, rng.permutation(y),
                                 k_features)["macro_f1"])
    null = np.array(null)
    return {"macro_f1": real, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": float(((null >= real).sum() + 1) / (len(null) + 1))}
