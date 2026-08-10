"""
features.py — one feature vector per CASE.

With 26 labelled cases, a 3D CNN would memorise the cohort. The evidence-backed
route for this dataset is the one the source paper takes: quantitative
ultrasound and texture descriptors, then a small regularised classifier. Every
feature here is cheap, interpretable, and computed inside the tumour mask.

Four families, each answering a different physical question:

  first-order   how bright and how variable is the tumour echo?
  speckle       Nakagami/Rayleigh statistics — how ordered are the scatterers?
                (pre/post chemotherapy necrosis changes scatterer organisation)
  QUS spectral  from RAW RF, before enveloping: spectral slope, intercept and
                midband fit relate to scatterer size and acoustic concentration.
                This is the information the envelope throws away, and the reason
                an RF dataset is more informative than a B-mode one.
  texture       GLCM on the log-compressed envelope — intra-tumour heterogeneity,
                which is precisely what the reference paper links to response.

Slice-level features are aggregated to the case with mean/std/percentiles, since
the case is the unit of prediction.
"""

from __future__ import annotations

import numpy as np

from .data import FS_HZ, Case

EPS = 1e-9


# -----------------------------------------------------------------------------
def _first_order(v: np.ndarray) -> dict:
    """Intensity distribution inside the ROI."""
    if v.size == 0:
        return {}
    p = np.percentile(v, [10, 25, 50, 75, 90])
    mean, std = float(v.mean()), float(v.std())
    return {
        "fo_mean": mean, "fo_std": std,
        "fo_cv": std / (abs(mean) + EPS),                 # speckle contrast
        "fo_skew": float(((v - mean) ** 3).mean() / (std ** 3 + EPS)),
        "fo_kurt": float(((v - mean) ** 4).mean() / (std ** 4 + EPS)),
        "fo_p10": float(p[0]), "fo_p25": float(p[1]), "fo_p50": float(p[2]),
        "fo_p75": float(p[3]), "fo_p90": float(p[4]),
        "fo_iqr": float(p[3] - p[1]),
        "fo_entropy": _entropy(v),
    }


def _entropy(v: np.ndarray, bins: int = 64) -> float:
    h, _ = np.histogram(v, bins=bins)
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _speckle(env: np.ndarray) -> dict:
    """Nakagami m and Rayleigh-departure statistics of the envelope.

    m ~ 1 is fully developed Rayleigh speckle (dense random scatterers);
    m < 1 indicates pre-Rayleigh (sparse/clustered) and m > 1 post-Rayleigh
    (periodic/coherent) conditions. Necrosis after chemotherapy changes
    scatterer organisation, so this is a physically motivated response marker.
    """
    if env.size == 0:
        return {}
    e2 = env ** 2
    m2, m4 = e2.mean(), (e2 ** 2).mean()
    nak_m = float((m2 ** 2) / max(m4 - m2 ** 2, EPS))
    r = float(env.mean() / (np.sqrt(e2.mean()) + EPS))    # 0.886 for Rayleigh
    return {"spk_nakagami_m": nak_m,
            "spk_nakagami_omega": float(m2),
            "spk_R": r,
            "spk_rayleigh_dev": abs(r - 0.8862),
            "spk_snr": float(env.mean() / (env.std() + EPS))}


def _qus_spectrum(rf_slice: np.ndarray, mask_slice: np.ndarray) -> dict:
    """Quantitative-ultrasound parameters from the RAW RF power spectrum.

    Fits a straight line to the mean log power spectrum over the transducer's
    usable band (3-18 MHz around the 10 MHz centre). Slope tracks effective
    scatterer size, midband fit tracks acoustic concentration. These are the
    features that justify using RF rather than B-mode at all.
    """
    rows = np.where(mask_slice.any(axis=1))[0]
    if rows.size < 16:
        return {}
    cols = np.where(mask_slice.any(axis=0))[0]
    seg = rf_slice[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    n = seg.shape[0]
    if n < 16:
        return {}

    win = np.hanning(n)[:, None]
    P = np.abs(np.fft.rfft(seg * win, axis=0)) ** 2
    P = P.mean(axis=1)
    f = np.fft.rfftfreq(n, d=1.0 / FS_HZ) / 1e6                    # MHz
    band = (f >= 3.0) & (f <= 18.0)
    if band.sum() < 4:
        return {}

    logP = 10.0 * np.log10(np.maximum(P[band], EPS))
    fb = f[band]
    slope, intercept = np.polyfit(fb, logP, 1)
    return {
        "qus_slope": float(slope),                 # dB/MHz
        "qus_intercept": float(intercept),         # dB
        "qus_midband": float(np.polyval([slope, intercept], fb.mean())),
        "qus_centroid": float((fb * P[band]).sum() / (P[band].sum() + EPS)),
        "qus_bw": float(np.sqrt(((fb - (fb * P[band]).sum() /
                                  (P[band].sum() + EPS)) ** 2 * P[band]).sum()
                                / (P[band].sum() + EPS))),
        "qus_peak_f": float(fb[np.argmax(P[band])]),
    }


def _glcm(img_u8: np.ndarray, mask: np.ndarray) -> dict:
    """GLCM texture of the log-envelope inside the ROI bounding box.

    Intra-tumour heterogeneity is the property the reference paper ties to
    treatment response, and GLCM is the standard way to quantify it.
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        return {}
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size < 8 or cols.size < 8:
        return {}
    patch = img_u8[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    patch = (patch // 8).astype(np.uint8)                 # 32 grey levels
    g = graycomatrix(patch, distances=[1, 3], angles=[0, np.pi / 2],
                     levels=32, symmetric=True, normed=True)
    out = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy",
                 "correlation", "ASM"):
        out[f"glcm_{prop}"] = float(graycoprops(g, prop).mean())
    return out


def _shape(mask: np.ndarray) -> dict:
    """Coarse tumour geometry.

    Kept deliberately small and reported separately: tumour SIZE may correlate
    with response for reasons unrelated to texture, and with 26 cases a size
    feature can dominate. Check its importance before trusting the model.
    """
    vox = int(mask.sum())
    per_slice = mask.reshape(-1, mask.shape[-1]).sum(0)
    nz = per_slice[per_slice > 0]
    return {
        "shp_voxels": float(vox),
        "shp_n_slices_with_tumour": float((per_slice > 0).sum()),
        "shp_area_mean": float(nz.mean()) if nz.size else 0.0,
        "shp_area_std": float(nz.std()) if nz.size else 0.0,
        "shp_area_max": float(nz.max()) if nz.size else 0.0,
    }


# -----------------------------------------------------------------------------
def case_features(case: Case) -> dict:
    """Aggregate slice-level descriptors into ONE vector for the case."""
    env = case.envelope()
    logenv = case.log_envelope()
    mask = case.mask_bool()

    per_slice: list[dict] = []
    for s in range(case.n_slices):
        m = mask[:, :, s]
        if m.sum() < 32:                       # too little tumour to describe
            continue
        d = {}
        d.update(_first_order(logenv[:, :, s][m]))
        d.update(_speckle(env[:, :, s][m]))
        d.update(_qus_spectrum(case.rf[:, :, s], m))
        d.update(_glcm(logenv[:, :, s], m))
        per_slice.append(d)

    feats: dict[str, float] = {}
    if per_slice:
        keys = sorted({k for d in per_slice for k in d})
        for k in keys:
            vals = np.array([d[k] for d in per_slice if k in d], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            # Aggregate across slices: the case is the unit of prediction, and
            # the spread across slices is itself a heterogeneity measure.
            feats[f"{k}_mean"] = float(vals.mean())
            feats[f"{k}_std"] = float(vals.std())
            feats[f"{k}_min"] = float(vals.min())
            feats[f"{k}_max"] = float(vals.max())

    feats.update(_shape(mask))
    feats["n_slices"] = float(case.n_slices)
    return feats


def build_feature_table(cases):
    """(X DataFrame, y array, case_ids) — one row per case."""
    import pandas as pd
    rows, ys, ids = [], [], []
    for c in cases:
        rows.append(case_features(c))
        ys.append(c.label)
        ids.append(c.case_id)
    X = pd.DataFrame(rows, index=ids).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X, np.array(ys), ids
