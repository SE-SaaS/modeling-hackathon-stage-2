"""
data.py — loading the RF ultrasound cohort.

Real dataset layout (verified against the release):

    Training_Data/
      Progressive/        18 cases
        <Case>/
          <Case>_loc.hdr + .img        Analyze 7.5, int16, (289, W, n_slices, 1)
          Mask*/<Case>_Seg.hdr + .img  same shape, values {0, 1}
      Non-Progressive/    8 cases

So 26 labelled cases, 18 vs 8 — the label is the parent folder. Width varies
(648, 656, ...) and slice count varies (10, 11, ...), so nothing may assume a
fixed volume shape.

Two properties of this data that were measured, not assumed, and that the
feature code depends on:

  * **Axial axis = 0.** The 50 MHz sampling direction is the 289-sample axis.
    Verified spectrally: the mean power-spectrum centroid along axis 0 is
    8.5 MHz, consistent with the 10 MHz transducer, versus 4.6 MHz along axis 1.
    The Hilbert transform and all spectral features run along this axis; get it
    wrong and you measure the beam profile instead of the pulse.
  * **There is a large DC offset** (slice means around -11000 on a +/-32767
    int16 range). It must be removed per A-line before any spectral work, or
    the DC bin swamps the tissue signal.

The tumour mask covers only ~1% of the volume, so features are computed inside
the mask; using the whole frame would drown the tumour in background.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FS_HZ = 50e6          # sampling frequency, from the acquisition description
F0_HZ = 10e6          # transducer centre frequency
AXIAL_AXIS = 0        # verified spectrally; see module docstring

CLASS_DIRS = {"Progressive": 1, "Non-Progressive": 0}


@dataclass
class Case:
    """One mouse / tumour volume — the unit of prediction.

    Slices are NOT independent samples: every slice of a case must stay in the
    same CV partition or the estimate is meaningless.
    """

    case_id: str
    rf: np.ndarray                  # (H_axial, W, n_slices) float32, DC removed
    mask: np.ndarray | None = None  # same shape, bool
    label: int | None = None        # 1 = progressive/stable, 0 = non-progressive
    meta: dict = field(default_factory=dict)

    @property
    def n_slices(self) -> int:
        return self.rf.shape[2]

    def envelope(self) -> np.ndarray:
        """|analytic signal| along the axial axis — the B-mode envelope.

        Texture must be computed on this, not on raw RF: raw RF oscillates at
        the carrier frequency, so its 'texture' is mostly the pulse itself.
        """
        from scipy.signal import hilbert
        return np.abs(hilbert(self.rf, axis=AXIAL_AXIS))

    def log_envelope(self, dr_db: float = 60.0) -> np.ndarray:
        """Log-compressed envelope mapped to uint8 0..255 (displayed B-mode).

        GLCM/LBP need a bounded integer range, and log compression is what makes
        speckle statistics comparable across depth and gain settings.
        """
        env = self.envelope()
        env = env / max(float(env.max()), 1e-12)
        db = 20.0 * np.log10(np.maximum(env, 1e-6))
        return ((np.clip(db, -dr_db, 0.0) + dr_db) / dr_db * 255.0).astype(np.uint8)

    def mask_bool(self) -> np.ndarray:
        if self.mask is None:
            return np.ones_like(self.rf, dtype=bool)
        return self.mask.astype(bool)


# =============================================================================
# loading
# =============================================================================
def _read_analyze(hdr_path: Path) -> np.ndarray:
    """Analyze 7.5 volume -> (H, W, n_slices) float32, singleton axes dropped."""
    import nibabel as nib
    arr = np.asarray(nib.load(str(hdr_path)).get_fdata(), dtype=np.float32)
    return np.squeeze(arr)


def _find_pair(case_dir: Path) -> tuple[Path, Path | None]:
    """Locate the RF header and its segmentation header inside a case folder.

    The mask folder is inconsistently named across cases ('Mask', 'Mask04_B',
    ...), so we search for any *_Seg.hdr below the case rather than assuming.
    """
    rf = next((p for p in case_dir.glob("*_loc.hdr")), None)
    if rf is None:
        cands = [p for p in case_dir.glob("*.hdr") if "_Seg" not in p.name]
        rf = cands[0] if cands else None
    if rf is None:
        raise FileNotFoundError(f"no RF .hdr in {case_dir}")
    seg = next((p for p in case_dir.rglob("*_Seg.hdr")), None)
    return rf, seg


def discover_cases(root) -> list[tuple[str, Path, int]]:
    """[(case_id, case_dir, label)] for every labelled case under Training_Data."""
    root = Path(root)
    train = root / "Training_Data" if (root / "Training_Data").exists() else root
    out = []
    for cls_dir, label in CLASS_DIRS.items():
        d = train / cls_dir
        if not d.exists():
            continue
        for case_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            out.append((case_dir.name, case_dir, label))
    return out


def load_case(case_id: str, case_dir: Path, label: int | None = None) -> Case:
    """Load one case, remove the DC offset, attach the mask."""
    rf_hdr, seg_hdr = _find_pair(Path(case_dir))
    rf = _read_analyze(rf_hdr)
    if rf.ndim == 2:                       # a single-slice case
        rf = rf[:, :, None]

    # Remove the per-A-line DC offset. Measured slice means are ~-11000 on an
    # int16 range; leaving that in puts all the spectral energy in the DC bin.
    rf = rf - rf.mean(axis=AXIAL_AXIS, keepdims=True)

    mask = None
    if seg_hdr is not None:
        m = _read_analyze(seg_hdr)
        if m.ndim == 2:
            m = m[:, :, None]
        if m.shape == rf.shape:
            mask = m > 0
        else:
            mask = None                     # shape mismatch -> fall back to full frame

    return Case(case_id=case_id, rf=rf.astype(np.float32), mask=mask, label=label,
                meta={"rf_file": str(rf_hdr), "seg_file": str(seg_hdr) if seg_hdr else None,
                      "shape": tuple(rf.shape)})


def load_cohort(root, verbose: bool = True) -> list[Case]:
    cases = []
    for case_id, case_dir, label in discover_cases(root):
        c = load_case(case_id, case_dir, label)
        cases.append(c)
        if verbose:
            frac = float(c.mask_bool().mean())
            print(f"  {case_id:16s} label={label} shape={c.rf.shape} "
                  f"mask={frac*100:.2f}%")
    return cases
