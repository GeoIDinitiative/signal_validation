"""
swcc_gapaware.py  —  gap-aware sliding-window cross-correlation (best-practice core)
==================================================================================
Improves the SWCC so it uses ALL of the cleaned signal and is intrinsically
gap-sensitive, instead of requiring each window to fit entirely inside one segment.

Idea: place the cleaned bandpassed signal on the FULL continuous 1 Hz time grid with
NaN in every gap (excised earthquakes, recording breaks, dropped short runs), then
compute a NaN-aware normalised Pearson r in which each window is correlated over only
its finite samples. A window is scored iff at least `min_valid_frac` of it is finite.

Why this is better, and the parameter rationale (best practice):
  · TIME STEP (hop): 1 sample. SWCC for *detection* is matched filtering — matched-
    filter theory says evaluate the correlation at every lag so the optimal alignment
    is never missed. (The 50–75 % "overlap" rule of thumb is for *spectral* estimation
    (Welch/STFT), not detection.) We compute every position via FFT, so hop = 1 is free.
  · WINDOW = template length (the matched filter is the template itself).
  · MIN_VALID_FRAC: fraction of a window that must be real signal (default 0.8). This is
    the single intuitive "gap sensitivity" knob: 1.0 = old behaviour (whole window inside
    one segment); lower = tolerate windows that straddle small gaps / segment edges, using
    more of the signal. Missing samples are excluded from the correlation, not zero-filled.
  · PEAK SEPARATION = window length (non-overlapping detections) — principled, vs an
    arbitrary fixed value.
  · Normalisation: Pearson r over the valid samples only (amplitude-invariant shape match).

Exactness: on a fully-valid window this reduces *exactly* to the old per-segment
swcc_segment (validated below to ~1e-12).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import correlate

from swcc_comprehensive import swcc_segment, load_clean, load_template

CLEAN = Path("/home/owen/tilt_validation/clean_bandpassed")


def swcc_gapaware(template, x, valid=None, min_valid_frac=0.8):
    """
    NaN-aware normalised sliding cross-correlation (Pearson r) of `template` against
    signal `x` (full grid; NaN or `valid=False` marks gaps).

    Returns r per window start (length len(x)-len(template)+1); NaN where a window has
    fewer than `min_valid_frac` finite samples or zero variance.
    """
    t = np.asarray(template, float).ravel()
    x = np.asarray(x, float).ravel()
    M, L = len(t), len(x)
    if M > L:
        return np.empty(0)
    if valid is None:
        valid = np.isfinite(x)
    v = valid.astype(float)
    x0 = np.where(valid, x, 0.0)

    # window sums of the signal over valid samples (x0 = 0 inside gaps)
    csx  = np.concatenate(([0.0], np.cumsum(x0)))
    csxx = np.concatenate(([0.0], np.cumsum(x0 * x0)))
    csv  = np.concatenate(([0.0], np.cumsum(v)))
    idx = np.arange(L - M + 1)
    Sx  = csx[idx + M]  - csx[idx]
    Sxx = csxx[idx + M] - csxx[idx]
    n   = csv[idx + M]  - csv[idx]

    # template sums restricted to each window's valid positions (depends on the mask)
    St  = correlate(v,  t,     mode="valid", method="fft")   # Σ t[k]·v[i+k]
    Stt = correlate(v,  t * t, mode="valid", method="fft")   # Σ t[k]²·v[i+k]
    C   = correlate(x0, t,     mode="valid", method="fft")   # Σ t[k]·x[i+k] (gaps→0)

    with np.errstate(invalid="ignore", divide="ignore"):
        cov   = C   - Sx * St / n
        var_x = Sxx - Sx * Sx / n
        var_t = Stt - St * St / n
        r = cov / np.sqrt(var_x * var_t)

    bad = (n < min_valid_frac * M) | (var_x <= 0) | (var_t <= 0) | ~np.isfinite(r)
    r[bad] = np.nan
    return r


# ── full-grid reconstruction ─────────────────────────────────────────────────
def to_full_grid(clean_df):
    """Place a clean_bandpassed dataframe on a continuous 1 Hz grid; gaps become NaN."""
    dt = pd.to_datetime(clean_df["datetime"])
    t0, t1 = dt.iloc[0], dt.iloc[-1]
    grid = pd.date_range(t0, t1, freq="1s")
    s = pd.Series(np.nan, index=grid)
    s.loc[dt.values] = clean_df["bandpassed"].to_numpy()
    return s.index.to_numpy(), s.to_numpy()


# ── validation + coverage demo ───────────────────────────────────────────────
def validate():
    print("Exactness gate: gap-aware vs old swcc_segment on a fully-valid window")
    d = load_clean("ingv", "ECPN", "dir")
    seg = d[d.segment_id == d.segment_id.iloc[0]]["bandpassed"].to_numpy(float)[:6000]
    tpl = load_template("ingv", "ECPN", "sim1", "template1")
    r_old = swcc_segment(tpl, seg)
    r_new = swcc_gapaware(tpl, seg, valid=np.ones(len(seg), bool), min_valid_frac=1.0)
    m = np.nanmax(np.abs(r_old - r_new))
    print(f"  max|Δr| = {m:.2e}   {'PASS' if m < 1e-9 else 'FAIL'}")
    return m < 1e-9


def coverage_demo():
    print("\nCoverage: windows scored, segment-contained (old) vs gap-aware (new)")
    tpl = load_template("ingv", "ECPN", "sim1", "template1"); M = len(tpl)
    for ds, st in [("ingv", "ECPN"), ("experiment", "EMAS")]:
        d = load_clean(ds, st, "dir")
        # old: sum over segments of (len-M+1) for segments >= M
        sl = d.groupby("segment_id").size()
        old = int((sl[sl >= M] - M + 1).clip(lower=0).sum())
        # new: gap-aware on the full grid
        _, x = to_full_grid(d)
        r = swcc_gapaware(tpl, x, min_valid_frac=0.8)
        new = int(np.isfinite(r).sum())
        print(f"  {ds}/{st}: old={old:,} windows   new={new:,} windows   "
              f"(+{100*(new-old)/max(old,1):.0f}%)   grid={len(x):,}, valid={int(np.isfinite(x).sum()):,}")


if __name__ == "__main__":
    validate()
    coverage_demo()
