"""
swcc_vector.py  —  two-component (vector) sliding-window matched filter
=======================================================================
Generalises the gap-aware scalar SWCC (swcc_gapaware.py) to the full 2-D tilt vector.
Observed and template are carried as COMPLEX series  z = c1 + i·c2  (c1,c2 = the two
recorded / modelled tilt axes).  Per window the normalised complex Pearson correlation is

      R_i = Σ conj(t−⟨t⟩)·(x−⟨x⟩)  /  ( ‖t−⟨t⟩‖ · ‖x−⟨x⟩‖ ) ,   |R_i| ≤ 1

computed gap/NaN-aware in O(L log L) (FFT cross-correlation + running complex moments),
reducing — term for term — to the same algebra as `swcc_gapaware` (numerator
`A − conj(St)·Sx/n`, denominators are the |·|² window energies).

Two detection statistics come from R:
  · |R|     — ROTATION-INVARIANT: unchanged by any constant rotation of either frame
              (x→x·e^{iφ} or t→t·e^{iψ} only rotate R's phase).  Needs NO orientation,
              so it is defined for every station (incl. ECIT/ECOR which lack an axis
              azimuth).  This is the universal `vec` detector wired into the pipeline.
  · Re(R)   — ORIENTED: meaningful once both series are in a COMMON geographic frame
              (observed rotated via tilt_station_orientation.get_rotation; template in
              geographic E/N).  Rewards correct polarity + orientation; range [−1,1].
  · arg(R)  — recovered relative rotation, used to VALIDATE the construction against the
              known instrument azimuth (vector_orientation.py).

`swcc_score()` dispatches on dtype so the existing detectors can stay unchanged: a real
template → scalar |r| (swcc_gapaware); a complex template → vector |R|.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import correlate

from swcc_gapaware import swcc_gapaware

warnings.filterwarnings("ignore")

BASE = Path("/home/owen/tilt_validation")
CONT = BASE / "continuous_bandpassed"
DOF_DIR = BASE / "tilt_templates_dofs"


# ── core: gap-aware complex SWCC ─────────────────────────────────────────────--

def swcc_vector_gapaware(template, x, valid=None, min_valid_frac=0.8):
    """NaN/gap-aware normalised complex Pearson correlation of complex `template`
    against complex signal `x`.  Returns complex R per window start
    (len(x)-len(template)+1); NaN where a window has < min_valid_frac finite samples
    or zero variance.  |R| is rotation-invariant; Re(R) is the oriented statistic."""
    t = np.asarray(template, complex).ravel()
    x = np.asarray(x, complex).ravel()
    M, L = len(t), len(x)
    if M > L:
        return np.empty(0, complex)
    if valid is None:
        valid = np.isfinite(x.real) & np.isfinite(x.imag)
    v = valid.astype(float)
    x0 = np.where(valid, x, 0.0).astype(complex)

    # window sums via cumulative sums
    csx = np.concatenate(([0.0 + 0j], np.cumsum(x0)))
    csxx = np.concatenate(([0.0], np.cumsum(np.abs(x0) ** 2)))      # real energy Σ|x|²
    csv = np.concatenate(([0.0], np.cumsum(v)))
    idx = np.arange(L - M + 1)
    Sx = csx[idx + M] - csx[idx]                                   # complex Σ x
    Sxx = csxx[idx + M] - csxx[idx]                                # real   Σ|x|²
    n = csv[idx + M] - csv[idx]                                    # valid count

    # template terms restricted to each window's valid positions (depend on the mask)
    A = correlate(x0, t, mode="valid", method="fft")              # Σ x[i+k] conj(t[k])
    St = correlate(v, np.conj(t), mode="valid", method="fft")     # Σ v[i+k] t[k]   (complex)
    Stt = correlate(v, np.abs(t) ** 2, mode="valid", method="fft")  # Σ v[i+k]|t[k]|² (real)
    Stt = Stt.real

    with np.errstate(invalid="ignore", divide="ignore"):
        num = A - np.conj(St) * Sx / n
        var_x = Sxx - (np.abs(Sx) ** 2) / n
        var_t = Stt - (np.abs(St) ** 2) / n
        R = num / np.sqrt(var_x * var_t)

    bad = (n < min_valid_frac * M) | (var_x <= 0) | (var_t <= 0) | ~np.isfinite(R)
    R[bad] = np.nan
    return R


def swcc_vector_loop_reference(template, x):
    """Slow O(L·M) reference (fully-valid windows) — the definition, for the gate."""
    t = np.asarray(template, complex).ravel()
    x = np.asarray(x, complex).ravel()
    M, L = len(t), len(x)
    if M > L:
        return np.empty(0, complex)
    tc = t - t.mean()
    nt = np.sqrt(np.vdot(tc, tc).real)
    out = np.zeros(L - M + 1, complex)
    for i in range(L - M + 1):
        w = x[i:i + M]
        wc = w - w.mean()
        nw = np.sqrt(np.vdot(wc, wc).real)
        out[i] = np.vdot(tc, wc) / (nt * nw) if nt > 0 and nw > 0 else np.nan
    return out


def swcc_score(template, sig, min_valid_frac=0.8):
    """Detection magnitude that slots into the existing scalar detectors:
       complex template → vector |R| ; real template → scalar |r| (swcc_gapaware)."""
    if np.iscomplexobj(template):
        return np.abs(swcc_vector_gapaware(template, sig, min_valid_frac=min_valid_frac))
    return np.abs(swcc_gapaware(template, sig, min_valid_frac=min_valid_frac))


# ── complex surrogate for the null ───────────────────────────────────────────--

def phase_randomize_complex(z, rng):
    """Full-spectrum phase-randomised surrogate of a complex series: keeps |Z(f)| (the
    complex auto-spectrum, incl. the +f/−f asymmetry that encodes polarisation) and
    randomises phases — the complex analogue of credibility_checks.phase_randomize."""
    z = np.asarray(z, complex)
    Z = np.fft.fft(z)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(Z)))
    return np.fft.ifft(Z * ph)


def surrogate(host, rng):
    """Dispatch: complex host → phase_randomize_complex; real host → phase_randomize."""
    if np.iscomplexobj(host):
        return phase_randomize_complex(host, rng)
    from credibility_checks import phase_randomize
    return phase_randomize(host, rng)


# ── loaders (complex template + complex observed) ────────────────────────────--

def load_vec_template(dataset, station, sim, template):
    """Complex 2-component band-passed template  dir_bp + i·ortho_bp  (proj_x + i·proj_y)."""
    f = DOF_DIR / dataset / f"{station}_{sim}_{template}_dof.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d["dir_bp"].to_numpy(float) + 1j * d["ortho_bp"].to_numpy(float)


def load_cont_complex(dataset, station):
    """Merge the dir (c1) and dir2 (c2) continuous feathers into one complex-`bandpassed`
    dataframe (datetime, bandpassed = c1 + i·c2, veto = c1.veto | c2.veto)."""
    f1 = CONT / dataset / f"{station}_dir_0p001-0p01Hz_cont_bp.feather"
    f2 = CONT / dataset / f"{station}_dir2_0p001-0p01Hz_cont_bp.feather"
    if not (f1.exists() and f2.exists()):
        return None
    d1 = pd.read_feather(f1); d2 = pd.read_feather(f2)
    d1["datetime"] = pd.to_datetime(d1["datetime"]); d2["datetime"] = pd.to_datetime(d2["datetime"])
    m = d1.merge(d2[["datetime", "bandpassed", "veto"]], on="datetime",
                 how="inner", suffixes=("", "_2"))
    out = pd.DataFrame({
        "datetime": m["datetime"],
        "bandpassed": m["bandpassed"].to_numpy(float) + 1j * m["bandpassed_2"].to_numpy(float),
        "veto": m["veto"].to_numpy(bool) | m["veto_2"].to_numpy(bool),
    })
    return out


def to_grid_complex(d):
    """Place a complex-`bandpassed` dataframe on a continuous 1 Hz grid (gaps→NaN)."""
    t0, t1 = d["datetime"].iloc[0], d["datetime"].iloc[-1]
    grid = pd.date_range(t0, t1, freq="1s")
    x = pd.Series(np.nan + 0j, index=grid, dtype=complex)
    x.loc[d["datetime"].values] = d["bandpassed"].to_numpy()
    v = pd.Series(False, index=grid); v.loc[d["datetime"].values] = d["veto"].to_numpy()
    return grid.values, x.to_numpy(), v.to_numpy(bool)


# ── validation gate ──────────────────────────────────────────────────────────
def validate():
    print("swcc_vector — numerical-equivalence gate (FFT vs complex loop)")
    rng = np.random.default_rng(0)
    M, L = 400, 6000
    t = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    x = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    Rf = swcc_vector_gapaware(t, x, valid=np.ones(L, bool), min_valid_frac=1.0)
    Rl = swcc_vector_loop_reference(t, x)
    d = np.nanmax(np.abs(Rf - Rl))
    print(f"  max|ΔR| (fully valid)        = {d:.2e}   {'PASS' if d < 1e-9 else 'FAIL'}")

    # rotation invariance of |R|: rotate the observed frame by an arbitrary angle
    phi = 0.7
    Rrot = swcc_vector_gapaware(t, x * np.exp(1j * phi), valid=np.ones(L, bool), min_valid_frac=1.0)
    d2 = np.nanmax(np.abs(np.abs(Rf) - np.abs(Rrot)))
    print(f"  |R| invariance under rotation = {d2:.2e}   {'PASS' if d2 < 1e-9 else 'FAIL'}")
    # ...and arg(R) tracks the rotation by exactly φ
    da = np.nanmax(np.abs(((np.angle(Rrot) - np.angle(Rf) - phi + np.pi) % (2 * np.pi)) - np.pi))
    print(f"  arg(R) shift recovers φ=0.7   = {da:.2e}   {'PASS' if da < 1e-9 else 'FAIL'}")

    # |R| ≤ 1 (Cauchy–Schwarz)
    mx = np.nanmax(np.abs(Rf))
    print(f"  max|R|                        = {mx:.6f}   {'PASS' if mx <= 1 + 1e-9 else 'FAIL'}")

    # gap-aware: NaN-masked vector path reduces to the loop on the finite samples
    val = np.ones(L, bool); val[1000:1100] = False; val[3000:3050] = False
    xg = x.copy(); xg[~val] = np.nan
    Rg = swcc_vector_gapaware(t, xg, min_valid_frac=0.8)
    # a window entirely inside a fully-valid stretch must match the loop exactly
    i = 4000
    wref = swcc_vector_loop_reference(t, x[i:i + M])[0]
    dg = abs(Rg[i] - wref)
    print(f"  gap-aware vs loop @ clean win = {dg:.2e}   {'PASS' if dg < 1e-9 else 'FAIL'}")
    return d < 1e-9 and d2 < 1e-9 and da < 1e-9 and mx <= 1 + 1e-9 and dg < 1e-9


if __name__ == "__main__":
    ok = validate()
    print("\nALL GATES PASS ✅" if ok else "\nGATE FAILURE ❌")
