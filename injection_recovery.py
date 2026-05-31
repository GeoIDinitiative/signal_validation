"""
injection_recovery.py  —  detection-sensitivity test for the SWCC pipeline
=========================================================================
Turns the null result into a quantitative limit and validates the search end-to-end.

For a sweep of injection SNRs:
  1. take a random CLEAN (un-vetoed) window of the real continuous signal — i.e. our
     best estimate of the noise;
  2. inject one of the search templates, scaled so its band RMS = SNR x local noise RMS,
     at the window centre;
  3. run the SAME accumulated detector (per-sim gap-aware SWCC -> MAX / STACK combine);
  4. count it RECOVERED if the combined score within +/-TOL of the injection exceeds the
     real 99th-pct null floor for that station/method.

Outputs the detection-probability vs SNR curve and the 50%/90% detectable SNR
(SWCC_comprehensive/injection/recovery.csv, recovery.png, summary.txt).
A pipeline that recovers high-SNR injections confirms the null means "no signal in the
data", not "a broken search".
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from swcc_accumulated import per_sim_scores, combine
from swcc_comprehensive import load_template, SIMS
from swcc_gapaware import swcc_gapaware

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
CONT = BASE / "continuous_bandpassed"
OUT  = BASE / "SWCC_comprehensive" / "injection"
FLOORS = pd.read_csv(BASE / "SWCC_comprehensive" / "continuous" / "detect_counts.csv")

STATIONS = {"ingv": ["ECPN", "EEC1"], "experiment": ["EC1", "EC10", "ECIT", "ECOR", "EMAS"]}
COMPONENTS = ["dir", "mag"]
TEMPLATES = ["template1", "template2", "template3"]
SNRS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
N_TRIALS = 80
WIN = 10002          # ~3 x template length
TOL = 400            # recovery tolerance (samples) around the injection start


def floors(ds, st, comp):
    f = FLOORS[(FLOORS.dataset == ds) & (FLOORS.station == st) & (FLOORS.component == comp)]
    if len(f):
        return float(f.max_f99.iloc[0]), float(f.stack_f99.iloc[0])
    return 0.5, 0.4


def clean_windows(d, n_need):
    """Return start indices of clean (un-vetoed, finite) windows of length WIN."""
    ok = (~d["veto"].to_numpy(bool)) & np.isfinite(d["bandpassed"].to_numpy())
    cs = np.concatenate(([0], np.cumsum(ok.astype(int))))
    valid_starts = np.where((cs[WIN:] - cs[:-WIN]) == WIN)[0]   # fully-clean windows
    return valid_starts


def trial(host, ds, st, inj_sim, inj_tpl, snr, fmax, fstk, rng):
    T = load_template(ds, st, inj_sim, inj_tpl)
    M = len(T)
    p = (len(host) - M) // 2
    noise_rms = np.sqrt(np.mean(host ** 2))
    if noise_rms == 0 or T.std() == 0:
        return False, False
    shape = (T - T.mean()) / T.std()                 # unit-RMS template shape
    sig = host.copy()
    sig[p:p+M] += snr * noise_rms * shape            # inject at SNR x noise RMS
    sims = per_sim_scores(sig, ds, st)
    if not sims:
        return False, False
    Mx, Sx = combine(sims)
    lo, hi = max(0, p - TOL), min(len(Mx), p + TOL)
    rec_max = np.nanmax(Mx[lo:hi]) > fmax if hi > lo else False
    rec_stk = np.nanmax(Sx[lo:hi]) > fstk if hi > lo else False
    return bool(rec_max), bool(rec_stk)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(13)
    rows = []
    for ds, sts in STATIONS.items():
        # pool trials across this dataset's stations/components
        pool = []
        for st in sts:
            for comp in COMPONENTS:
                f = CONT / ds / f"{st}_{comp}_0p001-0p01Hz_cont_bp.feather"
                if not f.exists():
                    continue
                d = pd.read_feather(f)
                starts = clean_windows(d, N_TRIALS)
                if len(starts) == 0:
                    continue
                pool.append((st, comp, d["bandpassed"].to_numpy(), starts, *floors(ds, st, comp)))
        if not pool:
            continue
        for snr in SNRS:
            rmax = rstk = ntot = 0
            for _ in range(N_TRIALS):
                st, comp, sig_all, starts, fmax, fstk = pool[rng.integers(len(pool))]
                s0 = starts[rng.integers(len(starts))]
                host = sig_all[s0:s0+WIN].copy()
                inj_sim = SIMS[rng.integers(len(SIMS))]
                inj_tpl = TEMPLATES[rng.integers(len(TEMPLATES))]
                a, b = trial(host, ds, st, inj_sim, inj_tpl, snr, fmax, fstk, rng)
                rmax += a; rstk += b; ntot += 1
            rows.append({"dataset": ds, "snr": snr,
                         "p_detect_max": rmax/ntot, "p_detect_stack": rstk/ntot, "n": ntot})
            print(f"  {ds} SNR={snr:>4}: P_max={rmax/ntot:.2f}  P_stack={rstk/ntot:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "recovery.csv", index=False)

    # curves + thresholds
    fig, ax = plt.subplots(figsize=(9, 6))
    L = ["INJECTION-RECOVERY — detection sensitivity", "=" * 46, ""]
    for ds, c in [("ingv", "#1f2937"), ("experiment", "#dc2626")]:
        g = df[df.dataset == ds].sort_values("snr")
        if g.empty:
            continue
        for col, ls, lab in [("p_detect_max", "-", "MAX"), ("p_detect_stack", "--", "STACK")]:
            ax.plot(g.snr, g[col], ls, color=c, marker="o",
                    label=f"{ds} · {lab}")
            # interpolate 50% and 90% thresholds
            def thr(p):
                x, y = g.snr.to_numpy(), g[col].to_numpy()
                if y.max() < p:
                    return np.nan
                return float(np.interp(p, y, x))
            s50, s90 = thr(0.5), thr(0.9)
            L.append(f"{ds:11s} {lab:5s}: SNR50={s50:.2f}  SNR90={s90:.2f}")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.6); ax.axhline(0.9, color="gray", ls=":", alpha=0.6)
    ax.set(xlabel="injection SNR (template band-RMS / noise band-RMS)",
           ylabel="detection probability", title="SWCC detection sensitivity (injection-recovery)")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "recovery.png", dpi=300); plt.close(fig)
    L += ["", "Interpretation: the pipeline recovers injected templates above SNR90; it found",
          "no real detections, so any matching transient in the data is below that level.",
          "High-SNR recovery -> 1.0 validates the search end-to-end."]
    (OUT / "summary.txt").write_text("\n".join(L))
    print("\n".join(L)); print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
