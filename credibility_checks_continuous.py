"""
credibility_checks_continuous.py  —  stage 8: Design-B validation figures
========================================================================
Updates the old credibility figures for the continuous (Design-B) pipeline and the
current two-tier accumulated null:

  1 filter_response.png   Butterworth 0.001-0.01 Hz magnitude + impulse response (settling)
  2 dering.png            naive bandpass-through-earthquake (rings) vs Design-B
                          (detectable event excluded → no ring leak)
  3 null_test.png         phase-randomised surrogate distribution of the ACCUMULATED
                          MAX/STACK score on the continuous signal, with the 95th
                          (detection) and 99th (significance) floors marked, and the
                          real-data score for comparison — i.e. how the floors are set.

Output: SWCC_comprehensive/credibility/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, sosfreqz

from swcc_accumulated import per_sim_scores, combine
from credibility_checks import phase_randomize
from build_clean_bandpassed_continuous import SRC

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
CONT = BASE / "continuous_bandpassed"
OUT  = BASE / "SWCC_comprehensive" / "credibility"
FS, BAND, ORDER = 1.0, (0.001, 0.01), 4
N_SURR = 300


def sos():
    nyq = 0.5 * FS
    return butter(ORDER, [BAND[0]/nyq, BAND[1]/nyq], btype="bandpass", output="sos")
SOS = sos()


def filter_response():
    w, h = sosfreqz(SOS, worN=8192, fs=FS)
    mag = 20*np.log10(np.maximum(np.abs(h), 1e-12))
    imp = np.zeros(20000); imp[10000] = 1
    ir = sosfiltfilt(SOS, imp)
    nz = np.where(np.abs(ir) > 0.01*np.abs(ir).max())[0]
    settle = (nz.max()-nz.min()) if len(nz) else 0
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].semilogx(w, mag, color="#2563eb")
    for fc in BAND: ax[0].axvline(fc, ls="--", c="k", alpha=0.5)
    ax[0].axhline(-3, ls=":", c="r", label="-3 dB")
    ax[0].set(xlim=(1e-4, 0.5), ylim=(-80, 5), xlabel="Frequency (Hz)", ylabel="dB",
              title=f"Butterworth order {ORDER} bandpass {BAND} Hz"); ax[0].legend(); ax[0].grid(which="both", alpha=0.3)
    ax[1].plot(ir, color="#dc2626"); ax[1].set(xlabel="sample (s)", ylabel="impulse response",
              title=f"settling ≈ {settle} s"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "filter_response.png", dpi=300); plt.close(fig)
    print(f"  filter_response.png (settling ≈ {settle} s)")


def dering():
    ds, path, fmt, dcol = SRC["ECPN"]
    raw = pd.read_feather(path); raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.sort_values("datetime").drop_duplicates("datetime")
    t0 = pd.Timestamp("2023-02-06 01:23:00")                       # Turkey M7.9
    m = (raw.datetime >= t0-pd.Timedelta("3h")) & (raw.datetime <= t0+pd.Timedelta("3h"))
    sub = raw[m]; x = sub[dcol].to_numpy(float)
    t = (sub.datetime - sub.datetime.iloc[0]).dt.total_seconds().to_numpy()/60
    naive = sosfiltfilt(SOS, x - x.mean())
    cont = pd.read_feather(CONT/ds/"ECPN_dir_0p001-0p01Hz_cont_bp.feather")
    cont["datetime"] = pd.to_datetime(cont["datetime"])
    cm = (cont.datetime >= sub.datetime.iloc[0]) & (cont.datetime <= sub.datetime.iloc[-1])
    csub = cont[cm]; ct = (csub.datetime - sub.datetime.iloc[0]).dt.total_seconds().to_numpy()/60
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(t, naive, color="#9ca3af", lw=0.8, label="naive: bandpass straight through the M7.9 (rings)")
    ax.plot(ct, csub.bandpassed.to_numpy(), color="#2563eb", lw=0.9, label="Design-B (event excluded → no ring)")
    ax.axvline((t0-sub.datetime.iloc[0]).total_seconds()/60, ls="--", c="k", alpha=0.6, label="M7.9 ETA")
    ax.set(title="De-ringing: detectable-earthquake exclusion prevents filter ringing",
           xlabel="minutes from window start", ylabel="bandpassed (east)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "dering.png", dpi=300); plt.close(fig)
    print("  dering.png")


def longest_clean(d):
    ok = (~d["veto"].to_numpy(bool)) & np.isfinite(d["bandpassed"].to_numpy())
    best_a = best = cur_a = cur = 0
    for i, v in enumerate(ok):
        if v:
            cur_a = cur_a if cur else i; cur += 1
            if cur > best: best, best_a = cur, cur_a
        else:
            cur = 0
    return d["bandpassed"].to_numpy()[best_a:best_a+min(best, 25000)]


def null_test(ds="ingv", st="ECPN"):
    d = pd.read_feather(CONT/ds/f"{st}_dir_0p001-0p01Hz_cont_bp.feather")
    host = longest_clean(d)
    rng = np.random.default_rng(7)
    surr = {"max": [], "stack": []}
    for _ in range(N_SURR):
        sc = per_sim_scores(phase_randomize(host, rng), ds, st)
        if not sc: continue
        M, S = combine(sc)
        surr["max"].append(np.nanmax(M)); surr["stack"].append(np.nanmax(S))
    real = per_sim_scores(host, ds, st)
    rM, rS = combine(real); real_val = {"max": np.nanmax(rM), "stack": np.nanmax(rS)}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, m, c in zip(ax, ["max", "stack"], ["#2563eb", "#f59e0b"]):
        v = np.array(surr[m]); p95, p99 = np.percentile(v, [95, 99])
        a.hist(v, bins=30, color=c, alpha=0.5)
        a.axvline(p95, ls="--", c="g", lw=2, label=f"detection floor 95th={p95:.2f}")
        a.axvline(p99, ls="-", c="purple", lw=2, label=f"significance floor 99th={p99:.2f}")
        a.axvline(real_val[m], ls="-", c="k", lw=2, label=f"real data max={real_val[m]:.2f}")
        a.set(title=f"{ds}/{st} {m.upper()}: surrogate score distribution",
              xlabel="accumulated score", ylabel="surrogates"); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle("Null test: two-tier floors from phase-randomised surrogates (continuous data)", fontweight="600")
    fig.tight_layout(); fig.savefig(OUT / "null_test.png", dpi=300); plt.close(fig)
    print(f"  null_test.png ({ds}/{st}: real MAX={real_val['max']:.2f} vs 99th floor)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Design-B credibility checks →", OUT)
    filter_response()
    dering()
    null_test()


if __name__ == "__main__":
    main()
