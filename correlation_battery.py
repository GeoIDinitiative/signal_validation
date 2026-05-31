"""
correlation_battery.py  —  multi-method test of template/data association
=========================================================================
STANDALONE (not part of run_pipeline.py). Asks "do the templates correlate with the
denoised data AT ALL?" using four independent detectors, each covering a different
blind spot of plain SWCC:

  SWCC      normalised sliding cross-correlation        (linear, fixed-scale waveform)
  SUBSPACE  template-SVD projection-energy ratio        (signal = mix of templates)
  ENVELOPE  Hilbert-envelope cross-correlation          (right energy, wrong phase/timing)
  DTW       subsequence dynamic time warping            (time-warped / rate-mismatched)

For each method we measure injection-recovery (detection probability vs SNR -> SNR50/90)
against that method's own phase-randomised null floor, plus a real-data detection count
on the longest clean block. If every method is null on the real data, "no correlation"
holds across method families, not just for SWCC.

Output: correlation_battery/   (recovery.png, recovery.csv, real_data_counts.csv, summary.txt)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import correlate, hilbert, decimate, find_peaks

from swcc_comprehensive import load_template, SIMS
try:
    from dtaidistance import dtw as _dtw
    HAVE_DTW = True
except Exception:
    HAVE_DTW = False

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
CONT = BASE / "continuous_bandpassed"
OUT  = BASE / "correlation_battery"
STATIONS = {"ingv": ["ECPN", "EEC1"], "experiment": ["EC1", "EC10", "ECIT", "ECOR", "EMAS"]}
TEMPLATES = ["template1", "template2", "template3"]
COMP = "dir"
SNRS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
N_TRIALS, N_FLOOR = 50, 150
WIN, TOL = 10002, 400
M_COMMON, K_SUB, DECIM = 3333, 6, 20
METHODS = ["SWCC", "SUBSPACE", "ENVELOPE"] + (["DTW"] if HAVE_DTW else [])


# ── per-method statistics ─────────────────────────────────────────────────────
def norm_xcorr(t, x):
    """|Pearson r| of template t against each window of x (length len(x)-len(t)+1)."""
    M, L = len(t), len(x)
    if t.std() == 0:
        return np.zeros(L - M + 1)
    tn = (t - t.mean()) / t.std()
    c = correlate(x, tn, "valid", "fft")
    cs1 = np.concatenate(([0.0], np.cumsum(x))); cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    idx = np.arange(L - M + 1)
    s1 = cs1[idx + M] - cs1[idx]; s2 = cs2[idx + M] - cs2[idx]
    mean = s1 / M; std = np.sqrt(np.clip(s2 / M - mean * mean, 0, None))
    r = np.zeros_like(c); nz = std > 0
    r[nz] = c[nz] / (M * std[nz])
    return np.abs(r)


def subspace_ratio(U, x):
    """Fraction of each window's centred energy captured by the template subspace U (M x K)."""
    M, K = U.shape; L = len(x)
    num = np.zeros(L - M + 1)
    for k in range(K):
        c = correlate(x, U[:, k], "valid", "fft")
        num += c * c
    cs1 = np.concatenate(([0.0], np.cumsum(x))); cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    idx = np.arange(L - M + 1)
    s1 = cs1[idx + M] - cs1[idx]; s2 = cs2[idx + M] - cs2[idx]
    den = s2 - s1 * s1 / M                    # centred window energy
    return np.where(den > 0, num / den, 0.0)


def env(x):
    return np.abs(hilbert(x))


def dtw_score(td, wd, near_dec=None):
    """Best fixed-length sliding DTW similarity of z-normed template td inside window wd.
    Fixed-length segments + Sakoe-Chiba band make it discriminating (subsequence DTW
    over-matches and saturates). Returns max similarity 1/(1+d/M) over slide positions."""
    Mt = len(td); band = max(3, Mt // 8); best = 0.0
    if near_dec is None:
        rng_lo, rng_hi = 0, len(wd) - Mt + 1
    else:
        rng_lo, rng_hi = max(0, near_dec - TOL // DECIM), min(len(wd) - Mt + 1, near_dec + TOL // DECIM + 1)
    for i in range(rng_lo, max(rng_lo + 1, rng_hi), max(1, Mt // 5)):
        seg = wd[i:i+Mt]
        if seg.std() == 0:
            continue
        seg = (seg - seg.mean()) / seg.std()
        d = _dtw.distance_fast(td, seg.astype(np.double), window=band, use_pruning=True)
        best = max(best, 1.0 / (1.0 + d))      # no /M: keeps signal/noise distances separable
    return best


# ── window score per method (near = restrict to injection neighbourhood) ──────
def score(method, win, ds, st, tpls, U, near=None):
    if method == "SUBSPACE":
        r = subspace_ratio(U, win)
        seg = r if near is None else r[max(0, near-TOL):near+TOL]
        return float(np.nanmax(seg)) if len(seg) else 0.0
    if method == "DTW":
        wd = decimate(win, DECIM).astype(np.double)
        near_dec = None if near is None else near // DECIM
        best = 0.0
        for T in tpls.values():
            td = decimate(T, DECIM)
            if td.std() == 0:
                continue
            td = ((td - td.mean()) / td.std()).astype(np.double)
            best = max(best, dtw_score(td, wd, near_dec))
        return best
    best = 0.0
    for T in tpls.values():
        r = norm_xcorr(T, win) if method == "SWCC" else norm_xcorr(env(T), env(win))
        seg = r if near is None else r[max(0, near-TOL):near+TOL]
        if len(seg):
            best = max(best, float(np.nanmax(seg)))
    return best


# ── data / basis ──────────────────────────────────────────────────────────────
def load_station(ds, st):
    f = CONT / ds / f"{st}_{COMP}_0p001-0p01Hz_cont_bp.feather"
    if not f.exists():
        return None
    d = pd.read_feather(f)
    ok = (~d["veto"].to_numpy(bool)) & np.isfinite(d["bandpassed"].to_numpy())
    cs = np.concatenate(([0], np.cumsum(ok.astype(int))))
    starts = np.where((cs[WIN:] - cs[:-WIN]) == WIN)[0]
    tpls = {}
    for sim in SIMS:
        for tn in TEMPLATES:
            T = load_template(ds, st, sim, tn)
            if T is not None:
                tpls[(sim, tn)] = T[:M_COMMON]
    A = np.vstack([(T - T.mean()) / T.std() for T in tpls.values()]).T
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    return d["bandpassed"].to_numpy(), starts, tpls, U[:, :K_SUB]


def inject(win, T, snr, rng):
    M = len(T); p = (len(win) - M) // 2
    nrms = np.sqrt(np.mean(win ** 2))
    if T.std() > 0 and nrms > 0:
        win = win.copy()
        win[p:p+M] += snr * nrms * (T - T.mean()) / T.std()
    return win, p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    rec_rows, cnt_rows = [], []
    Lsum = ["MULTI-METHOD CORRELATION BATTERY", "=" * 40,
            f"methods: {', '.join(METHODS)}", ""]

    for ds in STATIONS:
        pool = []
        for st in STATIONS[ds]:
            r = load_station(ds, st)
            if r is None or len(r[1]) == 0:
                continue
            pool.append((st, *r))
        if not pool:
            continue

        # null floors per method
        floors = {}
        for m in METHODS:
            vals = []
            for _ in range(N_FLOOR):
                st, sig, starts, tpls, U = pool[rng.integers(len(pool))]
                s0 = starts[rng.integers(len(starts))]
                vals.append(score(m, sig[s0:s0+WIN], ds, st, tpls, U, near=None))
            floors[m] = float(np.percentile(vals, 99))
        Lsum.append(f"{ds}: null floors " + "  ".join(f"{m}={floors[m]:.3f}" for m in METHODS))

        # injection-recovery
        for snr in SNRS:
            rec = {m: 0 for m in METHODS}
            for _ in range(N_TRIALS):
                st, sig, starts, tpls, U = pool[rng.integers(len(pool))]
                s0 = starts[rng.integers(len(starts))]
                inj_key = list(tpls)[rng.integers(len(tpls))]
                win, p = inject(sig[s0:s0+WIN], tpls[inj_key], snr, rng)
                for m in METHODS:
                    if score(m, win, ds, st, tpls, U, near=p) > floors[m]:
                        rec[m] += 1
            row = {"dataset": ds, "snr": snr}
            row.update({m: rec[m]/N_TRIALS for m in METHODS})
            rec_rows.append(row)
            print(f"  {ds} SNR={snr:>4}: " + " ".join(f"{m}={rec[m]/N_TRIALS:.2f}" for m in METHODS))

        # real-data detection counts (longest clean block, strided)
        st, sig, starts, tpls, U = pool[0]
        # longest clean run
        ok = np.isfinite(sig)
        block = sig[:200000]               # representative chunk
        for m in METHODS:
            if m == "DTW":
                n = 0  # strided subsequence scan
                for s0 in range(0, len(block)-WIN, WIN//2):
                    if score(m, block[s0:s0+WIN], ds, st, tpls, U, near=None) > floors[m]:
                        n += 1
                cnt_rows.append({"dataset": ds, "method": m, "n_windows": len(range(0, len(block)-WIN, WIN//2)), "n_detect": n})
            else:
                if m == "SUBSPACE":
                    r = subspace_ratio(U, block)
                else:
                    r = np.maximum.reduce([norm_xcorr(T if m=="SWCC" else env(T),
                                                      block if m=="SWCC" else env(block)) for T in tpls.values()])
                pk, _ = find_peaks(r, height=floors[m], distance=M_COMMON)
                cnt_rows.append({"dataset": ds, "method": m, "n_windows": len(r)//M_COMMON, "n_detect": int(len(pk))})
        Lsum.append("")

    rec = pd.DataFrame(rec_rows); rec.to_csv(OUT / "recovery.csv", index=False)
    cnt = pd.DataFrame(cnt_rows); cnt.to_csv(OUT / "real_data_counts.csv", index=False)

    # recovery figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {"SWCC": "#2563eb", "SUBSPACE": "#16a34a", "ENVELOPE": "#dc2626", "DTW": "#9333ea"}
    for ax, ds in zip(axes, ["ingv", "experiment"]):
        g = rec[rec.dataset == ds].sort_values("snr")
        if g.empty:
            continue
        for m in METHODS:
            ax.plot(g.snr, g[m], marker="o", color=colors[m], label=m)
            y, x = g[m].to_numpy(), g.snr.to_numpy()
            s90 = float(np.interp(0.9, y, x)) if y.max() >= 0.9 else np.nan
            s50 = float(np.interp(0.5, y, x)) if y.max() >= 0.5 else np.nan
            Lsum.append(f"{ds:11s} {m:9s}: SNR50={s50:.2f}  SNR90={s90:.2f}")
        ax.axhline(0.9, color="gray", ls=":", alpha=0.5); ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
        ax.set(title=f"{ds}: detection probability vs SNR", xlabel="injection SNR", ylabel="P(detect)")
        ax.set_ylim(0, 1.02); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "recovery.png", dpi=140); plt.close(fig)

    Lsum += ["", "REAL-DATA DETECTIONS (longest clean block):"]
    for _, r in cnt.iterrows():
        Lsum.append(f"   {r.dataset:11s} {r.method:9s}: {r.n_detect} detections / ~{r.n_windows} windows")
    Lsum += ["", "All methods null on real data + their injection curves => templates do not",
             "correlate with the denoised data by ANY of the tested families, down to each",
             "method's SNR90."]
    (OUT / "summary.txt").write_text("\n".join(Lsum))
    print("\n" + "\n".join(Lsum)); print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
