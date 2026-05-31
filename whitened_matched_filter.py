"""
whitened_matched_filter.py  —  GW method 2: noise-PSD-weighted matched filter
============================================================================
Pearson r weights every in-band frequency equally; the OPTIMAL statistic for coloured
Gaussian noise is the whitened matched filter, ρ = <d|h> with the inner product
<a|b> = Σ ã*b̃ / Sₙ(f) — i.e. divide by the noise PSD before correlating. Our in-band
noise is mildly red (≈3x more power at 0.001 than 0.01 Hz), so whitening should give a
moderate sensitivity gain.

This estimates the noise ASD from a clean stretch, whitens both data and templates, and
compares injection-recovery (SNR50/90) of the whitened matched filter vs plain Pearson r.

Output: gw_methods/whitened_mf.png , whitened_mf.csv
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch, correlate
from scipy.interpolate import interp1d
from numpy.fft import rfft, irfft, rfftfreq

from swcc_comprehensive import load_template, SIMS

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
CONT = BASE / "continuous_bandpassed"
OUT  = BASE / "gw_methods"
BANKS = {"ingv": "ECPN", "experiment": "EMAS"}
TEMPLATES = ["template1", "template2", "template3"]
FS, BAND = 1.0, (0.001, 0.01)
SNRS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
N_TRIALS, N_FLOOR, WIN, TOL = 60, 200, 10002, 400


def clean_segment(ds, st):
    d = pd.read_feather(CONT / ds / f"{st}_dir_0p001-0p01Hz_cont_bp.feather")
    ok = (~d["veto"].to_numpy(bool)) & np.isfinite(d["bandpassed"].to_numpy())
    cs = np.concatenate(([0], np.cumsum(ok.astype(int))))
    starts = np.where((cs[WIN:] - cs[:-WIN]) == WIN)[0]
    return d["bandpassed"].to_numpy(), starts


def make_asd(x, starts, n=40):
    """Noise ASD interpolator from clean windows."""
    segs = [x[s:s+WIN] for s in starts[:n]]
    f, P = welch(np.concatenate(segs), fs=FS, nperseg=4096)
    P = np.maximum(P, P[P > 0].min())
    return interp1d(f, np.sqrt(P), bounds_error=False, fill_value=(np.sqrt(P[0]), np.sqrt(P[-1])))


def whiten(x, asd):
    X = rfft(x); fr = rfftfreq(len(x), 1/FS)
    W = np.zeros_like(fr)
    ib = (fr >= BAND[0]) & (fr <= BAND[1])
    W[ib] = 1.0 / asd(fr[ib])
    return irfft(X * W, n=len(x))


def nx(t, x):
    """max |Pearson r| of template t over window x."""
    Mt, L = len(t), len(x)
    tn = (t - t.mean()) / (t.std() + 1e-30)
    c = correlate(x, tn, "valid", "fft")
    cs1 = np.concatenate(([0.0], np.cumsum(x))); cs2 = np.concatenate(([0.0], np.cumsum(x*x)))
    idx = np.arange(L-Mt+1); s1 = cs1[idx+Mt]-cs1[idx]; s2 = cs2[idx+Mt]-cs2[idx]
    std = np.sqrt(np.clip(s2/Mt-(s1/Mt)**2, 0, None))
    r = np.zeros_like(c); nz = std > 0; r[nz] = c[nz]/(Mt*std[nz])
    return np.abs(r)


def score(method, win, bank, asd, near=None):
    best = 0.0
    for t in bank:
        if method == "white":
            r = nx(whiten(t, asd), whiten(win, asd))
        else:
            r = nx(t, win)
        seg = r if near is None else r[max(0, near-TOL):near+TOL]
        if len(seg):
            best = max(best, float(np.nanmax(seg)))
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5); rows = []
    for ds, st in BANKS.items():
        x, starts = clean_segment(ds, st)
        if len(starts) == 0:
            continue
        asd = make_asd(x, starts)
        bank = [load_template(ds, st, s, tn)[:3333] for s in SIMS for tn in TEMPLATES
                if load_template(ds, st, s, tn) is not None]
        floors = {"plain": [], "white": []}
        for _ in range(N_FLOOR):
            s0 = starts[rng.integers(len(starts))]
            w = x[s0:s0+WIN]
            for m in ["plain", "white"]:
                floors[m].append(score(m, w, bank, asd))
        floors = {m: float(np.percentile(v, 99)) for m, v in floors.items()}
        for snr in SNRS:
            rec = {"plain": 0, "white": 0}
            for _ in range(N_TRIALS):
                s0 = starts[rng.integers(len(starts))]
                win = x[s0:s0+WIN].copy()
                T = bank[rng.integers(len(bank))]; p = (WIN-len(T))//2
                nrms = np.sqrt(np.mean(win**2))
                win[p:p+len(T)] += snr*nrms*(T-T.mean())/T.std()
                for m in ["plain", "white"]:
                    if score(m, win, bank, asd, near=p) > floors[m]:
                        rec[m] += 1
            rows.append({"dataset": ds, "snr": snr, "plain": rec["plain"]/N_TRIALS,
                         "white": rec["white"]/N_TRIALS})
            print(f"  {ds} SNR={snr:>4}: plain={rec['plain']/N_TRIALS:.2f} white={rec['white']/N_TRIALS:.2f}")
    df = pd.DataFrame(rows); df.to_csv(OUT / "whitened_mf.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ds in zip(axes, ["ingv", "experiment"]):
        g = df[df.dataset == ds].sort_values("snr")
        if g.empty:
            continue
        ax.plot(g.snr, g.plain, "s--", color="#6b7280", label="Pearson r")
        ax.plot(g.snr, g.white, "o-", color="#2563eb", label="whitened MF")
        for col, c in [("plain", "#6b7280"), ("white", "#2563eb")]:
            y, xx = g[col].to_numpy(), g.snr.to_numpy()
            s90 = float(np.interp(0.9, y, xx)) if y.max() >= 0.9 else np.nan
            ax.text(0.02, 0.9 if col == "white" else 0.82, f"{col} SNR90={s90:.2f}",
                    transform=ax.transAxes, color=c, fontsize=9)
        ax.axhline(0.9, ls=":", c="gray"); ax.set(title=f"{ds}: whitened MF vs Pearson r",
                  xlabel="injection SNR", ylabel="P(detect)", ylim=(0, 1.02))
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "whitened_mf.png", dpi=300); plt.close(fig)
    print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
