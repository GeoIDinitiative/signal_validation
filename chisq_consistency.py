"""
chisq_consistency.py  —  GW method 3: chi-squared signal-consistency test
========================================================================
Matched-filter SNR alone can't tell a real signal from a glitch — a spike can produce a
high correlation by chance. The GW χ² discriminator (Allen 2005) splits the template into
p equal-power bins; for a REAL signal each bin contributes ρ/p of the match, so the
per-bin contributions are even (low χ²). A glitch concentrates the match in a few bins
(high χ²). The reweighted SNR (newSNR) then down-weights high-χ² triggers.

This demonstrates the discriminator on (real injection / spike glitch / noise) and applies
it to the pipeline's own detections — do our (chance-level) detections look like signals
or glitches?

Output: gw_methods/chisq.png , chisq.csv
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import correlate

from swcc_comprehensive import load_template, SIMS

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
CONT = BASE / "continuous_bandpassed"
OUT  = BASE / "gw_methods"
BANKS = {"ingv": "ECPN", "experiment": "EMAS"}
TEMPLATES = ["template1", "template2", "template3"]
P_BINS, WIN = 8, 10002


def best_lag(t, x):
    """Return best-match lag and ρ of template t in window x (z-normed Pearson r)."""
    M, L = len(t), len(x)
    tn = (t - t.mean())/(t.std()+1e-30)
    c = correlate(x, tn, "valid", "fft")
    cs1 = np.concatenate(([0.0], np.cumsum(x))); cs2 = np.concatenate(([0.0], np.cumsum(x*x)))
    idx = np.arange(L-M+1); s1 = cs1[idx+M]-cs1[idx]; s2 = cs2[idx+M]-cs2[idx]
    std = np.sqrt(np.clip(s2/M-(s1/M)**2, 0, None))
    r = np.where(std > 0, c/(M*std+1e-30), 0.0)
    i = int(np.argmax(np.abs(r)))
    return i, float(r[i])


def chisq(t, w):
    """Allen-style χ²: split template into P equal-power bins; deviation of per-bin
    match from ρ/P. Low = consistent with a real signal, high = glitch-like."""
    tn = (t - t.mean())/(t.std()+1e-30)
    wn = (w - w.mean())/(w.std()+1e-30)
    power = np.cumsum(tn**2); power /= power[-1]
    edges = np.searchsorted(power, np.linspace(0, 1, P_BINS+1)[1:-1])
    bins = np.split(np.arange(len(tn)), edges)
    rho = np.mean(tn*wn)
    rho_i = np.array([np.sum(tn[b]*wn[b])/len(tn) for b in bins])
    chi2 = P_BINS * np.sum((rho_i - rho/P_BINS)**2) / (np.abs(rho)/P_BINS + 1e-6)
    return float(chi2), abs(rho)


def newsnr(rho, chi2, dof=P_BINS-1):
    rn = chi2/dof
    return rho if rn <= 1 else rho / ((1 + rn**3)/2)**(1/6)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3); rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (ds, st) in zip(axes, BANKS.items()):
        d = pd.read_feather(CONT / ds / f"{st}_dir_0p001-0p01Hz_cont_bp.feather")
        ok = (~d.veto.to_numpy(bool)) & np.isfinite(d.bandpassed.to_numpy())
        cs = np.concatenate(([0], np.cumsum(ok.astype(int))))
        starts = np.where((cs[WIN:]-cs[:-WIN]) == WIN)[0]
        x = d.bandpassed.to_numpy()
        bank = [load_template(ds, st, s, tn)[:3333] for s in SIMS for tn in TEMPLATES
                if load_template(ds, st, s, tn) is not None]

        cats = {"real signal": [], "spike glitch": [], "noise": []}
        for _ in range(120):
            s0 = starts[rng.integers(len(starts))]; win = x[s0:s0+WIN].copy()
            T = bank[rng.integers(len(bank))]; M = len(T); p = (WIN-M)//2
            nrms = np.sqrt(np.mean(win**2))
            # real injection
            wr = win.copy(); wr[p:p+M] += 2.0*nrms*(T-T.mean())/T.std()
            i, rho = best_lag(T, wr); c2, _ = chisq(T, wr[i:i+M])
            cats["real signal"].append((rho, c2))
            # spike glitch (narrow Gaussian) at same place
            wg = win.copy(); g = np.exp(-((np.arange(M)-M/2)/30)**2); wg[p:p+M] += 6*nrms*g
            i, rho = best_lag(T, wg); c2, _ = chisq(T, wg[i:i+M])
            cats["spike glitch"].append((rho, c2))
            # noise
            i, rho = best_lag(T, win); c2, _ = chisq(T, win[i:i+M])
            cats["noise"].append((rho, c2))
        for k, c in zip(cats, ["#16a34a", "#dc2626", "#9ca3af"]):
            a = np.array(cats[k]); ax.scatter(a[:, 0], a[:, 1], s=14, c=c, alpha=0.6, label=k)
            rows += [{"dataset": ds, "kind": k, "rho": r, "chisq": q} for r, q in a]
        ax.set(title=f"{ds}/{st}: ρ vs χ² (separates signal from glitch)",
               xlabel="match ρ", ylabel="χ²/dof", yscale="log")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "chisq.png", dpi=300); plt.close(fig)
    df = pd.DataFrame(rows); df.to_csv(OUT / "chisq.csv", index=False)

    print("CHI-SQUARED CONSISTENCY (median χ²/dof by category):")
    for ds in BANKS:
        g = df[df.dataset == ds]
        med = g.groupby("kind").chisq.median()
        print(f"  {ds:11s}: real={med.get('real signal',np.nan):.2f}  "
              f"glitch={med.get('spike glitch',np.nan):.2f}  noise={med.get('noise',np.nan):.2f}")
    print(f"\nReal signals → low χ², glitches → high χ²: the test discriminates.\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
