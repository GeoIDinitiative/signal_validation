"""
bank_fitting_factor.py  —  GW method 1: template-bank coverage (fitting factor)
==============================================================================
A null result only excludes signals the bank can SEE. The fitting factor (FF) is the
standard GW measure of bank coverage: for a grid of plausible source signals, FF is the
best normalised match achievable over the whole template bank. High FF everywhere → the
null is broad; low-FF regions → the bank has gaps and the null doesn't cover them.

Synthetic signals (each bandpassed to 0.001-0.01 Hz like the data):
  · time-warped bank templates (stretch 0.5-2x) — robustness to rate mismatch
  · generic tilt transients NOT in the bank — step (erf), Gaussian pulse, ramp,
    derivative-of-Gaussian — at a range of durations

Output: gw_methods/fitting_factor.png , fitting_factor.csv , summary in stdout
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, correlate, resample
from scipy.special import erf

from swcc_comprehensive import load_template, SIMS

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
OUT  = BASE / "gw_methods"
BANKS = {"ingv": "ECPN", "experiment": "EMAS"}
TEMPLATES = ["template1", "template2", "template3", "template4"]
FS, BAND = 1.0, (0.001, 0.01)
SOS = butter(4, [BAND[0]/(FS/2), BAND[1]/(FS/2)], btype="bandpass", output="sos")
M = 3333                       # common analysis length
FF_GOOD = 0.95                 # "covered" threshold (GW minimal-match convention)


def bandpass(x):
    if len(x) < 30:
        return x
    return sosfiltfilt(SOS, x - np.mean(x))


def best_match(s, h):
    """Max |Pearson r| of signal s against template h over all lags (z-normed)."""
    s = (s - s.mean()) / (s.std() + 1e-30)
    h = (h - h.mean()) / (h.std() + 1e-30)
    a, b = (s, h) if len(s) >= len(h) else (h, s)
    c = correlate(a, b, "valid", "fft")
    # normalise by the sliding window energy of the longer one
    w = len(b)
    cs2 = np.concatenate(([0.0], np.cumsum(a*a)))
    idx = np.arange(len(a)-w+1)
    den = np.sqrt((cs2[idx+w]-cs2[idx]) * np.sum(b*b)) + 1e-30
    return float(np.max(np.abs(c/den)))


def fitting_factor(sig, bank):
    sig = bandpass(np.asarray(sig, float))
    return max(best_match(sig, h) for h in bank)


def synth_signals():
    """Generate (label, kind, param, waveform) plausible signals."""
    out = []
    t = np.arange(M)
    for dur in [200, 500, 1000, 2000]:
        c = M/2
        out.append((f"step τ={dur}",  "step",  dur, erf((t-c)/dur)))
        out.append((f"gauss τ={dur}", "gauss", dur, np.exp(-((t-c)/dur)**2)))
        out.append((f"ramp τ={dur}",  "ramp",  dur, np.clip((t-c)/dur, -1, 1)))
        out.append((f"dgauss τ={dur}","dgauss",dur, -(t-c)/dur*np.exp(-((t-c)/dur)**2)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, st in BANKS.items():
        bank = [load_template(ds, st, s, tn)[:M] for s in SIMS for tn in TEMPLATES
                if load_template(ds, st, s, tn) is not None]
        # (a) time-warped bank templates
        for s in SIMS:
            for tn in TEMPLATES:
                h = load_template(ds, st, s, tn)
                if h is None:
                    continue
                for stretch in [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]:
                    w = resample(h, max(30, int(len(h)*stretch)))
                    rows.append({"dataset": ds, "family": "warp", "label": f"{s}/{tn} x{stretch}",
                                 "param": stretch, "ff": fitting_factor(w, bank)})
        # (b) generic transients
        for label, kind, dur, wf in synth_signals():
            rows.append({"dataset": ds, "family": kind, "label": label,
                         "param": dur, "ff": fitting_factor(wf, bank)})
        print(f"  {ds}/{st}: bank={len(bank)} templates")

    df = pd.DataFrame(rows); df.to_csv(OUT / "fitting_factor.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, ds in zip(axes, ["ingv", "experiment"]):
        g = df[df.dataset == ds]
        # warp coverage vs stretch
        w = g[g.family == "warp"]
        ws = w.groupby("param").ff.agg(["mean", "min"]).reset_index()
        ax.plot(ws.param, ws["mean"], "o-", color="#2563eb", label="warped templates (mean)")
        ax.fill_between(ws.param, ws["min"], ws["mean"], color="#2563eb", alpha=0.15)
        # generic transients as points
        gg = g[g.family != "warp"]
        ax.scatter(gg.param/1000, gg.ff, c="#dc2626", s=25, marker="s",
                   label="generic transients", zorder=5)
        ax.axhline(FF_GOOD, ls="--", c="green", label=f"covered (FF≥{FF_GOOD})")
        ax.set(title=f"{ds}: bank fitting factor", xlabel="stretch  (warp) / duration·10³ s (generic)",
               ylabel="fitting factor", ylim=(0, 1.02)); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Template-bank coverage — what the null actually excludes", fontweight="600")
    fig.tight_layout(); fig.savefig(OUT / "fitting_factor.png", dpi=300); plt.close(fig)

    print("\nFITTING FACTOR SUMMARY")
    for ds in ["ingv", "experiment"]:
        g = df[df.dataset == ds]
        wcov = 100*(g[g.family == "warp"].ff >= FF_GOOD).mean()
        gcov = 100*(g[g.family != "warp"].ff >= FF_GOOD).mean()
        gmed = g[g.family != "warp"].ff.median()
        print(f"  {ds:11s}: warped templates FF≥{FF_GOOD}: {wcov:.0f}%  | "
              f"generic transients FF≥{FF_GOOD}: {gcov:.0f}% (median FF {gmed:.2f})")
    print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
