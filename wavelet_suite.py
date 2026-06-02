"""
wavelet_suite.py — Morlet wavelet (CWT) analysis suite (ports the etna_signals_phd wavelet_analysis
folder onto the denoised continuous tilt + the template bank).

Outputs (wavelet_analysis/):
  by_template/templates_<dataset>.png   4×4 sim×template Morlet scalograms (template time-frequency content)
  by_station/station_scalograms.png      representative clean-window scalogram per station
  distributions/wavelet_power_spectrum.png  time-averaged wavelet power vs frequency, all stations
  temporal/temporal_<station>.png        long-window scalogram over the record (decimated)
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_labels import slab, slabs, STATION_LABEL
from pathlib import Path

from swcc_comprehensive import load_template, SIMS

try:
    import pywt
    HAVE = True
except Exception:
    HAVE = False

warnings.filterwarnings("ignore")
import phd_env                                # branch-aware OUT
BASE = Path("/home/owen/tilt_validation")
CONT = BASE/"continuous_bandpassed"
OUT = phd_env.out(BASE/"wavelet_analysis")
for s in ["by_template", "by_station", "distributions", "temporal"]:
    (OUT/s).mkdir(parents=True, exist_ok=True)
TEMPL = ["template1", "template2", "template3", "template4"]
STATIONS = {"ingv": ["ECPN", "EEC1"], "experiment": ["EC1", "EC10", "ECIT", "ECOR", "EMAS"]}
DSCOL = {"ingv": "#1d4ed8", "experiment": "#dc2626"}
FREQS = np.logspace(np.log10(5e-4), np.log10(2e-2), 64)          # Hz, around the 0.001–0.01 ULP band
SCALES = pywt.central_frequency("morl")/FREQS if HAVE else None


def cwt_power(sig):
    coef, _ = pywt.cwt(np.nan_to_num(sig - np.nanmean(sig)), SCALES, "morl", sampling_period=1.0)
    return np.abs(coef)**2


def clean_window(ds, st, n, skip=0.3):
    f = CONT/ds/f"{st}_dir_0p001-0p01Hz_cont_bp.feather"
    if not f.exists():
        return None
    d = pd.read_feather(f)
    x = d.bandpassed.to_numpy(); m = (~d.veto.astype(bool)) & (~d.interp.astype(bool)) & np.isfinite(x)
    fin = m.astype(int); dd = np.diff(np.concatenate(([0], fin, [0])))
    starts, ends = np.where(dd == 1)[0], np.where(dd == -1)[0]
    i = np.argmax(ends-starts); a, b = starts[i], ends[i]
    a0 = a + int((b-a)*skip)
    return x[a0:min(a0+n, b)]


def scalo(ax, sig, title, tunit="min"):
    P = cwt_power(sig); t = np.arange(P.shape[1])/(60 if tunit == "min" else 3600)
    ax.pcolormesh(t, FREQS, P, shading="gouraud", cmap="magma")
    ax.set_yscale("log"); ax.axhline(0.001, color="w", ls=":", lw=0.5); ax.axhline(0.01, color="w", ls=":", lw=0.5)
    ax.set_title(title, fontsize=9)


def main():
    if not HAVE:
        print("pywt not available — skipping wavelet suite."); return

    # by_template: 4×4 sim×template scalograms for a representative station per dataset
    for ds, rep in [("ingv", "ECPN"), ("experiment", "EMAS")]:
        fig, ax = plt.subplots(4, 4, figsize=(15, 12))
        for i, sim in enumerate(SIMS):
            for j, tn in enumerate(TEMPL):
                tpl = load_template(ds, rep, sim, tn)
                if tpl is None:
                    ax[i, j].axis("off"); continue
                scalo(ax[i, j], tpl, f"{sim}/{tn}")
                if j == 0:
                    ax[i, j].set_ylabel("frequency (Hz)", fontsize=8)
                if i == 3:
                    ax[i, j].set_xlabel("minutes", fontsize=8)
        fig.suptitle(f"Morlet scalograms of the template bank — {ds} ({rep})", fontsize=15)
        fig.tight_layout(); fig.savefig(OUT/"by_template"/f"templates_{ds}.png", dpi=200); plt.close(fig)

    # by_station: representative clean-window scalogram per station
    stns = STATIONS["ingv"] + STATIONS["experiment"]
    fig, ax = plt.subplots(2, 4, figsize=(18, 8)); axf = ax.ravel()
    specs = {}
    for k, st in enumerate(stns):
        ds = "ingv" if st in STATIONS["ingv"] else "experiment"
        w = clean_window(ds, st, 14400)        # ~4 h
        if w is None or len(w) < 2000:
            axf[k].axis("off"); continue
        scalo(axf[k], w, f"{slab(st)} (4 h clean window)")
        axf[k].set_xlabel("minutes", fontsize=8); axf[k].set_ylabel("frequency (Hz)", fontsize=8)
        specs[(ds, st)] = cwt_power(w).mean(axis=1)
    for k in range(len(stns), len(axf)):
        axf[k].axis("off")
    fig.suptitle("Representative Morlet scalograms per station (denoised tilt)", fontsize=15)
    fig.tight_layout(); fig.savefig(OUT/"by_station"/"station_scalograms.png", dpi=200); plt.close(fig)

    # distributions: time-averaged wavelet power spectrum, all stations
    fig, ax = plt.subplots(figsize=(11, 6))
    for (ds, st), spec in specs.items():
        ax.loglog(FREQS, spec, color=DSCOL[ds], alpha=0.8, lw=1.2,
                  label=f"{slab(st)} ({'eruptive' if ds == 'ingv' else 'non-eruptive'})")
    ax.axvspan(0.001, 0.01, color="#fde68a", alpha=0.3, label="ULP search band")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("time-averaged wavelet power")
    ax.set_title("Wavelet power spectrum per station (eruptive = blue, non-eruptive = red)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(OUT/"distributions"/"wavelet_power_spectrum.png", dpi=300); plt.close(fig)

    # temporal: long decimated scalogram for one station per dataset
    for ds, st in [("ingv", "ECPN"), ("experiment", "EMAS")]:
        w = clean_window(ds, st, 172800)       # ~2 days
        if w is None or len(w) < 5000:
            continue
        wd = w[::4]                            # decimate ×4 (ULP band well above Nyquist of 0.125 Hz)
        coef, _ = pywt.cwt(np.nan_to_num(wd-np.nanmean(wd)), pywt.central_frequency("morl")/(FREQS*4),
                           "morl", sampling_period=4.0)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.pcolormesh(np.arange(len(wd))*4/3600, FREQS, np.abs(coef)**2, shading="gouraud", cmap="magma")
        ax.set_yscale("log"); ax.axhline(0.001, color="w", ls=":", lw=0.6); ax.axhline(0.01, color="w", ls=":", lw=0.6)
        ax.set_xlabel("hours from window start"); ax.set_ylabel("frequency (Hz)")
        ax.set_title(f"Temporal wavelet power — {slab(st)} (~2-day clean window, decimated ×4)")
        fig.tight_layout(); fig.savefig(OUT/"temporal"/f"temporal_{st}.png", dpi=300); plt.close(fig)

    print(f"wavelet suite → {OUT}")


if __name__ == "__main__":
    main()
