"""
swcc_plots_continuous.py  —  old-style SWCC figures on the Design-B continuous signal,
with the correlation line coloured by contamination.

For each (dataset, station, component, sim, template):
  · compute the gap-aware |r|(t) across the full continuous record;
  · a correlation position is CONTAMINATED if its window overlaps any veto zone
    (detectable-quake buffer, block-edge settling, or ±2 h of an M>=5.5 ETA);
  · draw the line GREY where contaminated and BLUE where clean;
  · keep the original styling: 0.2/0.5/0.7 threshold lines + null floor, red shaded
    volcanic windows (INGV), and clean-peak markers (cyan circle / red star).

Output: SWCC_comprehensive/<dataset>/<station>/<sim>/<station>_<comp>_<sim>_<template>_swcc.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

from swcc_gapaware import swcc_gapaware
from swcc_comprehensive import load_template, SIMS, THRESHOLD
from swcc_oldstyle_plots import load_volcanic_events, plot_volcanic_events_on_swcc

BASE = Path("/home/owen/tilt_validation")
CONT = BASE / "continuous_bandpassed"
SWCC = BASE / "SWCC_comprehensive"
VOLC = load_volcanic_events(BASE / "etna_volcanic_events_cleaned.csv")
FLOORS = pd.read_csv(SWCC / "continuous" / "template_floors.csv")

STATIONS = {"ingv": ["ECPN", "EEC1"], "experiment": ["EC1", "EC10", "ECIT", "ECOR", "EMAS"]}
COMPONENTS = ["dir", "mag"]
TEMPLATES = ["template1", "template2", "template3", "template4"]
MIN_VALID, PLOT_STEP = 0.8, 30          # min window validity; decimate line to ~30 s for display


def floor_of(ds, st, comp, sim, tname):
    """(detection floor = 95th-pct null, significance floor = 99th-pct null) for this template."""
    f = FLOORS[(FLOORS.dataset == ds) & (FLOORS.station == st) & (FLOORS.component == comp)
              & (FLOORS.sim == sim) & (FLOORS.template == tname)]
    if len(f):
        return float(f.floor_detect.iloc[0]), float(f.floor_signif.iloc[0])
    return 0.45, 0.49


def load_grid(ds, st, comp):
    f = CONT / ds / f"{st}_{comp}_0p001-0p01Hz_cont_bp.feather"
    if not f.exists():
        return None
    d = pd.read_feather(f); d["datetime"] = pd.to_datetime(d["datetime"])
    t0, t1 = d["datetime"].iloc[0], d["datetime"].iloc[-1]
    grid = pd.date_range(t0, t1, freq="1s")
    x = pd.Series(np.nan, index=grid); x.loc[d["datetime"].values] = d["bandpassed"].to_numpy()
    v = pd.Series(False, index=grid);  v.loc[d["datetime"].values] = d["veto"].to_numpy()
    return grid.values, x.to_numpy(), v.to_numpy(bool)


def window_contaminated(veto, M):
    """True at window-start i if veto[i:i+M] contains any vetoed sample."""
    cs = np.concatenate(([0], np.cumsum(veto.astype(np.int64))))
    n = len(veto) - M + 1
    return (cs[M:M+n] - cs[:n]) > 0


def plot_one(ds, st, comp, sim, tname, grid, x, veto):
    tpl = load_template(ds, st, sim, tname)
    if tpl is None:
        return False
    M = len(tpl)
    r = np.abs(swcc_gapaware(tpl, x, min_valid_frac=MIN_VALID))
    if r.size == 0 or np.all(np.isnan(r)):
        return False
    cont = window_contaminated(veto, M)[:len(r)]
    t = pd.to_datetime(grid[:len(r)])

    clean_line = np.where(cont, np.nan, r)        # blue where clean
    cont_line  = np.where(cont, r, np.nan)        # grey where contaminated
    sl = slice(None, None, PLOT_STEP)             # decimate for display

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(1, 1, figsize=(16, 5), facecolor="white")
    ax.set_facecolor("#f8f9fa")
    ax.plot(t[sl], cont_line[sl], color="#9ca3af", lw=0.9, alpha=0.8,
            label="Contaminated (quake/edge veto)", zorder=5)
    ax.plot(t[sl], clean_line[sl], color="#2563eb", lw=1.0, alpha=0.95,
            label="Clean signal", zorder=10)

    # two null-derived thresholds (replace the old fixed 0.2 / 0.5 / 0.7)
    fdet, fsig = floor_of(ds, st, comp, sim, tname)
    ax.axhline(fdet, color="#16a34a", ls="--", lw=2.0, alpha=0.9,
               label=f"Detection floor · 95th-pct null (r = {fdet:.2f})", zorder=14)
    ax.axhline(fsig, color="#7c3aed", ls="-", lw=2.4, alpha=0.95,
               label=f"Significance floor · 99th-pct null (r = {fsig:.2f})", zorder=14)

    # clean peaks only, tiered by the null floors
    s = np.where(np.isfinite(r) & ~cont, r, 0.0)
    pk, _ = find_peaks(s, height=fdet, distance=1000)
    if len(pk):
        pr = r[pk]; pt = t[pk]
        lo = (pr >= fdet) & (pr < fsig); hi = pr >= fsig
        if lo.any():
            ax.scatter(pt[lo], pr[lo], s=40, c="#00FFFF", marker="o", zorder=12,
                       edgecolors="black", linewidths=1.2,
                       label=f"Detections 95–99th pct (n={int(lo.sum())})")
        if hi.any():
            ax.scatter(pt[hi], pr[hi], s=170, c="#ff0000", marker="*", zorder=15,
                       edgecolors="#8b0000", linewidths=1.0,
                       label=f"Significant > 99th pct (n={int(hi.sum())})")

    nv = 0
    if ds == "ingv":
        nv = plot_volcanic_events_on_swcc(ax, VOLC, (pd.Timestamp(t[0]), pd.Timestamp(t[-1])))
    ax.set_ylabel("Correlation Coefficient |r|", fontsize=13, fontweight="600")
    ax.set_xlabel("Time (UTC)", fontsize=13, fontweight="600")
    title = (f"{ds.upper()} – {st} ({comp}) | {sim.replace('sim','Simulation ')} – "
             f"{tname.replace('template','Template ')}")
    if nv:
        title += f" | {nv} Volcanic Events"
    ax.set_title(title, fontsize=14, fontweight="700", pad=15)
    ax.grid(True, alpha=0.25); ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8.5, ncol=2, frameon=True, framealpha=0.95)
    ax.set_ylim(0, 1.05); ax.set_xlim(t[0], t[-1])
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    out = SWCC / ds / st / sim
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{st}_{comp}_{sim}_{tname}_swcc.png", dpi=300, facecolor="white")
    plt.close(); plt.style.use("default")
    return True


def main():
    n = 0
    for ds, sts in STATIONS.items():
        for st in sts:
            for comp in COMPONENTS:
                g = load_grid(ds, st, comp)
                if g is None:
                    continue
                grid, x, veto = g
                for sim in SIMS:
                    for tname in TEMPLATES:
                        if plot_one(ds, st, comp, sim, tname, grid, x, veto):
                            n += 1
                print(f"   {ds}/{st}/{comp}")
    print(f"\nUpdated SWCC figures (clean=blue, contaminated=grey): {n}")


if __name__ == "__main__":
    main()
