"""
perf_plots.py — old-PLOT-script-style performance figures, regenerated from the new pipeline.
Reproduces the original thesis performance analysis (Figs 41–46) with data-driven detections:
  P1  correlated peaks by station × configuration   (grouped bars)        ← Fig 41
  P2  correlated peaks by station × template         (grouped bars)        ← Fig 44
  P3  peak-correlation distribution by configuration (box-plots, per comp) ← Fig 42
  P4  peak-correlation distribution by template      (box-plots, per comp) ← Fig 45
  P5  SNR distribution by configuration              (box-plots, per comp) ← Fig 43
  P6  SNR distribution by template                   (box-plots, per comp) ← Fig 46

Inputs: quant/quant_full.csv (counts) + quant/quant_peaks.csv (per-peak r + SNR).
Output → phd_output/thesis_figures/{png,pdf}/  (+ appends to figure_map.csv)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path("/home/owen/tilt_validation/phd_output/thesis_figures")
Q = FIG / "quant"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.family": "serif", "font.size": 10, "axes.titleweight": "bold",
                     "axes.grid": True, "grid.alpha": 0.3})
COMPS = ["dir", "dir2", "mag", "vec"]
CLAB = {"dir": "X", "dir2": "Y", "mag": "magnitude", "vec": "vector |R|"}
CCOL = {"dir": "#1d4ed8", "dir2": "#0891b2", "mag": "#16a34a", "vec": "#dc2626"}
CFG = {f"sim{i}": f"Config. {i}" for i in range(1, 5)}
TPL = {f"template{i}": f"Templ. {i}" for i in range(1, 5)}
DS = {"ingv": "INGV (eruptive)", "experiment": "IMPROVE (quiescent)"}
SIMS = [f"sim{i}" for i in range(1, 5)]
TEMPLATES = [f"template{i}" for i in range(1, 5)]
MAP = []


def save(fig, name, replaces, caption):
    fig.savefig(FIG / "png" / f"{name}.png"); fig.savefig(FIG / "pdf" / f"{name}.pdf"); plt.close(fig)
    MAP.append({"new_figure": name, "replaces_thesis": replaces, "caption": caption})
    print(f"  ✓ {name}")


def bars(key, order, labs, name, replaces, title, caption):
    """Grouped count bars: x = station, grouped by `key` (configuration or template); directional channel."""
    d = pd.read_csv(Q / "quant_full.csv")
    d = d[d.component == "dir"]
    g = d.groupby(["station", "dataset", key]).n_detect.sum().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=False)
    for ax, ds in zip(axes, ["ingv", "experiment"]):
        gd = g[g.dataset == ds]
        stations = sorted(gd.station.unique())
        x = np.arange(len(stations)); w = 0.8 / max(len(order), 1)
        cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(order)))
        for i, k in enumerate(order):
            vals = [gd[(gd.station == s) & (gd[key] == k)].n_detect.sum() for s in stations]
            ax.bar(x + (i - (len(order) - 1) / 2) * w, vals, w, label=labs[k], color=cmap[i])
        ax.set_xticks(x); ax.set_xticklabels(stations, fontsize=8)
        ax.set_title(DS[ds]); ax.set_ylabel("correlated peaks above floor (X axis)")
    axes[0].legend(fontsize=8, title=key.capitalize())
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save(fig, name, replaces, caption)


def boxes(key, order, labs, name, replaces, valcol, ylabel, title, caption, floor_band=None):
    """Per-component grouped box-plots of a per-peak quantity, x = configuration/template."""
    p = pd.read_csv(Q / "quant_peaks.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, ds in zip(axes, ["ingv", "experiment"]):
        pdd = p[p.dataset == ds]
        for ki, kk in enumerate(order):
            for ci, comp in enumerate(COMPS):
                vals = pdd[(pdd[key] == kk) & (pdd.component == comp)][valcol].dropna().values
                if len(vals) < 3:
                    continue
                pos = ki + (ci - 1.5) * 0.19
                bp = ax.boxplot([vals], positions=[pos], widths=0.17, showfliers=False, patch_artist=True)
                for b in bp["boxes"]:
                    b.set_facecolor(CCOL[comp]); b.set_alpha(0.6)
                for m in bp["medians"]:
                    m.set_color("black")
        if floor_band:
            ax.axhspan(*floor_band, color="grey", alpha=0.12)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([labs[k] for k in order], fontsize=8)
        ax.set_title(DS[ds]); ax.set_ylabel(ylabel)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CCOL[c], alpha=0.6) for c in COMPS]
    axes[0].legend(handles, [CLAB[c] for c in COMPS], fontsize=8, title="component")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    save(fig, name, replaces, caption)


def main():
    if not (Q / "quant_peaks.csv").exists():
        print("quant_peaks.csv missing — run quantitative_assessment.py first."); return
    bars("configuration", SIMS, CFG, "P1_peaks_by_station_configuration", "Fig 41",
         "Correlated peaks above floor by station × configuration (directional channel)",
         "Number of above-floor matched-filter detections per station, grouped by configuration (directional "
         "channel, summed over templates), for the eruptive and quiescent periods — the floor-based equivalent "
         "of the original Fig 41. Per-component totals are in Tables A–C.")
    bars("template", TEMPLATES, TPL, "P2_peaks_by_station_template", "Fig 44",
         "Correlated peaks above floor by station × template (directional channel)",
         "Above-floor detections per station grouped by time-window template (directional channel, summed over "
         "configurations) — the floor-based equivalent of Fig 44.")
    boxes("configuration", SIMS, CFG, "P3_peakcorr_by_configuration", "Fig 42", "r",
          "peak correlation |r|", "Peak-correlation distribution by configuration, per component",
          "Distribution of above-floor matched-peak correlations by configuration and component (X/Y/magnitude/"
          "vector). The populations cluster just above the per-component floors (grey band) for every "
          "configuration — the floor-based equivalent of Fig 42.", floor_band=(0.36, 0.52))
    boxes("template", TEMPLATES, TPL, "P4_peakcorr_by_template", "Fig 45", "r",
          "peak correlation |r|", "Peak-correlation distribution by template, per component",
          "Above-floor peak-correlation distribution by template and component; Template 4's distribution sits "
          "lowest, consistent with its lower floor — the floor-based equivalent of Fig 45.", floor_band=(0.36, 0.52))
    boxes("configuration", SIMS, CFG, "P5_snr_by_configuration", "Fig 43", "snr_lin",
          "SNR (linear)", "SNR distribution by configuration, per component",
          "Distribution of the template SNR (linear) at above-floor peaks, by configuration and component — the "
          "equivalent of Fig 43; SNR is comparable across configurations.")
    boxes("template", TEMPLATES, TPL, "P6_snr_by_template", "Fig 46", "snr_lin",
          "SNR (linear)", "SNR distribution by template, per component",
          "Template-SNR (linear) distribution at above-floor peaks, by template and component — the equivalent "
          "of Fig 46.")
    fm = pd.read_csv(FIG / "figure_map.csv")
    fm = pd.concat([fm, pd.DataFrame(MAP)], ignore_index=True).drop_duplicates("new_figure", keep="last")
    fm.to_csv(FIG / "figure_map.csv", index=False)
    print(f"\n{len(MAP)} performance figures → {FIG}")


if __name__ == "__main__":
    main()
