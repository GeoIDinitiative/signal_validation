"""
swcc_rankings.py — SWCC peak-count / ranking bar-chart suite (ports the etna_signals_phd
SWCC_analysis breakdown onto the new floor-based, denoised-continuous pipeline).

Reads the floor-based per-(station,sim,template) counts (top_templates/sst_peak_counts.csv — peaks
above each combination's OWN null/significance floor) and the accumulated detection list, and renders
the full by-station / by-template / by-simulation / by-dataset / rankings breakdown.

Outputs (SWCC_analysis/):
  by_station/    peaks + match-quality per station (grouped by dataset)
  by_template/   peaks + match-quality + floor per template
  by_simulation/ peaks + match-quality per simulation
  by_dataset/    totals + length-normalised peak rate
  rankings/      top-25 SST ranking (floors annotated)
  comprehensive/ one-page multi-panel summary
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_labels import slab, slabs, dlab, STATION_LABEL
from pathlib import Path

import phd_env                                # branch-aware OUT / detections / components
warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
OUT = phd_env.out(BASE/"SWCC_analysis")
SST = pd.read_csv(phd_env.out(BASE/"SWCC_comprehensive"/"top_templates")/"sst_peak_counts.csv")
DETS = pd.read_csv(phd_env.dets_dir()/"all_detections_continuous.csv", parse_dates=["peak_time"])
if "component" in DETS.columns:              # real components only (branch-aware); vec has its own stage
    DETS = DETS[DETS.component.isin(phd_env.components(["dir", "mag"]))].reset_index(drop=True)
DSCOL = {"ingv": "#1f2937", "experiment": "#dc2626"}
for sub in ["by_station", "by_template", "by_simulation", "by_dataset", "rankings", "comprehensive"]:
    (OUT/sub).mkdir(parents=True, exist_ok=True)


def grouped_bar(ax, df, key, val, ylab, title, order=None):
    keys = order or sorted(df[key].unique()); x = np.arange(len(keys)); w = 0.38
    for j, ds in enumerate(["ingv", "experiment"]):
        s = df[df.dataset == ds].groupby(key)[val].sum().reindex(keys).fillna(0)
        ax.bar(x + (j-0.5)*w, s.values, w, color=DSCOL[ds], label=dlab(ds))
    ax.set_xticks(x); ax.set_xticklabels([slab(k) if k in STATION_LABEL else str(k).replace("template", "template ").replace("sim", "simulation ") for k in keys], rotation=25, ha="right")
    ax.set_ylabel(ylab); ax.set_title(title); ax.legend(); ax.grid(axis="y", alpha=0.3)


def quality_bar(ax, df, key, ylab, title, order=None):
    keys = order or sorted(df[key].unique()); x = np.arange(len(keys)); w = 0.38
    for j, ds in enumerate(["ingv", "experiment"]):
        s = df[df.dataset == ds].groupby(key).max_r.mean().reindex(keys).fillna(0)
        ax.bar(x + (j-0.5)*w, s.values, w, color=DSCOL[ds], label=dlab(ds))
    ax.set_xticks(x); ax.set_xticklabels([slab(k) if k in STATION_LABEL else str(k).replace("template", "template ").replace("sim", "simulation ") for k in keys], rotation=25, ha="right")
    ax.set_ylabel(ylab); ax.set_title(title); ax.legend(); ax.grid(axis="y", alpha=0.3)


def fig1(key, kind, folder, order=None):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    grouped_bar(ax[0], SST, key, "n_detect", "peaks above null floor", f"Peaks above NULL floor per {kind}", order)
    grouped_bar(ax[1], SST, key, "n_signif", "peaks above significance floor", f"Peaks above SIGNIFICANCE floor per {kind}", order)
    quality_bar(ax[2], SST, key, "mean peak |r|", f"Match quality per {kind}", order)
    fig.suptitle(f"SWCC template-match summary by {kind} (per-template data-driven floors, denoised tilt)")
    fig.tight_layout(); fig.savefig(OUT/folder/f"peaks_by_{kind}.png", dpi=300); plt.close(fig)


def main():
    fig1("station", "station", "by_station", order=["ECPN", "EEC1", "EC1", "EC10", "ECIT", "ECOR", "EMAS"])
    fig1("template", "template", "by_template", order=["template1", "template2", "template3", "template4"])
    fig1("sim", "simulation", "by_simulation", order=["sim1", "sim2", "sim3", "sim4"])

    # per-template floor panel (floors vary by station/period)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    keys = ["template1", "template2", "template3", "template4"]; x = np.arange(4); w = 0.38
    for j, ds in enumerate(["ingv", "experiment"]):
        s = SST[SST.dataset == ds].groupby("template").floor_signif.mean().reindex(keys)
        ax.bar(x+(j-0.5)*w, s.values, w, color=DSCOL[ds], label=dlab(ds))
    ax.set_xticks(x); ax.set_xticklabels(["template 1", "template 2", "template 3", "template 4"]); ax.set_ylabel("mean significance floor |r|")
    ax.set_xlabel("Template"); ax.set_title("Data-driven significance floor per template (low for the long template4)")
    ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(OUT/"by_template"/"floors_by_template.png", dpi=300); plt.close(fig)

    # by_dataset totals + length-normalised rate
    spans = {ds: (g.peak_time.max()-g.peak_time.min()).total_seconds()/86400 for ds, g in DETS.groupby("dataset")}
    agg = SST.groupby("dataset").agg(n_detect=("n_detect", "sum"), n_signif=("n_signif", "sum")).reindex(["ingv", "experiment"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(2)
    ax[0].bar(x-0.2, agg.n_detect, 0.4, color="#93c5fd", label="above null")
    ax[0].bar(x+0.2, agg.n_signif, 0.4, color="#1d4ed8", label="above significance")
    ax[0].set_xticks(x); ax[0].set_xticklabels([dlab(d) for d in agg.index]); ax[0].set_ylabel("total peaks")
    ax[0].set_title("Total template-match peaks per dataset"); ax[0].legend(); ax[0].grid(axis="y", alpha=0.3)
    rate = [agg.n_signif[ds]/spans[ds] for ds in agg.index]
    ax[1].bar(x, rate, color=[DSCOL[ds] for ds in agg.index])
    ax[1].set_xticks(x); ax[1].set_xticklabels([dlab(d) for d in agg.index]); ax[1].set_ylabel("significance peaks / day")
    ax[1].set_title("Length-normalised significance-peak rate"); ax[1].grid(axis="y", alpha=0.3)
    fig.suptitle("SWCC template matches by dataset (eruptive vs non-eruptive period)")
    fig.tight_layout(); fig.savefig(OUT/"by_dataset"/"peaks_by_dataset.png", dpi=300); plt.close(fig)

    # rankings: top-25 SST by n_signif, floors annotated
    d = SST.sort_values("n_signif", ascending=False).head(25).copy()
    d["sst"] = d.dataset.map({"ingv":"erup","experiment":"summ"})+"·"+d.station.replace({"EEC1":"EC1"})+"·"+d.sim.str.replace("sim","s")+"·"+d.template.str.replace("template","t")
    fig, ax = plt.subplots(figsize=(10, 9))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.n_signif, color=[DSCOL[x] for x in d.dataset])
    for yi, r in zip(y, d.itertuples()):
        ax.text(r.n_signif, yi, f"  floor={r.floor_signif:.2f}, max|r|={r.max_r:.2f}", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(d.sst, fontsize=8)
    ax.set_xlabel("peaks above significance floor"); ax.set_title("Top-25 Station·Sim·Template by significance-peak count")
    ax.grid(axis="x", alpha=0.3); fig.tight_layout(); fig.savefig(OUT/"rankings"/"top25_sst.png", dpi=300); plt.close(fig)

    # comprehensive one-pager
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))
    grouped_bar(ax[0, 0], SST, "station", "n_signif", "signif peaks", "By station",
                ["ECPN", "EEC1", "EC1", "EC10", "ECIT", "ECOR", "EMAS"])
    grouped_bar(ax[0, 1], SST, "template", "n_signif", "signif peaks", "By template", keys)
    grouped_bar(ax[1, 0], SST, "sim", "n_signif", "signif peaks", "By simulation", ["sim1", "sim2", "sim3", "sim4"])
    quality_bar(ax[1, 1], SST, "station", "mean peak |r|", "Match quality by station",
                ["ECPN", "EEC1", "EC1", "EC10", "ECIT", "ECOR", "EMAS"])
    fig.suptitle("SWCC comprehensive ranking summary (peaks above per-template data-driven floors)", fontsize=15)
    fig.tight_layout(); fig.savefig(OUT/"comprehensive"/"summary.png", dpi=300); plt.close(fig)

    print(f"SWCC ranking suite → {OUT}")
    print(f"  totals: above-null {int(SST.n_detect.sum()):,} | above-significance {int(SST.n_signif.sum()):,}")


if __name__ == "__main__":
    main()
