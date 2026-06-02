"""
candidate_characterization.py — characterisation/distribution plots of the clean detections
(ports the etna_signals_phd comprehensive_analysis/best_candidates suite onto the floor-based,
denoised-continuous pipeline). Complements the ULP mega-panels (per-event waveforms) with the
population-level views: score distributions, per-station/per-dataset breakdowns, margin above the
significance floor, score-vs-time, and a top-20 table.

All counts/credibility are relative to each detection's OWN data-driven floors (not 0.2/0.5).

Outputs (comprehensive_analysis/best_candidates/):
  01_score_distribution.png   significant-peak |r| histogram per dataset (+ floor band)
  02_by_station.png            count + |r| boxplot per station
  03_by_dataset.png            counts, |r| distribution, length-normalised rate
  04_margin_above_floor.png    (score − significance floor) per station — credibility margin
  05_score_vs_time.png         significant-peak |r| vs time, coloured by station
  06_top20_table.png           the 20 loudest detections (station/time/score/floor/margin)
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_labels import slab, slabs, STATION_LABEL
from pathlib import Path

import phd_env                                # branch-aware OUT / detections / components
warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
OUT = phd_env.out(BASE/"comprehensive_analysis"/"best_candidates"); OUT.mkdir(parents=True, exist_ok=True)
D = pd.read_csv(phd_env.dets_dir()/"all_detections_continuous.csv", parse_dates=["peak_time"])
if "component" in D.columns:                  # real components only (branch-aware); vec → vector_orientation.py
    D = D[D.component.isin(phd_env.components(["dir", "mag"]))].reset_index(drop=True)
SIG = D[D.significant].copy(); SIG["margin"] = SIG.score - SIG.floor_signif
DSCOL = {"ingv": "#1f2937", "experiment": "#dc2626"}
STORDER = ["ECPN", "EEC1", "EC1", "EC10", "ECIT", "ECOR", "EMAS"]


def main():
    # 01 score distribution per dataset
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for ds, c in DSCOL.items():
        s = SIG[SIG.dataset == ds].score
        ax.hist(s, bins=30, alpha=0.6, color=c, label=f"{ds} (n={len(s)})", density=True)
    ax.axvspan(SIG.floor_signif.min(), SIG.floor_signif.max(), color="grey", alpha=0.15,
               label=f"significance floor range [{SIG.floor_signif.min():.2f}–{SIG.floor_signif.max():.2f}]")
    ax.set_xlabel("significant-peak correlation |r|"); ax.set_ylabel("density")
    ax.set_title("Distribution of significant-peak correlation by dataset")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT/"01_score_distribution.png", dpi=300); plt.close(fig)

    # 02 by station: count + score boxplot
    order = [s for s in STORDER if s in SIG.station.unique()]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    cnt = SIG.groupby("station").size().reindex(order).fillna(0)
    cols = [DSCOL["ingv"] if s in ("ECPN", "EEC1") else DSCOL["experiment"] for s in order]
    ax[0].bar(np.arange(len(order)), cnt.values, color=cols)
    ax[0].set_xticks(np.arange(len(order))); ax[0].set_xticklabels(slabs(order), rotation=30, ha="right")
    ax[0].set_ylabel("significant detections"); ax[0].set_title("Significant detection count per station"); ax[0].grid(axis="y", alpha=0.3)
    data = [SIG[SIG.station == s].score.values for s in order]
    bp = ax[1].boxplot(data, labels=slabs(order), patch_artist=True)
    for patch, s in zip(bp["boxes"], order):
        patch.set_facecolor(DSCOL["ingv"] if s in ("ECPN", "EEC1") else DSCOL["experiment"]); patch.set_alpha(0.6)
    ax[1].set_ylabel("peak |r|"); ax[1].set_title("Significant-peak |r| distribution per station")
    ax[1].tick_params(axis="x", rotation=30); ax[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Per-station detection characterisation"); fig.tight_layout(); fig.savefig(OUT/"02_by_station.png", dpi=300); plt.close(fig)

    # 03 by dataset
    spans = {ds: (g.peak_time.max()-g.peak_time.min()).total_seconds()/86400 for ds, g in D.groupby("dataset")}
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for ds, c in DSCOL.items():
        ax[0].bar(ds, len(SIG[SIG.dataset == ds]), color=c)
    ax[0].set_ylabel("significant detections"); ax[0].set_title("Count per dataset"); ax[0].grid(axis="y", alpha=0.3)
    for ds, c in DSCOL.items():
        ax[1].hist(SIG[SIG.dataset == ds].score, bins=25, alpha=0.6, color=c, label=ds, density=True)
    ax[1].set_xlabel("peak |r|"); ax[1].set_ylabel("density"); ax[1].set_title("Score distribution"); ax[1].legend(); ax[1].grid(alpha=0.3)
    for ds, c in DSCOL.items():
        ax[2].bar(ds, len(SIG[SIG.dataset == ds])/spans[ds], color=c)
    ax[2].set_ylabel("significant detections / day"); ax[2].set_title("Length-normalised rate"); ax[2].grid(axis="y", alpha=0.3)
    fig.suptitle("Dataset comparison — winter INGV vs summer experiment"); fig.tight_layout(); fig.savefig(OUT/"03_by_dataset.png", dpi=300); plt.close(fig)

    # 04 margin above floor
    fig, ax = plt.subplots(figsize=(12, 5.5))
    data = [SIG[SIG.station == s].margin.values for s in order]
    bp = ax.boxplot(data, labels=slabs(order), patch_artist=True)
    for patch, s in zip(bp["boxes"], order):
        patch.set_facecolor(DSCOL["ingv"] if s in ("ECPN", "EEC1") else DSCOL["experiment"]); patch.set_alpha(0.6)
    ax.axhline(0, ls="--", c="k", alpha=0.6)
    ax.set_ylabel("score − significance floor"); ax.set_title("Credibility margin above the significance floor per station")
    ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT/"04_margin_above_floor.png", dpi=300); plt.close(fig)

    # 05 score vs time
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))
    for a, ds in zip(ax, ["ingv", "experiment"]):
        g = SIG[SIG.dataset == ds]
        for s in g.station.unique():
            gs = g[g.station == s]; a.scatter(gs.peak_time, gs.score, s=18, alpha=0.6, label=s)
        a.set_ylabel("peak |r|"); a.set_title(f"{ds}: significant-peak |r| over time"); a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3)
    ax[1].set_xlabel("date"); fig.tight_layout(); fig.savefig(OUT/"05_score_vs_time.png", dpi=300); plt.close(fig)

    # 06 top-20 table
    top = SIG.sort_values("score", ascending=False).head(20)[
        ["dataset", "station", "component", "method", "peak_time", "score", "floor_signif", "margin"]].copy()
    top["peak_time"] = top.peak_time.dt.strftime("%Y-%m-%d %H:%M")
    top["score"] = top.score.round(3); top["floor_signif"] = top.floor_signif.round(3); top["margin"] = top.margin.round(3)
    fig, ax = plt.subplots(figsize=(13, 7)); ax.axis("off")
    t = ax.table(cellText=top.values, colLabels=top.columns, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.5)
    for j in range(len(top.columns)):
        t[(0, j)].set_facecolor("#1f2937"); t[(0, j)].set_text_props(color="white", fontweight="bold")
    ax.set_title("Top-20 loudest significant detections", fontweight="bold", pad=20)
    fig.tight_layout(); fig.savefig(OUT/"06_top20_table.png", dpi=300); plt.close(fig)
    top.to_csv(OUT/"top20_candidates.csv", index=False)

    print(f"candidate characterisation ({len(SIG)} significant detections) → {OUT}")


if __name__ == "__main__":
    main()
