"""
top_template_plots.py — per-(station,sim,template) performance plots (ports the old etna_signals_phd
top-40 SST + by-sim/by-template analysis onto the NEW denoised continuous pipeline).

For every (dataset, station, sim, template) it correlates the bank template against the denoised
continuous directional tilt (gap-aware SWCC), counts peaks above correlation thresholds, and records
the peak correlation. From that it builds, all properly formatted (titles / axis labels / gridlines):

  top_templates/01_top40_all_thresholds.png   top-40 station-sim-template by peak count, |r|>=0.2
  top_templates/02_top40_moderate.png         "                                    0.2<=|r|<0.5
  top_templates/03_top40_high_quality.png     "                                    |r|>=0.5
  top_templates/performance_by_sim.png        peaks + mean peak-|r| per simulation (by dataset)
  top_templates/performance_by_template.png   peaks + mean peak-|r| per template   (by dataset)
  top_templates/station_template_heatmap.png  station x (sim·template) peak-count heatmap
  top_templates/sst_peak_counts.csv           the aggregated table

Output: SWCC_comprehensive/top_templates/
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

from swcc_gapaware import swcc_gapaware
from swcc_comprehensive import load_template, SIMS, STATIONS
from swcc_continuous import load_cont, to_grid

warnings.filterwarnings("ignore")
OUT = Path("/home/owen/tilt_validation/SWCC_comprehensive/top_templates"); OUT.mkdir(parents=True, exist_ok=True)
TEMPL = ["template1", "template2", "template3", "template4"]   # full bank (incl. the long template4)
COMP, THRESH, DIST = "dir", 0.2, 1000
DSCOL = {"ingv": "#1f2937", "experiment": "#dc2626"}


def compute():
    rows = []
    for ds, sts in STATIONS.items():
        for st in sts:
            d = load_cont(ds, st, COMP)
            if d is None:
                continue
            _, gx, _ = to_grid(d)
            for sim in SIMS:
                for tn in TEMPL:
                    tpl = load_template(ds, st, sim, tn)
                    if tpl is None:
                        continue
                    r = swcc_gapaware(tpl, gx)
                    if r.size == 0:
                        continue
                    ar = np.nan_to_num(np.abs(r))
                    pk, _ = find_peaks(ar, height=THRESH, distance=DIST)
                    v = ar[pk]
                    rows.append({"dataset": ds, "station": st, "sim": sim, "template": tn,
                                 "n_all": int(len(v)),
                                 "n_mod": int(((v >= 0.2) & (v < 0.5)).sum()),
                                 "n_high": int((v >= 0.5).sum()),
                                 "max_r": round(float(v.max()), 3) if len(v) else 0.0,
                                 "mean_r": round(float(v.mean()), 3) if len(v) else 0.0})
            print(f"  scored {ds}/{st}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sst_peak_counts.csv", index=False)
    return df


def _sst(df):
    return (df.dataset.str[:3] + "·" + df.station + "·" + df.sim.str.replace("sim", "s")
            + "·" + df.template.str.replace("template", "t"))


def top40(df, col, title, fname):
    d = df.copy(); d["sst"] = _sst(d)
    d = d.sort_values(col, ascending=False).head(40)
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(np.arange(len(d)), d[col].to_numpy(), color=[DSCOL[x] for x in d.dataset])
    ax.set_xticks(np.arange(len(d))); ax.set_xticklabels(d.sst, rotation=90, fontsize=7)
    ax.set_xlabel("Station · Sim · Template combination", fontweight="700", fontsize=12)
    ax.set_ylabel("Number of matched peaks", fontweight="700", fontsize=12)
    ax.set_title(title, fontweight="700", fontsize=15, pad=12)
    ax.grid(axis="y", alpha=0.3)
    ax.margins(x=0.005)
    handles = [plt.Rectangle((0, 0), 1, 1, color=DSCOL[k]) for k in DSCOL]
    ax.legend(handles, [f"{k} (winter)" if k == "ingv" else f"{k} (summer)" for k in DSCOL],
              fontsize=10, frameon=True)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=300); plt.close(fig)


def performance(df, key, kind, fname):
    g = (df.groupby([key, "dataset"]).agg(peaks=("n_all", "sum"), mean_r=("max_r", "mean")).reset_index())
    keys = sorted(df[key].unique())
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for j, ds in enumerate(["ingv", "experiment"]):
        sub = g[g.dataset == ds].set_index(key).reindex(keys).fillna(0)
        ax[0].bar(x + (j-0.5)*w, sub.peaks, w, color=DSCOL[ds], label=f"{ds}")
        ax[1].bar(x + (j-0.5)*w, sub.mean_r, w, color=DSCOL[ds], label=f"{ds}")
    for a, ylab, ttl in [(ax[0], "total matched peaks (|r|≥0.2)", f"Detection yield per {kind}"),
                         (ax[1], "mean peak correlation |r|", f"Match quality per {kind}")]:
        a.set_xticks(x); a.set_xticklabels([k.replace("template", "t").replace("sim", "sim ") for k in keys])
        a.set_xlabel(kind.capitalize(), fontweight="700"); a.set_ylabel(ylab, fontweight="700")
        a.set_title(ttl, fontweight="700", fontsize=13); a.grid(axis="y", alpha=0.3); a.legend(fontsize=9)
    fig.suptitle(f"Template-bank performance by {kind} (denoised continuous, directional tilt)",
                 fontweight="700", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=300); plt.close(fig)


def heatmap(df):
    d = df.copy()
    d["row"] = d.dataset.str[:3] + "/" + d.station
    d["col"] = d.sim.str.replace("sim", "s") + "·" + d.template.str.replace("template", "t")
    piv = d.pivot_table(index="row", columns="col", values="n_all", fill_value=0)
    piv = piv.reindex(sorted(piv.columns), axis=1)
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, int(piv.iat[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if piv.iat[i, j] < piv.to_numpy().max()*0.6 else "black")
    ax.set_xlabel("Sim · Template", fontweight="700"); ax.set_ylabel("Dataset / Station", fontweight="700")
    ax.set_title("Matched-peak count (|r|≥0.2) by station and template", fontweight="700", fontsize=14, pad=12)
    cb = fig.colorbar(im, ax=ax); cb.set_label("number of matched peaks", fontweight="700")
    fig.tight_layout(); fig.savefig(OUT / "station_template_heatmap.png", dpi=300); plt.close(fig)


def main():
    csv = OUT / "sst_peak_counts.csv"
    df = pd.read_csv(csv) if csv.exists() and "--reuse" in __import__("sys").argv else compute()
    top40(df, "n_all",  "Top 40 Station-Sim-Template Combinations — all thresholds (|r| ≥ 0.2)", "01_top40_all_thresholds.png")
    top40(df, "n_mod",  "Top 40 Station-Sim-Template Combinations — moderate (0.2 ≤ |r| < 0.5)", "02_top40_moderate.png")
    top40(df, "n_high", "Top 40 Station-Sim-Template Combinations — high quality (|r| ≥ 0.5)", "03_top40_high_quality.png")
    performance(df, "sim", "simulation", "performance_by_sim.png")
    performance(df, "template", "template", "performance_by_template.png")
    heatmap(df)
    best = df.sort_values("n_all", ascending=False).iloc[0]
    print(f"\nTop SST: {best.station}-{best.sim}-{best.template} ({best.n_all} peaks, max|r|={best.max_r})")
    print(f"total peaks (|r|≥0.2): {int(df.n_all.sum()):,}  | high-quality (|r|≥0.5): {int(df.n_high.sum()):,}")
    print(f"Outputs → {OUT}")


if __name__ == "__main__":
    main()
