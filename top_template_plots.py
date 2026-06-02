"""
top_template_plots.py — per-(station,sim,template) performance plots, thresholded by each
template's OWN data-driven null + significance floors (NOT the legacy fixed 0.2 / 0.5).

For every (dataset, station, sim, template) on the denoised continuous directional tilt it runs a
gap-aware SWCC and counts peaks relative to that template's two-tier floors from
SWCC_comprehensive/continuous/template_floors.csv (stage 2):
    floor_detect = 95th-pct of surrogate-max |r|  (the NULL floor — replaces 0.2)
    floor_signif = 99th-pct of surrogate-max |r|  (the SIGNIFICANCE floor — replaces 0.5)
These floors vary by station AND period (ingv winter vs experiment summer) AND template, so every
combination is judged against its own threshold.

Outputs (SWCC_comprehensive/top_templates/), all formatted (titles / axes / gridlines):
  01_top40_above_null_floor.png        top-40 SST by peaks ABOVE the per-template null floor
  02_top40_between_floors.png          top-40 SST by peaks between null and significance floors
  03_top40_above_significance_floor.png top-40 SST by peaks ABOVE the per-template significance floor
  performance_by_sim.png / performance_by_template.png   yield (above null) + mean peak-|r|
  station_template_heatmap.png         station × (sim·template) peaks-above-significance heatmap
  sst_peak_counts.csv                  the aggregated table (incl. the floor used per combination)
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_labels import slab, slabs, STATION_LABEL
from pathlib import Path
from scipy.signal import find_peaks

from swcc_gapaware import swcc_gapaware
from swcc_comprehensive import load_template, SIMS, STATIONS
from swcc_continuous import load_cont, to_grid

import phd_env                                # branch-aware OUT / floors / component
warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
OUT = phd_env.out(BASE/"SWCC_comprehensive"/"top_templates"); OUT.mkdir(parents=True, exist_ok=True)
FLOORS_CSV = phd_env.dets_dir()/"template_floors.csv"
TEMPL = ["template1", "template2", "template3", "template4"]
COMP, DIST = phd_env.components(["dir"])[0], 1000   # branch primary axis (X for components, mag, …)
DSCOL = {"ingv": "#1f2937", "experiment": "#dc2626"}


def load_floors():
    f = pd.read_csv(FLOORS_CSV)
    f = f[f.component == COMP]
    return {(r.dataset, r.station, r.sim, r.template): (r.floor_detect, r.floor_signif) for r in f.itertuples()}


def compute():
    FL = load_floors()
    rows = []
    for ds, sts in STATIONS.items():
        for st in sts:
            d = load_cont(ds, st, COMP)
            if d is None:
                continue
            _, gx, _ = to_grid(d)
            for sim in SIMS:
                for tn in TEMPL:
                    tpl = load_template(ds, st, sim, tn, COMP)
                    fl = FL.get((ds, st, sim, tn))
                    if tpl is None or fl is None:
                        continue
                    fdet, fsig = fl
                    r = swcc_gapaware(tpl, gx)
                    if r.size == 0 or not np.isfinite(fdet):
                        continue
                    ar = np.nan_to_num(np.abs(r))
                    pk, _ = find_peaks(ar, height=fdet, distance=DIST)   # peaks ABOVE the per-template null floor
                    v = ar[pk]
                    n_sig = int((v >= fsig).sum())
                    rows.append({"dataset": ds, "station": st, "sim": sim, "template": tn,
                                 "floor_detect": round(float(fdet), 4), "floor_signif": round(float(fsig), 4),
                                 "n_detect": int(len(v)), "n_signif": n_sig, "n_between": int(len(v))-n_sig,
                                 "max_r": round(float(v.max()), 3) if len(v) else 0.0,
                                 "mean_r": round(float(v.mean()), 3) if len(v) else 0.0})
            print(f"  scored {ds}/{st}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"sst_peak_counts.csv", index=False)
    return df


def _sst(df):
    return (df.dataset.map({"ingv":"erup","experiment":"summ"}) + "·" + df.station.replace({"EEC1":"EC1"})
            + "·" + df.sim.str.replace("sim", "s") + "·" + df.template.str.replace("template", "t"))


def top40(df, col, title, fname):
    d = df.copy(); d["sst"] = _sst(d)
    d = d.sort_values(col, ascending=False).head(40)
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(np.arange(len(d)), d[col].to_numpy(), color=[DSCOL[x] for x in d.dataset])
    ax.set_xticks(np.arange(len(d))); ax.set_xticklabels(d.sst, rotation=90, fontsize=7)
    ax.set_xlabel("Station · Sim · Template combination")
    ax.set_ylabel("Number of matched peaks")
    ax.set_title(title); ax.grid(axis="y", alpha=0.3); ax.margins(x=0.005)
    handles = [plt.Rectangle((0, 0), 1, 1, color=DSCOL[k]) for k in DSCOL]
    ax.legend(handles, [f"{k} (winter)" if k == "ingv" else f"{k} (summer)" for k in DSCOL])
    fig.tight_layout(); fig.savefig(OUT/fname, dpi=300); plt.close(fig)


def performance(df, key, kind, fname):
    g = df.groupby([key, "dataset"]).agg(peaks=("n_detect", "sum"), mean_r=("max_r", "mean")).reset_index()
    keys = sorted(df[key].unique()); x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for j, ds in enumerate(["ingv", "experiment"]):
        sub = g[g.dataset == ds].set_index(key).reindex(keys).fillna(0)
        ax[0].bar(x + (j-0.5)*w, sub.peaks, w, color=DSCOL[ds], label=ds)
        ax[1].bar(x + (j-0.5)*w, sub.mean_r, w, color=DSCOL[ds], label=ds)
    for a, ylab, ttl in [(ax[0], "peaks above the null floor", f"Detection yield per {kind}"),
                         (ax[1], "mean peak correlation |r|", f"Match quality per {kind}")]:
        a.set_xticks(x); a.set_xticklabels([k.replace("template", "t").replace("sim", "sim ") for k in keys])
        a.set_xlabel(kind.capitalize()); a.set_ylabel(ylab); a.set_title(ttl)
        a.grid(axis="y", alpha=0.3); a.legend(fontsize=9)
    fig.suptitle(f"Template-bank performance by {kind} (peaks above each template's own null floor)")
    fig.tight_layout(); fig.savefig(OUT/fname, dpi=300); plt.close(fig)


def heatmap(df):
    d = df.copy()
    d["row"] = d.dataset.str[:3] + "/" + d.station
    d["col"] = d.sim.str.replace("sim", "s") + "·" + d.template.str.replace("template", "t")
    piv = d.pivot_table(index="row", columns="col", values="n_signif", fill_value=0)
    piv = piv.reindex(sorted(piv.columns), axis=1)
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=9)
    mx = max(1, piv.to_numpy().max())
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, int(piv.iat[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if piv.iat[i, j] < mx*0.6 else "black")
    ax.set_xlabel("Sim · Template"); ax.set_ylabel("Dataset / Station")
    ax.set_title("Peaks above the per-template SIGNIFICANCE floor (99th-pct), by station and template")
    cb = fig.colorbar(im, ax=ax); cb.set_label("peaks above significance floor")
    fig.tight_layout(); fig.savefig(OUT/"station_template_heatmap.png", dpi=300); plt.close(fig)


def main():
    csv = OUT/"sst_peak_counts.csv"
    df = pd.read_csv(csv) if csv.exists() and "--reuse" in __import__("sys").argv else compute()
    top40(df, "n_detect", "Top 40 Station-Sim-Template — peaks above each template's NULL floor (95th-pct)",
          "01_top40_above_null_floor.png")
    top40(df, "n_between", "Top 40 Station-Sim-Template — peaks between the null and significance floors",
          "02_top40_between_floors.png")
    top40(df, "n_signif", "Top 40 Station-Sim-Template — peaks above each template's SIGNIFICANCE floor (99th-pct)",
          "03_top40_above_significance_floor.png")
    performance(df, "sim", "simulation", "performance_by_sim.png")
    performance(df, "template", "template", "performance_by_template.png")
    heatmap(df)
    best = df.sort_values("n_signif", ascending=False).iloc[0]
    print(f"\nfloors used: per (station, sim, template) from template_floors.csv "
          f"(range {df.floor_detect.min():.2f}–{df.floor_detect.max():.2f} detect / "
          f"{df.floor_signif.min():.2f}–{df.floor_signif.max():.2f} signif)")
    print(f"Top SST by significance: {best.station}-{best.sim}-{best.template} "
          f"({best.n_signif} peaks above signif floor {best.floor_signif}, max|r|={best.max_r})")
    print(f"totals: above-null {int(df.n_detect.sum()):,} | above-significance {int(df.n_signif.sum()):,}")
    print(f"Outputs → {OUT}")


if __name__ == "__main__":
    main()
