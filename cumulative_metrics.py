"""
cumulative_metrics.py — cumulative detection-rate curves over time (ports the etna_signals_phd
cumulative_metrics onto the floor-based detection). FULLY ALIGNED: uses the same significant /
above-null detections (per-station data-driven floors) as the main pipeline.

Cumulative curves expose whether detections accumulate at a steady (noise-like) rate or in bursts
tied to events. The INGV panel overlays the eruptive phases.

Outputs (comprehensive_analysis/cumulative_metrics/):
  cumulative_<dataset>.png   cumulative significant detections per station + all-station above-null
                             vs above-significance, with eruptive overlays (INGV)
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_labels import slab, slabs, STATION_LABEL
from matplotlib.dates import date2num
from pathlib import Path

sys.path.insert(0, "/home/owen/tilt_validation")
from swcc_oldstyle_plots import load_volcanic_events

import phd_env                                # branch-aware OUT / detections / components
warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
OUT = phd_env.out(BASE/"comprehensive_analysis"/"cumulative_metrics"); OUT.mkdir(parents=True, exist_ok=True)
DETS = pd.read_csv(phd_env.dets_dir()/"all_detections_continuous.csv", parse_dates=["peak_time"])
if "component" in DETS.columns:              # real components only (branch-aware); vec has its own stage
    DETS = DETS[DETS.component.isin(phd_env.components(["dir", "mag"]))].reset_index(drop=True)
EV = load_volcanic_events(BASE/"etna_volcanic_events_cleaned.csv")
SPANS = EV[EV.end_datetime > EV.start_datetime + pd.Timedelta(hours=1)]
PERIODS = {"ingv": ("2022-11-14", "2023-03-01"), "experiment": ("2023-07-23", "2023-08-04")}


def cum(times):
    t = np.sort(times.values)
    return t, np.arange(1, len(t)+1)


def main():
    for ds, g0 in DETS.groupby("dataset"):
        method = "max"
        gm = g0[g0.method == method]
        fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        # per-station cumulative significant
        for st in sorted(gm.station.unique()):
            s = gm[(gm.station == st) & gm.significant]
            if len(s):
                t, c = cum(s.peak_time); ax[0].step(t, c, where="post", label=slab(st), lw=1.6)
        if ds == "ingv":
            for _, e in SPANS.iterrows():
                ax[0].axvspan(e.start_datetime, e.end_datetime, color="#dc2626", alpha=0.10)
                ax[1].axvspan(e.start_datetime, e.end_datetime, color="#dc2626", alpha=0.10)
        ax[0].set_ylabel("cumulative significant detections")
        ax[0].set_title(f"{ds}: cumulative significant detections per station (MAX, above 99th-pct floor)")
        ax[0].legend(fontsize=9, ncol=3); ax[0].grid(alpha=0.3)
        # all-station above-null vs above-significance
        td, cd = cum(gm.peak_time)
        ts, cs = cum(gm[gm.significant].peak_time)
        ax[1].step(td, cd, where="post", color="#93c5fd", lw=2, label="above null floor (detect)")
        ax[1].step(ts, cs, where="post", color="#1d4ed8", lw=2, label="above significance floor")
        ax[1].set_ylabel("cumulative detections (all stations)"); ax[1].set_xlabel("date")
        ax[1].set_title(f"{ds}: cumulative detection rate, all stations"
                        + (" (red = eruptive phases)" if ds == "ingv" else ""))
        ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
        ax[1].set_xlim(pd.Timestamp(PERIODS[ds][0]), pd.Timestamp(PERIODS[ds][1]))
        fig.tight_layout(); fig.savefig(OUT/f"cumulative_{ds}.png", dpi=300); plt.close(fig)
        print(f"  cumulative {ds}: {len(gm[gm.significant])} significant (max)")
    print(f"cumulative metrics → {OUT}")


if __name__ == "__main__":
    main()
