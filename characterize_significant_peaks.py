"""
characterize_significant_peaks.py
=================================
Interrogate the null-surviving (significant) SWCC detections to judge whether they
are genuine volcanic tilt transients rather than residual artefacts.

Inputs : SWCC_comprehensive/all_peaks_flagged.csv   (significant flag from null floor)
         etna_volcanic_events_cleaned.csv            (winter/INGV eruptive chronology)

Analyses
  1. Collapse multi-template/multi-component hits to one detection per (station, time)
     so a single transient isn't counted many times.
  2. Cross-station SYNCHRONY: cluster station-detections within TOL_S; a real source
     should appear at several stations at once (key test for the summer period where
     no volcanic catalogue exists).
  3. Volcanic-event COINCIDENCE: for INGV detections, nearest catalogue event + Δt.
  4. Temporal clustering by day + a timeline figure.

Outputs (SWCC_comprehensive/characterization/):
  significant_timeline.png, synchronous_significant_events.csv,
  station_detections.csv, characterization_summary.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/owen/tilt_validation")
OUT  = BASE / "SWCC_comprehensive" / "characterization"
PEAKS = BASE / "SWCC_comprehensive" / "all_peaks_flagged.csv"
VOLC  = BASE / "etna_volcanic_events_cleaned.csv"

TOL_S        = 600     # cross-station / same-detection tolerance (s)
DEDUP_S      = 600     # collapse hits at one station within this window
PERIODS = {
    "ingv":       ("2022-11-14", "2023-03-01"),
    "experiment": ("2023-07-23", "2023-08-03"),
}


def load_significant():
    p = pd.read_csv(PEAKS, parse_dates=["peak_time"])
    return p[p.significant].sort_values("peak_time").reset_index(drop=True)


def collapse_station_detections(sig):
    """One detection per (station, time-cluster); keep the strongest |r| and its template."""
    rows = []
    for (ds, st), g in sig.groupby(["dataset", "station"]):
        g = g.sort_values("peak_time")
        cluster_start = None
        bucket = []
        def flush(b):
            if not b:
                return
            b = pd.DataFrame(b)
            best = b.loc[b.abs_r.idxmax()]
            rows.append({
                "dataset": ds, "station": st,
                "time": b.peak_time.min(),
                "time_end": b.peak_time.max(),
                "n_hits": len(b),
                "best_abs_r": best.abs_r,
                "best_sim": best.sim, "best_template": best.template,
                "components": ",".join(sorted(b.component.unique())),
            })
        for _, r in g.iterrows():
            if cluster_start is None or (r.peak_time - cluster_start).total_seconds() <= DEDUP_S:
                bucket.append(r); cluster_start = bucket[0].peak_time
            else:
                flush(bucket); bucket = [r]; cluster_start = r.peak_time
        flush(bucket)
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def find_synchronous(dets):
    """Cluster station-detections across stations within TOL_S -> synchronous events."""
    events = []
    for ds, g in dets.groupby("dataset"):
        g = g.sort_values("time").reset_index(drop=True)
        used = np.zeros(len(g), bool)
        for i in range(len(g)):
            if used[i]:
                continue
            t0 = g.time[i]
            members = g[(~used) & ((g.time - t0).abs().dt.total_seconds() <= TOL_S)]
            used[members.index] = True
            stations = sorted(members.station.unique())
            events.append({
                "dataset": ds, "time": members.time.min(),
                "n_stations": len(stations), "stations": ",".join(stations),
                "n_detections": len(members), "max_abs_r": members.best_abs_r.max(),
            })
    ev = pd.DataFrame(events).sort_values(["n_stations", "max_abs_r"], ascending=False)
    return ev.reset_index(drop=True)


def load_volcanic():
    v = pd.read_csv(VOLC)
    v["start"] = pd.to_datetime(v["Date"] + " " + v["Time"], errors="coerce")
    return v.dropna(subset=["start"])


def volcanic_coincidence(dets, volc):
    ingv = dets[dets.dataset == "ingv"].copy()
    if ingv.empty or volc.empty:
        return ingv
    starts = volc["start"].to_numpy()
    out = []
    for _, d in ingv.iterrows():
        dt = (d.time.to_datetime64() - starts) / np.timedelta64(1, "h")
        j = np.argmin(np.abs(dt))
        out.append({**d.to_dict(),
                    "nearest_event": volc.iloc[j]["Event_Type"],
                    "dt_hours": round(float(dt[j]), 1)})
    return pd.DataFrame(out)


def timeline_fig(sig, dets, volc):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for ax, (ds, (lo, hi)) in zip(axes, PERIODS.items()):
        s = sig[sig.dataset == ds]
        stations = sorted(s.station.unique())
        ymap = {st: i for i, st in enumerate(stations)}
        if ds == "ingv":
            for _, e in volc.iterrows():
                if pd.Timestamp(lo) <= e.start <= pd.Timestamp(hi):
                    ax.axvline(e.start, color="#16a34a", alpha=0.25, lw=1, zorder=1)
        for st in stations:
            ss = s[s.station == st]
            ax.scatter(ss.peak_time, [ymap[st]] * len(ss),
                       s=20 + 120 * (ss.abs_r - 0.6), alpha=0.7,
                       c=ss.abs_r, cmap="viridis", vmin=0.6, vmax=0.95, zorder=3)
        ax.set_yticks(range(len(stations))); ax.set_yticklabels(stations)
        ax.set_title(f"{ds}: significant detections "
                     f"(green = volcanic events)" if ds == "ingv" else
                     f"{ds}: significant detections")
        ax.set_xlim(pd.Timestamp(lo), pd.Timestamp(hi)); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "significant_timeline.png", dpi=140); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sig = load_significant()
    dets = collapse_station_detections(sig)
    dets.to_csv(OUT / "station_detections.csv", index=False)
    sync = find_synchronous(dets)
    sync.to_csv(OUT / "synchronous_significant_events.csv", index=False)
    volc = load_volcanic()
    coinc = volcanic_coincidence(dets, volc)
    timeline_fig(sig, dets, volc)

    L = ["CHARACTERIZATION OF SIGNIFICANT SWCC DETECTIONS", "=" * 55,
         f"significant peaks (raw)     : {len(sig)}",
         f"station-level detections    : {len(dets)}  (multi-template hits collapsed)",
         ""]
    L.append("Station-level detections by dataset/station:")
    for (ds, st), g in dets.groupby(["dataset", "station"]):
        L.append(f"  {ds:11s} {st:5s} : {len(g):3d}  (best|r| up to {g.best_abs_r.max():.3f})")
    L.append("")
    multi = sync[sync.n_stations >= 2]
    L.append(f"SYNCHRONY: {len(multi)} multi-station events (>=2 stations within {TOL_S}s)")
    if len(multi):
        L.append(multi.head(20).to_string(index=False))
    else:
        L.append("  none — significant detections are NOT cross-station coincident.")
    L.append("")
    if len(coinc):
        near = coinc[coinc.dt_hours.abs() <= 24]
        L.append(f"VOLCANIC COINCIDENCE (INGV): {len(near)}/{len(coinc)} detections "
                 f"within 24 h of a catalogue event")
        L.append("  (timezone of catalogue vs UTC peaks unverified — treat as indicative)")
        L.append(coinc[["station", "time", "best_abs_r", "nearest_event", "dt_hours"]]
                 .to_string(index=False))
    txt = "\n".join(L)
    (OUT / "characterization_summary.txt").write_text(txt)
    print(txt)
    print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
