"""
peak_density_map.py — temporal distribution of clean correlated events vs Etna eruptive activity.

Collates every CLEAN peak above the null (detect, 95th-pct floor) and significance (99th-pct floor)
levels for all stations, bins them in time, and renders density heatmaps against the documented
eruptive-event overlays — to see whether the template-correlated events cluster around eruptions.
Also runs a permutation test: is the significant-peak rate higher during eruptive-active intervals
than during quiescent ones?

Outputs (eruptive_temporal/):
  01_eruptive_period_density.png   INGV winter eruptive window: eruptive timeline + daily peak
                                   density (pooled) + station x time heatmap, with overlays
  02_all_stations_timeline.png     all 7 stations x full timeline heatmap, eruptive overlays
  clustering_test.txt              eruptive-active vs quiescent peak-rate + permutation p
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import date2num, DateFormatter
from pathlib import Path

sys.path.insert(0, "/home/owen/tilt_validation")   # so the pipeline modules import from this subfolder
from swcc_oldstyle_plots import load_volcanic_events
from plot_labels import slab, slabs, STATION_LABEL

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
OUT = BASE/"eruptive_temporal"; OUT.mkdir(parents=True, exist_ok=True)
DETS = BASE/"SWCC_comprehensive"/"continuous"/"all_detections_continuous.csv"
EV = load_volcanic_events(BASE/"etna_volcanic_events_cleaned.csv")
WIN = (pd.Timestamp("2022-11-14"), pd.Timestamp("2023-03-01"))     # INGV eruptive window
FULL = (pd.Timestamp("2022-11-14"), pd.Timestamp("2023-08-04"))    # both campaigns
ING = ["ECPN", "EEC1"]; EXP = ["EC1", "EC10", "ECIT", "ECOR", "EMAS"]
SPANS = EV[EV.end_datetime > EV.start_datetime + pd.Timedelta(hours=1)]   # effusive/eruptive periods


def overlay_eruptions(ax, lo, hi, label=True):
    for i, (_, e) in enumerate(SPANS.iterrows()):
        ax.axvspan(date2num(max(e.start_datetime, lo)), date2num(min(e.end_datetime, hi)),
                   color="#dc2626", alpha=0.12, lw=0, label="eruptive phase" if (label and i == 0) else None)
    for _, e in EV.iterrows():
        if lo <= e.start_datetime <= hi:
            ax.axvline(date2num(e.start_datetime), color="#7f1d1d", alpha=0.25, lw=0.6)


def daily_counts(times, edges):
    return np.histogram(date2num(pd.DatetimeIndex(times)), bins=edges)[0]


def main():
    d = pd.read_csv(DETS, parse_dates=["peak_time"])
    if "component" in d.columns:               # scalar temporal density; vec → vector_orientation.py
        d = d[d.component.isin(["dir", "mag"])].reset_index(drop=True)
    sig = d[d.significant]                     # above 99th significance floor
    det = d                                    # above 95th null floor (all find_peaks output)

    # ── clustering test: significant-peak rate, eruptive-active vs quiescent (INGV) ──
    ing_sig = sig[sig.station.isin(ING)]
    active = np.zeros(len(ing_sig), bool)
    for _, e in EV.iterrows():
        active |= ((ing_sig.peak_time >= e.start_datetime) & (ing_sig.peak_time <= e.end_datetime)).to_numpy()
    span_days = (WIN[1]-WIN[0]).days
    active_secs = sum((min(e.end_datetime, WIN[1])-max(e.start_datetime, WIN[0])).total_seconds()
                      for _, e in SPANS.iterrows() if e.end_datetime > WIN[0] and e.start_datetime < WIN[1])
    frac_time_active = active_secs/((WIN[1]-WIN[0]).total_seconds())
    obs_frac = active.mean()
    # permutation: circularly shift peak times within the window, recompute active fraction
    rng = np.random.default_rng(1); span = (WIN[1]-WIN[0]).total_seconds()
    t0s = (ing_sig.peak_time - WIN[0]).dt.total_seconds().to_numpy()
    null = np.empty(2000)
    ev_lo = [(max(e.start_datetime, WIN[0])-WIN[0]).total_seconds() for _, e in SPANS.iterrows()]
    ev_hi = [(min(e.end_datetime, WIN[1])-WIN[0]).total_seconds() for _, e in SPANS.iterrows()]
    for k in range(len(null)):
        sh = (t0s + rng.uniform(0, span)) % span
        a = np.zeros(len(sh), bool)
        for lo, hi in zip(ev_lo, ev_hi):
            a |= (sh >= lo) & (sh <= hi)
        null[k] = a.mean()
    p = float((null >= obs_frac).mean())
    L = ["CLEAN-PEAK ↔ ERUPTION TEMPORAL ASSOCIATION (INGV winter)", "=" * 52,
         f"significant peaks (INGV): {len(ing_sig)}",
         f"fraction of the window that is eruptive-active: {frac_time_active:.2f}",
         f"fraction of significant peaks falling in eruptive-active intervals: {obs_frac:.2f}",
         f"  (if peaks were unrelated to eruptions, expect ≈ {frac_time_active:.2f})",
         f"permutation test (circular time-shift, 2000×): p = {p:.3f}",
         "",
         ("→ Significant clustering of correlated events around eruptive activity." if p < 0.05 else
          "→ NO significant clustering: correlated events are distributed in time independently of the"),
         ("" if p < 0.05 else "  eruptive phases (consistent with the overall null — the matches are not eruption-driven).")]
    (OUT/"clustering_test.txt").write_text("\n".join(L)); print("\n".join(L))

    # ── Figure 1: eruptive-period density (INGV) ──
    edges = date2num(pd.date_range(WIN[0], WIN[1], freq="1D"))
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.5, 1.1, 1.0], hspace=0.12)
    lo, hi = WIN
    ax_ev = fig.add_subplot(gs[0]); overlay_eruptions(ax_ev, lo, hi)
    ax_ev.set_yticks([]); ax_ev.set_title("Clean correlated-event density vs Etna eruptive activity — INGV winter period", fontsize=14)
    ax_ev.set_ylabel("eruptive\noverlay", fontsize=9); ax_ev.set_xlim(date2num(lo), date2num(hi)); ax_ev.legend(loc="upper right", fontsize=8)

    ax_m = fig.add_subplot(gs[1], sharex=ax_ev)
    cd = daily_counts(det[det.station.isin(ING)].peak_time, edges)
    cs = daily_counts(ing_sig.peak_time, edges)
    ctr = edges[:-1] + 0.5
    ax_m.fill_between(ctr, cd, step="mid", color="#93c5fd", alpha=0.7, label="above null floor (detect)")
    ax_m.fill_between(ctr, cs, step="mid", color="#1d4ed8", alpha=0.9, label="above significance floor")
    overlay_eruptions(ax_m, lo, hi, label=False)
    ax_m.set_ylabel("clean peaks / day\n(ECPN + EC1 eruptive)"); ax_m.legend(loc="upper right", fontsize=8); ax_m.grid(alpha=0.3)

    ax_h = fig.add_subplot(gs[2], sharex=ax_ev)
    grid = np.vstack([daily_counts(ing_sig[ing_sig.station == s].peak_time, edges) for s in ING])
    im = ax_h.imshow(grid, aspect="auto", cmap="magma", extent=[edges[0], edges[-1], len(ING)-0.5, -0.5],
                     interpolation="nearest")
    overlay_eruptions(ax_h, lo, hi, label=False)
    ax_h.set_yticks(range(len(ING))); ax_h.set_yticklabels(slabs(ING))
    ax_h.set_ylabel("station"); ax_h.set_xlabel("date")
    ax_h.xaxis.set_major_formatter(DateFormatter("%b %d")); ax_h.set_xlim(date2num(lo), date2num(hi))
    cb = fig.colorbar(im, ax=[ax_ev, ax_m, ax_h], fraction=0.025, pad=0.01); cb.set_label("significant peaks / day")
    ax_h.text(0.01, -0.32, f"eruptive-active peak fraction {obs_frac:.2f} (expected {frac_time_active:.2f} if random) — "
              f"permutation p={p:.3f} → {'clustered' if p<0.05 else 'no clustering'}",
              transform=ax_h.transAxes, fontsize=9)
    fig.savefig(OUT/"01_eruptive_period_density.png", dpi=300); plt.close(fig)

    # ── Figure 2: all stations, full timeline ──
    edges2 = date2num(pd.date_range(FULL[0], FULL[1], freq="2D"))
    stations = ING + EXP
    grid2 = np.vstack([daily_counts(sig[sig.station == s].peak_time, edges2) for s in stations])
    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(np.log1p(grid2), aspect="auto", cmap="viridis",
                   extent=[edges2[0], edges2[-1], len(stations)-0.5, -0.5], interpolation="nearest")
    overlay_eruptions(ax, FULL[0], FULL[1])
    ax.set_yticks(range(len(stations)))
    ax.set_yticklabels([slab(s) if s in ('EEC1','EC1') else f"{s} ({'eruptive' if s in ING else 'non-eruptive'})" for s in stations])
    ax.axhline(len(ING)-0.5, color="white", lw=1)
    ax.set_xlabel("date"); ax.set_ylabel("station")
    ax.set_title("Significant-peak temporal density — all stations vs eruptive overlays (full observation span)", fontsize=14)
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y")); ax.set_xlim(edges2[0], edges2[-1])
    ax.legend(loc="upper right", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.set_label("log(1 + significant peaks / 2-day bin)")
    fig.tight_layout(); fig.savefig(OUT/"02_all_stations_timeline.png", dpi=300); plt.close(fig)
    print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
