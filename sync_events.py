"""
sync_events.py — gallery of the actual cross-station SYNCHRONOUS coincidences (ports the
etna_signals_phd synchronous_events folder onto the floor-based detection + the main synchrony
procedure). FULLY ALIGNED: it uses the SAME significant detections (above each station's 99th-pct
floor) and the SAME TOL_S=600 s coincidence window as swcc_continuous's synchrony test.

For each coincidence (≥2 stations whose significant detections fall within TOL_S), it plots the
participating stations' denoised tilt around the event so the coincidence can be inspected — is it
a coherent network event, or chance alignment of unrelated peaks? (Given the null synchrony result,
expect the latter and expect very few coincidences.)

Outputs (synchronous_events/):
  coincidences.csv          every coincidence: time, stations, scores
  sync_event_<...>.png      one gallery panel per coincidence
  SUMMARY.txt               counts + the synchrony p-values (from synchrony.csv)
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_labels import slab, slabs, STATION_LABEL
from pathlib import Path

from swcc_continuous import load_cont, TOL_S      # SAME coincidence window as the main synchrony test

import phd_env                                # branch-aware OUT / detections / components
warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
OUT = phd_env.out(BASE/"synchronous_events"); OUT.mkdir(parents=True, exist_ok=True)
for p in OUT.glob("sync_event_*.png"):
    p.unlink()
DETS = pd.read_csv(phd_env.dets_dir()/"all_detections_continuous.csv", parse_dates=["peak_time"])
if "component" in DETS.columns:              # real components only (branch-aware); vec has its own stage
    DETS = DETS[DETS.component.isin(phd_env.components(["dir", "mag"]))].reset_index(drop=True)
SYNC = phd_env.dets_dir()/"synchrony.csv"
STATIONS = {"ingv": ["ECPN", "EEC1"], "experiment": ["EC1", "EC10", "ECIT", "ECOR", "EMAS"]}
COL = {"max": "#2563eb", "stack": "#f59e0b"}
MAXPANELS = 60


def dedup(times, tol):
    out = []
    for t in sorted(times):
        if not out or (t-out[-1]).total_seconds() > tol:
            out.append(t)
    return out


def groups(stt, tol):
    rows = sorted((t, st) for st, ts in stt.items() for t in ts)
    used = [False]*len(rows); out = []
    for i in range(len(rows)):
        if used[i]:
            continue
        mem = {rows[i][1]: rows[i][0]}; used[i] = True
        for j in range(i+1, len(rows)):
            if (rows[j][0]-rows[i][0]).total_seconds() <= tol:
                mem.setdefault(rows[j][1], rows[j][0]); used[j] = True
            else:
                break
        if len(mem) >= 2:
            out.append(mem)
    return out


def main():
    sig = DETS[DETS.significant]
    rows = []; n_panel = 0
    for method in ["max", "stack"]:
        for ds, sts in STATIONS.items():
            stt = {}
            for st in sts:
                ts = sig[(sig.dataset == ds) & (sig.station == st) & (sig.method == method)].peak_time.tolist()
                if ts:
                    stt[st] = dedup(ts, TOL_S)         # de-dup dir/mag within TOL_S, as in the main test
            if len(stt) < 2:
                continue
            for g in groups(stt, TOL_S):
                t0 = min(g.values())
                rows.append({"dataset": ds, "method": method, "time": t0,
                             "stations": "+".join(sorted(g)), "n_stations": len(g)})
                if n_panel >= MAXPANELS:
                    continue
                lo = min(g.values()) - pd.Timedelta(minutes=30); hi = max(g.values()) + pd.Timedelta(minutes=60)
                mem = sorted(g)
                fig, ax = plt.subplots(len(mem), 1, figsize=(11, 2.4*len(mem)), squeeze=False)
                for a, st in zip(ax[:, 0], mem):
                    d = load_cont(ds, st, "dir")
                    seg = d[(d.datetime >= lo) & (d.datetime <= hi)]
                    a.plot(seg.datetime, seg.bandpassed, lw=0.8, color=COL[method])
                    a.axvline(g[st], ls="--", c="k"); a.set_ylabel(f"{slab(st)}\nbandpassed"); a.grid(alpha=0.3)
                ax[0, 0].set_title(f"{ds} · {method} · synchronous coincidence {t0:%Y-%m-%d %H:%M} · "
                                   f"stations {'+'.join(slabs(mem))} (within {TOL_S}s)")
                ax[-1, 0].set_xlabel("time")
                fig.tight_layout()
                fig.savefig(OUT/f"sync_event_{ds}_{method}_{t0:%Y%m%d_%H%M}.png", dpi=200); plt.close(fig)
                n_panel += 1
    df = pd.DataFrame(rows); df.to_csv(OUT/"coincidences.csv", index=False)

    syncp = pd.read_csv(SYNC) if SYNC.exists() else pd.DataFrame()
    L = ["SYNCHRONOUS-EVENT GALLERY (aligned to the main floor-based synchrony test)", "=" * 60,
         f"coincidence window TOL_S = {TOL_S}s (identical to swcc_continuous)",
         f"total cross-station coincidences of significant detections: {len(df)}", ""]
    for ds in STATIONS:
        for method in ["max", "stack"]:
            n = len(df[(df.dataset == ds) & (df.method == method)]) if not df.empty else 0
            p = syncp[(syncp.dataset == ds) & (syncp.method == method)].p.iloc[0] if len(syncp) and \
                len(syncp[(syncp.dataset == ds) & (syncp.method == method)]) else np.nan
            L.append(f"   {ds:11s} {method:5s}: {n} coincidence(s)   synchrony p = {p:.3f}")
    L += ["", "Each coincidence is plotted; with the synchrony p-values non-significant, these are chance",
          "alignments of unrelated single-station peaks, not coherent network events (consistent with null)."]
    (OUT/"SUMMARY.txt").write_text("\n".join(L)); print("\n".join(L)); print(f"\n{n_panel} panels → {OUT}")


if __name__ == "__main__":
    main()
