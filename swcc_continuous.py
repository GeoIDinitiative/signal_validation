"""
swcc_continuous.py  —  Design B detection on the full continuous signal
======================================================================
Runs the gap-aware accumulated SWCC (per-datetime MAX / STACK score) on the
continuous_bandpassed signal (the whole record, ~100% coverage), then applies the
peak VETO in post-processing:
  · drop any peak whose window overlaps a veto zone (detectable-quake buffer or
    block-edge settling), and
  · drop any peak within ±CODA_H of an M >= CODA_MAG earthquake ETA (coda safety,
    even for events left in the signal).
Then: per-method null floors, cross-station synchrony with a circular-shift null,
and a signal-usage report — all directly comparable to the cut-based result.

Output: SWCC_comprehensive/continuous/  (summary.txt, synchrony.png)
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

from swcc_accumulated import per_sim_scores, combine, PEAK_SEP
from swcc_comprehensive import STATIONS, COMPONENTS, THRESHOLD
from credibility_checks import phase_randomize
from swcc_vector import load_cont_complex, surrogate   # vector (complex) observed + dtype-aware null
import phd_env                                          # branch-aware OUT / components / tracks

# detection "tracks": the established scalar detectors (dir, mag) and the new vector |R| detector.
# Kept separate so the scalar synchrony result is unchanged and the vector result is directly comparable.
# In a phd_output branch run these collapse to one track named after the branch (phd_env).
_LEGACY_TRACKS = {"scalar": ["dir", "mag"], "vec": ["vec"]}
TRACKS = phd_env.tracks(_LEGACY_TRACKS)
COMPS = phd_env.components(["dir", "mag", "vec"])
COMP2TRACK = phd_env.comp_track_map(_LEGACY_TRACKS)

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
CONT = BASE / "continuous_bandpassed"
OUT  = phd_env.out(BASE / "SWCC_comprehensive" / "continuous")
N_NULL, N_PERM, TOL_S = 300, 2000, 600
VETO_PRE = 1800                  # s — also veto contamination just BEFORE a peak (bandpass ringing
#                                  precedes the correlation peak); makes the veto symmetric, not forward-only
CODA_MAG, CODA_H = 5.5, 2.0      # veto peaks within ±2 h of any M>=5.5 ETA
CODA_VETO_ON = "--no-coda" not in sys.argv   # toggle the coda peak-veto for the confirmation test

eq = pd.read_csv(BASE / "earthquakes_merged_utc.csv")
eq["p_wave_eta"] = pd.to_datetime(eq["p_wave_eta"])
CODA_ETAS = eq[eq.magnitude >= CODA_MAG]["p_wave_eta"].to_numpy()


def load_cont(ds, st, comp):
    if comp == "vec":                                  # complex observed: dir (c1) + i·dir2 (c2)
        return load_cont_complex(ds, st)
    f = CONT / ds / f"{st}_{comp}_0p001-0p01Hz_cont_bp.feather"
    if not f.exists():
        return None
    d = pd.read_feather(f); d["datetime"] = pd.to_datetime(d["datetime"])
    return d


def to_grid(d):
    t0, t1 = d["datetime"].iloc[0], d["datetime"].iloc[-1]
    grid = pd.date_range(t0, t1, freq="1s")
    cplx = np.iscomplexobj(d["bandpassed"].to_numpy())          # vec path carries a complex signal
    x = pd.Series((np.nan + 0j) if cplx else np.nan, index=grid,
                  dtype=complex if cplx else float)
    x.loc[d["datetime"].values] = d["bandpassed"].to_numpy()
    v = pd.Series(False, index=grid); v.loc[d["datetime"].values] = d["veto"].to_numpy()
    return grid.values, x.to_numpy(), v.to_numpy(bool)


def near_coda(t):
    if not CODA_VETO_ON or len(CODA_ETAS) == 0:
        return False
    dt = np.abs((np.datetime64(t) - CODA_ETAS) / np.timedelta64(1, "h"))
    return bool(dt.min() <= CODA_H)


def detect(gt, score, veto, fdet, fsig, M=PEAK_SEP):
    """find_peaks is wired to the DETECTION floor (95th-pct null). Peaks above the SIGNIFICANCE
    floor (99th-pct null) get the `significant` flag — but note this is a per-window, single-station
    SCREENING flag (above-floor), NOT a detection claim: see audit finding C (tilt_experiments/
    10_floor_recalibration). The trustworthy significances are the time-slide FAR, the circular-shift
    synchrony, and the ULP-morphology gate. Returns detection records after veto + coda filters."""
    s = np.where(np.isfinite(score), score, 0.0)
    pk, _ = find_peaks(s, height=fdet, distance=PEAK_SEP)   # min detection = null floor
    recs = []
    for i in pk:
        if veto[max(0, i-VETO_PRE):min(i+M, len(veto))].any():   # matched window + pre-ring buffer overlaps veto
            continue
        t = gt[i]
        if near_coda(t):                              # coda safety
            continue
        recs.append({"peak_time": pd.Timestamp(t), "score": float(score[i]),
                     "significant": bool(score[i] > fsig)})
    return recs


def null_floor(host, ds, st, comp="dir", n=N_NULL, seed=5):
    """Two-tier floors per method: (95th-pct = detection, 99th-pct = significance)."""
    rng = np.random.default_rng(seed)
    mx, stk = [], []
    for _ in range(n):
        xs = surrogate(host, rng)                      # complex-aware for comp='vec'
        sc = per_sim_scores(xs, ds, st, comp)
        if not sc:
            continue
        m, s = combine(sc)
        mx.append(np.nanmax(m)); stk.append(np.nanmax(s))
    def tier(a):
        return (float(np.percentile(a, 95)), float(np.percentile(a, 99))) if a else (np.nan, np.nan)
    return tier(mx), tier(stk)


def count_coincidences(stt, tol=TOL_S):
    rows = sorted((x, st) for st, t in stt.items() for x in t)
    used = [False]*len(rows); n = 0
    for i in range(len(rows)):
        if used[i]:
            continue
        members = {rows[i][1]}; used[i] = True
        for j in range(i+1, len(rows)):
            if (rows[j][0]-rows[i][0]).total_seconds() <= tol:
                members.add(rows[j][1]); used[j] = True
            else:
                break
        if len(members) >= 2:
            n += 1
    return n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["DESIGN B — full-continuous-signal detection (gap-aware accumulated SWCC + veto)",
         "=" * 70, ""]

    # cache scores + detections (synchrony uses the SIGNIFICANT set), separated by track
    det = {m: {tr: {ds: {} for ds in STATIONS} for tr in TRACKS} for m in ("max", "stack")}
    spans = {ds: {} for ds in STATIONS}
    usage, counts, all_dets = [], [], []
    for ds, sts in STATIONS.items():
        for st in sts:
            per_st_times = {tr: {"max": [], "stack": []} for tr in TRACKS}
            for comp in COMPS:
                d = load_cont(ds, st, comp)
                if d is None:
                    continue
                gt, gx, gv = to_grid(d)
                sims = per_sim_scores(gx, ds, st, comp)
                if not sims:
                    continue
                M, S = combine(sims)
                # null host = longest clean (un-vetoed, finite) stretch
                fin = np.isfinite(gx) & ~gv
                host = gx[fin][:20000] if fin.sum() > 20000 else gx[fin]
                (m95, m99), (s95, s99) = null_floor(host, ds, st, comp)
                dM = detect(gt, M, gv, m95, m99)
                dS = detect(gt, S, gv, s95, s99)
                sM = [r["peak_time"] for r in dM if r["significant"]]
                sS = [r["peak_time"] for r in dS if r["significant"]]
                tr = COMP2TRACK.get(comp, comp)
                per_st_times[tr]["max"].append(pd.Series(sM))
                per_st_times[tr]["stack"].append(pd.Series(sS))
                spans[ds][st] = (gt[0], gt[-1])
                for method, recs, f95, f99 in [("max", dM, m95, m99), ("stack", dS, s95, s99)]:
                    for r in recs:
                        all_dets.append({"dataset": ds, "station": st, "component": comp,
                                         "method": method, "peak_time": r["peak_time"],
                                         "score": round(r["score"], 4),
                                         "floor_detect": round(f95, 3), "floor_signif": round(f99, 3),
                                         "significant": r["significant"]})
                counts.append({"dataset": ds, "station": st, "component": comp,
                               "max_detect": len(dM), "max_signif": len(sM),
                               "stack_detect": len(dS), "stack_signif": len(sS),
                               "max_f95": round(m95, 3), "max_f99": round(m99, 3),
                               "stack_f95": round(s95, 3), "stack_f99": round(s99, 3)})
                if comp == "dir":
                    usage.append({"dataset": ds, "station": st,
                                  "analysed": int(np.isfinite(gx).sum()),
                                  "grid": len(gx), "veto_pct": 100*gv.mean()})
            for tr in TRACKS:
                for m in ("max", "stack"):
                    if per_st_times[tr][m]:
                        allt = pd.concat(per_st_times[tr][m]).sort_values().reset_index(drop=True)
                        keep = []
                        for x in allt:
                            if not keep or (x-keep[-1]).total_seconds() > TOL_S:
                                keep.append(x)
                        det[m][tr][ds][st] = pd.to_datetime(pd.Series(keep))
        print(f"   detected {ds}")
    sfx0 = '' if CODA_VETO_ON else '_nocoda'
    pd.DataFrame(counts).to_csv(OUT / f"detect_counts{sfx0}.csv", index=False)
    pd.DataFrame(all_dets).to_csv(OUT / f"all_detections_continuous{sfx0}.csv", index=False)

    L.append("(1) SIGNAL USAGE (continuous):")
    u = pd.DataFrame(usage)
    for _, r in u.iterrows():
        L.append(f"   {r.dataset:11s} {r.station:5s}: {r.analysed:>9,} analysed "
                 f"({100*r.analysed/r.grid:.0f}% of span, {r.veto_pct:.1f}% vetoed)")
    L.append("")
    c = pd.DataFrame(counts)
    L.append("(1b) DETECTIONS  (find_peaks wired to null floors: detect=95th, signif=99th)")
    L.append("     NB: 'signif' = above per-window 99th-pct floor = SCREENING ONLY, not a detection")
    L.append("     claim (single-station amplitude floor is a weak discriminator — audit finding C).")
    L.append("     Real significance = synchrony (sec.2) + time-slide FAR + ULP-morphology gate.")
    for ds in STATIONS:
        cs = c[c.dataset == ds]
        L.append(f"   {ds:11s}: MAX  detect={cs.max_detect.sum()} above-floor={cs.max_signif.sum()} | "
                 f"STACK detect={cs.stack_detect.sum()} above-floor={cs.stack_signif.sum()}")
    L.append("")

    fig, axes = plt.subplots(len(TRACKS), 2, figsize=(14, 5*len(TRACKS)), squeeze=False)
    rng = np.random.default_rng(11)
    sync_rows = []
    for row, tr in enumerate(TRACKS):
        for col, method in enumerate(["max", "stack"]):
            L.append(f"(2) SYNCHRONY — TRACK {tr.upper()} · METHOD {method.upper()}")
            for ds in STATIONS:
                stt = {k: v for k, v in det[method][tr][ds].items() if len(v)}
                ndet = {k: len(v) for k, v in stt.items()}
                if len(stt) < 2:
                    L.append(f"   {ds}: <2 stations with detections ({ndet})")
                    sync_rows.append({"track": tr, "dataset": ds, "method": method,
                                      "n_stations": len(stt), "observed": 0, "chance": np.nan, "p": np.nan})
                    continue
                obs = count_coincidences(stt)
                null = np.empty(N_PERM)
                for k in range(N_PERM):
                    sh = {}
                    for st, t in stt.items():
                        t0, t1 = spans[ds][st]
                        span = (pd.Timestamp(t1)-pd.Timestamp(t0)).total_seconds()
                        off = rng.uniform(0, span)
                        nt = ((pd.to_datetime(t)-pd.Timestamp(t0)).dt.total_seconds()+off) % span
                        sh[st] = pd.to_datetime(pd.Timestamp(t0)+pd.to_timedelta(nt, unit="s"))
                    null[k] = count_coincidences(sh)
                p = float((null >= obs).mean())
                sync_rows.append({"track": tr, "dataset": ds, "method": method, "n_stations": len(stt),
                                  "observed": obs, "chance": round(float(null.mean()), 3), "p": p})
                L.append(f"   {ds:11s}: observed={obs} chance={null.mean():.2f} p={p:.4f}  det/station={ndet}")
                axes[row][col].hist(null, bins=range(0, max(6, obs+2)), alpha=0.5, label=f"{ds} chance")
                axes[row][col].axvline(obs, ls="--", lw=2, label=f"{ds} obs={obs} (p={p:.3f})")
            axes[row][col].set_title(f"{tr.upper()} · {method.upper()} detector")
            axes[row][col].set_xlabel("number of coincident cross-station events")
            axes[row][col].set_ylabel("count over time-shift permutations")
            axes[row][col].legend(fontsize=8); axes[row][col].grid(alpha=0.3)
            L.append("")

    sfx = "" if CODA_VETO_ON else "_nocoda"
    pd.DataFrame(sync_rows).to_csv(OUT / f"synchrony{sfx}.csv", index=False)
    fig.suptitle("Cross-station synchrony — observed vs time-shift-null distribution")
    fig.tight_layout(); fig.savefig(OUT / f"synchrony{sfx}.png", dpi=300); plt.close(fig)
    (OUT / f"summary{sfx}.txt").write_text("\n".join(L))
    print("\n".join(L)); print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
