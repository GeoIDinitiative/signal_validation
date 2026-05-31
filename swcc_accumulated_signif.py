"""
swcc_accumulated_signif.py
==========================
(1) Signal-usage audit: how much of the post-earthquake signal the gap-aware
    accumulated SWCC actually analyses.
(2) Is there real signal?  EXCESS-OVER-CHANCE via cross-station SYNCHRONY.
    For each combine method (MAX, STACK) we collapse detections to station level
    and count cross-station coincidences (<= TOL_S apart). The null is per-station
    random circular time-shifts (preserves each station's own detection clustering,
    destroys cross-station alignment) — so a high observed count vs the null means
    the detections are coherent across independent sensors, i.e. real signal.
    (Coincidence-excess is used instead of raw count-excess because the latter is
    confounded by the 99th-percentile floor calibration; coincidence is not.)
Outputs: SWCC_comprehensive/accumulated/significance_test.txt + .png, signal_usage.csv
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

from swcc_gapaware import to_full_grid
from swcc_accumulated import per_sim_scores, combine, PEAK_SEP
from swcc_comprehensive import load_clean, STATIONS, COMPONENTS, THRESHOLD

warnings.filterwarnings("ignore")
OUT = Path("/home/owen/tilt_validation/SWCC_comprehensive/accumulated")
RAW = Path("/home/owen/tilt_validation/tilt_raw_pwave_removed")
FLOORS = pd.read_csv(OUT / "method_comparison.csv")
TOL_S = 600
N_PERM = 2000


def floor_of(ds, st, comp, method):
    f = FLOORS[(FLOORS.dataset == ds) & (FLOORS.station == st)
              & (FLOORS.component == comp) & (FLOORS.method == method)].floor
    return float(f.iloc[0]) if len(f) else np.nan


# ── (1) signal usage ──────────────────────────────────────────────────────────
def signal_usage():
    rows = []
    for ds, sts in STATIONS.items():
        for st in sts:
            raw = pd.read_feather(RAW / f"{st}_tilt_raw_pwave_removed.feather")
            raw["datetime"] = pd.to_datetime(raw["datetime"])
            n_post = raw.drop_duplicates("datetime").shape[0]
            cl = load_clean(ds, st, "dir")
            n_used = len(cl) if cl is not None else 0
            rows.append({"dataset": ds, "station": st, "post_EQ_unique": n_post,
                         "analysed": n_used, "pct_analysed": 100 * n_used / max(n_post, 1)})
    return pd.DataFrame(rows)


# ── cache the real accumulated scores once ────────────────────────────────────
def build_cache():
    cache = {}
    for ds, sts in STATIONS.items():
        for st in sts:
            for comp in COMPONENTS:
                cl = load_clean(ds, st, comp)
                if cl is None:
                    continue
                gt, gx = to_full_grid(cl)
                sims = per_sim_scores(gx, ds, st)
                if not sims:
                    continue
                M, S = combine(sims)
                cache[(ds, st, comp)] = (gt, M, S)
        print(f"   scored {ds}")
    return cache


def detect_times(gt, score, floor):
    s = np.where(np.isfinite(score), score, 0.0)
    pk, _ = find_peaks(s, height=THRESHOLD, distance=PEAK_SEP)
    sig = pk[score[pk] > floor]
    return pd.to_datetime(gt[:len(score)][sig])


def station_detections(cache, method):
    res = {ds: {} for ds in STATIONS}
    spans = {ds: {} for ds in STATIONS}
    for (ds, st, comp), (gt, M, S) in cache.items():
        score = M if method == "max" else S
        t = detect_times(gt, score, floor_of(ds, st, comp, method))
        res[ds].setdefault(st, []).append(pd.Series(t))
        spans[ds][st] = (gt[0], gt[-1])
    # union components per station, dedupe within TOL
    for ds in res:
        for st in list(res[ds]):
            allt = pd.concat(res[ds][st]).sort_values().reset_index(drop=True)
            keep = []
            for x in allt:
                if not keep or (x - keep[-1]).total_seconds() > TOL_S:
                    keep.append(x)
            res[ds][st] = pd.to_datetime(pd.Series(keep))
    return res, spans


def count_coincidences(station_times, tol=TOL_S):
    rows = []
    for st, t in station_times.items():
        for x in t:
            rows.append((x, st))
    if not rows:
        return 0
    rows.sort(key=lambda r: r[0])
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


def synchrony(cache, method, seed=11):
    rng = np.random.default_rng(seed)
    res, spans = station_detections(cache, method)
    out = {}
    for ds in STATIONS:
        stt = {k: v for k, v in res[ds].items() if len(v)}
        ndet = {k: len(v) for k, v in stt.items()}
        if len(stt) < 2:
            out[ds] = dict(observed=0, chance_mean=0.0, p=np.nan, ndet=ndet, null=np.zeros(1))
            continue
        obs = count_coincidences(stt)
        null = np.empty(N_PERM)
        for k in range(N_PERM):
            shifted = {}
            for st, t in stt.items():
                t0, t1 = spans[ds][st]
                span_s = (pd.Timestamp(t1) - pd.Timestamp(t0)).total_seconds()
                off = rng.uniform(0, span_s)
                nt = ((pd.to_datetime(t) - pd.Timestamp(t0)).dt.total_seconds() + off) % span_s
                shifted[st] = pd.to_datetime(pd.Timestamp(t0) + pd.to_timedelta(nt, unit="s"))
            null[k] = count_coincidences(shifted)
        out[ds] = dict(observed=obs, chance_mean=float(null.mean()),
                       p=float((null >= obs).mean()), ndet=ndet, null=null)
    return out


def main():
    L = ["GAP-AWARE ACCUMULATED SWCC — signal usage + significance", "=" * 60, ""]

    usage = signal_usage(); usage.to_csv(OUT / "signal_usage.csv", index=False)
    L.append("(1) SIGNAL USAGE (post-earthquake signal actually analysed)")
    for _, r in usage.iterrows():
        L.append(f"   {r.dataset:11s} {r.station:5s}: {r.analysed:>9,}/{r.post_EQ_unique:>9,}"
                 f"  ({r.pct_analysed:.0f}% analysed)")
    L.append(f"   OVERALL: {100*usage.analysed.sum()/usage.post_EQ_unique.sum():.0f}% of the "
             f"post-EQ signal is in long-enough segments to be analysed")
    L.append("")

    print("scoring (once)…"); cache = build_cache()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for col, method in enumerate(["max", "stack"]):
        L.append(f"(2) SYNCHRONY EXCESS — METHOD = {method.upper()}")
        sy = synchrony(cache, method)
        for ds, d in sy.items():
            L.append(f"   {ds:11s}: observed coincidences={d['observed']}  "
                     f"chance={d['chance_mean']:.2f}  p={d['p']:.4f}   "
                     f"detections/station={d['ndet']}")
            ax = axes[col]
            mx = max(6, d["observed"]+2)
            ax.hist(d["null"], bins=range(0, mx+1), alpha=0.5, label=f"{ds} chance")
            ax.axvline(d["observed"], ls="--", lw=2,
                       label=f"{ds} observed={d['observed']} (p={d['p']:.3f})")
        axes[col].set_title(f"{method.upper()}: cross-station coincidences vs chance")
        axes[col].set_xlabel("coincident events"); axes[col].set_ylabel("permutations")
        axes[col].legend(fontsize=8); axes[col].grid(alpha=0.3)
        L.append("")

    fig.tight_layout(); fig.savefig(OUT / "significance_test.png", dpi=140); plt.close(fig)
    (OUT / "significance_test.txt").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nOutputs → {OUT}/significance_test.txt /.png /signal_usage.csv")


if __name__ == "__main__":
    main()
