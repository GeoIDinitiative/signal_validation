"""
swcc_accumulated.py  —  gap-aware SWCC with per-datetime accumulated detection score
===================================================================================
Builds on swcc_gapaware: for each (dataset, station, component) it computes, on the
full continuous grid, a gap-aware |r|(t) for every (sim, template) [short templates
T1-3], then ACCUMULATES them into one detection score per datetime:

  per-sim score   s_sim(t) = max over templates of |r|        (best template in a sim)
  MAX-combine     M(t)     = max over sims of s_sim(t)         (best of all 16)
  STACK-combine   S(t)     = mean over sims of s_sim(t)        (coherent average, √N)

Each combine has its OWN phase-randomised null floor (max-of-16 has a higher chance
baseline than mean-of-4, so a shared threshold would be unfair). We then compare how
many credible (null-surviving) detections each method yields.

Parameters (best practice): hop = 1 sample (matched filter), window = template length,
min_valid_frac = 0.8 (gap knob), peak separation = window length.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

from swcc_gapaware import swcc_gapaware, to_full_grid
from swcc_comprehensive import load_clean, load_template, SIMS, STATIONS, COMPONENTS, THRESHOLD
from credibility_checks import phase_randomize
from swcc_vector import swcc_score, surrogate          # vector (complex) detector + dtype-aware null

warnings.filterwarnings("ignore")

OUT = Path("/home/owen/tilt_validation/SWCC_comprehensive/accumulated")
TEMPLATES = ["template1", "template2", "template3"]
MIN_VALID = 0.8
PEAK_SEP = 3333          # = window length (non-overlapping detections)
N_SURR = 200


def per_sim_scores(sig, ds, st, comp="dir"):
    """dict sim -> s_sim(t) = nanmax over templates of |r|, all clipped to common length."""
    out = {}
    for sim in SIMS:
        rs = []
        for tname in TEMPLATES:
            tpl = load_template(ds, st, sim, tname, comp)
            if tpl is None:
                continue
            r = swcc_score(tpl, sig, min_valid_frac=MIN_VALID)   # |r| (scalar) or |R| (vector)
            if r.size:
                rs.append(r)
        if rs:
            L = min(len(r) for r in rs)
            out[sim] = np.nanmax(np.vstack([r[:L] for r in rs]), axis=0)
    return out


def combine(scores):
    """Return (max_combine, stack_combine) per datetime."""
    L = min(len(s) for s in scores.values())
    A = np.vstack([s[:L] for s in scores.values()])
    return np.nanmax(A, axis=0), np.nanmean(A, axis=0)


def null_floors(sig_host, ds, st, comp="dir", n=N_SURR, seed=3):
    """99th-pct null floor for each method from phase-randomised surrogates of the host."""
    rng = np.random.default_rng(seed)
    mx, stk, ps = [], [], []
    for _ in range(n):
        xs = surrogate(sig_host, rng)                 # complex-aware for comp='vec'
        sc = per_sim_scores(xs, ds, st, comp)
        if not sc:
            continue
        m, s = combine(sc)
        mx.append(np.nanmax(m)); stk.append(np.nanmax(s))
        ps.append(np.nanmax([np.nanmax(v) for v in sc.values()]))
    p = lambda a: float(np.percentile(a, 99)) if a else np.nan
    return {"max": p(mx), "stack": p(stk), "persim": p(ps)}


def detect(score, floor, sep=PEAK_SEP):
    s = np.where(np.isfinite(score), score, 0.0)
    pk, _ = find_peaks(s, height=THRESHOLD, distance=sep)
    sig = pk[score[pk] > floor] if np.isfinite(floor) else np.array([], int)
    return pk, sig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    rep = {}     # representative score traces for plotting
    for dataset, stations in STATIONS.items():
        for station in stations:
            for comp in COMPONENTS:
                clean = load_clean(dataset, station, comp)
                if clean is None:
                    continue
                grid_t, grid_x = to_full_grid(clean)
                sims = per_sim_scores(grid_x, dataset, station, comp)
                if not sims:
                    continue
                M, S = combine(sims)
                # host for the null = longest fully-valid run
                sl = clean.groupby("segment_id").size()
                host = clean[clean.segment_id == sl.idxmax()]["bandpassed"].to_numpy(float)
                floors = null_floors(host, dataset, station, comp)

                methods = {"persim_mean": None, "max": (M, floors["max"]),
                           "stack": (S, floors["stack"])}
                # per-sim: average significant across the 4 sims
                ps_sig = []
                for sim, sc in sims.items():
                    pk, sig = detect(sc, floors["persim"])
                    ps_sig.append(len(sig))
                rows.append({"dataset": dataset, "station": station, "component": comp,
                             "method": "persim_mean", "floor": round(floors["persim"], 3),
                             "n_peaks": int(np.mean([len(detect(sc, floors["persim"])[0]) for sc in sims.values()])),
                             "n_significant": float(np.mean(ps_sig))})
                for name, (score, fl) in [("max", (M, floors["max"])), ("stack", (S, floors["stack"]))]:
                    pk, sig = detect(score, fl)
                    rows.append({"dataset": dataset, "station": station, "component": comp,
                                 "method": name, "floor": round(fl, 3),
                                 "n_peaks": len(pk), "n_significant": len(sig)})
                if station in ("ECPN", "EMAS") and comp == "dir":
                    rep[station] = (grid_t, M, S, floors)
                print(f"  {dataset}/{station}/{comp}: floors max={floors['max']:.3f} "
                      f"stack={floors['stack']:.3f} persim={floors['persim']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "method_comparison.csv", index=False)
    summ = (df.groupby(["dataset", "method"])
            .agg(n_significant=("n_significant", "sum"), mean_floor=("floor", "mean"))
            .reset_index())
    print("\nCREDIBLE-DETECTION YIELD BY METHOD (sum over stations/components):")
    print(summ.to_string(index=False))
    summ.to_csv(OUT / "method_comparison_summary.csv", index=False)

    # representative per-datetime score figure
    if rep:
        fig, axes = plt.subplots(len(rep), 1, figsize=(14, 4*len(rep)), squeeze=False)
        for ax, (st, (t, M, S, fl)) in zip(axes[:, 0], rep.items()):
            L = min(len(t), len(M), len(S))
            ax.plot(t[:L], M[:L], lw=0.6, color="#2563eb", label="MAX-combine score")
            ax.plot(t[:L], S[:L], lw=0.6, color="#f59e0b", alpha=0.8, label="STACK-combine score")
            ax.axhline(fl["max"], ls="--", c="#2563eb", alpha=0.7, label=f"max null {fl['max']:.2f}")
            ax.axhline(fl["stack"], ls="--", c="#f59e0b", alpha=0.7, label=f"stack null {fl['stack']:.2f}")
            ax.set_title(f"{st} (dir): per-datetime accumulated detection score")
            ax.set_ylabel("score |r|"); ax.legend(fontsize=8, ncol=2, loc="upper right"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "accumulated_score_traces.png", dpi=140); plt.close(fig)
    print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
