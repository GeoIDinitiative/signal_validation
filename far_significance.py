"""
far_significance.py  —  GW method 4: false-alarm-rate significance via time-slides
=================================================================================
GW quotes significance as a FALSE-ALARM RATE: how often background (noise) produces a
coincidence at least as loud as the candidate. The background is built by TIME-SLIDING
the stations relative to each other (circular shifts) past any plausible signal delay, so
every coincidence in the slid data is by construction accidental. The loudest real
coincidence's FAR (events / year) gives its significance.

Statistic: a cross-station coincidence's loudness = sum of the participating stations'
detection scores. Real coincidences are ranked against the time-slide background.

Output: gw_methods/far.png , far.csv
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
DETS = BASE / "SWCC_comprehensive" / "continuous" / "all_detections_continuous.csv"
OUT  = BASE / "gw_methods"
TOL_S, N_SLIDES, METHOD = 600, 500, "max"


def coincidences(stations):
    """List of (time, loudness=sum of scores) for clusters spanning >=2 stations."""
    rows = sorted((t, st, sc) for st, df in stations.items()
                  for t, sc in zip(df["sec"], df["score"]))
    out = []
    used = [False]*len(rows)
    for i in range(len(rows)):
        if used[i]:
            continue
        members = {rows[i][1]: rows[i][2]}; used[i] = True
        for j in range(i+1, len(rows)):
            if rows[j][0] - rows[i][0] <= TOL_S:
                members.setdefault(rows[j][1], rows[j][2]); used[j] = True
            else:
                break
        if len(members) >= 2:
            out.append((rows[i][0], sum(members.values())))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dets = pd.read_csv(DETS, parse_dates=["peak_time"])
    dets = dets[dets.method == METHOD]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, ds in zip(axes, ["ingv", "experiment"]):
        g = dets[dets.dataset == ds]
        if g.empty:
            continue
        t0 = g.peak_time.min(); t1 = g.peak_time.max()
        span = (t1 - t0).total_seconds()
        livetime_yr = span / (365.25*86400)
        stations = {}
        for st, sdf in g.groupby("station"):
            sec = (sdf.peak_time - t0).dt.total_seconds().to_numpy()
            stations[st] = pd.DataFrame({"sec": sec, "score": sdf.score.to_numpy()})
        if len(stations) < 2:
            continue
        real = coincidences(stations)
        real_loud = max((c[1] for c in real), default=0.0)

        # time-slide background
        rng = np.random.default_rng(9)
        bg = []
        for _ in range(N_SLIDES):
            slid = {st: df.assign(sec=(df.sec + rng.uniform(0, span)) % span).sort_values("sec")
                    for st, df in stations.items()}
            bg += [c[1] for c in coincidences(slid)]
        bg = np.array(bg)
        bg_livetime_yr = N_SLIDES * livetime_yr

        # FAR curve: events/yr with loudness >= x
        xs = np.linspace(0, max(real_loud, bg.max() if len(bg) else 1)*1.05, 200)
        far = np.array([(bg >= x).sum() for x in xs]) / max(bg_livetime_yr, 1e-9)
        ax.semilogy(xs, np.maximum(far, 1e-3), color="#2563eb", label="background FAR")
        if real_loud > 0:
            far_loud = (bg >= real_loud).sum() / max(bg_livetime_yr, 1e-9)
            ax.axvline(real_loud, ls="--", c="#dc2626",
                       label=f"loudest real (FAR={far_loud:.1f}/yr)")
            rows.append({"dataset": ds, "n_stations": len(stations), "n_real_coinc": len(real),
                         "loudest": round(real_loud, 3), "far_per_yr": round(far_loud, 2),
                         "livetime_yr": round(livetime_yr, 3)})
        ax.set(title=f"{ds}: FAR vs loudness ({METHOD})", xlabel="coincidence loudness (Σ score)",
               ylabel="false-alarm rate (per year)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "far.png", dpi=300); plt.close(fig)
    df = pd.DataFrame(rows); df.to_csv(OUT / "far.csv", index=False)
    print("FALSE-ALARM RATE (time-slide background):")
    print(df.to_string(index=False) if len(df) else "  (no >=2-station coincidences)")
    print("\nA loudest-event FAR of many/year = consistent with background = NOT significant.")
    print(f"Outputs → {OUT}")


if __name__ == "__main__":
    main()
