"""
flag_significant_peaks.py
=========================
Apply the phase-randomised null floor to the SWCC peaks so detections are judged
against an empirical significance level rather than the nominal r=0.2 threshold
(see SWCC_comprehensive/credibility/null_test.*).

For each dataset it builds a null distribution of peak |r| from phase-randomised
surrogates of a representative host segment, takes the 99th percentile as the
significance floor, then flags every peak in all_peaks.csv as significant or not.

Output:
  SWCC_comprehensive/all_peaks_flagged.csv
  SWCC_comprehensive/significance_summary.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

from swcc_comprehensive import (swcc_segment, load_clean, load_template,
                                SIMS, TEMPLATES, THRESHOLD, PEAK_DISTANCE)
from credibility_checks import phase_randomize

BASE = Path("/home/owen/tilt_validation")
OUT  = BASE / "SWCC_comprehensive"

# representative host per dataset (long, well-sampled segment)
HOSTS = {"ingv": ("ECPN", "dir"), "experiment": ("EMAS", "dir")}
N_SURR = 200
SEED = 1


def null_p99(dataset, station, comp, n_surr=N_SURR, seed=SEED):
    rng = np.random.default_rng(seed)
    df = load_clean(dataset, station, comp)
    sid = df.groupby("segment_id").size().idxmax()
    host = df[df.segment_id == sid]["bandpassed"].to_numpy(float)
    tpls = [load_template(dataset, station, s, t) for s in SIMS for t in TEMPLATES]
    tpls = [t for t in tpls if t is not None and len(t) <= len(host)]
    maxr = []
    for _ in range(n_surr):
        xs = phase_randomize(host, rng)
        best = 0.0
        for tpl in tpls:
            r = np.abs(swcc_segment(tpl, xs))
            if r.size:
                pk, _ = find_peaks(r, height=THRESHOLD, distance=PEAK_DISTANCE)
                if len(pk):
                    best = max(best, r[pk].max())
        maxr.append(best)
    return float(np.percentile(maxr, 99)), len(host), len(tpls)


def main():
    peaks = pd.read_csv(OUT / "all_peaks.csv")
    lines = ["SWCC SIGNIFICANCE (phase-randomised null, 99th-pct floor)", "=" * 58]
    floors = {}
    for ds, (st, cp) in HOSTS.items():
        p99, hlen, ntpl = null_p99(ds, st, cp)
        floors[ds] = p99
        lines.append(f"{ds:11s} host={st}/{cp} seg_len={hlen} tpls={ntpl}  "
                     f"|r|_99 floor = {p99:.3f}")
    lines.append("")

    peaks["null_floor"] = peaks["dataset"].map(floors)
    peaks["significant"] = peaks["abs_r"] > peaks["null_floor"]
    peaks.to_csv(OUT / "all_peaks_flagged.csv", index=False)

    lines.append(f"{'dataset/station/comp':28s} {'peaks':>7s} {'signif':>7s} {'%':>6s}")
    for (ds, st, cp), g in peaks.groupby(["dataset", "station", "component"]):
        n, ns = len(g), int(g["significant"].sum())
        lines.append(f"{ds+'/'+st+'/'+cp:28s} {n:7d} {ns:7d} {100*ns/n:6.1f}")
    tot, tots = len(peaks), int(peaks["significant"].sum())
    lines += ["", f"TOTAL peaks={tot}  significant={tots}  ({100*tots/tot:.1f} %)",
              "",
              "Significant peaks are those whose |r| exceeds what a same-spectrum,",
              "random-phase signal produces at the 99th percentile — i.e. unlikely to",
              "be chance narrowband alignment. The nominal r=0.2 threshold retains many",
              "chance peaks and should not be used as the detection criterion."]
    (OUT / "significance_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {OUT/'all_peaks_flagged.csv'} and significance_summary.txt")


if __name__ == "__main__":
    main()
