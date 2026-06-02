"""
compute_floors.py  —  two-tier null floors per (dataset, station, component, sim, template)
==========================================================================================
For every template, derive the detection/significance thresholds from the SAME
phase-randomised surrogate test used for the conclusions:
    detection floor   = 95th percentile of surrogate-max |r|   (replaces the old 0.2)
    significance floor = 99th percentile of surrogate-max |r|   (replaces the old 0.5)
Both are data-driven (no arbitrary constants). Host = the longest clean (un-vetoed)
stretch of that station's continuous signal.

Output: SWCC_comprehensive/continuous/template_floors.csv
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from swcc_gapaware import swcc_gapaware
from swcc_comprehensive import load_template, SIMS
from credibility_checks import phase_randomize
from swcc_vector import swcc_score, surrogate, load_cont_complex   # vector |R| + complex host/null
import phd_env                                                     # branch-aware OUT / components

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
CONT = BASE / "continuous_bandpassed"
OUT  = phd_env.out(BASE / "SWCC_comprehensive" / "continuous")
STATIONS = {"ingv": ["ECPN", "EEC1"], "experiment": ["EC1", "EC10", "ECIT", "ECOR", "EMAS"]}
COMPONENTS = phd_env.components(["dir", "mag", "vec"])
TEMPLATES = ["template1", "template2", "template3", "template4"]
N_SURR, HOST_MAX = 300, 30000


def clean_host(d):
    """Longest run of consecutive un-vetoed, finite samples (capped at HOST_MAX)."""
    ok = (~d["veto"].to_numpy(bool)) & np.isfinite(d["bandpassed"].to_numpy())
    best_a = best_len = cur_a = cur = 0
    for i, v in enumerate(ok):
        if v:
            if cur == 0:
                cur_a = i
            cur += 1
            if cur > best_len:
                best_len, best_a = cur, cur_a
        else:
            cur = 0
    return d["bandpassed"].to_numpy()[best_a:best_a+min(best_len, HOST_MAX)]


def main():
    rows = []
    for ds, sts in STATIONS.items():
        for st in sts:
            for comp in COMPONENTS:
                if comp == "vec":                         # complex host: dir (c1) + i·dir2 (c2)
                    d = load_cont_complex(ds, st)
                    if d is None:
                        continue
                else:
                    f = CONT / ds / f"{st}_{comp}_0p001-0p01Hz_cont_bp.feather"
                    if not f.exists():
                        continue
                    d = pd.read_feather(f)
                host = clean_host(d)
                if len(host) < 8000:
                    continue
                rng = np.random.default_rng(7)
                for sim in SIMS:
                    for tname in TEMPLATES:
                        tpl = load_template(ds, st, sim, tname, comp)
                        if tpl is None or len(tpl) > len(host):
                            continue
                        maxima = []
                        for _ in range(N_SURR):
                            r = swcc_score(tpl, surrogate(host, rng), min_valid_frac=0.8)
                            if r.size and np.isfinite(r).any():
                                maxima.append(np.nanmax(r))
                        if not maxima:
                            continue
                        d95, d99 = np.percentile(maxima, [95, 99])
                        rows.append({"dataset": ds, "station": st, "component": comp,
                                     "sim": sim, "template": tname,
                                     "floor_detect": round(float(d95), 4),
                                     "floor_signif": round(float(d99), 4)})
                print(f"   {ds}/{st}/{comp}")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "template_floors.csv", index=False)
    print(f"\nwrote {len(df)} template floors → {OUT/'template_floors.csv'}")
    print(df.groupby("dataset")[["floor_detect", "floor_signif"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
