"""
quantitative_assessment.py — floor-based performance numbers from the new pipeline, per component.
Computes, for every dataset × station × component {X=dir, Y=dir2, magnitude=mag, vector=|R|} ×
configuration(sim) × template, the number of matched-filter peaks above the data-driven detection
floor (n_detect) and significance floor (n_signif), plus max/mean correlation. Aggregates by
template, by configuration and by station. These replace the fixed-threshold Tables 8–11.

Output → phd_output/thesis_figures/quant/  (quant_full.csv + by_template/configuration/station + summary.txt)
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

sys.path.insert(0, "/home/owen/tilt_validation")        # so the pipeline modules import when run by path
from swcc_comprehensive import load_template, SIMS, STATIONS, template_snr
from swcc_continuous import load_cont, to_grid
from swcc_vector import swcc_score

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
OUT = BASE / "phd_output" / "thesis_figures" / "quant"
OUT.mkdir(parents=True, exist_ok=True)
TEMPLATES = ["template1", "template2", "template3", "template4"]
COMPS = ["dir", "dir2", "mag", "vec"]
CLABEL = {"dir": "X (tilt-x)", "dir2": "Y (tilt-y)", "mag": "magnitude", "vec": "vector |R|"}
CFG = {f"sim{i}": f"Configuration {i}" for i in range(1, 5)}
TPL = {f"template{i}": f"Template {i}" for i in range(1, 5)}
DSL = {"ingv": "INGV (eruptive)", "experiment": "IMPROVE (quiescent)"}
PEAK_DIST = 1000


def floor_lookup():
    """(ds,st,comp,sim,tn) -> (floor_detect, floor_signif), merging legacy (dir/mag/vec) + dir2."""
    fl = {}
    lg = pd.read_csv(BASE / "SWCC_comprehensive" / "continuous" / "template_floors.csv")
    cb = pd.read_csv(BASE / "phd_output" / "components" / "SWCC_comprehensive" / "continuous" / "template_floors.csv")
    cb = cb[cb.component == "dir2"]
    for d in (lg, cb):
        for r in d.itertuples():
            fl[(r.dataset, r.station, r.component, r.sim, r.template)] = (r.floor_detect, r.floor_signif)
    return fl


def main():
    FL = floor_lookup()
    rows, peak_rows = [], []
    for ds, sts in STATIONS.items():
        for st in sts:
            for comp in COMPS:
                d = load_cont(ds, st, comp)
                if d is None:
                    continue
                gt, gx, gv = to_grid(d)
                valid = np.isfinite(gx) if not np.iscomplexobj(gx) else (np.isfinite(gx.real) & np.isfinite(gx.imag))
                sig_snr = gx if not np.iscomplexobj(gx) else np.abs(gx)   # real signal for the SNR estimate
                for sim in SIMS:
                    for tn in TEMPLATES:
                        tpl = load_template(ds, st, sim, tn, comp)
                        if tpl is None:
                            continue
                        fkey = FL.get((ds, st, comp, sim, tn))
                        if fkey is None:
                            continue
                        fdet, fsig = fkey
                        score = swcc_score(tpl, gx, min_valid_frac=0.8)
                        if score.size == 0:
                            continue
                        s = np.where(np.isfinite(score), score, 0.0)
                        s[~valid[:len(s)]] = 0.0
                        s[gv[:len(s)]] = 0.0
                        pk, _ = find_peaks(s, height=fdet, distance=PEAK_DIST)
                        vals = score[pk]
                        n_det = int(len(pk))
                        n_sig = int(np.sum(vals >= fsig))
                        M = len(tpl)
                        for k in pk:                              # per-peak correlation + SNR
                            snr_db = template_snr(sig_snr, int(k), M)
                            if not np.isfinite(snr_db):
                                continue
                            peak_rows.append({"dataset": ds, "station": st, "component": comp,
                                              "configuration": sim, "template": tn,
                                              "r": round(float(score[k]), 4),
                                              "snr_db": round(float(snr_db), 3),
                                              "snr_lin": round(float(10 ** (snr_db / 20.0)), 4)})
                        rows.append({
                            "dataset": ds, "station": st, "component": comp,
                            "configuration": sim, "template": tn,
                            "floor_detect": round(float(fdet), 4), "floor_signif": round(float(fsig), 4),
                            "n_detect": n_det, "n_signif": n_sig,
                            "max_r": round(float(np.nanmax(vals)), 4) if n_det else np.nan,
                            "mean_r": round(float(np.nanmean(vals)), 4) if n_det else np.nan,
                        })
                print(f"  {ds}/{st}/{comp}: done")
    full = pd.DataFrame(rows)
    full.to_csv(OUT / "quant_full.csv", index=False)
    pd.DataFrame(peak_rows).to_csv(OUT / "quant_peaks.csv", index=False)   # per-peak r + SNR (for distributions)
    print(f"  saved {len(peak_rows)} per-peak rows → quant_peaks.csv")

    # ── aggregations: by template / configuration / station, per component × dataset ──
    def agg(key):
        g = (full.groupby([key, "component", "dataset"])
             .agg(n_detect=("n_detect", "sum"), n_signif=("n_signif", "sum"),
                  mean_r=("mean_r", "mean"), max_r=("max_r", "max"),
                  floor_detect=("floor_detect", "mean")).reset_index())
        return g

    for key, name in [("template", "by_template"), ("configuration", "by_configuration"),
                      ("station", "by_station")]:
        g = agg(key); g.to_csv(OUT / f"quant_{name}.csv", index=False)
        # readable pivot: n_detect, rows=key, cols=component, for each dataset
        for ds in ("ingv", "experiment"):
            piv = (g[g.dataset == ds].pivot(index=key, columns="component", values="n_detect")
                   .reindex(columns=COMPS))
            piv.to_csv(OUT / f"pivot_{name}_{ds}_ndetect.csv")

    # ── summary ──
    L = ["QUANTITATIVE PERFORMANCE — floor-based detections, per component", "=" * 64, "",
         "Components: X=tilt-x, Y=tilt-y, magnitude=√(x²+y²), vector=|R| (2-component).",
         "n_detect = peaks above the data-driven 95th-pct detection floor; n_signif = above 99th-pct.", ""]
    for key, name in [("template", "BY TEMPLATE"), ("configuration", "BY CONFIGURATION"),
                      ("station", "BY STATION")]:
        g = agg(key)
        L.append(f"### {name} — total n_detect (both datasets), per component")
        piv = g.groupby([key, "component"]).n_detect.sum().unstack().reindex(columns=COMPS)
        piv.columns = [CLABEL[c] for c in piv.columns]
        L.append(piv.to_string()); L.append("")
    L.append(f"GRAND TOTAL above-floor detections: {int(full.n_detect.sum())} "
             f"(significance-floor: {int(full.n_signif.sum())})")
    (OUT / "summary.txt").write_text("\n".join(L))
    print("\n" + "\n".join(L))
    print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
