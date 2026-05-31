"""
swcc_template4.py  —  dedicated long-template (template4) SWCC procedure
=======================================================================
template4 is ~10,001 samples (~2.8 h), 3x longer than templates 1-3. It cannot be
hosted by the short post-excision segments (median ~83 min), so it is removed from
the main swcc_comprehensive run (TEMPLATES = T1-3) and handled here with parameters
matched to its length:

  · window length  = the template's own ~10,001 samples (so only segments that long
    can host it; experiment has none, INGV has 2 per station);
  · peak spacing   = PEAK_DISTANCE_T4 (scaled up with the longer template);
  · null floor     = computed from T4-only surrogates of the longest qualifying
    segment (a separate significance level from the T1-3 floor).

Outputs (kept separate, then merged so downstream sees everything):
  · SWCC_comprehensive/template4/t4_peaks_flagged.csv, t4_significance.txt
  · merges its rows into SWCC_comprehensive/all_peaks_flagged.csv with a
    `procedure` column ("main_T1-3" vs "T4_long") so the two are distinguishable.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

from swcc_comprehensive import (swcc_segment, load_clean, load_template,
                                template_snr, SIMS, STATIONS, COMPONENTS, THRESHOLD)
from credibility_checks import phase_randomize

SWCC = Path("/home/owen/tilt_validation/SWCC_comprehensive")
T4_DIR = SWCC / "template4"
TEMPLATE = "template4"
PEAK_DISTANCE_T4 = 3000      # ~0.3 x template length (matches T1-3's 1000/3333 ratio)
N_SURR = 200
SEED = 2


def t4_templates(dataset, station):
    """The four template4 waveforms (one per sim) for this station."""
    out = []
    for sim in SIMS:
        t = load_template(dataset, station, sim, TEMPLATE)
        if t is not None:
            out.append((sim, t))
    return out


def null_floor_t4(dataset, station, comp, clean, tpls, seed=SEED, n=N_SURR):
    """99th-pct of max|r| from T4-only phase-randomised surrogates of the longest host."""
    M = max(len(t) for _, t in tpls)
    seglens = clean.groupby("segment_id").size()
    hosts = seglens[seglens >= M]
    if hosts.empty:
        return None
    host = clean[clean.segment_id == hosts.idxmax()]["bandpassed"].to_numpy(float)
    rng = np.random.default_rng(seed)
    maxr = []
    for _ in range(n):
        xs = phase_randomize(host, rng)
        best = 0.0
        for _, t in tpls:
            if len(t) > len(xs):
                continue
            r = np.abs(swcc_segment(t, xs))
            if r.size:
                pk, _ = find_peaks(r, height=THRESHOLD, distance=PEAK_DISTANCE_T4)
                if len(pk):
                    best = max(best, r[pk].max())
        maxr.append(best)
    return float(np.percentile(maxr, 99))


def main():
    T4_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    floors, notes = {}, []
    for dataset, stations in STATIONS.items():
        for station in stations:
            for comp in COMPONENTS:
                clean = load_clean(dataset, station, comp)
                if clean is None:
                    continue
                tpls = t4_templates(dataset, station)
                if not tpls:
                    continue
                M = max(len(t) for _, t in tpls)
                seglens = clean.groupby("segment_id").size()
                hosts = list(seglens[seglens >= M].index)
                if not hosts:
                    notes.append(f"{dataset}/{station}/{comp}: 0 segments >= {M} samples (T4 not evaluable)")
                    continue
                floor = floors.get(dataset)
                if floor is None:
                    floor = null_floor_t4(dataset, station, comp, clean, tpls)
                    floors[dataset] = floor
                for sim, tpl in tpls:
                    for sid in hosts:
                        seg = clean[clean.segment_id == sid]
                        sig = seg["bandpassed"].to_numpy(float)
                        if len(sig) < len(tpl):
                            continue
                        seg_dt = seg["datetime"].to_numpy()
                        edge = (seg["edge"].to_numpy(bool) if "edge" in seg.columns
                                else np.zeros(len(sig), bool))
                        r = swcc_segment(tpl, sig)
                        peaks, _ = find_peaks(np.abs(r), height=THRESHOLD,
                                              distance=PEAK_DISTANCE_T4)
                        for pk in peaks:
                            rows.append({
                                "dataset": dataset, "station": station, "component": comp,
                                "sim": sim, "template": TEMPLATE, "segment_id": int(sid),
                                "peak_time": pd.Timestamp(seg_dt[pk]),
                                "r": float(r[pk]), "abs_r": float(abs(r[pk])),
                                "snr_db": float(template_snr(sig, int(pk), len(tpl))),
                                "in_edge": bool(edge[pk:min(pk+len(tpl), len(edge))].any()),
                                "null_floor": floor, "procedure": "T4_long",
                            })

    t4 = pd.DataFrame(rows)
    if len(t4):
        t4["significant"] = t4["abs_r"] > t4["null_floor"]
    t4.to_csv(T4_DIR / "t4_peaks_flagged.csv", index=False)

    # ── significance report ──────────────────────────────────────────────────
    lines = ["TEMPLATE-4 (long, ~10,001 samples) SEPARATE SWCC", "=" * 50,
             f"peak spacing = {PEAK_DISTANCE_T4} samples | surrogates = {N_SURR}", ""]
    for ds, fl in floors.items():
        lines.append(f"{ds}: T4 null floor (99th pct) = {fl:.3f}" if fl else f"{ds}: no host segment")
    lines.append("")
    if len(t4):
        for (ds, st, cp), g in t4.groupby(["dataset", "station", "component"]):
            lines.append(f"  {ds}/{st}/{cp}: {len(g)} peaks, {int(g.significant.sum())} significant "
                         f"(max|r|={g.abs_r.max():.3f})")
    else:
        lines.append("  No T4 peaks (no segment long enough on any station).")
    lines += [""] + notes
    (T4_DIR / "t4_significance.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    # ── merge into the combined flagged peaks (tagged by procedure) ──────────
    main_path = SWCC / "all_peaks_flagged.csv"
    if main_path.exists():
        base = pd.read_csv(main_path)
        if "procedure" not in base.columns:
            base["procedure"] = "main_T1-3"
        base = base[base["procedure"] != "T4_long"]            # avoid double-append on re-run
        combined = pd.concat([base, t4], ignore_index=True) if len(t4) else base
        combined.to_csv(main_path, index=False)
        print(f"\nmerged {len(t4)} T4 rows into {main_path.name} "
              f"(total {len(combined)}, procedures: {sorted(combined.procedure.unique())})")


if __name__ == "__main__":
    main()
