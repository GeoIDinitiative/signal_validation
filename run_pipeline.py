#!/usr/bin/env python3
"""
run_pipeline.py  —  single entry point for the full Design-B Etna tilt SWCC workflow
===================================================================================
Runs every stage in dependency order. Each stage is a self-contained script that is
also runnable on its own; this orchestrator just runs them as subprocesses (so a stage
always reads the files produced by the previous one — several scripts load CSVs at
import time, which rules out a plain import-and-call).

Stages
  1 continuous   build_clean_bandpassed_continuous.py  raw → continuous_bandpassed/ (veto mask, ~100% coverage)
  2 floors       compute_floors.py                     per-template two-tier null floors (95th/99th)
  3 detect       swcc_continuous.py                    gap-aware accumulated detection + synchrony (veto-aware)
  4 plots        swcc_plots_continuous.py              SWCC plots (clean=blue/contaminated=grey, null thresholds)
  5 injection    injection_recovery.py                 detection sensitivity — SNR50/SNR90
  6 network      network_matched_filter.py             cross-station coherent stack sensitivity
  7 analysis     swcc_analysis_plots.py                clean-peak analysis (overview, candidates, by-station)
  8 credibility  credibility_checks_continuous.py      validation figures (filter response, de-ringing, null test)
  9 contamination contamination_audit.py               data-driven veto validation (in-band STA/LTA vs veto)
 10 report        pipeline_report.py                   consolidated one-page report (REPORT.png/.md)
 11 fitfactor     bank_fitting_factor.py               GW: template-bank coverage (fitting factor)
 12 whiten        whitened_matched_filter.py           GW: whitened matched filter vs Pearson r
 13 chisq         chisq_consistency.py                 GW: chi-squared signal-consistency (glitch test)
 14 far           far_significance.py                  GW: false-alarm-rate via time-slide background

Dependencies: 2,3 need 1 · 4 needs 1,2 · 5 needs 1,3 · 6 needs 1 · 7 needs 3 · 8,9 need 1
              10 needs 3,5,6 · 11 templates only · 12,13 need 1 · 14 needs 3.
The multi-method correlation battery (correlation_battery.py, incl. DTW) is intentionally
standalone and not part of this run.

Usage
  python3 run_pipeline.py              # run all 6 stages
  python3 run_pipeline.py --from 3     # resume from stage 3 (reuse earlier outputs)
  python3 run_pipeline.py --only 5 6   # run just those stages
  python3 run_pipeline.py --list       # list stages and exit

Note: the older cut-first scripts (build_clean_bandpassed.py, swcc_comprehensive.py,
flag_significant_peaks.py, swcc_template4.py, characterize_significant_peaks.py,
credibility_checks.py, swcc_accumulated*.py) are superseded by Design B and remain
runnable standalone; they are intentionally not part of this default run.
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

BASE = Path("/home/owen/tilt_validation")

STAGES = [
    ("continuous", "build_clean_bandpassed_continuous.py", "raw → continuous_bandpassed/ (veto, ~100% coverage)"),
    ("floors",     "compute_floors.py",                    "per-template two-tier null floors (95th/99th)"),
    ("detect",     "swcc_continuous.py",                   "gap-aware accumulated detection + synchrony"),
    ("plots",      "swcc_plots_continuous.py",             "SWCC plots: clean=blue/contaminated=grey, null thresholds"),
    ("injection",  "injection_recovery.py",                "detection sensitivity (SNR50/SNR90)"),
    ("network",    "network_matched_filter.py",            "cross-station coherent stack sensitivity (scalar dir + vector |R|)"),
    ("vectororient","vector_orientation.py",                "two-component vector filter: xlsx-orientation validation + real-data |R|/Re(R) null"),
    ("analysis",   "swcc_analysis_plots.py",               "clean-peak analysis: overview, candidates, by-station"),
    ("credibility","credibility_checks_continuous.py",      "validation figures: filter response, de-ringing, null test"),
    ("contamination","contamination_audit.py",              "data-driven veto validation (in-band STA/LTA vs veto)"),
    ("report",     "pipeline_report.py",                   "consolidated one-page report (REPORT.png/.md)"),
    ("fitfactor",  "bank_fitting_factor.py",               "GW: template-bank coverage (fitting factor)"),
    ("whiten",     "whitened_matched_filter.py",           "GW: whitened matched filter vs Pearson r"),
    ("chisq",      "chisq_consistency.py",                 "GW: chi-squared signal-consistency (glitch test)"),
    ("far",        "far_significance.py",                  "GW: false-alarm-rate via time-slide background"),
    ("templateperf","top_template_plots.py",               "per-(station,sim,template) top-40 SST + by-sim/by-template performance plots"),
    ("eec1ec1",    "eec1_ec1_comparison/make_comparison.py", "EEC1(winter) vs EC1(summer) same-instrument comparison suite (7 figs)"),
    ("eec1stats",  "eec1_ec1_comparison/statistical_tests.py", "EEC1 vs EC1 comprehensive statistical tests (distribution/spectral/thermal/detection)"),
    ("eruptivemap","eruptive_temporal/peak_density_map.py",    "clean-peak temporal density vs eruptive overlays + clustering permutation test"),
    ("rankings",   "swcc_rankings.py",                      "SWCC peak-count/ranking bar charts by station/template/sim/dataset (floor-based)"),
    ("candchar",   "candidate_characterization.py",         "detection characterisation: score/station/dataset distributions, margins, top-20"),
    ("wavelet",    "wavelet_suite.py",                      "Morlet CWT suite: template/station scalograms, power spectrum, temporal"),
    # NOTE: template_exceedance/exceedance_rates.py is intentionally NOT a stage — the per-template
    # count-enrichment null is methodologically flawed (degenerate circular-shift + unrepresentative
    # quiet null segment); see template_exceedance/SUMMARY.txt (RETRACTED). Kept standalone as a lesson.
    ("syncevents", "sync_events.py",                        "gallery of cross-station synchronous coincidences (same TOL_S=600 as the synchrony test)"),
    ("cumulative", "cumulative_metrics.py",                 "cumulative detection-rate curves per station/dataset with eruptive overlays"),
]


def run_stage(i, name, script, desc):
    print(f"\n{'='*72}\n[{i}/{len(STAGES)}] {name:10s} — {desc}\n{'='*72}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(BASE / script)], cwd=str(BASE))
    dt = time.time() - t0
    ok = r.returncode == 0
    print(f"{'✓' if ok else '✗'} stage {i} ({name}) "
          f"{'done' if ok else f'FAILED (exit {r.returncode})'} in {dt:.0f}s", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Design-B Etna tilt SWCC pipeline")
    ap.add_argument("--from", dest="start", type=int, default=1, help="resume from stage N")
    ap.add_argument("--only", type=int, nargs="+", help="run only these stage numbers")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    a = ap.parse_args()

    if a.list:
        print("Design-B pipeline stages:")
        for i, (n, s, d) in enumerate(STAGES, 1):
            print(f"  {i}  {n:10s}  {s:36s}  {d}")
        return

    selected = a.only if a.only else list(range(a.start, len(STAGES) + 1))
    print(f"Running stages: {selected}")
    t0 = time.time()
    for i, (n, s, d) in enumerate(STAGES, 1):
        if i in selected:
            if not run_stage(i, n, s, d):
                print(f"\nAborting at stage {i} ({n}).")
                sys.exit(1)
    print(f"\n✅ pipeline complete — {len(selected)} stage(s) in {time.time()-t0:.0f}s")
    print("   outputs: continuous_bandpassed/, SWCC_comprehensive/{continuous,injection,network,ingv,experiment}/")


if __name__ == "__main__":
    main()
