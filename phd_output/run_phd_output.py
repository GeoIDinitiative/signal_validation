"""
phd_output/run_phd_output.py — three-branch evidence build + shared validation + synthesis
==========================================================================================
Builds phd_output/ as recommended by the thesis review: present the template search as three
explicit evidence branches and add the cross-facet direction-agreement synthesis.

  components  → SWCC on X and Y axes separately   (primary)      [dir, dir2]
  magnitude   → SWCC on √(x²+y²)                   (supporting)   [mag]
  vector      → 2-component |R| + oriented Re(R)   (direction)    [vec]

Each branch runs its component-specific stages (branch-routed via phd_env / the PHD_BRANCH env var)
into phd_output/<branch>/. Component-agnostic validation (filter response, contamination veto,
template-bank fitting-factor, whitened/χ² glitch tests, EEC1-vs-EC1, eruptive density) is mirrored
ONCE into phd_output/_shared/ from the existing single-pipeline outputs. Finally the synthesis ties
the branches together (cross-branch comparison + the strict co-detection / direction-agreement test).

The existing single pipeline (run_pipeline.py) is untouched: it just runs with PHD_BRANCH unset.

Run:  python3 phd_output/run_phd_output.py            # all branches + shared + synthesis
      python3 phd_output/run_phd_output.py --only components magnitude
      python3 phd_output/run_phd_output.py --skip-shared
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

BASE = Path("/home/owen/tilt_validation")
PHD = BASE / "phd_output"

# component-specific stages run per branch (in dependency order)
REAL_STAGES = [                       # components & magnitude (real components dir/dir2/mag)
    "compute_floors.py", "swcc_continuous.py", "swcc_plots_continuous.py",
    "injection_recovery.py", "network_matched_filter.py", "swcc_analysis_plots.py",
    "far_significance.py", "top_template_plots.py", "swcc_rankings.py",
    "candidate_characterization.py", "wavelet_suite.py", "sync_events.py",
    "cumulative_metrics.py", "pipeline_report.py",
]
VEC_STAGES = [                        # vector branch (complex |R|; real-trace panels not applicable)
    "compute_floors.py", "swcc_continuous.py", "injection_recovery.py",
    "network_matched_filter.py", "vector_orientation.py", "far_significance.py",
    "candidate_characterization.py", "cumulative_metrics.py", "sync_events.py",
    "pipeline_report.py",
]
BRANCH_STAGES = {"components": REAL_STAGES, "magnitude": REAL_STAGES, "vector": VEC_STAGES}

# component-agnostic legacy outputs mirrored into _shared/ (relative to BASE)
SHARED_DIRS = [
    "SWCC_comprehensive/credibility",
    "SWCC_comprehensive/contamination_audit",
    "gw_methods",
    "eec1_ec1_comparison/stats",
    "eruptive_temporal",
]


def run(script, branch):
    env = dict(os.environ, PHD_BRANCH=branch)
    tag = branch or "synthesis"
    print(f"\n{'='*72}\n[{tag}] {script}\n{'='*72}", flush=True)
    return subprocess.run([sys.executable, str(BASE / script)], env=env, cwd=str(BASE)).returncode == 0


def mirror_shared():
    dst_root = PHD / "_shared"
    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\n[_shared] mirroring component-agnostic validation\n{'='*72}", flush=True)
    for rel in SHARED_DIRS:
        src = BASE / rel
        if not src.exists():
            print(f"  · skip (missing): {rel}  — run run_pipeline.py once to generate it")
            continue
        dst = dst_root / rel
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        print(f"  · {rel}")
    (dst_root / "README.txt").write_text(
        "phd_output/_shared — component-agnostic validation shared by all three branches\n"
        "(filter response & null test, contamination veto, template-bank fitting-factor,\n"
        "whitened & chi-squared glitch tests, EEC1-vs-EC1 same-instrument stats, eruptive density).\n"
        "Mirrored from the single-pipeline outputs; identical regardless of branch.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(BRANCH_STAGES), help="run only these branches")
    ap.add_argument("--skip-shared", action="store_true")
    ap.add_argument("--skip-synthesis", action="store_true")
    a = ap.parse_args()
    branches = a.only or list(BRANCH_STAGES)

    for branch in branches:
        for s in BRANCH_STAGES[branch]:
            if not run(s, branch):
                print(f"\n✗ FAILED: {branch}/{s}"); sys.exit(1)

    if not a.skip_shared:
        mirror_shared()

    if not a.skip_synthesis:
        for s in ["phd_output/direction_synthesis.py", "phd_output/cross_branch_compare.py"]:
            if not run(s, ""):
                print(f"\n✗ FAILED: {s}"); sys.exit(1)

    print(f"\n✅ phd_output complete → {PHD}")
    print("   branches:", ", ".join(branches), "| + _shared + synthesis")


if __name__ == "__main__":
    main()
