"""
phd_env.py — branch-aware output routing + component selection for the phd_output/ build
=========================================================================================
Reads the PHD_BRANCH environment variable. When it is UNSET, every helper returns the LEGACY
value, so the existing single pipeline (run_pipeline.py) behaves byte-identically. When set to one
of {components, magnitude, vector, _shared}, stage outputs are redirected under phd_output/<branch>/
and the active component / track set becomes the branch's.

Branch → components (see the approved plan):
  components → [dir, dir2]   X and Y axes, correlated separately (primary evidence)
  magnitude → [mag]          √(x²+y²)                            (supporting)
  vector    → [vec]          2-component |R| / oriented Re(R)    (direction)
  _shared   → component-agnostic stages run once; they READ the primary (components) branch's
              detections via dets_dir().

Stages opt in with three calls, each carrying their legacy default:
    OUT        = phd_env.out(BASE / "SWCC_comprehensive" / "continuous")
    COMPONENTS = phd_env.components(["dir", "mag", "vec"])
    TRACKS     = phd_env.tracks({"scalar": ["dir", "mag"], "vec": ["vec"]})
and read cross-stage detection CSVs from phd_env.dets_dir().
"""

import os
from pathlib import Path

BASE = Path("/home/owen/tilt_validation")
PHD_ROOT = BASE / "phd_output"
BRANCH = os.environ.get("PHD_BRANCH") or None

BRANCH_COMPONENTS = {
    "components": ["dir", "dir2"],
    "magnitude": ["mag"],
    "vector":    ["vec"],
}
PRIMARY_BRANCH = "components"          # whose detections the _shared agnostic stages consume


def active():
    return BRANCH is not None


def components(default):
    """Active branch's component list, or `default` (legacy)."""
    if BRANCH in BRANCH_COMPONENTS:
        return list(BRANCH_COMPONENTS[BRANCH])
    return default


def tracks(default):
    """Synchrony / recovery 'tracks'. Branch mode = one track named after the branch; else legacy."""
    if BRANCH in BRANCH_COMPONENTS:
        return {BRANCH: list(BRANCH_COMPONENTS[BRANCH])}
    return default


def comp_track_map(default_tracks):
    """component → track-name, derived from tracks(default_tracks)."""
    return {c: t for t, cs in tracks(default_tracks).items() for c in cs}


def out(legacy):
    """Redirect a stage's legacy output dir under phd_output/<branch>/<leaf> (leaf = path relative
    to BASE). Legacy mode returns `legacy` unchanged."""
    legacy = Path(legacy)
    if BRANCH is None:
        return legacy
    try:
        leaf = legacy.relative_to(BASE)
    except ValueError:
        leaf = Path(legacy.name)
    return PHD_ROOT / BRANCH / leaf


def dets_dir():
    """Directory holding all_detections_continuous.csv / detect_counts.csv to READ.
    Branch mode → own branch; _shared → primary branch; legacy → SWCC_comprehensive/continuous."""
    if BRANCH in BRANCH_COMPONENTS:
        return PHD_ROOT / BRANCH / "SWCC_comprehensive" / "continuous"
    if BRANCH == "_shared":
        return PHD_ROOT / PRIMARY_BRANCH / "SWCC_comprehensive" / "continuous"
    return BASE / "SWCC_comprehensive" / "continuous"
