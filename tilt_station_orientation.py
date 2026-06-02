"""
tilt_station_orientation.py  —  per-station instrument→geographic rotation for the vector filter
================================================================================================
Parses tilt_station_orientation/Tilt_Conversion_AxisOrientation.xlsx and exposes, per station, the
2×2 rotation R_st that maps the recorded instrument axes (x, y) onto geographic (East, North):

        [E]            [ sinθx   sinθy ] [x]
        [N]  =  g · [ cosθx   cosθy ] [y] ,    θy = θx − 90°,   g = −orientation

where θx is the compass azimuth (clockwise from North) of the Tilt-X axis and `orientation` (±1) is
the sheet's handedness flag.  This global-sign convention reproduces the sheet's worked "Tilt E /
Tilt N" columns (unit input x=y=1) exactly — validated in `_self_test()` to < 1e-3 for every listed
station (PDN, CDV, …, EC1, EMAS).

Coverage for our analysis stations:
  · INGV  ECPN, EEC1 — recorded directly in geographic east/north → rotation = IDENTITY.
  · expt  EC1 (θx=122°), EC10 (θx=122°), EMAS (θx=273°) — covered by the sheet.
  · expt  ECIT, ECOR — NOT in the sheet → get_rotation returns None (oriented Re(R) undefined;
    these stations fall back to the rotation-invariant |R|, which needs no orientation).

Used by the two-component (vector) matched filter (swcc_vector.py) for the *oriented* statistic
Re(R) and for the orientation-consistency validation (recovered arg(R) vs the sheet azimuth).
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("/home/owen/tilt_validation")
XLSX = BASE / "tilt_station_orientation" / "Tilt_Conversion_AxisOrientation.xlsx"

# our observed-station code  →  sheet code  (PDN=EPDN, ECP=ECPN per the user's note)
NAME_MAP = {"ECPN": "ECP", "EPDN": "PDN", "EC1": "EC1", "EC10": "EC10", "EMAS": "EMAS"}
# INGV stations are recorded in geographic east/north already → identity rotation, no sheet needed
GEOGRAPHIC = {"ECPN", "EEC1"}


def _parse_az_NdegE(s):
    """'N296°E' / 'N224.5°E' → 296.0 / 224.5 (compass degrees, clockwise from North)."""
    m = re.search(r"N\s*([\d.]+)\s*°?\s*E", str(s))
    return float(m.group(1)) if m else np.nan


def _load_table():
    """Return {sheet_station: (orientation:±1, az_X_deg)} merged from both sheets,
    plus the Conversion sheet's worked (TiltE, TiltN) for the unit-input validation gate."""
    az, gate = {}, {}
    conv = pd.read_excel(XLSX, sheet_name="Conversion", header=None)
    for _, r in conv.iterrows():
        st = str(r[0]).strip()
        try:
            orient = int(r[1]); azx = float(r[2])
        except (ValueError, TypeError):
            continue                                   # header / blank rows
        az[st] = (1 if orient >= 0 else -1, azx)
        gate[st] = (float(r[7]), float(r[8]))          # Tilt E, Tilt N for x=y=1
    stn = pd.read_excel(XLSX, sheet_name="Station", header=None)
    for _, r in stn.iterrows():
        st = str(r[0]).strip()
        azx = _parse_az_NdegE(r[1])
        if np.isfinite(azx) and st not in az:          # Conversion sheet wins where both exist
            try:
                orient = int(r[3])
            except (ValueError, TypeError):
                orient = 1
            az[st] = (1 if orient >= 0 else -1, azx)
    return az, gate


_AZ, _GATE = _load_table()


def _rotation_from_az(orient, az_x_deg):
    """2×2 instrument(x,y)→geographic(E,N) matrix (see module docstring)."""
    thx = np.radians(az_x_deg)
    thy = np.radians(az_x_deg - 90.0)
    g = -orient
    return g * np.array([[np.sin(thx), np.sin(thy)],
                         [np.cos(thx), np.cos(thy)]])


def get_rotation(station):
    """Return the 2×2 (x,y)→(E,N) rotation for `station`, or None if orientation is unknown.
    INGV stations (already geographic) return the identity."""
    if station in GEOGRAPHIC:
        return np.eye(2)
    code = NAME_MAP.get(station, station)
    if code not in _AZ:
        return None
    return _rotation_from_az(*_AZ[code])


def axis_azimuth(station):
    """Compass azimuth (deg, cw from North) of the Tilt-X axis, or None/0 for geographic."""
    if station in GEOGRAPHIC:
        return 0.0
    code = NAME_MAP.get(station, station)
    return _AZ[code][1] if code in _AZ else None


def covered(station):
    return get_rotation(station) is not None


def _self_test():
    print("tilt_station_orientation — reproduction gate (Tilt E/N for unit input x=y=1)")
    worst = 0.0
    for code, (tE, tN) in _GATE.items():
        R = _rotation_from_az(*_AZ[code])
        e, n = R @ np.array([1.0, 1.0])
        d = max(abs(e - tE), abs(n - tN)); worst = max(worst, d)
        print(f"  {code:5s}  az_X={_AZ[code][1]:5.1f}°  got=({e:+.4f},{n:+.4f})  "
              f"sheet=({tE:+.4f},{tN:+.4f})  Δ={d:.1e}")
    print(f"\n  max|Δ| = {worst:.2e}   {'PASS' if worst < 1e-3 else 'FAIL'}")
    print("\nCoverage for analysis stations:")
    for st in ["ECPN", "EEC1", "EC1", "EC10", "ECIT", "ECOR", "EMAS"]:
        R = get_rotation(st)
        tag = ("identity (geographic)" if st in GEOGRAPHIC
               else f"az_X={axis_azimuth(st)}°" if R is not None else "NO ORIENTATION → |R| only")
        print(f"  {st:5s}: {'covered' if R is not None else 'uncovered':9s}  {tag}")
    return worst < 1e-3


if __name__ == "__main__":
    _self_test()
