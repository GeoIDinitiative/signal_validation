"""
rebuild_templates_from_dofs.py — rebuild the tilt templates from the FULL-RESOLUTION simulation DOFs
(proj_x, proj_y) instead of the coarse 11-level `tilt/` magnitude file the originals were built from.

For each (dataset, station, sim, template) it reads the simulation's full-resolution tilt DOFs for the
template window, and writes the directional, orthogonal and magnitude templates, each with its RAW
(un-filtered) and BAND-PASSED (0.001–0.01 Hz) form:
    dir_raw   = proj_x            dir_bp   = bandpass(proj_x)      (tilt-x DOF — observed 'dir' channel)
    ortho_raw = proj_y            ortho_bp = bandpass(proj_y)      (tilt-y DOF — 2nd axis for the VECTOR filter)
    mag_raw   = √(proj_x²+proj_y²) mag_bp  = bandpass(mag_raw)     (tilt magnitude — observed 'mag' channel)
The complex 2-component template used by the vector matched filter is  dir_bp + i·ortho_bp.

Output: tilt_templates_dofs/<dataset>/<station>_<sim>_<template>_dof.csv
(NOTE: does NOT touch tilt_templates_csv/ or re-run detection — by design; eyeball the new templates first.)
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

warnings.filterwarnings("ignore")
SIM = Path("/home/owen/Signal_Validation/solid_dofs/tilt")
OUT = Path("/home/owen/tilt_validation/tilt_templates_dofs")
SOS = butter(4, [0.001/0.5, 0.01/0.5], btype="bandpass", output="sos")
STATIONS = {"ingv": ["ECPN", "EEC1"], "experiment": ["EC1", "EC10", "ECIT", "ECOR", "EMAS"]}
OBS2SIM = {"ECPN": "ECPN", "EEC1": "EC1", "EC1": "EC1", "EC10": "EC10",
           "ECIT": "ECIT", "ECOR": "ECOR", "EMAS": "EMAS"}
SIMS = ["sim1", "sim2", "sim3", "sim4"]
WINDOWS = {"template1": (0, 3333), "template2": (3333, 6666),
           "template3": (6666, 10000), "template4": (0, 10000)}
bp = lambda x: sosfiltfilt(SOS, x - np.mean(x))


def loadcol(path):
    if not path.exists():
        return None
    a = np.loadtxt(path)
    return a[:, 1] if a.ndim == 2 else np.ravel(a)


def main():
    n = 0
    for ds, sts in STATIONS.items():
        (OUT/ds).mkdir(parents=True, exist_ok=True)
        for st in sts:
            node = OBS2SIM[st]
            for sim in SIMS:
                px = loadcol(SIM/sim/"proj_x"/f"{node}.txt")
                py = loadcol(SIM/sim/"proj_y"/f"{node}.txt")
                if px is None or py is None:
                    print(f"  missing DOFs for {ds}/{st} (node {node}) {sim}"); continue
                mag = np.hypot(px, py)
                for tn, (i0, i1) in WINDOWS.items():
                    dr, orr, mr = px[i0:i1], py[i0:i1], mag[i0:i1]
                    df = pd.DataFrame({"time_seconds": np.arange(len(dr), dtype=float),
                                       "dir_raw": dr, "dir_bp": bp(dr),
                                       "ortho_raw": orr, "ortho_bp": bp(orr),
                                       "mag_raw": mr, "mag_bp": bp(mr)})
                    df.to_csv(OUT/ds/f"{st}_{sim}_{tn}_dof.csv", index=False)
                    n += 1
            print(f"  rebuilt {ds}/{st} (sim node {node})")
    print(f"\n{n} DOF-based templates → {OUT}")
    print("columns: time_seconds, dir_raw, dir_bp, ortho_raw, ortho_bp, mag_raw, mag_bp")
    print("         (full-resolution proj_x, proj_y, √(px²+py²); complex template = dir_bp + i·ortho_bp)")


if __name__ == "__main__":
    main()
