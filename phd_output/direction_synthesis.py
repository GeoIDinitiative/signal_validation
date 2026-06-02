"""
phd_output/synthesis/direction_synthesis.py — cross-facet co-detection with direction agreement
================================================================================================
The strongest single piece of evidence the thesis can offer for "the modelled magma-driven tilt is
present in the data" is a JOINT match: at the same time a station's

    X correlation is high   AND   Y correlation is high   AND   magnitude correlation is high
    AND the observed tilt-vector DIRECTION matches the modelled template direction.

For every detected candidate window (union of the three branches' significant detections) we pick the
template that best matches the full 2-D tilt (max |R|), then at that template/lag measure:
    rX   = |r|(proj_x , X)        rY = |r|(proj_y , Y)        rMag = |r|(√(px²+py²) , magnitude)
    rVec = |R|                    dAz = |arg(m·R)|  (deg)  — the angle between the observed and
                                   modelled tilt vectors, m = instrument→geographic rotation
                                   (exact for EC1/EC10/EMAS + INGV; ECIT/ECOR have no orientation → dAz = —).
An event PASSES at threshold Δ iff rX, rY and rMag each exceed their per-template detection floor AND
dAz ≤ Δ. We report survivors at Δ = 10/20/30/45°, against a null built from random clean windows.

Outputs (phd_output/synthesis/): codetection.csv, codetection_summary.txt, direction_agreement.png
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/owen/tilt_validation")
sys.path.insert(0, str(BASE))
from swcc_vector import swcc_vector_gapaware, load_cont_complex          # noqa: E402
from swcc_gapaware import swcc_gapaware                                  # noqa: E402
from swcc_comprehensive import load_template, SIMS                       # noqa: E402
from tilt_station_orientation import get_rotation                       # noqa: E402

warnings.filterwarnings("ignore")
PHD = BASE / "phd_output"
OUT = PHD / "synthesis"
STATIONS = {"ingv": ["ECPN", "EEC1"], "experiment": ["EC1", "EC10", "ECIT", "ECOR", "EMAS"]}
TEMPLATES = ["template1", "template2", "template3"]
DELTAS = [10, 20, 30, 45]
TOL = 400                 # lag search half-window (samples) around the detection
N_NULL = 80               # random clean windows per station for the null
MAXC = 120                # cap candidates per station (sampled if more)


def complex_m(st):
    R = get_rotation(st)
    return None if R is None else R[0, 0] + 1j * R[1, 0]


def load_floors():
    """Per-(dataset,station,component,sim,template) detection floor from each branch."""
    fl = {}
    for branch in ("components", "magnitude"):
        f = PHD / branch / "SWCC_comprehensive" / "continuous" / "template_floors.csv"
        if not f.exists():
            continue
        for r in pd.read_csv(f).itertuples():
            fl[(r.dataset, r.station, r.component, r.sim, r.template)] = float(r.floor_detect)
    return fl


def load_candidates(ds, st):
    """Significant detection times for this station across all branches (deduped)."""
    times = []
    for branch in ("components", "magnitude", "vector"):
        f = PHD / branch / "SWCC_comprehensive" / "continuous" / "all_detections_continuous.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, parse_dates=["peak_time"])
        d = d[(d.dataset == ds) & (d.station == st) & d.significant]
        times += list(d.peak_time)
    if not times:
        return []
    t = pd.Series(sorted(pd.to_datetime(times))).drop_duplicates()
    keep = []
    for x in t:                                    # dedupe within ~TOL seconds
        if not keep or (x - keep[-1]).total_seconds() > TOL:
            keep.append(x)
    return keep


def station_frame(ds, st):
    """Merged per-datetime frame: X (dir bp), Y (dir2 bp), MAG (mag bp), datetime. None if absent."""
    cz = load_cont_complex(ds, st)                 # bandpassed = dir + i·dir2
    if cz is None:
        return None
    fmag = BASE / "continuous_bandpassed" / ds / f"{st}_mag_0p001-0p01Hz_cont_bp.feather"
    if not fmag.exists():
        return None
    dm = pd.read_feather(fmag); dm["datetime"] = pd.to_datetime(dm["datetime"])
    m = cz.merge(dm[["datetime", "bandpassed"]], on="datetime", how="inner", suffixes=("", "_m"))
    z = m["bandpassed"].to_numpy()
    return {"dt": m["datetime"].to_numpy(), "X": z.real, "Y": z.imag,
            "Z": z, "MAG": m["bandpassed_m"].to_numpy(float)}


def evaluate(fr, i0, ds, st, m, floors):
    """At window start i0, find the best-|R| template and report rX, rY, rMag, rVec, dAz + floors."""
    M = 3333
    lo, hi = max(0, i0 - TOL), i0 + M + TOL
    Zw, Xw, Yw, Mw = fr["Z"][lo:hi], fr["X"][lo:hi], fr["Y"][lo:hi], fr["MAG"][lo:hi]
    if len(Zw) < M + 1 or not np.isfinite(Zw).all():
        return None
    best = None
    for sim in SIMS:
        for tn in TEMPLATES:
            zt = load_template(ds, st, sim, tn, "vec")
            if zt is None:
                continue
            R = swcc_vector_gapaware(zt[:M], Zw, min_valid_frac=0.8)
            if R.size == 0 or not np.isfinite(R).any():
                continue
            k = int(np.nanargmax(np.abs(R)))
            if best is None or np.abs(R[k]) > best[0]:
                best = (float(np.abs(R[k])), sim, tn, k, R[k])
    if best is None:
        return None
    rVec, sim, tn, k, Rk = best
    tx = load_template(ds, st, sim, tn, "dir")[:M]
    ty = load_template(ds, st, sim, tn, "dir2")[:M]
    tm = load_template(ds, st, sim, tn, "mag")[:M]
    rX = float(np.abs(swcc_gapaware(tx, Xw, min_valid_frac=0.8)[k]))
    rY = float(np.abs(swcc_gapaware(ty, Yw, min_valid_frac=0.8)[k]))
    rM = float(np.abs(swcc_gapaware(tm, Mw, min_valid_frac=0.8)[k]))
    dAz = float(abs(np.degrees(np.angle(m * Rk)))) if m is not None else np.nan
    fX = floors.get((ds, st, "dir", sim, tn), np.inf)
    fY = floors.get((ds, st, "dir2", sim, tn), np.inf)
    fM = floors.get((ds, st, "mag", sim, tn), np.inf)
    return dict(dataset=ds, station=st, sim=sim, template=tn, rX=rX, rY=rY, rMag=rM, rVec=rVec,
                dAz=dAz, passX=rX > fX, passY=rY > fY, passMag=rM > fM)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    floors = load_floors()
    if not floors:
        print("No branch template_floors found — run the components & magnitude branches first.")
        return
    rng = np.random.default_rng(5)
    rows, null_rows = [], []

    for ds, sts in STATIONS.items():
        for st in sts:
            fr = station_frame(ds, st)
            if fr is None:
                continue
            m = complex_m(st)
            dt = pd.to_datetime(fr["dt"])
            # candidate windows
            cand = load_candidates(ds, st)
            if len(cand) > MAXC:
                cand = list(pd.Series(cand).sample(MAXC, random_state=1))
            for t0 in cand:
                i0 = int(np.searchsorted(dt.values, np.datetime64(t0)))
                ev = evaluate(fr, i0, ds, st, m, floors)
                if ev:
                    ev["peak_time"] = pd.Timestamp(t0); rows.append(ev)
            # null windows (random, clean)
            n = len(fr["X"])
            for _ in range(N_NULL):
                i0 = int(rng.integers(0, max(1, n - 4200)))
                ev = evaluate(fr, i0, ds, st, m, floors)
                if ev:
                    null_rows.append(ev)
            print(f"  {ds}/{st}: candidates={len(cand)} covered={m is not None}")

    cd = pd.DataFrame(rows)
    nd = pd.DataFrame(null_rows)
    cd.to_csv(OUT / "codetection.csv", index=False)
    nd.to_csv(OUT / "codetection_null.csv", index=False)

    # ── AND-gate survivor counts vs Δ ─────────────────────────────────────────
    L = ["CROSS-FACET CO-DETECTION — X & Y & magnitude all above floor AND direction agreement",
         "=" * 80, "",
         "An event passes iff rX>floorX AND rY>floorY AND rMag>floorMag AND |Δazimuth| ≤ Δ.",
         "Direction (Δazimuth) is exact for EC1/EC10/EMAS + INGV; ECIT/ECOR (no orientation) can",
         "satisfy the amplitude AND-gate but have no azimuth and never count toward the angle test.", ""]

    def survivors(df, delta):
        if df.empty:
            return 0
        amp = df.passX & df.passY & df.passMag
        ang = df.dAz.notna() & (df.dAz <= delta)
        return int((amp & ang).sum())

    n_amp = int((cd.passX & cd.passY & cd.passMag).sum()) if not cd.empty else 0
    L.append(f"candidate events evaluated: {len(cd)}   (amplitude AND-gate X&Y&mag passed: {n_amp})")
    L.append("")
    L.append("  Δ (deg) | co-detections (observed) | per-window null rate")
    for d in DELTAS:
        obs = survivors(cd, d)
        nullrate = (survivors(nd, d) / len(nd)) if len(nd) else float("nan")
        L.append(f"   {d:4d}   |        {obs:4d}            |   {nullrate:.4f}")
    L += ["",
          "Expectation under the established null: ~0 surviving co-detections (a chance amplitude",
          "match in all three facets that ALSO points the right way is very unlikely).",
          f"Orientation-covered stations: EC1, EC10, EMAS + INGV ECPN/EEC1."]
    (OUT / "codetection_summary.txt").write_text("\n".join(L))
    print("\n" + "\n".join(L))

    # ── figure: dAz vs joint-|r|, candidates vs null ──────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    if not cd.empty:
        joint = (cd.rX * cd.rY * cd.rMag) ** (1 / 3)
        amp = cd.passX & cd.passY & cd.passMag
        a0 = ax[0]
        a0.scatter(cd.dAz, joint, c=np.where(amp, "#dc2626", "#9ca3af"), s=18, alpha=0.7)
        for d in DELTAS:
            a0.axvline(d, ls=":", c="gray", alpha=0.5)
        a0.set(xlabel="direction disagreement |Δazimuth| (deg)",
               ylabel="joint correlation (rX·rY·rMag)$^{1/3}$",
               title="Candidate events — red = X&Y&magnitude all above floor")
        a0.grid(alpha=0.3)
    if not nd.empty and nd.dAz.notna().any():
        a1 = ax[1]
        a1.hist(nd.dAz.dropna(), bins=np.arange(0, 181, 10), color="#2563eb", alpha=0.6,
                density=True, label="null (random clean windows)")
        if not cd.empty and cd.dAz.notna().any():
            a1.hist(cd.dAz.dropna(), bins=np.arange(0, 181, 10), color="#dc2626", alpha=0.5,
                    density=True, label="candidate events")
        a1.set(xlabel="direction disagreement |Δazimuth| (deg)", ylabel="density",
               title="Δazimuth distribution (uniform ⇒ no directional consistency)")
        a1.legend(fontsize=9); a1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "direction_agreement.png", dpi=300); plt.close(fig)
    print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
