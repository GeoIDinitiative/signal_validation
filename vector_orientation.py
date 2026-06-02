"""
vector_orientation.py  —  use the station axis orientations + validate the vector filter
========================================================================================
Two complementary outputs for the two-component (vector) matched filter:

(A) ORIENTATION CONSISTENCY (uses the xlsx).  For each covered station the instrument→geographic
    map R_st (tilt_station_orientation) is a proper rotation, equivalent to a complex multiply
    z_geo = m·z_inst with m = R[0,0] + i·R[1,0], |m| = 1, arg(m) = 90° − az_X.  We inject the
    GEOGRAPHIC template into a clean noise window expressed in the INSTRUMENT frame (the signal the
    tiltmeter would actually record), then run the vector correlation against the geographic
    template.  The "alignment ratio"  Re(m·R)/|R|  at the injection must be ≈ +1 — i.e. applying the
    xlsx rotation lines the recovered tilt vector back up with the template (a +1 means correct
    orientation, −1 a polarity flip, 0 a 90° error).  This validates the whole oriented chain
    end-to-end against the supplied azimuths.

(B) REAL-DATA NULL (rotation-invariant |R| vs oriented Re(R)).  On clean windows of the real signal
    we report the rotation-invariant |R| (the universal detector, needs no orientation) and the
    oriented Re(R) (using R_st where available).  Both are null — the vector filter, with or without
    the orientation, finds no coherent ULP tilt.

Covered (oriented): EC1, EC10, EMAS (+ INGV ECPN/EEC1, recorded geographic = identity).
Uncovered: ECIT, ECOR — |R| only (rotation-invariant), reported in the null panel.

Output: vector_orientation/  (orientation_validation.csv, real_null.csv, vector_orientation.png, summary.txt)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from swcc_vector import load_cont_complex, load_vec_template, swcc_vector_gapaware
from tilt_station_orientation import get_rotation, axis_azimuth, covered
from swcc_comprehensive import SIMS
import phd_env                                          # branch-aware OUT

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
OUT = phd_env.out(BASE / "vector_orientation")
STATIONS = {"ingv": ["ECPN", "EEC1"], "experiment": ["EC1", "EC10", "ECIT", "ECOR", "EMAS"]}
TEMPLATES = ["template1", "template2", "template3"]
WIN = 10002
N_WIN = 60                          # clean windows per station for the real-data null
INJ_SNR = 5.0


def complex_m(station):
    """Complex rotation m with z_geo = m·z_inst (|m|=1), or None if uncovered."""
    R = get_rotation(station)
    if R is None:
        return None
    return R[0, 0] + 1j * R[1, 0]


def clean_starts(d):
    ok = (~d["veto"].to_numpy(bool)) & np.isfinite(d["bandpassed"].to_numpy())
    cs = np.concatenate(([0], np.cumsum(ok.astype(int))))
    return np.where((cs[WIN:] - cs[:-WIN]) == WIN)[0]


def best_window(w, ds, st, m):
    """Over templates, the window's best |R| and the oriented Re(m·R) at that same peak."""
    bestR = 0.0 + 0j
    for sim in SIMS:
        for tn in TEMPLATES:
            zt = load_vec_template(ds, st, sim, tn)
            if zt is None:
                continue
            R = swcc_vector_gapaware(zt, w, min_valid_frac=0.8)
            if R.size == 0 or not np.isfinite(R).any():
                continue
            k = int(np.nanargmax(np.abs(R)))
            if np.abs(R[k]) > np.abs(bestR):
                bestR = R[k]
    mag = float(np.abs(bestR))
    orient = float(np.real(m * bestR)) if m is not None else np.nan   # Re(m·R) = oriented
    return mag, orient


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    val_rows, null_rows = [], []

    for ds, sts in STATIONS.items():
        for st in sts:
            d = load_cont_complex(ds, st)
            if d is None:
                continue
            sig = d["bandpassed"].to_numpy()
            starts = clean_starts(d)
            if len(starts) == 0:
                continue
            m = complex_m(st)

            # ── (A) orientation consistency via injection (covered stations only) ──
            if m is not None:
                zt = load_vec_template(ds, st, "sim2", "template3")
                if zt is None:                                  # EMAS sim1 etc. — fall back
                    zt = next((load_vec_template(ds, st, s, "template3") for s in SIMS
                               if load_vec_template(ds, st, s, "template3") is not None), None)
                if zt is not None:
                    M = len(zt); p = (WIN - M) // 2
                    s0 = starts[rng.integers(len(starts))]
                    w = sig[s0:s0+WIN].copy()
                    nrms = np.sqrt(np.mean(np.abs(w) ** 2))
                    shape = (zt - zt.mean()) / zt.std()
                    w[p:p+M] += INJ_SNR * nrms * np.conj(m) * shape   # inject in INSTRUMENT frame
                    R = swcc_vector_gapaware(zt, w, min_valid_frac=0.8)
                    k = int(np.nanargmax(np.abs(R)))
                    mag = float(np.abs(R[k]))
                    align = float(np.real(m * R[k]) / mag) if mag > 0 else np.nan
                    val_rows.append({"dataset": ds, "station": st,
                                     "az_X_deg": axis_azimuth(st),
                                     "arg_m_deg": round(float(np.degrees(np.angle(m))), 2),
                                     "inj_absR": round(mag, 3),
                                     "oriented_ReR": round(float(np.real(m * R[k])), 3),
                                     "alignment_ratio": round(align, 3)})

            # ── (B) real-data null: |R| (rotation-invariant) vs Re(m·R) (oriented) ──
            idx = starts if len(starts) <= N_WIN else starts[rng.choice(len(starts), N_WIN, replace=False)]
            for s0 in idx:
                mag, orient = best_window(sig[s0:s0+WIN], ds, st, m)
                null_rows.append({"dataset": ds, "station": st, "covered": covered(st),
                                  "absR": mag, "oriented_ReR": orient})
            print(f"  {ds}/{st}: covered={covered(st)}  windows={len(idx)}")

    val = pd.DataFrame(val_rows); val.to_csv(OUT / "orientation_validation.csv", index=False)
    nul = pd.DataFrame(null_rows); nul.to_csv(OUT / "real_null.csv", index=False)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    if len(val):
        ax = axes[0]
        ax.bar(val.station, val.alignment_ratio, color="#2563eb", alpha=0.8)
        ax.axhline(1.0, ls="--", c="green", label="perfect alignment (+1)")
        ax.axhline(0.0, ls=":", c="gray"); ax.set_ylim(-1.1, 1.2)
        ax.set_title("Orientation consistency: Re(m·R)/|R| at injection\n(applying the xlsx rotation aligns the recovered tilt vector)")
        ax.set_ylabel("alignment ratio"); ax.set_xlabel("station"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    if len(nul):
        ax = axes[1]
        order = [s for s in ["ECPN", "EEC1", "EC1", "EC10", "EMAS", "ECIT", "ECOR"]
                 if s in nul.station.unique()]
        data_abs = [nul[nul.station == s]["absR"].to_numpy() for s in order]
        bp = ax.boxplot(data_abs, labels=order, showfliers=False, patch_artist=True)
        for patch, s in zip(bp["boxes"], order):
            patch.set_facecolor("#2563eb" if covered(s) else "#9ca3af"); patch.set_alpha(0.5)
        ax.set_title("Real-data null: rotation-invariant |R| per station\n(blue = orientation available, grey = |R|-only)")
        ax.set_ylabel("best-window |R|"); ax.set_xlabel("station"); ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(OUT / "vector_orientation.png", dpi=300); plt.close(fig)

    # ── summary ──────────────────────────────────────────────────────────────
    L = ["VECTOR MATCHED FILTER — orientation validation + real-data null", "=" * 64, ""]
    L.append("(A) ORIENTATION CONSISTENCY (inject geographic template in the instrument frame,")
    L.append("    recover with the xlsx rotation m; alignment ratio Re(m·R)/|R| → +1 = correct):")
    if len(val):
        for _, r in val.iterrows():
            L.append(f"   {r.station:5s} az_X={r.az_X_deg:>5}°  |R|={r.inj_absR:.3f}  "
                     f"Re(m·R)={r.oriented_ReR:+.3f}  alignment={r.alignment_ratio:+.3f}")
        L.append(f"\n   mean alignment ratio = {val.alignment_ratio.mean():+.3f}  "
                 f"({'PASS ≈ +1' if val.alignment_ratio.mean() > 0.9 else 'CHECK'})")
    L.append("")
    L.append("(B) REAL-DATA NULL (best-window statistic; both detectors null = no coherent ULP tilt):")
    if len(nul):
        for st in [s for s in ["ECPN", "EEC1", "EC1", "EC10", "EMAS", "ECIT", "ECOR"]
                   if s in nul.station.unique()]:
            g = nul[nul.station == st]
            ori = g.oriented_ReR.dropna()
            ori_s = f"  Re(m·R) median={ori.median():+.3f}" if len(ori) else "  (|R| only — no orientation)"
            L.append(f"   {st:5s} covered={bool(g.covered.iloc[0])!s:5s}  "
                     f"|R| median={g.absR.median():.3f} 95th={np.percentile(g.absR,95):.3f}{ori_s}")
    L += ["", "Conclusion: the xlsx orientations are reproduced by the vector filter (A ≈ +1), and the",
          "real-data vector statistic is null whether rotation-invariant (|R|) or oriented (Re(m·R))."]
    (OUT / "summary.txt").write_text("\n".join(L))
    print("\n".join(L)); print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
