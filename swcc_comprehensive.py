"""
swcc_comprehensive.py
=====================
Stage 3 of the rebuilt pipeline (see PIPELINE_AUDIT.md).

Vectorised, SEGMENT-AWARE sliding-window cross-correlation (normalised Pearson r)
of the simulation tilt templates against the clean_bandpassed observed signals,
for both components (directional + magnitude) and both periods (ingv, experiment).

Key fixes vs the old swcc_edit.py:
  * O(N) running-moment + FFT cross-correlation instead of a pure-Python O(N*M)
    loop  (audit A4)  — validated to match the old definition to ~1e-9.
  * Correlation is computed WITHIN continuous segments only; a window never spans
    a gap, so there are no splice/through-quake artefacts (audit A5).
  * Runs both 'dir' and 'mag' products and writes a comparison (audit A6).

Run:
  python3 swcc_comprehensive.py --validate     # numerical-equivalence gate only
  python3 swcc_comprehensive.py                # full run
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks, correlate

# ── paths / params ──────────────────────────────────────────────────────────--
BASE_DIR  = Path("/home/owen/tilt_validation")
CLEAN_DIR = BASE_DIR / "clean_bandpassed"
TPL_DIR   = BASE_DIR / "tilt_templates_csv"
OUT_DIR   = BASE_DIR / "SWCC_comprehensive"

THRESHOLD     = 0.2
PEAK_DISTANCE = 1000          # samples between detected peaks (ULP spacing)
FS            = 1.0
SNR_GUARD_S   = 600.0         # shorter than old (segments are short after excision)
SNR_HALFWIN_S = 2000.0
SIMS      = ["sim1", "sim2", "sim3", "sim4"]
# template4 (~10,001 samples, ~2.8 h) is handled separately by swcc_template4.py:
# it is too long to fit the short post-excision segments, so it needs its own
# long-segment procedure (different window length, peak spacing and null floor).
TEMPLATES = ["template1", "template2", "template3"]

STATIONS = {
    "ingv":       ["ECPN", "EEC1"],
    "experiment": ["EC1", "EC10", "ECIT", "ECOR", "EMAS"],
}
COMPONENTS = {"dir": "directional", "mag": "magnitude"}


# ── core SWCC (vectorised, normalised Pearson r) ─────────────────────────────--

def swcc_segment(template, signal):
    """
    Normalised sliding-window cross-correlation (Pearson r) of `template` against
    one continuous `signal` segment.  Returns r for each window start (len L-M+1).

    Identical definition to the old loop:
        r_i = dot(template_norm, (window_i - mean_i)/std_i) / M
    but computed in O(L log L) via FFT cross-correlation + running moments.
    """
    t = np.asarray(template, float).ravel()
    x = np.asarray(signal, float).ravel()
    M, L = len(t), len(x)
    if M > L:
        return np.empty(0)

    t_std = t.std()
    if t_std == 0:
        return np.empty(0)
    t_norm = (t - t.mean()) / t_std            # sum(t_norm) == 0

    # cross-correlation term  c_i = sum_k t_norm[k] * x[i+k]
    c = correlate(x, t_norm, mode="valid", method="fft")   # length L-M+1

    # running window mean / std via cumulative sums
    cs1 = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    idx = np.arange(L - M + 1)
    s1 = cs1[idx + M] - cs1[idx]
    s2 = cs2[idx + M] - cs2[idx]
    mean = s1 / M
    var = s2 / M - mean * mean
    std = np.sqrt(np.clip(var, 0.0, None))

    r = np.zeros_like(c)
    nz = std > 0
    r[nz] = c[nz] / (M * std[nz])
    return r


def swcc_loop_reference(template, signal):
    """Slow O(N*M) reference — the OLD definition — used only for the validation gate."""
    t = np.asarray(template, float).ravel()
    x = np.asarray(signal, float).ravel()
    M, L = len(t), len(x)
    if M > L:
        return np.empty(0)
    tn = (t - t.mean()) / t.std()
    out = np.zeros(L - M + 1)
    for i in range(L - M + 1):
        w = x[i:i + M]
        sd = w.std()
        out[i] = 0.0 if sd == 0 else np.dot(tn, (w - w.mean()) / sd) / M
    return out


def template_snr(signal, peak_idx, M, fs=FS,
                 guard_s=SNR_GUARD_S, halfwin_s=SNR_HALFWIN_S):
    """SNR (dB) = 20log10( sqrt(P_signal / P_noise) ) using guarded flanking windows."""
    n = len(signal)
    i0, i1 = peak_idx, min(peak_idx + M, n)
    seg = signal[i0:i1]
    if len(seg) == 0:
        return np.nan
    p_sig = np.mean(seg ** 2)
    g, h = int(guard_s * fs), int(halfwin_s * fs)
    parts = []
    a0, a1 = max(0, i0 - g - h), max(0, i0 - g)
    b0, b1 = min(n, i1 + g), min(n, i1 + g + h)
    if a1 > a0:
        parts.append(signal[a0:a1])
    if b1 > b0:
        parts.append(signal[b0:b1])
    if not parts:
        return np.nan
    p_noise = np.mean(np.concatenate(parts) ** 2)
    if p_noise <= 0 or p_sig <= 0:
        return np.nan
    return 20.0 * np.log10(np.sqrt(p_sig / p_noise))


# ── loaders ──────────────────────────────────────────────────────────────────

def load_clean(dataset, station, comp_tag):
    f = CLEAN_DIR / dataset / f"{station}_{comp_tag}_0p001-0p01Hz_clean_bp.feather"
    if not f.exists():
        return None
    df = pd.read_feather(f)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def template_station(dataset, station):
    """INGV observed EEC1 is matched against the templates stored under name 'EC1'."""
    if dataset == "ingv" and station == "EEC1":
        return "EC1"
    return station


DOF_DIR = BASE_DIR / "tilt_templates_dofs"   # full-resolution templates rebuilt from the sim DOFs (proj_x/proj_y)


def load_template(dataset, station, sim, template, comp="dir"):
    """Component-aware band-passed template from the full-resolution DOFs:
    comp='dir' → proj_x (tilt-x, for the observed directional channel);
    comp='mag' → √(proj_x²+proj_y²) (tilt magnitude);
    comp='vec' → COMPLEX 2-component template proj_x + i·proj_y (for the vector matched filter).
    DOF files are keyed by OBSERVED station."""
    f = DOF_DIR / dataset / f"{station}_{sim}_{template}_dof.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if comp == "vec":
        return d["dir_bp"].to_numpy(float) + 1j * d["ortho_bp"].to_numpy(float)
    col = {"mag": "mag_bp", "dir2": "ortho_bp"}.get(comp, "dir_bp")   # dir2 = proj_y (Y axis)
    return d[col].to_numpy(float)


# ── validation gate ──────────────────────────────────────────────────────────

def validate():
    print("Numerical-equivalence gate (vectorised vs old loop)…")
    df = load_clean("ingv", "ECPN", "dir")
    tpl = load_template("ingv", "ECPN", "sim1", "template1")
    seg = df[df.segment_id == df.segment_id.iloc[0]]["bandpassed"].to_numpy(float)
    seg = seg[:6000]
    r_fast = swcc_segment(tpl, seg)
    r_slow = swcc_loop_reference(tpl, seg)
    max_abs = np.max(np.abs(r_fast - r_slow))
    print(f"  windows={len(r_fast)}  max|Δr|={max_abs:.3e}")
    ok = max_abs < 1e-9
    print("  PASS ✅" if ok else "  FAIL ❌")
    return ok


# ── main run ─────────────────────────────────────────────────────────────────

def run_station_component(dataset, station, comp_tag):
    df = load_clean(dataset, station, comp_tag)
    if df is None:
        print(f"  {station}/{comp_tag}: no clean file"); return None
    seg_ids = df.segment_id.unique()
    seg_arrays = {sid: df[df.segment_id == sid] for sid in seg_ids}

    peak_rows = []
    for sim in SIMS:
        for tname in TEMPLATES:
            tpl = load_template(dataset, station, sim, tname)
            if tpl is None:
                continue
            M = len(tpl)
            for sid, seg_df in seg_arrays.items():
                sig = seg_df["bandpassed"].to_numpy(float)
                if len(sig) < M:
                    continue
                seg_dt = seg_df["datetime"].to_numpy()
                edge = (seg_df["edge"].to_numpy(bool) if "edge" in seg_df.columns
                        else np.zeros(len(sig), bool))
                r = swcc_segment(tpl, sig)
                if r.size == 0:
                    continue
                peaks, props = find_peaks(np.abs(r), height=THRESHOLD,
                                          distance=PEAK_DISTANCE)
                for pk in peaks:
                    # window spans [pk, pk+M): edge-contaminated if it overlaps a settling zone
                    in_edge = bool(edge[pk:min(pk + M, len(edge))].any())
                    peak_rows.append({
                        "dataset": dataset, "station": station, "component": comp_tag,
                        "sim": sim, "template": tname, "segment_id": int(sid),
                        "peak_time": pd.Timestamp(seg_dt[pk]),
                        "r": float(r[pk]), "abs_r": float(abs(r[pk])),
                        "snr_db": float(template_snr(sig, int(pk), M)),
                        "in_edge": in_edge,
                    })
    pk_df = pd.DataFrame(peak_rows)
    out_sub = OUT_DIR / dataset
    out_sub.mkdir(parents=True, exist_ok=True)
    pk_df.to_csv(out_sub / f"{station}_{comp_tag}_peaks.csv", index=False)
    print(f"  {station}/{comp_tag}: {len(pk_df)} peaks "
          f"(segments={len(seg_ids)}, templates={len(SIMS)*len(TEMPLATES)})")
    return pk_df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not validate():
        print("Aborting: validation gate failed."); sys.exit(1)

    all_peaks = []
    for dataset, stations in STATIONS.items():
        print(f"\n=== {dataset} ===")
        for station in stations:
            for comp_tag in COMPONENTS:
                pk = run_station_component(dataset, station, comp_tag)
                if pk is not None and len(pk):
                    all_peaks.append(pk)

    if not all_peaks:
        print("No peaks found."); return
    peaks = pd.concat(all_peaks, ignore_index=True)
    peaks.to_csv(OUT_DIR / "all_peaks.csv", index=False)

    # ── summary table per dataset/station/component ──────────────────────────
    summ = (peaks.groupby(["dataset", "station", "component"])
            .agg(n_peaks=("abs_r", "size"),
                 median_abs_r=("abs_r", "median"),
                 max_abs_r=("abs_r", "max"),
                 median_snr_db=("snr_db", "median"))
            .reset_index())
    summ.to_csv(OUT_DIR / "summary_by_station_component.csv", index=False)
    print("\nSUMMARY\n" + summ.to_string(index=False))

    # ── component comparison figure (dir vs mag) ─────────────────────────────
    make_comparison_figure(peaks)
    print(f"\nOutputs → {OUT_DIR}")


def make_comparison_figure(peaks):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ds in zip(axes, ["ingv", "experiment"]):
        sub = peaks[peaks.dataset == ds]
        data, labels = [], []
        for comp in ["dir", "mag"]:
            vals = sub[sub.component == comp]["abs_r"].to_numpy()
            if len(vals):
                data.append(vals); labels.append(f"{comp}\n(n={len(vals)})")
        if data:
            bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
            for patch, c in zip(bp["boxes"], ["#2563eb", "#dc2626"]):
                patch.set_facecolor(c); patch.set_alpha(0.5)
        ax.axhline(THRESHOLD, ls="--", c="k", alpha=0.6, label=f"threshold {THRESHOLD}")
        ax.set_title(f"{ds}: peak |r| by component"); ax.set_ylabel("|r|")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "component_comparison.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    if "--validate" in sys.argv:
        sys.exit(0 if validate() else 1)
    main()
