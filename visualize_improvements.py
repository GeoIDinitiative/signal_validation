"""
visualize_improvements.py
=========================
Build a gallery of BEFORE/AFTER figures showing every improvement made this session,
on a common time window per station.

Output: pipeline_improvements/
  <ST>_01_raw_vs_denoised.png    raw component      vs despiked+detrended (conditioning)
  <ST>_02_raw_vs_bandpassed.png  raw component      vs new clean bandpass
  <ST>_03_old_vs_new_bandpass.png  OLD mag-bandpass vs NEW mag clean-bandpass (same window)
  <ST>_04_waterfall.png          raw → despiked → detrended → bandpassed (stacked)
  <ST>_05_dering.png             naive across-gap filtering vs per-segment (de-ringing)
  GLOBAL_06_threshold_vs_null.png  peak |r| histogram: old r=0.2 threshold vs null floor
  GLOBAL_07_significance_bars.png  per-station peaks vs null-surviving counts

Stages are recomputed with the SAME operators as build_clean_bandpassed.py so the
intermediate (despiked / detrended) steps can be shown explicitly.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import sosfiltfilt, detrend as scipy_detrend

from build_clean_bandpassed import hampel_despike, SOS, FS, BAND

BASE = Path("/home/owen/tilt_validation")
OUT  = BASE / "pipeline_improvements"
CLEAN = BASE / "clean_bandpassed"
OLDBP = BASE / "bandpassed_exports_001_01_csv_only_utc"
RAWPWR = BASE / "tilt_raw_pwave_removed"

# station → (dataset, raw_feather, directional_col)
SRC = {
    "ECPN": ("ingv",       "/home/owen/Signals/experiment/INGV/ECPN.feather", "east"),
    "EMAS": ("experiment", "/home/owen/Signals/experiment/school-data/INGV_feather/EMAS.feather", "x"),
    "ECOR": ("experiment", "/home/owen/Signals/experiment/school-data/INGV_feather/ECOR.feather", "x"),
}
OLD_ANCH = {"ingv": pd.Timestamp("2022-11-14 22:00:00"),
            "experiment": pd.Timestamp("2023-07-23 23:00:00")}
WINDOW_S = 7200   # 2 h shown


def window_for(station):
    ds, _, _ = SRC[station]
    d = pd.read_feather(CLEAN / ds / f"{station}_dir_0p001-0p01Hz_clean_bp.feather")
    seg = d[d.segment_id == d.groupby("segment_id").size().idxmax()]
    t0 = pd.Timestamp(seg.datetime.min())
    return t0, t0 + pd.Timedelta(seconds=WINDOW_S)


def raw_window(station, t0, t1, col):
    ds, path, _ = SRC[station]
    df = pd.read_feather(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime")
    m = (df.datetime >= t0) & (df.datetime <= t1)
    sub = df[m]
    return sub["datetime"].to_numpy(), sub[col].to_numpy(float)


def stages(values):
    """Recompute conditioning stages identically to build_clean_bandpassed."""
    des, _ = hampel_despike(values)
    det = scipy_detrend(des, type="linear")
    bp = sosfiltfilt(SOS, det)
    return des, det, bp


def raw_global_start(station):
    """First raw datetime (sorted, de-duplicated) — used to sample-align the old export."""
    _, path, _ = SRC[station]
    df = pd.read_feather(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").drop_duplicates("datetime")["datetime"].iloc[0]


def old_bandpass_window(station, t0, t1):
    """
    The old export stores only time_seconds (no datetime). It is, sample-for-sample,
    the bandpass of the raw record (verified corr≈0.99), so we anchor it to the raw
    start: old_datetime = raw_start + (time_seconds - time_seconds[0]). This sample-
    aligns it with the new product (the hand-set swcc anchors carry an extra UTC shift
    that would mis-place the curve relative to the raw-datetime axis used here).
    """
    ds, _, _ = SRC[station]
    f = OLDBP / ds / f"{station}_0p001-0p01Hz_bp.csv"
    df = pd.read_csv(f)
    ts = df["time_seconds"].to_numpy(float)
    anchor = raw_global_start(station) - pd.Timedelta(seconds=float(ts[0]))
    dt = anchor + pd.to_timedelta(ts, unit="s")
    m = (dt >= t0) & (dt <= t1)
    return dt[m].to_numpy(), df["bandpassed"].to_numpy()[m]


def new_mag_window(station, t0, t1):
    ds, _, _ = SRC[station]
    d = pd.read_feather(CLEAN / ds / f"{station}_mag_0p001-0p01Hz_clean_bp.feather")
    d["datetime"] = pd.to_datetime(d["datetime"])
    m = (d.datetime >= t0) & (d.datetime <= t1)
    return d.datetime[m].to_numpy(), d["bandpassed"].to_numpy()[m]


# ── figure helpers ───────────────────────────────────────────────────────────
def side_by_side(t, left, right, ltitle, rtitle, suptitle, fname, sharey=False):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5), sharey=sharey)
    ax[0].plot(t, left, lw=0.7, color="#6b7280"); ax[0].set_title(ltitle)
    ax[1].plot(t, right, lw=0.7, color="#2563eb"); ax[1].set_title(rtitle)
    for a in ax:
        a.grid(alpha=0.3); a.tick_params(axis="x", rotation=30)
    fig.suptitle(suptitle, fontweight="600")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=140); plt.close(fig)


def overlay_two(tl, l, tr, r, llabel, rlabel, suptitle, fname):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
    ax[0].plot(tl, l, lw=0.7, color="#dc2626"); ax[0].set_title(llabel)
    ax[1].plot(tr, r, lw=0.7, color="#2563eb"); ax[1].set_title(rlabel)
    ylim = ax[0].get_ylim()
    for a in ax:
        a.grid(alpha=0.3); a.tick_params(axis="x", rotation=30)
    fig.suptitle(suptitle, fontweight="600")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=140); plt.close(fig)


def waterfall(station, t, raw, des, det, bp):
    fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for a, y, lab, c in zip(
        ax, [raw, des, det, bp],
        ["1. raw (P-wave-excised)", "2. despiked (Hampel/MAD)",
         "3. detrended (linear)", f"4. bandpassed {BAND} Hz"],
        ["#6b7280", "#9333ea", "#0891b2", "#2563eb"]):
        a.plot(t, y, lw=0.7, color=c); a.set_ylabel(lab, fontsize=9); a.grid(alpha=0.3)
    ax[-1].tick_params(axis="x", rotation=30)
    fig.suptitle(f"{station}: full conditioning waterfall (raw → bandpassed)",
                 fontweight="600")
    fig.tight_layout(); fig.savefig(OUT / f"{station}_04_waterfall.png", dpi=140)
    plt.close(fig)


def dering(station):
    ds, _, col = SRC[station]
    df = pd.read_feather(RAWPWR / f"{station}_tilt_raw_pwave_removed.feather")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    sec = (df.datetime - df.datetime.iloc[0]).dt.total_seconds().to_numpy(float)
    x = df[col].to_numpy(float)
    gaps = np.where(np.diff(sec) > 1.5)[0]
    pick = next(((max(0, g-5000), g, min(len(x), g+5000)) for g in gaps
                 if g-max(0, g-5000) > 4500 and min(len(x), g+5000)-g > 4500), None)
    if pick is None:
        return
    a, g, b = pick
    seg = x[a:b]
    naive = sosfiltfilt(SOS, seg - seg.mean())
    correct = np.full(len(seg), np.nan)
    for sl in (slice(0, g-a), slice(g-a, b-a)):
        xx = seg[sl] - seg[sl].mean()
        if len(xx) > 100:
            correct[sl] = sosfiltfilt(SOS, xx)
    t = sec[a:b] - sec[a]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t, naive, color="#dc2626", lw=0.8, label="OLD: one filter across the gap (rings)")
    ax.plot(t, correct, color="#2563eb", lw=0.8, label="NEW: per-segment filtering")
    ax.axvline(t[g-a], ls="--", c="k", alpha=0.6, label="gap (excised window)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set(title=f"{station}: de-ringing at a gap", xlabel="s within stretch",
           ylabel=f"bandpassed ({col})")
    fig.tight_layout(); fig.savefig(OUT / f"{station}_05_dering.png", dpi=140); plt.close(fig)


def global_swcc():
    p = pd.read_csv(BASE / "SWCC_comprehensive" / "all_peaks_flagged.csv")
    floor = p.groupby("dataset")["null_floor"].first().mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(p.abs_r, bins=60, color="#9ca3af", alpha=0.85)
    ax.axvline(0.2, ls="--", c="#dc2626", lw=2,
               label=f"OLD threshold r=0.2  (keeps {len(p)})")
    ax.axvline(floor, ls="--", c="#16a34a", lw=2,
               label=f"NEW null floor ≈{floor:.2f}  (keeps {int(p.significant.sum())})")
    ax.set(xlabel="peak |r|", ylabel="count",
           title="SWCC credibility: old r=0.2 threshold vs phase-randomised null floor")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "GLOBAL_06_threshold_vs_null.png", dpi=140)
    plt.close(fig)

    g = (p.groupby("station")
         .agg(peaks=("abs_r", "size"), significant=("significant", "sum"))
         .reset_index().sort_values("significant", ascending=False))
    x = np.arange(len(g)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x-w/2, g.peaks, w, label="peaks (r>0.2)", color="#9ca3af")
    ax.bar(x+w/2, g.significant, w, label="significant (null-surviving)", color="#2563eb")
    ax.set_xticks(x); ax.set_xticklabels(g.station)
    ax.set(ylabel="count", title="Per-station: chance peaks vs credible detections")
    ax.grid(alpha=0.3, axis="y"); ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 3)   # headroom (log axis) so legend clears the bars
    ax.legend(loc="upper right", framealpha=1.0, edgecolor="gray")
    fig.tight_layout(); fig.savefig(OUT / "GLOBAL_07_significance_bars.png", dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for st in SRC:
        ds, _, col = SRC[st]
        t0, t1 = window_for(st)
        t, raw = raw_window(st, t0, t1, col)
        if len(raw) < 1000:
            print(f"{st}: too few raw samples in window"); continue
        des, det, bp = stages(raw)
        print(f"{st}: window {t0} .. {t1}  ({len(raw)} samples)")

        side_by_side(t, raw, det,
                     f"{st} raw ({col})", f"{st} denoised (despiked+detrended)",
                     f"{st}: raw vs denoised", f"{st}_01_raw_vs_denoised.png")
        side_by_side(t, raw, bp,
                     f"{st} raw ({col})", f"{st} bandpassed {BAND} Hz",
                     f"{st}: raw vs bandpassed", f"{st}_02_raw_vs_bandpassed.png")
        waterfall(st, t, raw, des, det, bp)

        # old vs new bandpass on the magnitude channel (same component, fair compare)
        to, yo = old_bandpass_window(st, t0, t1)
        tn, yn = new_mag_window(st, t0, t1)
        if len(yo) and len(yn):
            overlay_two(to, yo, tn, yn,
                        f"OLD mag-bandpass (n={len(yo)})",
                        f"NEW mag clean-bandpass (n={len(yn)})",
                        f"{st}: old vs new bandpass (magnitude)",
                        f"{st}_03_old_vs_new_bandpass.png")
        dering(st)

    global_swcc()
    print(f"\nGallery → {OUT}")


if __name__ == "__main__":
    main()
