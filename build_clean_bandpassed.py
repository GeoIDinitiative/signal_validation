"""
build_clean_bandpassed.py
=========================
Stage 1+2 of the rebuilt tilt pipeline (see PIPELINE_AUDIT.md).

Input  : tilt_raw_pwave_removed/<ST>_tilt_raw_pwave_removed.feather
         (UTC datetimes; P-wave windows already excised)
Output : clean_bandpassed/<dataset>/<ST>_<component>_0p001-0p01Hz_clean_bp.feather
         + matching .meta.json, + clean_bandpassed/clean_summary.txt

For every station and EACH component (directional + magnitude):
  1. Sort, drop duplicate timestamps, build a UTC-relative seconds axis.
  2. SEGMENT at gaps  (split wherever consecutive samples are > GAP_SEC apart).
     Gaps come from both recording breaks and excised earthquake windows.
  3. Per segment, if long enough:
        a. DESPIKE  (Hampel / rolling-MAD; glitches -> local median)
        b. DETREND  (linear; kills DC + drift before filtering)
        c. BANDPASS (Butterworth order 4, sos, zero-phase sosfiltfilt, 0.001-0.01 Hz)
     Segments shorter than MIN_SEG_SAMPLES are dropped (cannot be filtered cleanly).
  4. Reassemble onto the true UTC axis; gaps are preserved, each run keeps a segment_id.

Design rationale: per-segment filtering avoids the ringing that the old pipeline
produced by running sosfiltfilt straight across multi-million-second gaps (audit A3),
and operating on already-excised raw avoids splice/through-quake artifacts (audit A5).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, detrend as scipy_detrend

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/owen/tilt_validation")
IN_DIR   = BASE_DIR / "tilt_raw_pwave_removed"
OUT_DIR  = BASE_DIR / "clean_bandpassed"

# ── parameters ──────────────────────────────────────────────────────────────--
FS              = 1.0          # Hz (1 s nominal sampling within a segment)
BAND            = (0.001, 0.01)
ORDER           = 4
GAP_SEC         = 1.5         # split into a new segment when spacing exceeds this
MIN_SEG_SAMPLES = 4000        # ~4000 s; must exceed filter settling to bandpass cleanly
DESPIKE_WIN     = 31          # rolling window (samples) for Hampel median/MAD
DESPIKE_K       = 6.0         # outlier if |x-med| > K * 1.4826 * MAD
EDGE_SETTLE_S   = 1716        # filter settling length (s); used for odd-pad + edge flag

# ── station → (dataset, directional_col) ; magnitude handled separately ──────--
STATIONS = {
    "ECPN": ("ingv",        "east"),
    "EEC1": ("ingv",        "east"),
    "EC1":  ("experiment",  "x"),
    "EC10": ("experiment",  "x"),
    "ECIT": ("experiment",  "x"),
    "ECOR": ("experiment",  "x"),
    "EMAS": ("experiment",  "x"),
}


# ── helpers ─────────────────────────────────────────────────────────────────--

def load_clean_raw(station):
    """Load the P-wave-removed feather; return df sorted, de-duplicated on datetime."""
    df = pd.read_feather(IN_DIR / f"{station}_tilt_raw_pwave_removed.feather")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    return df


def get_component_series(df, station, component):
    """Return a float numpy array for the requested component ('directional'|'magnitude')."""
    _, dir_col = STATIONS[station]
    if component == "directional":
        return df[dir_col].to_numpy(float), dir_col
    # magnitude
    if "mag" in df.columns:
        return df["mag"].to_numpy(float), "mag"
    # EC1 has no 'mag' -> compute from x,y
    if {"x", "y"}.issubset(df.columns):
        m = np.sqrt(df["x"].to_numpy(float) ** 2 + df["y"].to_numpy(float) ** 2)
        return m, "sqrt(x^2+y^2)"
    raise ValueError(f"{station}: cannot build magnitude")


def segment_indices(seconds):
    """Return list of (start, end) index slices for continuous runs (gap > GAP_SEC splits)."""
    breaks = np.where(np.diff(seconds) > GAP_SEC)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends   = np.concatenate((breaks + 1, [len(seconds)]))
    return list(zip(starts, ends))


def hampel_despike(x, win=DESPIKE_WIN, k=DESPIKE_K):
    """Vectorised Hampel filter: replace outliers (|x-med| > k*1.4826*MAD) with rolling median."""
    s = pd.Series(x)
    med = s.rolling(win, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(win, center=True, min_periods=1).median()
    thresh = k * 1.4826 * mad
    out = s.where((s - med).abs() <= thresh, med)
    n_repl = int(((s - med).abs() > thresh).sum())
    return out.to_numpy(float), n_repl


def make_sos():
    nyq = 0.5 * FS
    lo, hi = BAND[0] / nyq, BAND[1] / nyq
    return butter(ORDER, [lo, hi], btype="bandpass", output="sos")


SOS = make_sos()


def process_component(df, station, component):
    """Run segment→despike→detrend→bandpass for one component. Returns out_df, stats."""
    values, src_col = get_component_series(df, station, component)
    dt = df["datetime"].to_numpy()
    seconds = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds().to_numpy(float)

    segs = segment_indices(seconds)
    out_dt, out_sec, out_bp, out_sid, out_edge = [], [], [], [], []
    n_spikes = 0
    n_used = n_dropped = 0
    seg_id = 0

    for (a, b) in segs:
        seg_len = b - a
        if seg_len < MIN_SEG_SAMPLES:
            n_dropped += 1
            continue
        x = values[a:b].astype(float)
        # guard against NaNs inside a segment
        if not np.all(np.isfinite(x)):
            x = pd.Series(x).interpolate(limit_direction="both").to_numpy(float)
            if not np.all(np.isfinite(x)):
                n_dropped += 1
                continue
        x, ns = hampel_despike(x)
        n_spikes += ns
        x = scipy_detrend(x, type="linear")
        # even-reflection padding by ~one settling length suppresses the edge transient
        # at BOTH ends. (Even, not odd: odd padding doubles a non-zero edge value and can
        # blow the end transient up ~7x on noisy data; even mirrors within the data range
        # and is stable. Tukey tapering kills transients better still but distorts the
        # waveform shape the SWCC matches, so it is avoided here.)
        padlen = min(EDGE_SETTLE_S, seg_len - 1)
        xf = sosfiltfilt(SOS, x, padtype="even", padlen=padlen)
        # flag the (still slightly unreliable) settling zones at each segment end
        edge = np.zeros(seg_len, dtype=bool)
        e = min(EDGE_SETTLE_S, seg_len)
        edge[:e] = True
        edge[seg_len - e:] = True

        out_dt.append(dt[a:b])
        out_sec.append(seconds[a:b])
        out_bp.append(xf)
        out_sid.append(np.full(seg_len, seg_id, dtype=int))
        out_edge.append(edge)
        seg_id += 1
        n_used += 1

    if n_used == 0:
        return None, dict(src_col=src_col, n_seg=len(segs), n_used=0, n_dropped=n_dropped,
                          n_spikes=n_spikes, n_out=0)

    out_df = pd.DataFrame({
        "datetime":     np.concatenate(out_dt),
        "time_seconds": np.concatenate(out_sec),
        "bandpassed":   np.concatenate(out_bp),
        "segment_id":   np.concatenate(out_sid),
        "edge":         np.concatenate(out_edge),
    })
    stats = dict(src_col=src_col, n_seg=len(segs), n_used=n_used, n_dropped=n_dropped,
                 n_spikes=n_spikes, n_out=len(out_df))
    return out_df, stats


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    summary = [
        "CLEAN BANDPASSED — Stage 1+2 summary",
        "=" * 60,
        f"Band {BAND[0]}-{BAND[1]} Hz | Butterworth order {ORDER} | zero-phase sosfiltfilt",
        f"Gap split > {GAP_SEC}s | min segment {MIN_SEG_SAMPLES} samples | "
        f"Hampel win {DESPIKE_WIN}, k {DESPIKE_K}",
        "=" * 60, "",
    ]

    for station, (dataset, dir_col) in STATIONS.items():
        in_path = IN_DIR / f"{station}_tilt_raw_pwave_removed.feather"
        if not in_path.exists():
            print(f"skip {station}: {in_path} missing")
            continue
        df = load_clean_raw(station)
        out_subdir = OUT_DIR / dataset
        out_subdir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\n{station} ({dataset})  rows={len(df):,}")
        summary.append(f"{station} ({dataset})  raw rows={len(df):,}")

        for component in ("directional", "magnitude"):
            out_df, st = process_component(df, station, component)
            tag = "dir" if component == "directional" else "mag"
            line = (f"  {component:11s} [{st['src_col']:14s}] "
                    f"segs={st['n_seg']:4d} used={st['n_used']:4d} drop={st['n_dropped']:4d} "
                    f"spikes={st['n_spikes']:6d} out_rows={st['n_out']:,}")
            print(line); summary.append(line)
            if out_df is None:
                continue
            base = out_subdir / f"{station}_{tag}_0p001-0p01Hz_clean_bp"
            out_df.to_feather(base.with_suffix(".feather"))
            meta = {
                "dataset": dataset, "station": station, "component": component,
                "source_column": st["src_col"], "fs": FS, "band": list(BAND),
                "butter_order": ORDER, "method": "per-segment detrend+Hampel+sosfiltfilt",
                "gap_split_s": GAP_SEC, "min_segment_samples": MIN_SEG_SAMPLES,
                "despike_window": DESPIKE_WIN, "despike_k": DESPIKE_K,
                "edge_settle_s": EDGE_SETTLE_S, "pad": "even-reflection, padlen=settling",
                "n_segments_total": st["n_seg"], "n_segments_used": st["n_used"],
                "n_segments_dropped": st["n_dropped"], "n_spikes_replaced": st["n_spikes"],
                "n_output_rows": st["n_out"],
                "columns": ["datetime", "time_seconds", "bandpassed", "segment_id", "edge"],
            }
            base.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        summary.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "clean_summary.txt").write_text("\n".join(summary))
    print(f"\n{'='*60}\nSummary → {OUT_DIR/'clean_summary.txt'}")


if __name__ == "__main__":
    main()
