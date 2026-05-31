"""
build_tilt_raw_pwave_removed.py
================================
Applies asymmetric P-wave exclusion buffers (from pwave_buffer.py) to every
raw tilt signal and saves the cleaned result.

For each station:
  1. Load raw feather/CSV
  2. Sort and align datetimes to UTC
  3. Build exclusion mask from the earthquake catalog
     – buffer = calculate_pwave_buffers(distance_km, magnitude)
     – each event excludes  [p_wave_eta − pre_min, p_wave_eta + post_min]
     – overlapping intervals are merged before masking (efficient)
  4. Drop excluded rows (segments are removed, not NaN-filled)
  5. Write  <STATION>_tilt_raw_pwave_removed.feather
  6. Write  <STATION>_removal_log.csv   (one row per excluded interval,
             with originating earthquake metadata)
  7. Append station summary to  removal_summary.txt

Output directory:  etna_signals_phd/tilt_raw_pwave_removed/
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
BASE_DIR  = Path("/home/owen/tilt_validation")
OUT_DIR   = BASE_DIR / "tilt_raw_pwave_removed"
EQ_CSV    = BASE_DIR / "earthquakes_merged_utc.csv"

sys.path.insert(0, str(BASE_DIR))
from pwave_buffer import calculate_pwave_buffers

# ── raw signal sources ────────────────────────────────────────────────────────
RAW_SOURCES = {
    # label: (path, file_format, signal_columns, dataset_tag)
    "ECPN": (
        Path("/home/owen/Signals/experiment/INGV/ECPN.feather"),
        "feather",
        ["east", "north", "mag"],
        "ingv",
    ),
    "EEC1": (
        Path("/home/owen/Signals/experiment/INGV/EEC1.feather"),
        "feather",
        ["east", "north", "mag"],
        "ingv",
    ),
    "EC1": (
        Path("/home/owen/Signals/experiment/EC1.csv"),
        "csv",
        [" x", "y"],         # note leading space in original header
        "experiment",
    ),
    "EC10": (
        Path("/home/owen/Signals/experiment/school-data/INGV_feather/EC10.feather"),
        "feather",
        ["x", "y", "mag"],
        "experiment",
    ),
    "ECIT": (
        Path("/home/owen/Signals/experiment/school-data/INGV_feather/ECIT.feather"),
        "feather",
        ["x", "y", "mag"],
        "experiment",
    ),
    "ECOR": (
        Path("/home/owen/Signals/experiment/school-data/INGV_feather/ECOR.feather"),
        "feather",
        ["x", "y", "mag"],
        "experiment",
    ),
    "EMAS": (
        Path("/home/owen/Signals/experiment/school-data/INGV_feather/EMAS.feather"),
        "feather",
        ["x", "y", "mag"],
        "experiment",
    ),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_raw(path, fmt):
    """Load feather or CSV and return dataframe with parsed 'datetime' column."""
    if fmt == "feather":
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def build_exclusion_intervals(eq_df, t_start, t_end):
    """
    For every earthquake whose p_wave_eta falls within [t_start, t_end],
    compute (exclusion_start, exclusion_end, pre_min, post_min, magnitude, distance).
    Returns a list of dicts, sorted by exclusion_start.
    """
    intervals = []
    for _, q in eq_df.iterrows():
        eta  = q["p_wave_eta"]
        if not (t_start <= eta <= t_end):
            continue
        mag  = q.get("magnitude",   np.nan)
        dist = q.get("distance_km", np.nan)
        pre, post = calculate_pwave_buffers(
            None if pd.isna(dist) else dist,
            None if pd.isna(mag)  else mag,
        )
        ex_start = eta - pd.Timedelta(minutes=pre)
        ex_end   = eta + pd.Timedelta(minutes=post)
        intervals.append({
            "eq_p_wave_eta":   eta,
            "eq_magnitude":    mag,
            "eq_distance_km":  dist,
            "pre_min":         pre,
            "post_min":        post,
            "exclusion_start": ex_start,
            "exclusion_end":   ex_end,
        })
    intervals.sort(key=lambda x: x["exclusion_start"])
    return intervals


def merge_intervals(intervals):
    """
    Merge overlapping/adjacent exclusion intervals.
    Returns list of (merged_start, merged_end, [contributing_rows]).
    """
    if not intervals:
        return []
    merged = []
    cur_start = intervals[0]["exclusion_start"]
    cur_end   = intervals[0]["exclusion_end"]
    cur_rows  = [intervals[0]]

    for iv in intervals[1:]:
        if iv["exclusion_start"] <= cur_end:
            cur_end  = max(cur_end, iv["exclusion_end"])
            cur_rows.append(iv)
        else:
            merged.append((cur_start, cur_end, cur_rows))
            cur_start = iv["exclusion_start"]
            cur_end   = iv["exclusion_end"]
            cur_rows  = [iv]
    merged.append((cur_start, cur_end, cur_rows))
    return merged


def apply_exclusion_mask(df, merged_intervals):
    """
    Return (clean_df, mask) where mask is a boolean Series
    (True = clean, False = excluded).
    """
    dt = df["datetime"].values  # numpy datetime64 array
    mask = np.ones(len(df), dtype=bool)

    for (ex_start, ex_end, _) in merged_intervals:
        lo = np.searchsorted(dt, np.datetime64(ex_start, "ns"))
        hi = np.searchsorted(dt, np.datetime64(ex_end,   "ns"), side="right")
        mask[lo:hi] = False

    return df[mask].reset_index(drop=True), mask


def build_removal_log(intervals):
    """Build a per-earthquake log — time-based metrics only (sampling rate may vary)."""
    rows = []
    for iv in intervals:
        duration_min = (iv["exclusion_end"] - iv["exclusion_start"]).total_seconds() / 60
        rows.append({
            "eq_p_wave_eta":    iv["eq_p_wave_eta"],
            "eq_magnitude":     iv["eq_magnitude"],
            "eq_distance_km":   iv["eq_distance_km"],
            "pre_min":          iv["pre_min"],
            "post_min":         iv["post_min"],
            "exclusion_start":  iv["exclusion_start"],
            "exclusion_end":    iv["exclusion_end"],
            "window_duration_min": round(duration_min, 2),
        })
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "P-WAVE REMOVAL SUMMARY",
        "=" * 60,
        f"Earthquake catalog : {EQ_CSV}",
        f"Buffer logic       : pwave_buffer.calculate_pwave_buffers",
        f"Output directory   : {OUT_DIR}",
        "=" * 60,
        "",
    ]

    eq_df = pd.read_csv(EQ_CSV)
    eq_df["p_wave_eta"] = pd.to_datetime(eq_df["p_wave_eta"])

    for station, (path, fmt, sig_cols, dataset) in RAW_SOURCES.items():
        print(f"\n{'='*60}")
        print(f"  {station}  ({dataset})")

        # ── load ─────────────────────────────────────────────────────────────
        df = load_raw(path, fmt)
        t_start = df["datetime"].iloc[0]
        t_end   = df["datetime"].iloc[-1]
        n_total = len(df)
        print(f"  Loaded {n_total:,} rows  [{t_start}  →  {t_end}]")

        # ── build intervals ───────────────────────────────────────────────────
        intervals = build_exclusion_intervals(eq_df, t_start, t_end)
        merged    = merge_intervals(intervals)
        print(f"  Earthquakes in window : {len(intervals)}")
        print(f"  Merged intervals      : {len(merged)}")

        # ── apply mask ────────────────────────────────────────────────────────
        keep_cols = ["datetime"] + [c for c in sig_cols if c in df.columns]
        df_out, mask = apply_exclusion_mask(df[keep_cols], merged)

        n_removed   = int(np.sum(~mask))
        n_clean     = len(df_out)
        # Time-based metrics (meaningful regardless of variable sampling rate)
        signal_duration_h = (t_end - t_start).total_seconds() / 3600
        removed_time_h    = sum(
            (e - s).total_seconds() / 3600
            for s, e, _ in merged
            if s <= t_end and e >= t_start
        )
        pct_time_removed  = 100 * removed_time_h / signal_duration_h
        print(f"  Samples removed       : {n_removed:,} / {n_total:,}  (rows; rate may vary)")
        print(f"  Time removed          : {removed_time_h:.1f} h / {signal_duration_h:.1f} h  ({pct_time_removed:.1f} %)")
        print(f"  Clean samples         : {n_clean:,}")

        # ── rename EC1 column ─────────────────────────────────────────────────
        if " x" in df_out.columns:
            df_out = df_out.rename(columns={" x": "x"})

        # ── save cleaned signal ───────────────────────────────────────────────
        out_feather = OUT_DIR / f"{station}_tilt_raw_pwave_removed.feather"
        df_out.reset_index(drop=True).to_feather(out_feather)
        print(f"  Saved → {out_feather.name}")

        # ── save removal log ──────────────────────────────────────────────────
        log_df = build_removal_log(intervals)
        log_csv = OUT_DIR / f"{station}_removal_log.csv"
        log_df.to_csv(log_csv, index=False)
        print(f"  Log   → {log_csv.name}  ({len(log_df)} events)")

        # ── station summary ───────────────────────────────────────────────────
        summary_lines += [
            f"{station} ({dataset})",
            f"  Signal window  : {t_start}  →  {t_end}  ({signal_duration_h:.1f} h)",
            f"  Total rows     : {n_total:,}  (variable sampling rate — use time metrics)",
            f"  Time removed   : {removed_time_h:.1f} h  ({pct_time_removed:.1f} % of signal duration)",
            f"  Time retained  : {signal_duration_h - removed_time_h:.1f} h",
            f"  EQ intervals   : {len(intervals)} individual  →  {len(merged)} merged",
            "",
        ]

    # ── write summary ─────────────────────────────────────────────────────────
    summary_path = OUT_DIR / "removal_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"\n{'='*60}")
    print(f"Summary → {summary_path}")
    print(f"All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
