#!/usr/bin/env python3
"""
Find Synchronous Events Across All Stations
Uses Bagagli methods for event detection with SWCC plotting style
Filters to templates 1-3 only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, detrend as scipy_detrend
from scipy.interpolate import interp1d
import os

# ==========================
# Configuration
# ==========================
MERGED_CSV = "/home/owen/tilt_validation/merged_csv/peaks_p_filtered.csv"

TEMPLATES_DIR = "tilt_templates"
DENOISED_DIR = "/home/owen/denoised_exports_003_001_csv_only/ingv/"
OUTPUT_DIR = "/home/owen/tilt_validation/synchronous_events"
EARTHQUAKE_CSV = "/home/owen/tilt_validation/earthquakes_with_p_waves.csv"

# Time tolerance for synchronous events (seconds)
# Events within this window are considered synchronous
# Set to match template/segment size for proper overlap detection
SYNC_TOLERANCE_SEC = 3333  # Match segment duration

# Minimum number of stations for synchronous event
MIN_STATIONS = 2

# Quality filter criteria
MIN_AVG_CORRELATION = 0.0  # Minimum average peak_corr for event
MIN_AVG_SNR_LINEAR = 0.0   # Minimum average snr_linear for event


# P-wave screening window (seconds)
# Screen out events that occur within this window around P-wave arrivals
P_WAVE_BEFORE_SEC = 600   # 10 minutes before P-wave
P_WAVE_AFTER_SEC = 600   # 10 minutes after P-wave

# Segment duration for plotting (seconds)
SEGMENT_DURATION_S = 3333


# Filter parameters
BAND_HZ = (0.001, 0.01)
FILTER_ORDER = 4

# Templates to include (1-3 only)
TEMPLATES_TO_INCLUDE = ["template1", "template2", "template3","template4"]

# ==========================
# Helper Functions
# ==========================
def load_tilt_templates(outdir: str):
    """
    Load templates from directory structure
    Returns (templates, all_stations, all_sims)
    """
    templates = {}
    all_sims, all_stations = set(), set()
    
    for tpl in ["T1", "T2", "T3", "T4"]:
        tpl_dir = os.path.join(outdir, tpl)
        if not os.path.isdir(tpl_dir):
            continue
        templates[tpl] = {}
        
        for sim_dir in os.listdir(tpl_dir):
            sim_path = os.path.join(tpl_dir, sim_dir)
            if not os.path.isdir(sim_path):
                continue
            all_sims.add(sim_dir)
            templates[tpl][sim_dir] = {}
            
            for fname in os.listdir(sim_path):
                if not fname.endswith('.npy'):
                    continue
                station = fname.replace('.npy', '')
                all_stations.add(station)
                fpath = os.path.join(sim_path, fname)
                try:
                    arr = np.load(fpath)
                    t = np.arange(len(arr))
                    templates[tpl][sim_dir][station] = {'x': arr, 't': t}
                except Exception as e:
                    print(f"Warning: Could not load {fpath}: {e}")
    
    return templates, sorted(all_stations), sorted(all_sims)


def find_synchronous_events(df: pd.DataFrame, tolerance_sec: int, min_stations: int = 2):
    """
    Find synchronous events across stations using time-based clustering.
    
    Args:
        df: DataFrame with peak_time_dt column
        tolerance_sec: Time tolerance window in seconds
        min_stations: Minimum number of stations required for an event
    
    Returns:
        DataFrame with event_id column added
    """
    df = df.copy()
    df['peak_time_dt'] = pd.to_datetime(df['peak_time_dt'])
    df = df.sort_values('peak_time_dt').reset_index(drop=True)
    
    # Initialize event IDs
    df['event_id'] = -1
    event_counter = 0
    tolerance = pd.Timedelta(seconds=tolerance_sec)
    
    # Group peaks within tolerance window
    unassigned = df['event_id'] == -1
    
    while unassigned.any():
        # Get first unassigned peak
        idx = df.loc[unassigned].index[0]
        ref_time = df.loc[idx, 'peak_time_dt']
        
        # Find all peaks within tolerance
        time_diff = (df['peak_time_dt'] - ref_time).abs()
        in_window = (time_diff <= tolerance) & unassigned
        
        # Check if enough stations
        stations_in_window = df.loc[in_window, 'station'].unique()
        
        if len(stations_in_window) >= min_stations:
            # Assign event ID
            df.loc[in_window, 'event_id'] = event_counter
            event_counter += 1
        else:
            # Mark as isolated (no event ID)
            df.loc[in_window, 'event_id'] = -999
        
        unassigned = df['event_id'] == -1
    
    # Remove isolated events
    df = df[df['event_id'] >= 0].copy()
    
    return df


def filter_events_by_quality(df: pd.DataFrame, min_avg_corr: float, min_avg_snr: float):
    """
    Filter events based on quality criteria (average correlation and SNR).
    
    Args:
        df: DataFrame with event_id, peak_corr, and snr_linear columns
        min_avg_corr: Minimum average correlation threshold
        min_avg_snr: Minimum average SNR (linear) threshold
    
    Returns:
        Filtered DataFrame with only high-quality events
    """
    df = df.copy()
    
    # Calculate per-event averages
    event_stats = df.groupby('event_id').agg({
        'peak_corr': 'mean',
        'snr_linear': 'mean'
    }).reset_index()
    
    event_stats.columns = ['event_id', 'avg_correlation', 'avg_snr_linear']
    
    # Filter events by criteria
    good_events = event_stats[
        (event_stats['avg_correlation'] >= min_avg_corr) &
        (event_stats['avg_snr_linear'] >= min_avg_snr)
    ]['event_id'].values
    
    # Keep only good events
    df_filtered = df[df['event_id'].isin(good_events)].copy()
    
    # Merge stats back for reporting
    df_filtered = df_filtered.merge(event_stats, on='event_id', how='left')
    
    return df_filtered, event_stats


def screen_earthquake_contamination(df: pd.DataFrame, earthquake_csv: str, 
                                   before_sec: int, after_sec: int):
    """
    Screen out events that coincide with earthquake P-wave arrivals.
    
    Args:
        df: DataFrame with event_id and peak_time_dt columns
        earthquake_csv: Path to earthquake CSV with p_wave_eta column
        before_sec: Seconds before P-wave to screen
        after_sec: Seconds after P-wave to screen
    
    Returns:
        Filtered DataFrame, list of screened event_ids, earthquake DataFrame
    """
    df = df.copy()
    
    # Load earthquake data
    if not Path(earthquake_csv).exists():
        print(f"Warning: Earthquake file not found: {earthquake_csv}")
        return df, [], None
    
    quakes = pd.read_csv(earthquake_csv)
    
    # Parse P-wave arrival times
    if 'p_wave_eta' not in quakes.columns:
        print("Warning: p_wave_eta column not found in earthquake CSV")
        return df, [], None
    
    quakes['p_wave_eta'] = pd.to_datetime(quakes['p_wave_eta'], errors='coerce')
    quakes = quakes.dropna(subset=['p_wave_eta'])
    
    print(f"\nLoaded {len(quakes)} earthquakes with P-wave arrivals")
    
    # For each event, check if its full plotting window overlaps any P-wave guard window
    contaminated_events = []
    half_window = pd.Timedelta(seconds=SEGMENT_DURATION_S / 2)
    
    for event_id in df['event_id'].unique():
        event_group = df[df['event_id'] == event_id]
        earliest_peak = event_group['peak_time_dt'].min()
        if pd.isna(earliest_peak):
            continue
        event_start = earliest_peak - half_window
        event_end = earliest_peak + half_window
        
        # Check if any P-wave arrival falls within screening window
        for _, quake in quakes.iterrows():
            p_arrival = quake['p_wave_eta']
            screen_start = p_arrival - pd.Timedelta(seconds=before_sec)
            screen_end = p_arrival + pd.Timedelta(seconds=after_sec)
            
            # Check for overlap
            if (event_start <= screen_end) and (event_end >= screen_start):
                contaminated_events.append(event_id)
                break
    
    # Filter out contaminated events
    n_screened = len(contaminated_events)
    df_clean = df[~df['event_id'].isin(contaminated_events)].copy()
    
    print(f"Screened out {n_screened} events contaminated by earthquake P-waves")
    print(f"  (within -{before_sec//60}min to +{after_sec//60}min of P-wave arrival)")
    
    return df_clean, contaminated_events, quakes


def bandpass_signal(x, fs, f1=0.001, f2=0.01, order=4):
    """Apply bandpass filter to signal"""
    x = np.asarray(x, float)
    nyq = 0.5 * fs
    lo = max(f1 / nyq, 1e-6)
    hi = min(f2 / nyq, 0.999999)
    if not (0 < lo < hi < 1):
        return x
    sos = butter(order, [lo, hi], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def prep_signal(x, fs=None, detrend_type="linear", normalize=None, apply_filter=True, band=(0.001, 0.01)):
    """Preprocess signal: detrend, filter, normalize"""
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    
    # Detrend
    if detrend_type in ('linear', 'constant'):
        x = scipy_detrend(x, type=detrend_type)
    
    # Filter
    if apply_filter and fs is not None:
        x = bandpass_signal(x, fs, band[0], band[1])
    
    # Normalize
    if normalize == 'zscore':
        mu, sd = np.mean(x), np.std(x)
        if sd > 0:
            x = (x - mu) / sd
    elif normalize == 'maxabs':
        ma = np.max(np.abs(x))
        if ma > 0:
            x = x / ma
    
    return x


def resample_to_grid(x, t, t_grid):
    """Resample signal to common time grid"""
    x = np.asarray(x, float).ravel()
    t = np.asarray(t, float).ravel()
    if x.size < 1:
        return np.full_like(t_grid, np.nan, dtype=float)
    if x.size == 1 or (t[-1] - t[0]) == 0:
        return np.full_like(t_grid, x[0], dtype=float)
    f = interp1d(t, x, kind='linear', bounds_error=False, fill_value='extrapolate')
    return f(t_grid)


def load_denoised_station(station_name, denoised_dir):
    """Load denoised data for a station"""
    den_path = Path(denoised_dir) / f"{station_name}_0p001-0p01.csv"
    if not den_path.exists():
        return None
    
    df = pd.read_csv(den_path)
    if "denoised" not in df.columns:
        return None
    
    # Find datetime column
    dt_col = next((c for c in ["datetime", "time", "timestamp"] if c in df.columns), None)
    if dt_col is None:
        return None
    
    df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)
    
    return df


def plot_synchronous_event(event_id, event_group, templates, denoised_data, 
                           segment_duration_s, output_dir, band_hz, quakes_df=None,
                           p_wave_before_sec=300, p_wave_after_sec=1200):
    """
    Plot synchronous event using SWCC style with P-wave arrivals marked
    """
    # Get time window
    earliest_peak = event_group["peak_time_dt"].min()
    latest_peak = event_group["peak_time_dt"].max()
    
    # Center window on event with buffer
    half = segment_duration_s / 2
    start_time = earliest_peak - pd.Timedelta(seconds=half)
    end_time = earliest_peak + pd.Timedelta(seconds=half)
    
    print(f"\n{'='*60}")
    print(f"Event {event_id}: {earliest_peak} (span: {(latest_peak - earliest_peak).total_seconds():.1f}s)")
    print(f"Stations: {sorted(event_group['station'].unique())}")
    print(f"Templates: {sorted(event_group['template'].unique())}")
    
    # Find P-wave arrivals in this window
    p_waves_in_window = []
    if quakes_df is not None:
        for _, quake in quakes_df.iterrows():
            p_arrival = pd.to_datetime(quake['p_wave_eta'])
            if start_time <= p_arrival <= end_time:
                p_waves_in_window.append({
                    'time': p_arrival,
                    'magnitude': quake.get('magnitude', np.nan),
                    'distance_km': quake.get('distance_km', np.nan)
                })
    
    if p_waves_in_window:
        print(f"P-wave arrivals in window: {len(p_waves_in_window)}")
    
    # Common time grid for plotting (in minutes)
    t_common = np.linspace(0, segment_duration_s, 3000)
    
    # Station colors (SWCC style)
    station_colors = {
        "EEC1": "#1f77b4",  # blue
        "EC1": "#1f77b4",   # blue
        "ECPN": "#2ca02c",  # green
        "ECIT": "#ff7f0e",  # orange
        "EC10": "#d62728",  # red
        "ECOR": "#9467bd",  # purple
        "EMAS": "#8c564b"   # brown
    }
    
    # Template colors (warm palette)
    template_colors = ["#E74C3C", "#E67E22", "#F39C12", "#D35400", "#C0392B",
                      "#FF6B6B", "#FF8C42", "#FFA600", "#FF4D6D", "#FF6F00"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Template mapping
    template_map = {"template1": "T1", "template2": "T2", "template3": "T3"}
    
    # Track plotted items for legend
    plotted_stations = set()
    template_idx = 0
    
    # Plot each station
    for station in sorted(event_group['station'].unique()):
        station_upper = str(station).upper()
        station_peaks = event_group[event_group['station'] == station]
        
        # Load denoised data
        den_df = denoised_data.get(station_upper)
        if den_df is None:
            continue
        
        # Slice to event window
        seg = den_df[(den_df["datetime"] >= start_time) & (den_df["datetime"] <= end_time)].copy()
        if seg.empty:
            continue
        
        # Prepare observed signal
        x_raw = seg["denoised"].to_numpy(float)
        t_s = (seg["datetime"] - seg["datetime"].iloc[0]).dt.total_seconds().to_numpy()
        
        # Estimate fs
        fs = 1.0 / np.median(np.diff(t_s)) if len(t_s) > 1 else 1.0
        
        # Preprocess
        x = prep_signal(x_raw, fs=fs, apply_filter=True, band=band_hz)
        
        # Resample to common grid
        y_obs = resample_to_grid(x, t_s, t_common)
        
        # Plot observed
        color = station_colors.get(station_upper, "#000000")
        ax.plot(t_common / 60, y_obs, color=color, lw=2.0, ls='-',
               alpha=0.95, label=f"{station_upper} denoised", zorder=8)
        plotted_stations.add(station_upper)
        
        # Plot templates for this station
        for _, row in station_peaks.iterrows():
            sim = str(row.get("sim", ""))
            csv_tpl = str(row.get("template", "")).lower()
            tpl_key = template_map.get(csv_tpl)
            
            if tpl_key is None or tpl_key not in templates:
                continue
            
            # Find template
            sim_key = sim if sim in templates[tpl_key] else None
            if sim_key is None:
                for prefix in ["filt_", "raw_", ""]:
                    cand = f"{prefix}{sim}"
                    if cand in templates[tpl_key]:
                        sim_key = cand
                        break
            
            if sim_key is None:
                continue
            
            # Map station name for template lookup
            tpl_station = "EC1" if station_upper == "EEC1" else station_upper
            st_key = tpl_station if tpl_station in templates[tpl_key][sim_key] else None
            
            if st_key is None:
                for k in templates[tpl_key][sim_key].keys():
                    if k.upper() == tpl_station.upper():
                        st_key = k
                        break
            
            if st_key is None:
                continue
            
            # Get template
            tpl_data = templates[tpl_key][sim_key][st_key]
            tpl_x = np.asarray(tpl_data['x'], float)
            tpl_t = np.asarray(tpl_data.get('t', np.arange(len(tpl_x))), float)
            
            # Estimate template fs
            fs_tpl = (len(tpl_t) - 1) / (tpl_t[-1] - tpl_t[0]) if tpl_t.size > 1 and (tpl_t[-1] - tpl_t[0]) > 0 else 1.0
            
            # Preprocess template
            tpl_x_prep = prep_signal(tpl_x, fs=fs_tpl, apply_filter=True, band=band_hz)
            
            # Resample to grid (scale template time to segment duration)
            if tpl_t.size > 1:
                tpl_t_norm = (tpl_t - tpl_t[0]) / (tpl_t[-1] - tpl_t[0]) * segment_duration_s
            else:
                tpl_t_norm = np.linspace(0, segment_duration_s, len(tpl_x_prep))
            
            y_tpl = resample_to_grid(tpl_x_prep, tpl_t_norm, t_common)
            
            # Plot template
            col = template_colors[template_idx % len(template_colors)]
            template_idx += 1
            
            peak_corr = row.get("peak_corr", None)
            label = f"{station_upper} — {sim} {csv_tpl.upper()}"
            if peak_corr is not None and not np.isnan(peak_corr):
                label += f" (r={peak_corr:.2f})"
            
            ax.plot(t_common / 60, y_tpl, color=col, lw=1.8, ls='--',
                   alpha=0.9, label=label, zorder=6)
    
    # Mark peak times
    for _, row in event_group.iterrows():
        pk = pd.to_datetime(row["peak_time_dt"])
        station = str(row["station"]).upper()
        xpk = (pk - start_time).total_seconds() / 60
        color = station_colors.get(station, "#999999")
        ax.axvline(xpk, color=color, ls=":", lw=1.6, alpha=0.85)
        ax.text(xpk, ax.get_ylim()[1] * 0.96, station, rotation=90,
               ha="right", va="top", fontsize=8, color=color, alpha=0.95)
    
    # Mark P-wave arrivals
    p_wave_plotted = False
    for i, pw in enumerate(p_waves_in_window):
        x_pw = (pw['time'] - start_time).total_seconds() / 60
        ax.axvline(x_pw, color='#FF1493', ls='--', lw=2.5, alpha=0.8, zorder=10,
                  label='P-wave arrival' if not p_wave_plotted else None)
        p_wave_plotted = True
        
        # Annotate with magnitude
        mag = pw['magnitude']
        if not np.isnan(mag):
            ax.text(x_pw, ax.get_ylim()[0] * 0.95, f"M{mag:.1f}", 
                   rotation=90, ha="left", va="bottom", fontsize=9,
                   color='#FF1493', fontweight='bold', alpha=0.9)
    
    # Labels and title
    event_time_str = pd.to_datetime(earliest_peak).strftime("%Y-%m-%d %H:%M:%S")
    
    # Get quality metrics
    avg_corr = event_group['avg_correlation'].iloc[0] if 'avg_correlation' in event_group.columns else None
    avg_snr = event_group['avg_snr_linear'].iloc[0] if 'avg_snr_linear' in event_group.columns else None
    
    quality_str = ""
    if avg_corr is not None and avg_snr is not None:
        quality_str = f" | Avg corr={avg_corr:.3f}, SNR={avg_snr:.2f}"
    
    ax.set_title(
        f"Synchronous Event {event_id} @ {event_time_str}\n"
        f"Stations: {', '.join(sorted(plotted_stations))} | "
        f"BP {band_hz[0]}–{band_hz[1]} Hz | Templates 1-3{quality_str}",
        fontsize=12, fontweight="bold", pad=8
    )
    ax.set_xlabel("Time (minutes)", fontsize=11)
    ax.set_ylabel("Tilt (detrended, filtered)", fontsize=11)
    ax.grid(True, ls=":", alpha=0.35)
    
    # Legend
    leg = ax.legend(loc="upper right", fontsize=9, framealpha=1.0, facecolor="white")
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.9)
    
    plt.tight_layout()
    
    # Save
    out_path = Path(output_dir) / f"sync_event_{event_id:04d}.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# ==========================
# Main Execution
# ==========================
def main():
    print("="*60)
    print("Finding Synchronous Events (Templates 1-4)")
    print("="*60)
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load merged peaks
    print(f"\nLoading peaks from: {MERGED_CSV}")
    df = pd.read_csv(MERGED_CSV)
    print(f"Total peaks: {len(df)}")
    
    # Filter to templates 1-3
    df = df[df['template'].isin(TEMPLATES_TO_INCLUDE)].copy()
    print(f"Peaks with templates 1-3: {len(df)}")
    
    # Convert peak_time_dt to datetime
    df['peak_time_dt'] = pd.to_datetime(df['peak_time_dt'])
    
    # Find synchronous events
    print(f"\nFinding synchronous events (tolerance={SYNC_TOLERANCE_SEC}s, min_stations={MIN_STATIONS})...")
    df_events = find_synchronous_events(df, SYNC_TOLERANCE_SEC, MIN_STATIONS)
    
    n_events_raw = df_events['event_id'].nunique()
    print(f"Found {n_events_raw} raw synchronous events")
    
    # Apply quality filters
    print(f"\nApplying quality filters:")
    print(f"  - Minimum average correlation: {MIN_AVG_CORRELATION}")
    print(f"  - Minimum average SNR (linear): {MIN_AVG_SNR_LINEAR}")
    
    df_events, event_stats = filter_events_by_quality(
        df_events, MIN_AVG_CORRELATION, MIN_AVG_SNR_LINEAR
    )
    
    n_events = df_events['event_id'].nunique()
    n_filtered = n_events_raw - n_events
    print(f"Events after filtering: {n_events} ({n_filtered} filtered out)")
    
    if n_events == 0:
        print("\nNo events passed quality filters. Exiting.")
        return
    
    # Screen out earthquake-contaminated events
    df_events, screened_ids, quakes = screen_earthquake_contamination(
        df_events, EARTHQUAKE_CSV, P_WAVE_BEFORE_SEC, P_WAVE_AFTER_SEC
    )
    
    n_events_final = df_events['event_id'].nunique()
    n_quake_screened = n_events - n_events_final
    print(f"Events after earthquake screening: {n_events_final} ({n_quake_screened} screened)")
    
    if n_events_final == 0:
        print("\nNo events remaining after screening. Exiting.")
        return
    
    # Save synchronous events CSV
    csv_out = Path(OUTPUT_DIR) / "synchronous_events.csv"
    df_events.to_csv(csv_out, index=False)
    print(f"\nSaved events CSV: {csv_out}")
    
    # Save screened events for reference
    if screened_ids:
        screened_csv = Path(OUTPUT_DIR) / "earthquake_contaminated_events.csv"
        df_all = find_synchronous_events(df, SYNC_TOLERANCE_SEC, MIN_STATIONS)
        df_all, _ = filter_events_by_quality(df_all, MIN_AVG_CORRELATION, MIN_AVG_SNR_LINEAR)
        df_screened = df_all[df_all['event_id'].isin(screened_ids)]
        df_screened.to_csv(screened_csv, index=False)
        print(f"Saved screened events CSV: {screened_csv}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("Event Summary")
    print("="*60)
    for event_id in sorted(df_events['event_id'].unique()):
        event_group = df_events[df_events['event_id'] == event_id]
        stations = sorted(event_group['station'].unique())
        templates = sorted(event_group['template'].unique())
        time_span = (event_group['peak_time_dt'].max() - event_group['peak_time_dt'].min()).total_seconds()
        avg_corr = event_group['avg_correlation'].iloc[0]
        avg_snr = event_group['avg_snr_linear'].iloc[0]
        print(f"Event {event_id}: {len(stations)} stations, {len(event_group)} peaks, "
              f"span={time_span:.1f}s, templates={templates}, "
              f"avg_corr={avg_corr:.3f}, avg_snr={avg_snr:.2f}")
    
    # Load templates
    print(f"\n{'='*60}")
    print("Loading templates...")
    templates, all_stations, all_sims = load_tilt_templates(TEMPLATES_DIR)
    print(f"Templates: {list(templates.keys())}")
    
    # Load denoised data for each station
    print("\nLoading denoised data...")
    denoised_data = {}
    for station in ["EEC1", "ECPN", "ECIT", "EC10", "ECOR", "EMAS", "EC1"]:
        
        df_den = load_denoised_station(station, DENOISED_DIR)
        if df_den is not None:
            denoised_data[station] = df_den
            print(f"  {station}: {len(df_den)} samples")
    
    # Plot each synchronous event
    print(f"\n{'='*60}")
    print("Plotting synchronous events...")
    for event_id in sorted(df_events['event_id'].unique()):
        event_group = df_events[df_events['event_id'] == event_id]
        plot_synchronous_event(
            event_id, event_group, templates, denoised_data,
            SEGMENT_DURATION_S, OUTPUT_DIR, BAND_HZ, quakes_df=quakes,
            p_wave_before_sec=P_WAVE_BEFORE_SEC, p_wave_after_sec=P_WAVE_AFTER_SEC
        )
    
    print(f"\n{'='*60}")
    print(f"Complete! Results saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
