#!/usr/bin/env python3
import sys; print("FILE LOADING STARTED", flush=True, file=sys.stderr)
"""
SWCC with P-wave Filtering and Cumulative Analysis (ALL STATIONS) - OPTIMIZED

This script:
1. Loads denoised tilt data
2. Runs sliding window cross-correlation on FULL signal (single pass)
3. Identifies peaks that overlap with P-wave exclusion zones
4. Separates peaks into contaminated (in P-wave zones) vs eligible (clean)
5. Creates plots showing:
   - Correlation time series (contaminated peaks in yellow, eligible in green/red)
   - Cumulative correlation over time
   - Cumulative SNR over time (linear scale)
   - Volcanic events overlay (INGV only - red=start, green=stop)
6. Exports peaks and P-wave impact comparison

Note: Uses P-wave window flags to distinguish clean vs contaminated windows.

Focus: INGV (ECPN, EC1) and Experiment (EC1, ECIT, EMAS, EC10, ECOR) stations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Input paths
BANDPASSED_CSV_ROOT = "/home/owen/tilt_validation/bandpassed_exports_001_01_csv_only_utc"
TEMPLATES_CSV_ROOT = "/home/owen/tilt_validation/tilt_templates_csv"
EARTHQUAKE_CSV = "/home/owen/tilt_validation/earthquakes_merged_utc.csv"

# Output paths
OUTPUT_ROOT = "/home/owen/tilt_validation/SWCC_utc_fixed"

# SWCC parameters
THRESHOLD = 0.2  # Minimum correlation for peak detection
SAMPLING_RATE = 1.0  # Hz

# Performance optimization flags
SAVE_CORRELATION_CSV = False  # Keep False to avoid exporting full correlation timeseries (280MB per template!)

# P-wave filtering
USE_DISTANCE_BASED_BUFFER = True
DEFAULT_P_WAVE_BUFFER_MIN = 10.0

# Volcanic events overlay (INGV only)
VOLCANIC_EVENTS_CSV = "/home/owen/tilt_validation/etna_volcanic_events_cleaned.csv"

# Datasets and stations to analyze
DATASETS_STATIONS = {
    #"experiment": ["EC1"]
    "experiment": ["EC1", "ECIT", "EMAS", "EC10", "ECOR"],  # All 5 experiment stations
    #"ingv": ["ECPN", "EC1"]  # EC1 uses EEC1 bandpassed files (mapped in load function)
}

# SNR calculation parameters
SNR_GUARD_S = 3000.0
SNR_HALFWIN_S = 10000.0

# ============================================================================
# Helper Functions
# ============================================================================

from pwave_buffer import calculate_pwave_buffers


def load_bandpassed_signal(dataset, station):
    """Load bandpassed signal from CSV (matches swcc_fix.py format)."""
    # Map EC1 -> EEC1 for INGV bandpassed files only (templates use EC1, bandpassed uses EEC1)
    station_bp = "EEC1" if (station == "EC1" and dataset == "ingv") else station
    csv_path = Path(BANDPASSED_CSV_ROOT) / dataset / f"{station_bp}_0p001-0p01Hz_bp.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Bandpassed CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Use time_seconds column (matches swcc_fix.py)
    time_seconds = df['time_seconds'].values
    signal = df['bandpassed'].values

    # Convert time_seconds to datetime using start_datetime reference
    # INGV start: 2022-11-14 23:00:00 (from swcc_fix.py line 4831)
    if dataset == "ingv":
        start_dt = pd.Timestamp("2022-11-14 22:00:00")  # UTC+1 clock → UTC
    else:  # experiment
        start_dt = pd.Timestamp("2023-07-23 23:00:00")  # UTC+1 clock → UTC

    # Reconstruct datetime array
    time_dt = start_dt + pd.to_timedelta(time_seconds, unit='s')

    return time_dt.values, signal


def load_template(dataset, station, sim, template):
    """Load template from CSV (flat directory structure)."""
    # Templates are stored as: ECPN_sim1_template1_0p001-0p01Hz_tpl.csv
    csv_path = Path(TEMPLATES_CSV_ROOT) / dataset / f"{station}_{sim}_{template}_0p001-0p01Hz_tpl.csv"

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    # Template CSV has 'x' column for signal values
    return df['x'].values


def remove_p_wave_contaminated_periods(time_dt, signal, earthquake_df):
    """
    Remove all time periods that overlap with P-wave exclusion zones.
    Returns cleaned signal with gaps (as separate segments).
    """
    if earthquake_df is None or len(earthquake_df) == 0:
        return time_dt, signal, np.ones(len(time_dt), dtype=bool)

    # Create mask: True = clean, False = contaminated
    clean_mask = np.ones(len(time_dt), dtype=bool)

    # Mark all contaminated periods
    for _, quake in earthquake_df.iterrows():
        p_time = pd.to_datetime(quake['p_wave_eta'])

        magnitude = quake.get('magnitude', None)
        dist_km   = quake.get('distance_km', None) if USE_DISTANCE_BASED_BUFFER else None
        if dist_km is not None and pd.isna(dist_km):
            dist_km = None
        pre_min, post_min = calculate_pwave_buffers(dist_km, magnitude)

        exclusion_start = p_time - pd.Timedelta(minutes=pre_min)
        exclusion_end   = p_time + pd.Timedelta(minutes=post_min)

        contaminated = (time_dt >= exclusion_start) & (time_dt <= exclusion_end)
        clean_mask &= ~contaminated

    # Return only clean samples
    return time_dt[clean_mask], signal[clean_mask], clean_mask


def sliding_window_cross_correlation(template, signal):
    """
    Normalized sliding-window cross-correlation (Pearson r).
    Returns correlation coefficient for each window position.
    """
    template = np.asarray(template, float).ravel()
    signal = np.asarray(signal, float).ravel()

    n_template = len(template)
    n_signal = len(signal)

    if n_template > n_signal:
        return np.array([])

    # Normalize template
    template_mean = np.mean(template)
    template_std = np.std(template)
    if template_std == 0:
        return np.array([])
    template_norm = (template - template_mean) / template_std

    # Number of windows
    n_windows = n_signal - n_template + 1
    correlations = np.zeros(n_windows)

    # Simple for-loop - tested to be faster than vectorized approaches for this use case
    # Pre-compute template sum for marginal speedup
    for i in range(n_windows):
        window = signal[i:i + n_template]
        window_mean = window.mean()
        window_std = window.std()

        if window_std == 0:
            correlations[i] = 0.0
        else:
            window_norm = (window - window_mean) / window_std
            correlations[i] = np.dot(template_norm, window_norm) / n_template

    return correlations


def compute_template_snr_against_signal(signal, peak_idx, template_length, fs,
                                       noise_guard_s=3000.0, noise_halfwin_s=10000.0):
    """
    Compute SNR at a peak location by comparing signal power to nearby noise power.
    """
    n = len(signal)
    i0 = peak_idx
    i1 = min(peak_idx + template_length, n)

    # Signal segment
    signal_segment = signal[i0:i1]
    signal_power = np.mean(signal_segment**2) if len(signal_segment) > 0 else 0.0

    # Noise segments (before and after, with guard bands)
    guard_samples = int(noise_guard_s * fs)
    halfwin_samples = int(noise_halfwin_s * fs)

    noise_before_start = max(0, i0 - guard_samples - halfwin_samples)
    noise_before_end = max(0, i0 - guard_samples)
    noise_after_start = min(n, i1 + guard_samples)
    noise_after_end = min(n, i1 + guard_samples + halfwin_samples)

    noise_segments = []
    if noise_before_end > noise_before_start:
        noise_segments.append(signal[noise_before_start:noise_before_end])
    if noise_after_end > noise_after_start:
        noise_segments.append(signal[noise_after_start:noise_after_end])

    if len(noise_segments) == 0:
        return np.nan

    noise_all = np.concatenate(noise_segments)
    noise_power = np.mean(noise_all**2) if len(noise_all) > 0 else 1e-12

    # SNR in dB
    snr_linear = np.sqrt(signal_power / noise_power) if noise_power > 0 else 0.0
    snr_db = 20 * np.log10(snr_linear) if snr_linear > 0 else -np.inf

    return snr_db


def load_volcanic_events(csv_path: str) -> pd.DataFrame:
    """
    Load volcanic events from cleaned CSV file.

    Expected columns: Date, Time, End Date, End Time, Phase, Event_Type

    Parameters:
    -----------
    csv_path : str
        Path to volcanic events CSV file

    Returns:
    --------
    pd.DataFrame
        DataFrame with parsed start_datetime, end_datetime, and duration columns
    """
    df = pd.read_csv(csv_path)

    # Parse start and end datetimes
    df['start_datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['end_datetime'] = pd.to_datetime(df['End Date'] + ' ' + df['End Time'])

    # Add 'datetime' column for compatibility with plotting function
    df['datetime'] = df['start_datetime']

    # Calculate duration in hours
    df['duration_hours'] = (df['end_datetime'] - df['start_datetime']).dt.total_seconds() / 3600

    # Determine if it's a single instance or time window
    # If duration > 1 hour, it's a time window
    df['is_time_window'] = df['duration_hours'] > 1.0

    return df


def get_event_type_abbreviation(event_type: str) -> str:
    """Map event types to abbreviations."""
    abbrev_map = {
        'Vent Opening': 'VO',
        'Lava Flow': 'LF',
        'Weather Obscuration': 'WO',
        'Hornito Formation': 'HF',
        'Hornito Activity': 'HA',
        'Lava Flow Advance': 'LFA',
        'Volume Estimate': 'VE',
        'Effusion Decrease': 'ED',
        'Effusion Stop': 'ES',
        'Effusion Renewal': 'ER',
        'Variable Effusion': 'VEf',
        'Ash Emission': 'AE',
        'Degassing': 'DG',
        'Incandescence': 'IN',
        'Effusion Increase': 'EI',
        'Effusion Restart': 'ERs',
        'Cooling Begins': 'CB',
        'Effusion End': 'EE',
        'Aviation Code Change': 'ACC'
    }
    return abbrev_map.get(event_type, 'EFF')  # EFF = Effusive (default for unmapped events)


def plot_volcanic_events_on_swcc(ax, events_df: pd.DataFrame, time_range: tuple = None):
    """
    Plot volcanic events as vertical lines on an existing SWCC plot.
    Color codes by phase: red for 'start', green for 'stop'.
    Plots time windows as shaded regions for events with duration.
    Intelligently positions annotations to avoid overlap.

    Parameters:
    -----------
    ax : matplotlib axis
        The axis to plot on
    events_df : pd.DataFrame
        DataFrame with volcanic events (must have 'datetime', 'Event_Type', 'Phase' columns)
    time_range : tuple of datetime, optional
        (start_time, end_time) to filter events within the plot range

    Returns:
    --------
    int : Number of events plotted
    """
    if events_df is None or len(events_df) == 0:
        return 0

    # Filter events to plot time range if provided
    if time_range is not None:
        start_time, end_time = time_range
        # Convert to pandas Timestamp if needed
        if not isinstance(start_time, pd.Timestamp):
            start_time = pd.Timestamp(start_time)
        if not isinstance(end_time, pd.Timestamp):
            end_time = pd.Timestamp(end_time)

        mask = (events_df['datetime'] >= start_time) & (events_df['datetime'] <= end_time)
        events_to_plot = events_df[mask].copy()
    else:
        events_to_plot = events_df.copy()

    if len(events_to_plot) == 0:
        return 0

    # Get y-axis limits
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin

    # Sort events by datetime to handle overlaps properly
    events_to_plot = events_to_plot.sort_values('datetime').reset_index(drop=True)

    # Track annotation positions to avoid overlap
    annotation_positions = []
    min_time_spacing = pd.Timedelta(hours=12)  # Minimum time between annotations

    def find_non_overlapping_y(event_time, used_positions):
        """Find a y position that doesn't overlap with nearby annotations"""
        # Define possible y positions (relative to ymax)
        y_levels = [0.95, 0.88, 0.81, 0.74, 0.67, 0.60]

        for y_frac in y_levels:
            y_pos = ymin + y_range * y_frac
            # Check if this position is clear (no nearby annotations in time)
            is_clear = True
            for used_time, used_y in used_positions:
                time_diff = abs((event_time - used_time).total_seconds())
                if time_diff < min_time_spacing.total_seconds() and abs(used_y - y_pos) < y_range * 0.05:
                    is_clear = False
                    break
            if is_clear:
                return y_pos

        # If all positions are taken, use the top one anyway
        return ymin + y_range * 0.95

    # Color mapping for phase
    phase_colors = {
        'start': 'red',
        'stop': 'green'
    }

    # Plot each event
    for idx, row in events_to_plot.iterrows():
        event_time = row['datetime']
        event_type = row['Event_Type']
        phase = row.get('Phase', 'start').lower()
        abbrev = get_event_type_abbreviation(event_type)

        # Get color based on phase
        color = phase_colors.get(phase, 'purple')

        # Check if this is a time window event
        is_time_window = row.get('is_time_window', False)

        if is_time_window and 'end_datetime' in row:
            # Plot as shaded region for time windows
            end_time = row['end_datetime']
            ax.axvspan(event_time, end_time, color=color, alpha=0.2, zorder=1)
            # Add vertical lines at start and end
            ax.axvline(event_time, color=color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
            ax.axvline(end_time, color=color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
        else:
            # Plot as vertical line for instantaneous events
            ax.axvline(event_time, color=color, linestyle=':', linewidth=1.5, alpha=0.6, zorder=2)

        # Find non-overlapping position for annotation
        y_pos = find_non_overlapping_y(event_time, annotation_positions)
        annotation_positions.append((event_time, y_pos))

        # Annotate with abbreviation
        ax.text(event_time, y_pos, abbrev,
                rotation=90, verticalalignment='bottom', horizontalalignment='right',
                fontsize=7, color=color, alpha=0.9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.7, linewidth=0.5))

    return len(events_to_plot)


def plot_swcc_standalone(time_dt, correlation, peaks, peak_correlations,
                         template_name, sim, output_dir,
                         dataset, station,
                         volcanic_events_df=None,
                         correlation_contaminated=None, time_dt_contaminated=None,
                         peaks_contaminated=None, peak_correlations_contaminated=None,
                         peaks_eligible=None, peak_correlations_eligible=None):
    """
    Create standalone SWCC plot showing dual correlation overlay
    - Contaminated (orange) overlaid with Clean (blue)
    - Contaminated peaks (yellow, smaller), Eligible peaks (green/red, on top)
    - Volcanic events overlay (INGV only)

    Professional styling with improved colors, fonts, and layout.
    """
    # Set professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(1, 1, figsize=(16, 5), facecolor='white')
    ax.set_facecolor('#f8f9fa')

    # Professional color palette
    color_contaminated = '#ff8c42'  # Warm orange
    color_clean = '#2563eb'  # Professional blue
    color_peaks_contaminated = '#ffd93d'  # Bright yellow
    color_peaks_low = '#00FFFF'  # Cyan (changed from green)
    color_peaks_high = '#ff0000'  # Bright pure red (more vibrant)

    # Plot contaminated data FIRST - full time series
    if correlation_contaminated is not None and time_dt_contaminated is not None:
        ax.plot(time_dt_contaminated, np.abs(correlation_contaminated),
                color=color_contaminated, linewidth=1.0, alpha=0.7,
                label='Contaminated (with P-waves)', zorder=1)

    # Plot clean data on TOP - overlays orange where data exists
    ax.plot(time_dt, np.abs(correlation), color=color_clean, linewidth=1.2, alpha=0.95,
            label='Cleaned (P-waves removed)', zorder=10)

    # Reference threshold lines with improved styling
    threshold_color = '#1f2937' if dataset.lower() == 'ingv' else '#dc2626'
    ax.axhline(THRESHOLD, color=threshold_color, linestyle='--', linewidth=2.0, alpha=0.8,
               label=f'Detection Threshold (r = {THRESHOLD})', zorder=14)
    ax.axhline(0.5, color='#1e40af', linestyle='--', linewidth=2.5, alpha=0.85,
               label='High Correlation (r = 0.5)', zorder=14)
    ax.axhline(0.7, color='#7c3aed', linestyle='--', linewidth=2.5, alpha=0.85,
               label='Very High Correlation (r = 0.7)', zorder=14)

    # Plot contaminated peaks
    if peaks_contaminated is not None and len(peaks_contaminated) > 0:
        peak_times_cont = time_dt_contaminated[peaks_contaminated]
        ax.scatter(peak_times_cont, peak_correlations_contaminated, s=30,
                  c=color_peaks_contaminated, marker='D', zorder=11,
                  edgecolors='#92400e', linewidths=0.5, alpha=0.8,
                  label=f'Contaminated Peaks (n={len(peaks_contaminated)})')

    # Plot eligible peaks - separate by correlation value
    if peaks_eligible is not None and len(peaks_eligible) > 0:
        peak_times_elig = time_dt[peaks_eligible]

        # Separate into 0.2-0.5 and >0.5
        mask_low = (peak_correlations_eligible >= THRESHOLD) & (peak_correlations_eligible < 0.5)
        mask_high = peak_correlations_eligible >= 0.5

        # Plot moderate correlation peaks (0.2-0.5)
        if np.any(mask_low):
            ax.scatter(peak_times_elig[mask_low], peak_correlations_eligible[mask_low],
                      s=40, c=color_peaks_low, marker='o', zorder=12,
                      edgecolors='black', linewidths=1.5, alpha=1.0,
                      label=f'Clean Peaks: Moderate (r = 0.2–0.5, n={np.sum(mask_low)})')

        # Plot high correlation peaks (>0.5) - LARGER and MORE PROMINENT
        if np.any(mask_high):
            ax.scatter(peak_times_elig[mask_high], peak_correlations_eligible[mask_high],
                      s=150, c=color_peaks_high, marker='*', zorder=15,
                      edgecolors='#8b0000', linewidths=1.2, alpha=1.0,
                      label=f'Clean Peaks: High (r > 0.5, n={np.sum(mask_high)})')

    # Add volcanic events overlay for INGV dataset only
    num_volcanic_events = 0
    if dataset.lower() == "ingv" and volcanic_events_df is not None and len(volcanic_events_df) > 0:
        time_range = (time_dt[0] if isinstance(time_dt, np.ndarray) else time_dt.iloc[0],
                      time_dt[-1] if isinstance(time_dt, np.ndarray) else time_dt.iloc[-1])
        num_volcanic_events = plot_volcanic_events_on_swcc(ax, volcanic_events_df, time_range)

    # Professional axis styling
    ax.set_ylabel('Correlation Coefficient |r|', fontsize=13, fontweight='600', color='#1f2937')
    ax.set_xlabel('Time (UTC)', fontsize=13, fontweight='600', color='#1f2937')

    # Cleaner title
    title_parts = [f'{dataset.upper()} – Station {station}',
                   f'{sim.upper().replace("SIM", "Simulation ")} – {template_name.replace("template", "Template ")}']
    if num_volcanic_events > 0:
        title_parts.append(f'{num_volcanic_events} Volcanic Events')
    title_text = ' | '.join(title_parts)
    ax.set_title(title_text, fontsize=14, fontweight='700', pad=15, color='#111827')

    # Improved grid
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='#9ca3af')
    ax.set_axisbelow(True)

    # Better legend
    ax.legend(loc='upper right', fontsize=8.5, ncol=2, frameon=True,
             fancybox=True, shadow=True, framealpha=0.95, edgecolor='#d1d5db')

    ax.set_ylim(0, 1.05)
    ax.set_xlim(time_dt_contaminated[0] if time_dt_contaminated is not None else time_dt[0],
                time_dt_contaminated[-1] if time_dt_contaminated is not None else time_dt[-1])

    # Improve tick labels
    ax.tick_params(axis='both', labelsize=10, colors='#374151')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    # Save with higher DPI for publication quality
    output_file = output_dir / f"{station}_{sim}_{template_name}_swcc_standalone.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.style.use('default')  # Reset style

    print(f"    ✅ Saved standalone SWCC: {output_file.name}")


def plot_swcc_with_cumulative(time_dt, correlation, peaks, peak_correlations,
                              template_name, sim, output_dir,
                              dataset, station,
                              snr_values=None, cumulative_corr=None, cumulative_snr_linear=None,
                              volcanic_events_df=None,
                              correlation_contaminated=None, time_dt_contaminated=None,
                              peaks_contaminated=None, peak_correlations_contaminated=None,
                              peaks_eligible=None, peak_correlations_eligible=None,
                              comparison_data=None, pct_time_contaminated=None):
    """
    Create comprehensive plot with:
    - Top: Dual SWCC - Contaminated (orange) overlaid with Clean (blue)
           Contaminated peaks (yellow), Eligible peaks (green/red)
    - Bottom: Dual-axis cumulative plot (Correlation + SNR)
    - Volcanic events overlay (INGV only)

    Professional styling with improved colors, fonts, and layout.
    """
    # Set professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 10), facecolor='white')

    # Create grid spec for 2-panel layout
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.25)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Professional color palette (same as standalone)
    color_contaminated = '#ff8c42'
    color_clean = '#2563eb'
    color_peaks_contaminated = '#ffd93d'
    color_peaks_low = '#00FFFF'  # Cyan (changed from green)
    color_peaks_high = '#ff0000'  # Bright pure red (more vibrant)

    # Top: Correlation time series with DUAL SWCC overlay
    ax1.set_facecolor('#f8f9fa')

    # Plot contaminated data FIRST
    if correlation_contaminated is not None and time_dt_contaminated is not None:
        ax1.plot(time_dt_contaminated, np.abs(correlation_contaminated),
                color=color_contaminated, linewidth=1.0, alpha=0.7,
                label='Contaminated (with P-waves)', zorder=1)

    # Plot clean data on TOP
    ax1.plot(time_dt, np.abs(correlation), color=color_clean, linewidth=1.2, alpha=0.95,
             label='Cleaned (P-waves removed)', zorder=10)

    # Reference threshold lines
    threshold_color = '#1f2937' if dataset.lower() == 'ingv' else '#dc2626'
    ax1.axhline(THRESHOLD, color=threshold_color, linestyle='--', linewidth=2.0, alpha=0.8,
                label=f'Detection Threshold (r = {THRESHOLD})', zorder=14)
    ax1.axhline(0.5, color='#1e40af', linestyle='--', linewidth=2.5, alpha=0.85,
                label='High Corr. (r = 0.5)', zorder=14)
    ax1.axhline(0.7, color='#7c3aed', linestyle='--', linewidth=2.5, alpha=0.85,
                label='Very High Corr. (r = 0.7)', zorder=14)

    # Plot contaminated peaks
    if peaks_contaminated is not None and len(peaks_contaminated) > 0:
        peak_times_cont = time_dt_contaminated[peaks_contaminated]
        ax1.scatter(peak_times_cont, peak_correlations_contaminated, s=25,
                   c=color_peaks_contaminated, marker='D', zorder=11,
                   edgecolors='#92400e', linewidths=0.5, alpha=0.8,
                   label=f'Contam. Peaks (n={len(peaks_contaminated)})')

    # Plot eligible peaks
    if peaks_eligible is not None and len(peaks_eligible) > 0:
        peak_times_elig = time_dt[peaks_eligible]
        mask_low = (peak_correlations_eligible >= THRESHOLD) & (peak_correlations_eligible < 0.5)
        mask_high = peak_correlations_eligible >= 0.5

        if np.any(mask_low):
            ax1.scatter(peak_times_elig[mask_low], peak_correlations_eligible[mask_low],
                       s=35, c=color_peaks_low, marker='o', zorder=12,
                       edgecolors='black', linewidths=1.5, alpha=1.0,
                       label=f'Moderate (0.2–0.5, n={np.sum(mask_low)})')

        if np.any(mask_high):
            ax1.scatter(peak_times_elig[mask_high], peak_correlations_eligible[mask_high],
                       s=140, c=color_peaks_high, marker='*', zorder=15,
                       edgecolors='#8b0000', linewidths=1.2, alpha=1.0,
                       label=f'High (>0.5, n={np.sum(mask_high)})')

    # Add volcanic events overlay
    num_volcanic_events = 0
    if dataset.lower() == "ingv" and volcanic_events_df is not None and len(volcanic_events_df) > 0:
        time_range = (time_dt[0] if isinstance(time_dt, np.ndarray) else time_dt.iloc[0],
                      time_dt[-1] if isinstance(time_dt, np.ndarray) else time_dt.iloc[-1])
        num_volcanic_events = plot_volcanic_events_on_swcc(ax1, volcanic_events_df, time_range)

    ax1.set_ylabel('Correlation |r|', fontsize=12, fontweight='600', color='#1f2937')
    title_parts = [f'{dataset.upper()} – Station {station}',
                   f'{sim.upper().replace("SIM", "Sim ")} – {template_name.replace("template", "Tpl ")}']
    if num_volcanic_events > 0:
        title_parts.append(f'{num_volcanic_events} Volcanic Events')
    ax1.set_title(' | '.join(title_parts), fontsize=13, fontweight='700', pad=12, color='#111827')
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='#9ca3af')
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper right', fontsize=7.5, ncol=3, frameon=True,
              fancybox=True, shadow=True, framealpha=0.95, edgecolor='#d1d5db')
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='both', labelsize=10, colors='#374151')

    # Middle: Dual-axis cumulative plot (Correlation + SNR)
    ax2.set_facecolor('#f8f9fa')

    # Left axis: Cumulative Correlation
    if cumulative_corr is not None and len(cumulative_corr) > 0:
        ax2.plot(time_dt, cumulative_corr, color='#9333ea', linewidth=2.5, alpha=0.85,
                label='Cumulative Correlation', zorder=2)
        ax2.fill_between(time_dt, 0, cumulative_corr, color='#9333ea', alpha=0.1, zorder=1)
        ax2.set_ylabel('Cumulative Correlation |r|', fontsize=12, fontweight='600', color='#9333ea')
        ax2.tick_params(axis='y', labelcolor='#9333ea', labelsize=10, colors='#9333ea')

    # Right axis: Linear SNR
    ax2_right = ax2.twinx()
    ax2_right.set_facecolor('#f8f9fa')
    if cumulative_snr_linear is not None and len(cumulative_snr_linear) > 0 and len(peaks) > 0:
        peak_times = time_dt[peaks]
        # cumulative_snr_linear now contains linear SNR values directly (not dB)

        ax2_right.plot(peak_times, cumulative_snr_linear, color='#059669', linewidth=2.5, alpha=0.85,
                      marker='o', markersize=3, markerfacecolor='#047857', markeredgecolor='none',
                      label='Linear SNR', zorder=3)

        ax2_right.set_ylabel('SNR (Linear)', fontsize=12, fontweight='600', color='#059669')
        ax2_right.tick_params(axis='y', labelcolor='#059669', labelsize=10, colors='#059669')

        # Add horizontal reference line at SNR = 1.0 (equivalent to 0 dB)
        ax2_right.axhline(1.0, color='#6b7280', linestyle='--', linewidth=1.2, alpha=0.6, zorder=1,
                        label='SNR = 1.0 (0 dB)')

    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='#9ca3af', zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_xlabel('Time (UTC)', fontsize=12, fontweight='600', color='#1f2937')
    ax2.tick_params(axis='x', labelsize=10, colors='#374151')

    # Combined legend for both lines
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9,
              frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#d1d5db')

    plt.setp(ax1.get_xticklabels(), visible=False)  # Hide x-tick labels on top plot

    # Save combined plot with higher DPI
    output_file = output_dir / f"{station}_{sim}_{template_name}_swcc_cumulative.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.style.use('default')  # Reset style

    print(f"    ✅ Saved combined SWCC + Cumulative: {output_file.name}")


def plot_station_pwave_impact_summary(all_comparison_data, dataset, station, output_dir):
    """
    Create station-level P-wave impact summary plot.
    Shows peak counts for all 4 sims, grouped by template (4 templates per sim).

    One plot per station with 4 bins (one per sim).
    """
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(all_comparison_data)

    fig = plt.figure(figsize=(18, 12), facecolor='white')
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # Color scheme
    color_total = '#ff8c42'
    color_clean = '#2563eb'
    color_lost = '#dc2626'

    sims = ['sim1', 'sim2', 'sim3', 'sim4']
    templates = ['template1', 'template2', 'template3', 'template4']

    # ====================
    # Panel 1: Total Peaks per Sim
    # ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#f8f9fa')

    # Group by sim and sum across templates
    sim_totals = df.groupby('sim').agg({
        'n_peaks_contaminated': 'sum',
        'n_peaks_clean': 'sum',
        'n_peaks_lost': 'sum'
    }).reindex(sims)

    x = np.arange(len(sims))
    width = 0.25

    bars1 = ax1.bar(x - width, sim_totals['n_peaks_contaminated'], width,
                    label='Total (with P-waves)', color=color_total, alpha=0.8,
                    edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x, sim_totals['n_peaks_clean'], width,
                    label='Clean', color=color_clean, alpha=0.8,
                    edgecolor='black', linewidth=1.2)
    bars3 = ax1.bar(x + width, sim_totals['n_peaks_lost'], width,
                    label='Lost', color=color_lost, alpha=0.8,
                    edgecolor='black', linewidth=1.2)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Total Peaks (All Templates)', fontsize=12, fontweight='600')
    ax1.set_title(f'Peak Counts by Simulation', fontsize=13, fontweight='700', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.upper() for s in sims])
    ax1.legend(fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_axisbelow(True)

    # ====================
    # Panel 2: Peak Loss Percentage per Sim
    # ====================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#f8f9fa')

    pct_lost = (sim_totals['n_peaks_lost'] / sim_totals['n_peaks_contaminated'] * 100).values

    bars = ax2.bar(sims, pct_lost, color=color_lost, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    for bar, pct in zip(bars, pct_lost):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Peaks Lost (%)', fontsize=12, fontweight='600')
    ax2.set_title('P-wave Filtering Impact (%)', fontsize=13, fontweight='700', pad=10)
    ax2.set_xticklabels([s.upper() for s in sims])
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, 100)

    # ====================
    # Panels 3-6: Peak Counts by Template for Each Sim
    # ====================
    for sim_idx, sim in enumerate(sims):
        ax = fig.add_subplot(gs[(sim_idx // 2) + 1, sim_idx % 2])
        ax.set_facecolor('#f8f9fa')

        # Filter data for this sim
        sim_data = df[df['sim'] == sim].set_index('template').reindex(templates)

        x_templates = np.arange(len(templates))
        width = 0.25

        bars1 = ax.bar(x_templates - width, sim_data['n_peaks_contaminated'], width,
                       label='Total', color=color_total, alpha=0.8, edgecolor='black', linewidth=1.0)
        bars2 = ax.bar(x_templates, sim_data['n_peaks_clean'], width,
                       label='Clean', color=color_clean, alpha=0.8, edgecolor='black', linewidth=1.0)
        bars3 = ax.bar(x_templates + width, sim_data['n_peaks_lost'], width,
                       label='Lost', color=color_lost, alpha=0.8, edgecolor='black', linewidth=1.0)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_ylabel('Peaks', fontsize=11, fontweight='600')
        ax.set_title(f'{sim.upper()} - Peak Counts by Template', fontsize=12, fontweight='700', pad=8)
        ax.set_xticks(x_templates)
        ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
        if sim_idx == 0:  # Only show legend on first panel
            ax.legend(fontsize=8, framealpha=0.95, loc='upper right')
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_axisbelow(True)

    # Main title
    fig.suptitle(f'{dataset.upper()} – Station {station}\nP-wave Contamination Impact Summary (All Simulations & Templates)',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    # Save
    output_file = output_dir / f"{station}_pwave_impact_summary_all_sims.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"    ✅ Saved station-level P-wave impact summary: {output_file.name}")


def plot_pwave_impact(comparison_data, template_name, sim, output_dir, dataset, station):
    """
    Create comprehensive P-wave contamination impact visualization.

    Shows:
    - Peak count comparison (contaminated vs clean)
    - Correlation distribution comparison
    - Peak loss breakdown by correlation threshold
    - Time/sample contamination statistics
    """
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Color scheme
    color_contaminated = '#ff8c42'
    color_clean = '#2563eb'
    color_lost = '#dc2626'

    # ====================
    # Panel 1: Peak Count Bar Chart
    # ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#f8f9fa')

    n_contaminated = comparison_data['n_peaks_contaminated']
    n_clean = comparison_data['n_peaks_clean']
    n_lost = comparison_data['n_peaks_lost']
    pct_lost = comparison_data['pct_peaks_lost']

    categories = ['Total\nPeaks', 'Clean\nPeaks', 'Lost\nPeaks']
    values = [n_contaminated, n_clean, n_lost]
    colors = [color_contaminated, color_clean, color_lost]

    bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}\n({val/n_contaminated*100:.1f}%)' if n_contaminated > 0 else f'{int(val)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Number of Peaks', fontsize=12, fontweight='600')
    ax1.set_title(f'P-wave Filtering Impact on Peak Detection\n({pct_lost:.1f}% peaks lost)',
                 fontsize=13, fontweight='700', pad=10)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_axisbelow(True)

    # ====================
    # Panel 2: Peak Loss Breakdown
    # ====================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#f8f9fa')

    n_in_zone = comparison_data['n_peaks_contaminated_in_pwave_zone']
    n_out_zone = comparison_data['n_peaks_contaminated_outside_pwave_zone']

    # Pie chart of contamination reasons
    labels = [f'In P-wave Zone\n({n_in_zone})', f'Outside Zone\n({n_out_zone})']
    sizes = [n_in_zone, n_out_zone]
    colors_pie = ['#dc2626', '#f59e0b']

    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Contaminated Peak Breakdown', fontsize=13, fontweight='700', pad=10)

    # ====================
    # Panel 3: Correlation Distribution Comparison
    # ====================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#f8f9fa')

    # Bar chart comparing correlation statistics
    metrics = ['Mean', 'Median', 'Max']
    contaminated_vals = [
        comparison_data['peak_corr_mean_contaminated'],
        comparison_data['peak_corr_median_contaminated'],
        comparison_data['peak_corr_max_contaminated']
    ]
    clean_vals = [
        comparison_data['peak_corr_mean_clean'],
        comparison_data['peak_corr_median_clean'],
        comparison_data['peak_corr_max_clean']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax3.bar(x - width/2, contaminated_vals, width, label='With P-waves',
                    color=color_contaminated, alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax3.bar(x + width/2, clean_vals, width, label='P-waves Removed',
                    color=color_clean, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3.set_ylabel('Correlation Coefficient |r|', fontsize=12, fontweight='600')
    ax3.set_title('Peak Correlation Statistics Comparison', fontsize=13, fontweight='700', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend(fontsize=10, framealpha=0.95)
    ax3.grid(True, alpha=0.2, axis='y')
    ax3.set_axisbelow(True)
    ax3.set_ylim(0, 1.0)

    # ====================
    # Panel 4: High Correlation Peaks by Threshold
    # ====================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#f8f9fa')

    # Compare peaks above different thresholds
    thresholds = ['r ≥ 0.5', 'r ≥ 0.7']
    contaminated_counts = [
        comparison_data['n_windows_above_0.5_contaminated'],
        comparison_data['n_windows_above_0.7_contaminated']
    ]
    clean_counts = [
        comparison_data['n_windows_above_0.5_clean'],
        comparison_data['n_windows_above_0.7_clean']
    ]

    x = np.arange(len(thresholds))
    width = 0.35

    bars1 = ax4.bar(x - width/2, contaminated_counts, width, label='With P-waves',
                    color=color_contaminated, alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax4.bar(x + width/2, clean_counts, width, label='P-waves Removed',
                    color=color_clean, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax4.set_ylabel('Number of Windows', fontsize=12, fontweight='600')
    ax4.set_title('High Correlation Windows Above Thresholds', fontsize=13, fontweight='700', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(thresholds)
    ax4.legend(fontsize=10, framealpha=0.95)
    ax4.grid(True, alpha=0.2, axis='y')
    ax4.set_axisbelow(True)

    # Main title
    fig.suptitle(f'{dataset.upper()} – Station {station} | {sim.upper()} – {template_name}\nP-wave Contamination Impact Analysis',
                fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_file = output_dir / f"{station}_{sim}_{template_name}_pwave_impact.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"    ✅ Saved P-wave impact plot: {output_file.name}")


def plot_cumulative_standalone(time_dt, correlation, peaks, peak_correlations,
                               template_name, sim, output_dir,
                               dataset, station,
                               cumulative_corr=None, cumulative_snr_linear=None):
    """
    Create standalone dual-axis cumulative plot (Correlation + Linear SNR).

    Plots LINEAR SNR directly instead of converting to dB.
    Professional styling with improved colors, fonts, and compact legend in upper left.
    """
    # Set professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(1, 1, figsize=(16, 6), facecolor='white')
    ax.set_facecolor('#f8f9fa')

    # Left axis: Cumulative Correlation
    if cumulative_corr is not None and len(cumulative_corr) > 0:
        ax.plot(time_dt, cumulative_corr, color='#9333ea', linewidth=2.5, alpha=0.85,
                label='Cumulative Correlation', zorder=2)
        ax.fill_between(time_dt, 0, cumulative_corr, color='#9333ea', alpha=0.1, zorder=1)
        ax.set_ylabel('Cumulative Correlation |r|', fontsize=13, fontweight='600', color='#9333ea')
        ax.tick_params(axis='y', labelcolor='#9333ea', labelsize=10, colors='#9333ea')

    # Right axis: Linear SNR (plots snr_linear values directly)
    ax_right = ax.twinx()
    ax_right.set_facecolor('#f8f9fa')
    if cumulative_snr_linear is not None and len(cumulative_snr_linear) > 0 and len(peaks) > 0:
        peak_times = time_dt[peaks]
        # cumulative_snr_linear now contains linear SNR values directly (not dB)

        ax_right.plot(peak_times, cumulative_snr_linear, color='#059669', linewidth=2.5, alpha=0.85,
                     marker='o', markersize=3, markerfacecolor='#047857', markeredgecolor='none',
                     label='Linear SNR', zorder=3)

        ax_right.set_ylabel('SNR (Linear)', fontsize=13, fontweight='600', color='#059669')
        ax_right.tick_params(axis='y', labelcolor='#059669', labelsize=10, colors='#059669')

        # Add horizontal reference line at SNR = 1.0 (equivalent to 0 dB)
        ax_right.axhline(1.0, color='#6b7280', linestyle='--', linewidth=1.2, alpha=0.6, zorder=1,
                        label='SNR = 1.0 (0 dB)')

    # Axis styling
    ax.set_xlabel('Time (UTC)', fontsize=13, fontweight='600', color='#1f2937')
    ax.tick_params(axis='x', labelsize=10, colors='#374151')

    # Title
    title_parts = [f'{dataset.upper()} – Station {station}',
                   f'{sim.upper().replace("SIM", "Simulation ")} – {template_name.replace("template", "Template ")}',
                   'Cumulative Correlation & Linear SNR']
    ax.set_title(' | '.join(title_parts), fontsize=14, fontweight='700', pad=15, color='#111827')

    # Grid
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='#9ca3af', zorder=0)
    ax.set_axisbelow(True)

    # Combined legend - compact in upper left
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9,
             frameon=True, fancybox=False, shadow=False, framealpha=0.95,
             edgecolor='#9ca3af', borderpad=0.6, labelspacing=0.5)

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    # Save with higher DPI
    output_file = output_dir / f"{station}_{sim}_{template_name}_cumulative_standalone.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.style.use('default')  # Reset style

    print(f"    ✅ Saved standalone cumulative: {output_file.name}")


# ============================================================================
# Main Processing
# ============================================================================

def main():
    import sys
    print("="*80, flush=True, file=sys.stderr)
    print("SWCC WITH P-WAVE PRE-CLEANED DATA (ALL DATASETS & STATIONS)", flush=True, file=sys.stderr)
    print("="*80, flush=True, file=sys.stderr)
    print("\nThis script removes P-wave contaminated periods BEFORE running SWCC.", flush=True, file=sys.stderr)
    print("Gaps will be visible in the time series where P-waves were removed.\n", flush=True, file=sys.stderr)

    # Load earthquake data (shared across all datasets and stations)
    print(f"📊 Loading earthquake data...", flush=True, file=sys.stderr)
    earthquake_df = pd.read_csv(EARTHQUAKE_CSV)
    print(f"📊 Loaded CSV, processing...", flush=True, file=sys.stderr)
    earthquake_df['p_wave_eta'] = pd.to_datetime(earthquake_df['p_wave_eta'])
    print(f"  ✅ Loaded {len(earthquake_df):,} earthquakes\n")

    # Load volcanic events data (for INGV plots only)
    volcanic_events_df = None
    try:
        print(f"📊 Loading volcanic events data...")
        volcanic_events_df = load_volcanic_events(VOLCANIC_EVENTS_CSV)
        print(f"  ✅ Loaded {len(volcanic_events_df):,} volcanic events\n")
    except Exception as e:
        print(f"  ⚠️  Could not load volcanic events: {e}\n")
        volcanic_events_df = None

    # Process each dataset
    for DATASET, STATIONS in DATASETS_STATIONS.items():
        print("\n" + "="*80)
        print(f"PROCESSING DATASET: {DATASET.upper()}")
        print("="*80)
        print(f"Stations: {', '.join(STATIONS)}\n")

        # Track dataset-level summary rows
        dataset_summary_rows = []

        # Process each station in this dataset
        for STATION in STATIONS:
            print("\n" + "="*80)
            print(f"PROCESSING STATION: {STATION}")
            print("="*80 + "\n")

            # Create output directories
            output_root = Path(OUTPUT_ROOT)
            output_dir = output_root / DATASET / STATION
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load bandpassed signal
            print(f"📊 Loading bandpassed signal for {STATION}...")
            try:
                time_dt_full, signal_full = load_bandpassed_signal(DATASET, STATION)
                print(f"  ✅ Loaded {len(signal_full):,} samples\n")
            except FileNotFoundError as e:
                print(f"  ⚠️  {e}")
                print(f"  Skipping {STATION}\n")
                continue

            # ========================================
            # Build P-wave exclusion zones ONCE per station (independent of template length)
            # ========================================
            print(f"\n{'='*80}")
            print(f"Building P-wave exclusion zones for {STATION}...")
            print(f"{'='*80}\n")

            # Build sample-level clean mask from P-wave exclusion zones
            clean_mask = np.ones(len(time_dt_full), dtype=bool)
            if earthquake_df is not None and len(earthquake_df) > 0:
                for _, quake in earthquake_df.iterrows():
                    p_time = pd.to_datetime(quake['p_wave_eta'])

                    magnitude = quake.get('magnitude', None)
                    dist_km   = quake.get('distance_km', None) if USE_DISTANCE_BASED_BUFFER else None
                    if dist_km is not None and pd.isna(dist_km):
                        dist_km = None
                    pre_min, post_min = calculate_pwave_buffers(dist_km, magnitude)

                    exclusion_start = p_time - pd.Timedelta(minutes=pre_min)
                    exclusion_end   = p_time + pd.Timedelta(minutes=post_min)

                    contaminated = (time_dt_full >= exclusion_start) & (time_dt_full <= exclusion_end)
                    clean_mask &= ~contaminated

            # Count contaminated samples
            n_contaminated_samples = np.sum(~clean_mask)
            n_total_samples = len(clean_mask)
            pct_contaminated = 100 * n_contaminated_samples / n_total_samples
            print(f"  ✅ P-wave exclusion zones built:")
            print(f"     Contaminated samples: {n_contaminated_samples:,} / {n_total_samples:,} ({pct_contaminated:.2f}%)")
            print(f"     Clean samples: {np.sum(clean_mask):,} ({100-pct_contaminated:.2f}%)\n")

            # Remove P-wave contaminated periods from signal ONCE (same for all templates)
            print(f"🧹 Removing P-wave contaminated periods from signal...")
            time_dt_clean = time_dt_full[clean_mask]
            signal_clean = signal_full[clean_mask]
            n_removed = len(signal_full) - len(signal_clean)
            pct_removed = 100 * n_removed / len(signal_full)
            print(f"  ✅ Removed {n_removed:,} samples ({pct_removed:.2f}%)")
            print(f"  ✅ Clean signal: {len(signal_clean):,} samples\n")

            # Process each sim and template
            sims = ["sim1", "sim2", "sim3", "sim4"]
            templates = ["template1", "template2", "template3", "template4"]

            all_results = []
            per_station_summary_rows = []

            # Track all contaminated and clean peaks for station-level summaries
            all_contaminated_peaks = []
            all_clean_peaks = []
            station_pwave_impact_data = []  # For station-level summary plot

            for sim in sims:
                print(f"\n{'='*80}")
                print(f"Processing {sim.upper()}")
                print(f"{'='*80}\n")

                sim_dir = output_dir / sim
                sim_dir.mkdir(parents=True, exist_ok=True)

                per_sim_summary_rows = []

                for template_name in templates:
                    print(f"  🔄 {template_name}:")

                    # Load template
                    template = load_template(DATASET, STATION, sim, template_name)
                    if template is None:
                        print(f"    ⚠️  Template not found, skipping")
                        continue

                    # Run SWCC on FULL signal (contaminated, for comparison)
                    print(f"    Running SWCC on full signal (contaminated)...")
                    correlations_full = sliding_window_cross_correlation(template, signal_full)
                    corr_time_dt_full = time_dt_full[:len(correlations_full)]

                    if len(correlations_full) == 0:
                        print(f"    ⚠️  No correlation results on full signal")
                        continue

                    # Run SWCC on CLEANED signal (P-wave periods removed)
                    # Uses station-level signal_clean (same for all templates at this station)
                    print(f"    Running SWCC on cleaned signal...")
                    correlations = sliding_window_cross_correlation(template, signal_clean)

                    if len(correlations) == 0:
                        print(f"    ⚠️  No correlation results on cleaned signal")
                        continue

                    # Get time array for correlation windows (clean signal)
                    corr_time_dt = time_dt_clean[:len(correlations)]

                    # Detect peaks on CLEANED signal
                    # Use distance = 1000 samples (1000s at 1 Hz)
                    # Optimized for Ultra-Long Period (ULP) events with frequencies 0.01-0.001 Hz (periods 100-1000s)
                    distance = 1000
                    peaks, props = find_peaks(np.abs(correlations), height=THRESHOLD, distance=int(distance))
                    peak_heights = props.get("peak_heights", np.array([]))

                    print(f"    Found {len(peaks)} peaks above threshold {THRESHOLD} (cleaned)")

                    # Detect peaks on FULL signal (for comparison)
                    peaks_original, props_original = find_peaks(np.abs(correlations_full), height=THRESHOLD, distance=int(distance))
                    peak_heights_original = props_original.get("peak_heights", np.array([]))

                    print(f"    Found {len(peaks_original)} peaks above threshold {THRESHOLD} (full)")

                    # Mark windows as contaminated if ANY sample in the SWCC window is contaminated
                    # Use the STATION-LEVEL clean_mask computed above (same for all templates)
                    contaminated_samples = (~clean_mask).astype(int)
                    window_len = len(template)
                    if len(contaminated_samples) >= window_len:
                        window_contaminated = np.convolve(
                            contaminated_samples, np.ones(window_len, dtype=int), mode='valid'
                        ) > 0
                    else:
                        window_contaminated = np.array([], dtype=bool)
                    p_wave_window_mask = window_contaminated

                    # Identify which contaminated peaks overlap with P-wave exclusion zones
                    contaminated_mask = p_wave_window_mask[peaks_original] if len(peaks_original) > 0 else np.array([], dtype=bool)

                    # Separate contaminated vs eligible peaks (from FULL signal analysis)
                    peaks_contaminated = peaks_original[contaminated_mask]
                    peak_heights_contaminated = peak_heights_original[contaminated_mask]

                    # Eligible peaks are from the CLEANED signal (all peaks are eligible by definition)
                    peaks_eligible = peaks.copy()
                    peak_heights_eligible = peak_heights.copy()

                    print(f"    Contaminated peaks (in P-wave zones, from full signal): {len(peaks_contaminated)}")
                    print(f"    Eligible peaks (from cleaned signal): {len(peaks_eligible)}")

                    # Compute SNR for eligible peaks only (using cleaned signal)
                    snr_values = []
                    snr_linear_values = []
                    for peak_idx in peaks_eligible:
                        snr_db = compute_template_snr_against_signal(
                            signal_clean, peak_idx, len(template), SAMPLING_RATE,
                            SNR_GUARD_S, SNR_HALFWIN_S
                        )
                        snr_linear = 10.0**(snr_db/20.0) if np.isfinite(snr_db) else np.nan
                        snr_values.append(snr_db)
                        snr_linear_values.append(snr_linear)

                    # Compute cumulative metrics on CLEANED signal only
                    cumulative_corr = np.abs(correlations).cumsum()

                    # Use linear SNR values directly (not dB)
                    # Pass snr_linear_values to plotting function for display
                    if len(snr_linear_values) > 0:
                        # Simply use the linear SNR values directly
                        cumulative_snr_linear = np.array(snr_linear_values)
                    else:
                        cumulative_snr_linear = np.array([])

                    # Add simplified summary row for per-sim CSV (clean windows only)
                    corr_abs = np.abs(correlations)
                    if len(corr_abs) > 0:
                        avg_corr = float(np.mean(corr_abs))
                        min_corr = float(np.min(corr_abs))
                        max_corr = float(np.max(corr_abs))
                    else:
                        avg_corr = np.nan
                        min_corr = np.nan
                        max_corr = np.nan

                    # SNR at best peak (eligible only)
                    if len(peaks) > 0:
                        best_peak_idx = peaks[np.argmax(peak_heights)]
                        snr_db_best = compute_template_snr_against_signal(
                            signal_clean, best_peak_idx, len(template), SAMPLING_RATE,
                            SNR_GUARD_S, SNR_HALFWIN_S
                        )
                    else:
                        snr_db_best = np.nan

                    summary_row = {
                        "dataset": DATASET,
                        "station": STATION,
                        "sim": sim,
                        "template": template_name,
                        "avg_corr": avg_corr,
                        "min_corr": min_corr,
                        "max_corr": max_corr,
                        "snr_db_at_best_peak": float(snr_db_best) if np.isfinite(snr_db_best) else np.nan,
                        "n_peaks_ge_thresh": int(len(peaks)),
                        "threshold": float(THRESHOLD)
                    }
                    per_sim_summary_rows.append(summary_row)
                    per_station_summary_rows.append(summary_row)

                    # ========================================
                    # Create P-wave Filtering Impact Comparison data
                    # ========================================
                    corr_abs_full = np.abs(correlations_full)

                    comparison_data = {
                        'dataset': DATASET,
                        'station': STATION,
                        'sim': sim,
                        'template': template_name,
                        'threshold': THRESHOLD,
                        'peak_distance': distance,

                        # Peak counts
                        'n_peaks_contaminated': len(peaks_original),
                        'n_peaks_clean': len(peaks),
                        'n_peaks_lost': len(peaks_original) - len(peaks),
                        'pct_peaks_lost': 100 * (len(peaks_original) - len(peaks)) / len(peaks_original) if len(peaks_original) > 0 else 0.0,

                        # Contaminated peaks breakdown
                        'n_peaks_contaminated_in_pwave_zone': np.sum(contaminated_mask),
                        'n_peaks_contaminated_outside_pwave_zone': np.sum(~contaminated_mask),

                        # Correlation statistics - Contaminated (full signal)
                        'corr_max_contaminated': float(np.max(corr_abs_full)),
                        'corr_mean_contaminated': float(np.mean(corr_abs_full)),
                        'corr_median_contaminated': float(np.median(corr_abs_full)),
                        'corr_std_contaminated': float(np.std(corr_abs_full)),

                        # Correlation statistics - Clean (cleaned signal)
                        'corr_max_clean': float(np.max(corr_abs)) if len(corr_abs) > 0 else np.nan,
                        'corr_mean_clean': float(np.mean(corr_abs)) if len(corr_abs) > 0 else np.nan,
                        'corr_median_clean': float(np.median(corr_abs)) if len(corr_abs) > 0 else np.nan,
                        'corr_std_clean': float(np.std(corr_abs)) if len(corr_abs) > 0 else np.nan,

                        # Peak correlation statistics - Contaminated
                        'peak_corr_max_contaminated': float(np.max(peak_heights_original)) if len(peak_heights_original) > 0 else np.nan,
                        'peak_corr_mean_contaminated': float(np.mean(peak_heights_original)) if len(peak_heights_original) > 0 else np.nan,
                        'peak_corr_median_contaminated': float(np.median(peak_heights_original)) if len(peak_heights_original) > 0 else np.nan,
                        'peak_corr_std_contaminated': float(np.std(peak_heights_original)) if len(peak_heights_original) > 0 else np.nan,

                        # Peak correlation statistics - Clean (Eligible)
                        'peak_corr_max_clean': float(np.max(peak_heights)) if len(peak_heights) > 0 else np.nan,
                        'peak_corr_mean_clean': float(np.mean(peak_heights)) if len(peak_heights) > 0 else np.nan,
                        'peak_corr_median_clean': float(np.median(peak_heights)) if len(peak_heights) > 0 else np.nan,
                        'peak_corr_std_clean': float(np.std(peak_heights)) if len(peak_heights) > 0 else np.nan,

                        # SNR statistics (clean peaks only)
                        'snr_max_db_clean': float(np.nanmax(snr_values)) if len(snr_values) > 0 else np.nan,
                        'snr_mean_db_clean': float(np.nanmean(snr_values)) if len(snr_values) > 0 else np.nan,
                        'snr_median_db_clean': float(np.nanmedian(snr_values)) if len(snr_values) > 0 else np.nan,
                        'snr_std_db_clean': float(np.nanstd(snr_values)) if len(snr_values) > 0 else np.nan,

                        # Windows above thresholds
                        'n_windows_above_0.5_contaminated': int(np.sum(corr_abs_full >= 0.5)),
                        'n_windows_above_0.5_clean': int(np.sum(corr_abs >= 0.5)),
                        'n_windows_above_0.7_contaminated': int(np.sum(corr_abs_full >= 0.7)),
                        'n_windows_above_0.7_clean': int(np.sum(corr_abs >= 0.7)),
                        'pct_windows_above_0.5_contaminated': float(100 * np.sum(corr_abs_full >= 0.5) / len(corr_abs_full)),
                        'pct_windows_above_0.5_clean': float(100 * np.sum(corr_abs >= 0.5) / len(corr_abs)),
                        'pct_windows_above_0.7_contaminated': float(100 * np.sum(corr_abs_full >= 0.7) / len(corr_abs_full)),
                        'pct_windows_above_0.7_clean': float(100 * np.sum(corr_abs >= 0.7) / len(corr_abs)),

                        # Peak counts by correlation range (clean peaks only)
                        'n_peaks_0.2_to_0.5_clean': int(np.sum((peak_heights >= THRESHOLD) & (peak_heights < 0.5))),
                        'n_peaks_0.5_to_0.7_clean': int(np.sum((peak_heights >= 0.5) & (peak_heights < 0.7))),
                        'n_peaks_above_0.7_clean': int(np.sum(peak_heights >= 0.7)),
                        'pct_peaks_0.2_to_0.5_clean': float(100 * np.sum((peak_heights >= THRESHOLD) & (peak_heights < 0.5)) / len(peak_heights)) if len(peak_heights) > 0 else 0.0,
                        'pct_peaks_0.5_to_0.7_clean': float(100 * np.sum((peak_heights >= 0.5) & (peak_heights < 0.7)) / len(peak_heights)) if len(peak_heights) > 0 else 0.0,
                        'pct_peaks_above_0.7_clean': float(100 * np.sum(peak_heights >= 0.7) / len(peak_heights)) if len(peak_heights) > 0 else 0.0,

                        # Cumulative metrics
                        'cumulative_corr_final': float(cumulative_corr[-1]) if len(cumulative_corr) > 0 else np.nan,
                        'cumulative_snr_linear_final': float(cumulative_snr_linear[-1]) if len(cumulative_snr_linear) > 0 else np.nan,
                        'cumulative_snr_db_final': float(20 * np.log10(cumulative_snr_linear[-1])) if len(cumulative_snr_linear) > 0 and cumulative_snr_linear[-1] > 0 else np.nan,
                    }

                    # Create P-wave impact analysis plot
                    plot_pwave_impact(
                        comparison_data, template_name, sim, sim_dir,
                        DATASET, STATION
                    )

                    # Create standalone SWCC plot
                    plot_swcc_standalone(
                        corr_time_dt, correlations, peaks_eligible, peak_heights_eligible,
                        template_name, sim, sim_dir,
                        DATASET, STATION,
                        volcanic_events_df=volcanic_events_df,
                        correlation_contaminated=correlations_full,
                        time_dt_contaminated=corr_time_dt_full,
                        peaks_contaminated=peaks_contaminated,
                        peak_correlations_contaminated=peak_heights_contaminated,
                        peaks_eligible=peaks_eligible,
                        peak_correlations_eligible=peak_heights_eligible
                    )

                    # Create comprehensive plot with dual SWCC overlay + cumulative (2 panels)
                    plot_swcc_with_cumulative(
                        corr_time_dt, correlations, peaks, peak_heights,
                        template_name, sim, sim_dir,
                        DATASET, STATION,
                        snr_values=np.array(snr_values),
                        cumulative_corr=cumulative_corr,
                        cumulative_snr_linear=cumulative_snr_linear,
                        volcanic_events_df=volcanic_events_df,
                        correlation_contaminated=correlations_full,
                        time_dt_contaminated=corr_time_dt_full,
                        peaks_contaminated=peaks_contaminated,
                        peak_correlations_contaminated=peak_heights_contaminated,
                        peaks_eligible=peaks_eligible,
                        peak_correlations_eligible=peak_heights_eligible,
                        comparison_data=comparison_data,
                        pct_time_contaminated=pct_contaminated
                    )

                    # Create standalone cumulative plot (dual-axis: Correlation + SNR)
                    plot_cumulative_standalone(
                        corr_time_dt, correlations, peaks, peak_heights,
                        template_name, sim, sim_dir,
                        DATASET, STATION,
                        cumulative_corr=cumulative_corr,
                        cumulative_snr_linear=cumulative_snr_linear
                    )

                    # Export raw correlation time series
                    # Optionally save correlation timeseries CSV (can be 280MB per template!)
                    if SAVE_CORRELATION_CSV:
                        raw_corr_df = pd.DataFrame({
                            'window_index': np.arange(len(correlations)),
                            'time_dt': corr_time_dt,
                            'correlation': correlations,
                            'cumulative_correlation': cumulative_corr
                        })
                        raw_corr_csv = sim_dir / f"{STATION}_{sim}_{template_name}_correlation_timeseries.csv"
                        raw_corr_df.to_csv(raw_corr_csv, index=False)
                        print(f"    💾 Saved correlation CSV: {raw_corr_csv.name}")
                    else:
                        print(f"    ⚡ Skipped correlation CSV (SAVE_CORRELATION_CSV=False)")

                    # Export peaks with SNR (CLEAN peaks only)
                    peaks_data = []
                    if len(peaks) > 0:
                        for i, peak_idx in enumerate(peaks):
                            # peak_idx is in clean correlation space
                            peak_record = {
                                'dataset': DATASET,
                                'station': STATION,
                                'sim': sim,
                                'template': template_name,
                                'peak_index': peak_idx,
                                'peak_time_dt': corr_time_dt[peak_idx],
                                'peak_corr': float(np.abs(correlations[peak_idx])),
                                'snr_db': snr_values[i],
                                'snr_linear': snr_linear_values[i],
                                'cumulative_snr_linear': cumulative_snr_linear[i],
                                'cumulative_snr_db': 20 * np.log10(cumulative_snr_linear[i]) if cumulative_snr_linear[i] > 0 else -np.inf,
                                'peak_type': 'clean'
                            }
                            peaks_data.append(peak_record)
                            all_clean_peaks.append(peak_record)  # Collect for station-level CSV

                        peaks_df = pd.DataFrame(peaks_data)
                        peaks_csv = sim_dir / f"{STATION}_{sim}_{template_name}_peaks.csv"
                        peaks_df.to_csv(peaks_csv, index=False)

                    all_results.extend(peaks_data)

                    # Export CONTAMINATED peaks
                    contaminated_peaks_data = []
                    if len(peaks_contaminated) > 0:
                        for i, peak_idx in enumerate(peaks_contaminated):
                            # peak_idx is in full correlation space
                            contaminated_record = {
                                'dataset': DATASET,
                                'station': STATION,
                                'sim': sim,
                                'template': template_name,
                                'peak_index': peak_idx,
                                'peak_time_dt': corr_time_dt_full[peak_idx],
                                'peak_corr': float(peak_heights_contaminated[i]),
                                'snr_db': np.nan,  # SNR not computed for contaminated peaks
                                'snr_linear': np.nan,
                                'cumulative_snr_linear': np.nan,
                                'cumulative_snr_db': np.nan,
                                'peak_type': 'contaminated'
                            }
                            contaminated_peaks_data.append(contaminated_record)
                            all_contaminated_peaks.append(contaminated_record)  # Collect for station-level CSV

                        contaminated_df = pd.DataFrame(contaminated_peaks_data)
                        contaminated_csv = sim_dir / f"{STATION}_{sim}_{template_name}_contaminated_peaks.csv"
                        contaminated_df.to_csv(contaminated_csv, index=False)
                        print(f"    ✅ Saved {len(contaminated_peaks_data)} contaminated peaks: {contaminated_csv.name}")

                    # Save P-wave impact comparison CSV
                    comparison_df = pd.DataFrame([comparison_data])
                    comparison_csv = sim_dir / f"{STATION}_{sim}_{template_name}_pwave_impact_comparison.csv"
                    comparison_df.to_csv(comparison_csv, index=False)
                    print(f"    ✅ Saved P-wave impact comparison: {comparison_csv.name}")

                    # Collect for station-level summary plot
                    station_pwave_impact_data.append(comparison_data)

                    # Print comparison summary
                    print(f"    📊 P-wave Filtering Impact:")
                    print(f"       Contaminated: {len(peaks_contaminated)} peaks ({np.sum(contaminated_mask)} in P-wave zones)")
                    print(f"       Clean: {len(peaks)} eligible peaks")
                    print(f"       Lost: {len(peaks_original) - len(peaks)} peaks ({100 * (len(peaks_original) - len(peaks)) / len(peaks_original) if len(peaks_original) > 0 else 0:.1f}%)")

                print(f"    ✅ Complete: {len(correlations):,} windows, {len(peaks)} peaks above {THRESHOLD}")

                # Save per-sim summary CSV
                if per_sim_summary_rows:
                    per_sim_summary_df = pd.DataFrame(per_sim_summary_rows)
                    per_sim_summary_csv = sim_dir / f"{STATION}_{sim}_summary.csv"
                    per_sim_summary_df.to_csv(per_sim_summary_csv, index=False)
                    print(f"\n  ✅ Saved per-sim summary: {per_sim_summary_csv.name}")

        # Save combined results for this station
        if all_results:
            combined_df = pd.DataFrame(all_results)
            combined_csv = output_dir / f"{STATION}_all_peaks_combined.csv"
            combined_df.to_csv(combined_csv, index=False)
            print(f"\n✅ Saved combined peaks: {combined_csv}")

        # Save station-level CLEAN peaks CSV
        if all_clean_peaks:
            clean_peaks_df = pd.DataFrame(all_clean_peaks)
            clean_peaks_csv = output_dir / f"{STATION}_clean_peaks_all.csv"
            clean_peaks_df.to_csv(clean_peaks_csv, index=False)
            print(f"✅ Saved all clean peaks: {clean_peaks_csv} ({len(all_clean_peaks)} peaks)")

        # Save station-level CONTAMINATED peaks CSV
        if all_contaminated_peaks:
            contaminated_peaks_df = pd.DataFrame(all_contaminated_peaks)
            contaminated_peaks_csv = output_dir / f"{STATION}_contaminated_peaks_all.csv"
            contaminated_peaks_df.to_csv(contaminated_peaks_csv, index=False)
            print(f"✅ Saved all contaminated peaks: {contaminated_peaks_csv} ({len(all_contaminated_peaks)} peaks)")

        # Create station-level P-wave impact summary plot
        if station_pwave_impact_data:
            print(f"\n📊 Creating station-level P-wave impact summary plot...")
            plot_station_pwave_impact_summary(station_pwave_impact_data, DATASET, STATION, output_dir)

        # Save per-station summary (across all sims)
        if per_station_summary_rows:
            per_station_summary_df = pd.DataFrame(per_station_summary_rows)
            per_station_summary_csv = output_dir / f"{STATION}_all_sims_summary.csv"
            per_station_summary_df.to_csv(per_station_summary_csv, index=False)
            print(f"✅ Saved per-station summary: {per_station_summary_csv.name}")
            dataset_summary_rows.extend(per_station_summary_rows)

        # Save dataset-level summary CSV
        if dataset_summary_rows:
            dataset_summary_df = pd.DataFrame(dataset_summary_rows)
            dataset_summary_csv = Path(OUTPUT_ROOT) / DATASET / f"{DATASET}_all_stations_summary.csv"
            dataset_summary_df.to_csv(dataset_summary_csv, index=False)
            print(f"\n{'='*80}")
            print(f"✅ DATASET SUMMARY SAVED: {dataset_summary_csv}")
            print(f"{'='*80}\n")

    print(f"\n{'='*80}")
    print(f"✅ ALL DATASETS & STATIONS COMPLETE!")
    print(f"   Output directory: {OUTPUT_ROOT}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    print("=== SCRIPT STARTING ===", flush=True, file=sys.stderr)
    main()
