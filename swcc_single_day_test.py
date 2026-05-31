#!/usr/bin/env python3
"""
Single-Day SWCC Test - Complete Procedure
Runs the full swcc_edit.py analysis pipeline on a single day for ECPN station

This mimics the complete swcc_edit.py procedure but filtered to one day:
- Loads signal for 2022-12-04
- Builds P-wave exclusion zones
- Removes contaminated periods
- Runs SWCC on both full and cleaned signals
- Detects peaks and computes SNR
- Creates all comparison plots
- Saves peaks CSVs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks, butter, sosfiltfilt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import plotting functions from swcc_edit.py by copying them here
# (We'll use the same functions to ensure identical behavior)

# ============================================================================
# Configuration (Same as swcc_edit.py)
# ============================================================================

# Target
TEST_DATE = "2022-12-04"
STATION = "ECPN"
DATASET = "ingv"

# Paths
BANDPASSED_CSV_ROOT = Path("/home/owen/tilt_validation/bandpassed_exports_001_01_csv_only_utc")
TEMPLATES_CSV_ROOT = Path("/home/owen/tilt_validation/tilt_templates_csv")
EARTHQUAKE_CSV = Path("/home/owen/tilt_validation/earthquakes_merged_utc.csv")
VOLCANIC_EVENTS_CSV = Path("/home/owen/tilt_validation/etna_volcanic_events_cleaned.csv")

OUTPUT_ROOT = Path("/home/owen/tilt_validation/test_single_day_swcc_full")
OUTPUT_DIR = OUTPUT_ROOT / DATASET / STATION
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters (Same as swcc_edit.py)
THRESHOLD = 0.2
SAMPLING_RATE = 1.0
SNR_GUARD_S = 300
SNR_HALFWIN_S = 600
USE_DISTANCE_BASED_BUFFER = True
DEFAULT_P_WAVE_BUFFER_MIN = 10.0
SAVE_CORRELATION_CSV = False  # Disable to save space

# ============================================================================
# Copy core functions from swcc_edit.py
# ============================================================================

def calculate_distance_based_buffer(distance_km: float, magnitude: float = None) -> float:
    """Calculate P-wave buffer (from swcc_edit.py)."""
    if magnitude is None:
        return DEFAULT_P_WAVE_BUFFER_MIN

    if magnitude >= 5.5:
        base_buffer = 15.0
    elif magnitude >= 4.5:
        base_buffer = 10.0
    elif magnitude >= 3.5:
        base_buffer = 7.0
    else:
        base_buffer = 5.0

    return base_buffer


def sliding_window_cross_correlation(template, signal):
    """Sliding window cross-correlation (from swcc_edit.py)."""
    if len(signal) < len(template):
        return np.array([])

    template = template - np.mean(template)
    template_norm = np.linalg.norm(template)
    if template_norm == 0:
        return np.array([])
    template = template / template_norm

    n_windows = len(signal) - len(template) + 1
    correlations = np.zeros(n_windows)

    for i in range(n_windows):
        window = signal[i:i+len(template)]
        window = window - np.mean(window)
        window_norm = np.linalg.norm(window)
        if window_norm > 0:
            correlations[i] = np.dot(template, window / window_norm)

    return correlations


def compute_template_snr_against_signal(signal, peak_idx, template_len, fs, guard_s, halfwin_s):
    """Compute SNR (from swcc_edit.py)."""
    signal_start = max(0, peak_idx - int(halfwin_s * fs))
    signal_end = min(len(signal), peak_idx + template_len + int(halfwin_s * fs))

    if signal_end - signal_start < template_len:
        return np.nan

    signal_segment = signal[signal_start:signal_end]
    peak_power = np.var(signal[peak_idx:peak_idx+template_len])

    guard_samples = int(guard_s * fs)
    left_bg = signal_segment[:max(0, peak_idx - signal_start - guard_samples)]
    right_bg = signal_segment[min(len(signal_segment), peak_idx - signal_start + template_len + guard_samples):]

    background = np.concatenate([left_bg, right_bg]) if len(left_bg) > 0 and len(right_bg) > 0 else (left_bg if len(left_bg) > 0 else right_bg)

    if len(background) == 0:
        return np.nan

    bg_power = np.var(background)
    if bg_power <= 0:
        return np.nan

    snr_db = 10 * np.log10(peak_power / bg_power)
    return snr_db


def load_bandpassed_signal(dataset, station, target_date=None):
    """Load bandpassed signal (from swcc_edit.py), optionally filter to one day."""
    station_bp = "EEC1" if (station == "EC1" and dataset == "ingv") else station
    csv_path = Path(BANDPASSED_CSV_ROOT) / dataset / f"{station_bp}_0p001-0p01Hz_bp.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Bandpassed CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    time_seconds = df['time_seconds'].values
    signal = df['bandpassed'].values

    # Convert to datetime
    if dataset == "ingv":
        start_dt = pd.Timestamp("2022-11-14 22:00:00")  # UTC+1 clock → UTC
    else:
        start_dt = pd.Timestamp("2023-07-23 23:00:00")  # UTC+1 clock → UTC

    time_dt = pd.to_datetime(start_dt) + pd.to_timedelta(time_seconds, unit='s')

    # Filter to target date if specified
    if target_date:
        target_dt = pd.to_datetime(target_date)
        day_mask = time_dt.date == target_dt.date()
        time_dt = time_dt[day_mask]
        signal = signal[day_mask]
        print(f"  Filtered to {target_date}: {len(signal):,} samples")

    return time_dt, signal


def load_template(station, dataset, sim, template_name):
    """Load template (from swcc_edit.py)."""
    station_tpl = "EEC1" if (station == "EC1" and dataset == "experiment") else station
    template_path = TEMPLATES_CSV_ROOT / dataset / f"{station_tpl}_{sim}_{template_name}_0p001-0p01Hz_tpl.csv"

    if not template_path.exists():
        return None

    df = pd.read_csv(template_path)
    return df['x'].values


# ============================================================================
# Simplified plotting functions (basic versions)
# ============================================================================

def plot_swcc_comparison_simple(time_dt_full, correlations_full, peaks_full, peak_heights_full,
                                time_dt_clean, correlations_clean, peaks_clean, peak_heights_clean,
                                peaks_contaminated, peak_heights_contaminated, time_dt_cont,
                                template_name, sim, output_dir, dataset, station,
                                clean_mask, time_dt_original):
    """Simple comparison plot."""
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), facecolor='white')

    # Plot 1: Full signal correlation
    ax1 = axes[0]
    ax1.plot(time_dt_full, correlations_full, 'b-', linewidth=0.8, alpha=0.7, label='Full signal')

    if len(peaks_full) > 0:
        peak_times = time_dt_full[peaks_full]
        ax1.scatter(peak_times, correlations_full[peaks_full], c='blue', s=60, marker='o',
                   zorder=10, edgecolors='black', linewidth=1, label=f'Peaks (n={len(peaks_full)})')

    # Mark contaminated regions
    contaminated_regions = ~clean_mask
    if np.any(contaminated_regions):
        # Find contiguous regions
        changes = np.diff(np.concatenate([[0], contaminated_regions.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for start, end in zip(starts, ends):
            if start < len(time_dt_original) and end <= len(time_dt_original):
                ax1.axvspan(time_dt_original[start], time_dt_original[min(end, len(time_dt_original)-1)],
                           alpha=0.3, color='red', label='P-wave zone' if start == starts[0] else '')

    ax1.axhline(THRESHOLD, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(-THRESHOLD, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Correlation', fontweight='600')
    ax1.set_title(f'{dataset.upper()} - {station} | {sim} - {template_name}\nFull Signal (Contaminated)',
                 fontweight='700', pad=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 1)

    # Plot 2: Cleaned signal correlation
    ax2 = axes[1]
    ax2.plot(time_dt_clean, correlations_clean, 'g-', linewidth=0.8, alpha=0.7, label='Cleaned signal')

    if len(peaks_clean) > 0:
        peak_times = time_dt_clean[peaks_clean]
        ax2.scatter(peak_times, correlations_clean[peaks_clean], c='darkgreen', s=60, marker='o',
                   zorder=10, edgecolors='black', linewidth=1, label=f'Peaks (n={len(peaks_clean)})')

    ax2.axhline(THRESHOLD, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(-THRESHOLD, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Correlation', fontweight='600')
    ax2.set_xlabel('Time (UTC)', fontweight='600')
    ax2.set_title('Cleaned Signal (P-wave periods removed)', fontweight='700', pad=10, color='darkgreen')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)

    plt.tight_layout()
    output_file = output_dir / f"{station}_{sim}_{template_name}_swcc_comparison.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    💾 Saved: {output_file.name}")


# ============================================================================
# Main analysis function
# ============================================================================

def run_single_day_swcc():
    """Run complete SWCC procedure for single day."""

    print("\n" + "="*80)
    print(f"SINGLE-DAY SWCC TEST - COMPLETE PROCEDURE")
    print(f"Date: {TEST_DATE} | Station: {STATION} ({DATASET})")
    print("="*80)

    # Load signal for target day
    print(f"\n📂 Loading bandpassed signal for {TEST_DATE}...")
    time_dt_full, signal_full = load_bandpassed_signal(DATASET, STATION, TEST_DATE)

    # Load earthquakes
    print(f"\n🌍 Loading earthquake catalog...")
    df_quakes = pd.read_csv(EARTHQUAKE_CSV)
    df_quakes['datetime'] = pd.to_datetime(df_quakes['datetime'])
    df_quakes['p_wave_eta'] = pd.to_datetime(df_quakes['p_wave_eta'])

    # Filter to day
    day_start = pd.to_datetime(TEST_DATE)
    day_end = day_start + timedelta(days=1)
    df_quakes_day = df_quakes[
        (df_quakes['p_wave_eta'] >= day_start) &
        (df_quakes['p_wave_eta'] < day_end)
    ].copy()

    print(f"  ✅ Found {len(df_quakes_day)} earthquakes for {TEST_DATE}")

    # Build P-wave exclusion zones
    print(f"\n🚫 Building P-wave exclusion zones...")
    clean_mask = np.ones(len(signal_full), dtype=bool)

    for idx, quake in df_quakes_day.iterrows():
        p_time = pd.to_datetime(quake['p_wave_eta'])
        magnitude = quake['magnitude'] if 'magnitude' in quake and not pd.isna(quake['magnitude']) else None

        if USE_DISTANCE_BASED_BUFFER and 'distance_km' in quake and not pd.isna(quake['distance_km']):
            buffer_min = calculate_distance_based_buffer(quake['distance_km'], magnitude)
        else:
            buffer_min = DEFAULT_P_WAVE_BUFFER_MIN

        buffer_delta = pd.Timedelta(minutes=buffer_min)
        exclusion_start = p_time - buffer_delta
        exclusion_end = p_time + buffer_delta

        contaminated = (time_dt_full >= exclusion_start) & (time_dt_full <= exclusion_end)
        clean_mask[contaminated] = False

    n_contaminated = np.sum(~clean_mask)
    pct_contaminated = 100 * n_contaminated / len(signal_full)
    print(f"  ✅ Contaminated: {n_contaminated:,} / {len(signal_full):,} ({pct_contaminated:.2f}%)")

    # Remove contaminated periods
    print(f"\n🧹 Removing P-wave contaminated periods...")
    time_dt_clean = time_dt_full[clean_mask]
    signal_clean = signal_full[clean_mask]
    print(f"  ✅ Clean signal: {len(signal_clean):,} samples")

    # Process each sim and template
    sims = ["sim1", "sim2", "sim3", "sim4"]
    templates = ["template1", "template2", "template3", "template4"]

    all_results = []

    for sim in sims:
        print(f"\n{'='*80}")
        print(f"Processing {sim.upper()}")
        print(f"{'='*80}")

        sim_dir = OUTPUT_DIR / sim
        sim_dir.mkdir(parents=True, exist_ok=True)

        for template_name in templates:
            print(f"\n  🔄 {template_name}:")

            # Load template
            template = load_template(STATION, DATASET, sim, template_name)
            if template is None:
                print(f"    ⚠️  Template not found")
                continue

            print(f"    Template: {len(template)} samples ({len(template)/60:.1f} min)")

            # Run SWCC on FULL signal
            print(f"    Running SWCC on full signal...")
            correlations_full = sliding_window_cross_correlation(template, signal_full)
            corr_time_dt_full = time_dt_full[:len(correlations_full)]

            # Run SWCC on CLEANED signal
            print(f"    Running SWCC on cleaned signal...")
            correlations_clean = sliding_window_cross_correlation(template, signal_clean)
            corr_time_dt_clean = time_dt_clean[:len(correlations_clean)]

            # Detect peaks on CLEANED signal
            distance = 1000
            peaks_clean, props_clean = find_peaks(
                np.abs(correlations_clean),
                height=THRESHOLD,
                distance=distance
            )
            peak_heights_clean = props_clean.get("peak_heights", np.array([]))
            print(f"    Found {len(peaks_clean)} peaks (cleaned)")

            # Detect peaks on FULL signal
            peaks_full, props_full = find_peaks(
                np.abs(correlations_full),
                height=THRESHOLD,
                distance=distance
            )
            peak_heights_full = props_full.get("peak_heights", np.array([]))
            print(f"    Found {len(peaks_full)} peaks (full)")

            # Identify contaminated peaks
            contaminated_samples = (~clean_mask).astype(int)
            window_len = len(template)
            if len(contaminated_samples) >= window_len:
                window_contaminated = np.convolve(
                    contaminated_samples,
                    np.ones(window_len, dtype=int),
                    mode='valid'
                ) > 0
            else:
                window_contaminated = np.array([], dtype=bool)

            contaminated_mask = window_contaminated[peaks_full] if len(peaks_full) > 0 else np.array([], dtype=bool)
            peaks_contaminated = peaks_full[contaminated_mask]
            peak_heights_contaminated = peak_heights_full[contaminated_mask]

            print(f"    Contaminated peaks: {len(peaks_contaminated)}")
            print(f"    Eligible peaks: {len(peaks_clean)}")

            # Compute SNR for eligible peaks
            snr_values = []
            snr_linear_values = []
            for peak_idx in peaks_clean:
                snr_db = compute_template_snr_against_signal(
                    signal_clean, peak_idx, len(template),
                    SAMPLING_RATE, SNR_GUARD_S, SNR_HALFWIN_S
                )
                snr_linear = 10.0**(snr_db/20.0) if np.isfinite(snr_db) else np.nan
                snr_values.append(snr_db)
                snr_linear_values.append(snr_linear)

            # Save peaks CSV
            if len(peaks_clean) > 0:
                peaks_data = []
                for i, peak_idx in enumerate(peaks_clean):
                    peaks_data.append({
                        'dataset': DATASET,
                        'station': STATION,
                        'sim': sim,
                        'template': template_name,
                        'peak_index': peak_idx,
                        'peak_time_dt': corr_time_dt_clean[peak_idx],
                        'peak_corr': float(np.abs(correlations_clean[peak_idx])),
                        'snr_db': snr_values[i],
                        'snr_linear': snr_linear_values[i],
                        'peak_type': 'clean'
                    })

                peaks_df = pd.DataFrame(peaks_data)
                peaks_csv = sim_dir / f"{STATION}_{sim}_{template_name}_peaks.csv"
                peaks_df.to_csv(peaks_csv, index=False)
                print(f"    💾 Saved {len(peaks_data)} peaks: {peaks_csv.name}")

                all_results.extend(peaks_data)

            # Create plot
            plot_swcc_comparison_simple(
                corr_time_dt_full, correlations_full, peaks_full, peak_heights_full,
                corr_time_dt_clean, correlations_clean, peaks_clean, peak_heights_clean,
                peaks_contaminated, peak_heights_contaminated, corr_time_dt_full,
                template_name, sim, sim_dir, DATASET, STATION,
                clean_mask, time_dt_full
            )

    # Save combined results
    if len(all_results) > 0:
        combined_df = pd.DataFrame(all_results)
        combined_csv = OUTPUT_DIR / f"{STATION}_all_peaks_combined.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n✅ Saved combined results: {combined_csv}")
        print(f"   Total peaks: {len(all_results)}")

    print("\n" + "="*80)
    print("SINGLE-DAY TEST COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Date tested: {TEST_DATE}")
    print(f"Station: {STATION} ({DATASET})")
    print(f"\nCheck {OUTPUT_DIR} for:")
    print(f"  - Peak CSVs for each sim/template")
    print(f"  - SWCC comparison plots")
    print(f"  - Combined peaks CSV")
    print("="*80 + "\n")


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    run_single_day_swcc()
