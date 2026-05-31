#!/usr/bin/env python3
"""
Cumulative Correlation and SNR Analysis on Clean Time Series

This script:
1. Loads denoised time series data
2. Removes time periods contaminated by P-wave arrivals
3. Computes cumulative correlation and SNR from SWCC peaks (clean only)
4. Plots cumulative metrics over time for each station-sim-template combination

The goal is to see how correlation/SNR accumulate over time in clean data,
excluding earthquake-contaminated periods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Input paths
PEAKS_COMBINED_CSV = "/home/owen/tilt_validation/SWCC_test/all_peaks_combined.csv"
P_WAVE_PEAKS_CSV = "/home/owen/tilt_validation/SWCC_test/all_p_wave_peaks_combined.csv"
EARTHQUAKE_CSV = "/home/owen/tilt_validation/earthquakes_with_p_waves.csv"

# Denoised data directories
DENOISED_DIR_INGV = "/home/owen/denoised_exports_003_001_csv_only/ingv/"
DENOISED_DIR_EXP = "/home/owen/denoised_exports_003_001_csv_only/experiment/"

# Output directory
OUTPUT_DIR = "/home/owen/tilt_validation/cumulative_analysis"

# P-wave buffer (minutes) - must match what was used in SWCC processing
P_WAVE_BUFFER_MIN = 10.0  # Will use distance-based calculation if available
USE_DISTANCE_BASED_BUFFER = True

# Plotting options
PLOT_DPI = 150
FIGURE_SIZE = (16, 10)

# ============================================================================
# Helper Functions
# ============================================================================

def calculate_distance_based_buffer(distance_km: float, min_buffer_min: float = 10.0, max_buffer_min: float = 20.0) -> float:
    """
    Calculate P-wave guard buffer time based on earthquake distance.

    Matches the implementation in swcc_fix.py
    """
    if distance_km <= 200:
        return min_buffer_min
    elif distance_km >= 1000:
        return max_buffer_min
    else:
        # Linear interpolation between 200-1000 km
        return min_buffer_min + (max_buffer_min - min_buffer_min) * (distance_km - 200) / (1000 - 200)


def load_denoised_timeseries(station: str, dataset: str = "ingv") -> pd.DataFrame:
    """
    Load denoised time series for a station.

    Returns DataFrame with columns: datetime, denoised
    """
    base_dir = DENOISED_DIR_INGV if dataset == "ingv" else DENOISED_DIR_EXP

    # Station name mapping for files
    if dataset == "ingv" and station == "EEC1":
        file_station = "EC1"
    else:
        file_station = station

    filepath = Path(base_dir) / f"{file_station}_0p001-0p01.csv"

    if not filepath.exists():
        print(f"  ⚠️  Denoised file not found: {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Find datetime column
    dt_col = None
    for col in ["datetime", "time", "timestamp"]:
        if col in df.columns:
            dt_col = col
            break

    if dt_col is None or "denoised" not in df.columns:
        print(f"  ⚠️  Required columns not found in {filepath}")
        return None

    df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)

    return df[["datetime", "denoised"]].copy()


def create_p_wave_exclusion_mask(timeseries_df: pd.DataFrame, earthquake_df: pd.DataFrame) -> pd.Series:
    """
    Create a boolean mask indicating which time points should be excluded due to P-wave contamination.

    Returns:
        pd.Series (bool): True = clean data, False = P-wave contaminated (should be excluded)
    """
    if earthquake_df is None or len(earthquake_df) == 0:
        # No earthquakes - all data is clean
        return pd.Series(True, index=timeseries_df.index)

    # Start with all clean
    clean_mask = pd.Series(True, index=timeseries_df.index)

    # Mark contaminated periods
    for idx, quake in earthquake_df.iterrows():
        p_wave_time = pd.to_datetime(quake['p_wave_eta'])

        # Calculate buffer based on distance if available
        if USE_DISTANCE_BASED_BUFFER and 'distance_km' in quake and not pd.isna(quake['distance_km']):
            buffer_minutes = calculate_distance_based_buffer(quake['distance_km'])
        else:
            buffer_minutes = P_WAVE_BUFFER_MIN

        buffer_delta = pd.Timedelta(minutes=buffer_minutes)

        # Define exclusion window
        exclusion_start = p_wave_time - buffer_delta
        exclusion_end = p_wave_time + buffer_delta

        # Mark contaminated points
        contaminated = (timeseries_df['datetime'] >= exclusion_start) & (timeseries_df['datetime'] <= exclusion_end)
        clean_mask = clean_mask & ~contaminated

    return clean_mask


def compute_cumulative_metrics(peaks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative correlation and SNR from peaks DataFrame.

    Input: peaks_df with columns: peak_time_dt, peak_corr, snr_linear, snr_db
    Output: DataFrame with cumulative_corr, cumulative_snr_linear, cumulative_snr_db

    Note: Uses SNR linear for cumulative sum (always positive).
          SNR dB is converted from linear for proper averaging.
    """
    # Sort by time
    df = peaks_df.sort_values('peak_time_dt').copy()

    # Cumulative sum of correlation (absolute value)
    df['cumulative_corr'] = df['peak_corr'].abs().cumsum()

    # Cumulative sum of SNR LINEAR (always positive, more meaningful for sum)
    df['cumulative_snr_linear'] = df['snr_linear'].cumsum()

    # Convert cumulative linear back to dB for display
    # SNR_dB = 20 * log10(SNR_linear)
    df['cumulative_snr_db'] = 20 * np.log10(df['cumulative_snr_linear'].replace(0, np.nan))

    # Cumulative count
    df['cumulative_count'] = np.arange(1, len(df) + 1)

    # Cumulative average (use linear for SNR, convert to dB for display)
    df['cumulative_avg_corr'] = df['cumulative_corr'] / df['cumulative_count']
    df['cumulative_avg_snr_linear'] = df['cumulative_snr_linear'] / df['cumulative_count']
    df['cumulative_avg_snr_db'] = 20 * np.log10(df['cumulative_avg_snr_linear'].replace(0, np.nan))

    return df


def plot_cumulative_metrics_per_station(station: str, dataset: str, peaks_df: pd.DataFrame,
                                        earthquake_df: pd.DataFrame, output_dir: Path):
    """
    Plot cumulative metrics for all sim-template combinations for a given station.
    """
    # Filter peaks for this station
    station_peaks = peaks_df[peaks_df['station'] == station].copy()

    if len(station_peaks) == 0:
        print(f"  ⚠️  No peaks found for {station}")
        return

    # Get unique sim-template combinations
    sim_template_combos = station_peaks.groupby(['sim', 'template']).size().reset_index()[['sim', 'template']]

    n_combos = len(sim_template_combos)
    if n_combos == 0:
        return

    # Create figure with subplots
    n_cols = 2
    n_rows = (n_combos + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIGURE_SIZE, sharex=False)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    axes = axes.flatten()

    for idx, (_, row) in enumerate(sim_template_combos.iterrows()):
        sim = row['sim']
        template = row['template']

        # Filter peaks for this sim-template
        combo_peaks = station_peaks[(station_peaks['sim'] == sim) &
                                    (station_peaks['template'] == template)].copy()

        if len(combo_peaks) == 0:
            continue

        # Compute cumulative metrics
        combo_peaks = compute_cumulative_metrics(combo_peaks)

        ax = axes[idx]

        # Create twin axis for SNR
        ax2 = ax.twinx()

        # Plot cumulative correlation
        line1 = ax.plot(combo_peaks['peak_time_dt'], combo_peaks['cumulative_corr'],
                       'b-', linewidth=2, label='Cumulative Correlation', alpha=0.8)

        # Plot cumulative SNR
        line2 = ax2.plot(combo_peaks['peak_time_dt'], combo_peaks['cumulative_snr_db'],
                        'r-', linewidth=2, label='Cumulative SNR (dB)', alpha=0.8)

        # Mark P-wave arrivals in time range
        if earthquake_df is not None and len(earthquake_df) > 0:
            time_range = (combo_peaks['peak_time_dt'].min(), combo_peaks['peak_time_dt'].max())
            p_waves_in_range = earthquake_df[
                (earthquake_df['p_wave_eta'] >= time_range[0]) &
                (earthquake_df['p_wave_eta'] <= time_range[1])
            ]

            for _, quake in p_waves_in_range.iterrows():
                p_time = pd.to_datetime(quake['p_wave_eta'])
                ax.axvline(p_time, color='orange', linestyle='--', alpha=0.3, linewidth=1)

        # Labels and title
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Cumulative Correlation', color='b', fontsize=10)
        ax2.set_ylabel('Cumulative SNR (dB)', color='r', fontsize=10)
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.set_title(f'{sim} - {template}\n{len(combo_peaks)} clean peaks', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=8)

        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for idx in range(len(sim_template_combos), len(axes)):
        axes[idx].set_visible(False)

    # Main title
    fig.suptitle(f'{dataset.upper()} - {station} - Cumulative Correlation & SNR (Clean Data Only)',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f"{dataset}_{station}_cumulative_metrics.png"
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  ✅ Saved: {output_file.name}")
    plt.close()


def plot_global_cumulative_metrics(peaks_df: pd.DataFrame, earthquake_df: pd.DataFrame,
                                   output_dir: Path, dataset: str):
    """
    Plot global cumulative metrics across all stations, sims, and templates.
    """
    # Compute cumulative metrics for all peaks
    all_peaks = peaks_df.copy()
    all_peaks = compute_cumulative_metrics(all_peaks)

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))

    # Plot 1: Cumulative correlation
    ax1.plot(all_peaks['peak_time_dt'], all_peaks['cumulative_corr'],
            'b-', linewidth=2, label='Cumulative Correlation', alpha=0.8)
    ax1.set_ylabel('Cumulative Correlation', fontsize=12, fontweight='bold')
    ax1.set_title(f'{dataset.upper()} - Global Cumulative Correlation ({len(all_peaks)} clean peaks)',
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: Cumulative SNR (converted from linear sum)
    ax2.plot(all_peaks['peak_time_dt'], all_peaks['cumulative_snr_db'],
            'r-', linewidth=2, label='Cumulative SNR (dB from linear sum)', alpha=0.8)
    ax2.set_ylabel('Cumulative SNR (dB)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{dataset.upper()} - Global Cumulative SNR (from linear sum)',
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Plot 3: Cumulative average correlation & SNR
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(all_peaks['peak_time_dt'], all_peaks['cumulative_avg_corr'],
                    'b-', linewidth=2, label='Avg Correlation', alpha=0.8)
    line2 = ax3_twin.plot(all_peaks['peak_time_dt'], all_peaks['cumulative_avg_snr_db'],
                         'r-', linewidth=2, label='Avg SNR (dB)', alpha=0.8)

    ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Correlation', color='b', fontsize=12, fontweight='bold')
    ax3_twin.set_ylabel('Average SNR (dB)', color='r', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    ax3.set_title(f'{dataset.upper()} - Cumulative Averages',
                 fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left', fontsize=10)

    # Mark P-wave arrivals
    if earthquake_df is not None and len(earthquake_df) > 0:
        time_range = (all_peaks['peak_time_dt'].min(), all_peaks['peak_time_dt'].max())
        p_waves_in_range = earthquake_df[
            (earthquake_df['p_wave_eta'] >= time_range[0]) &
            (earthquake_df['p_wave_eta'] <= time_range[1])
        ]

        for ax in [ax1, ax2, ax3]:
            for _, quake in p_waves_in_range.iterrows():
                p_time = pd.to_datetime(quake['p_wave_eta'])
                ax.axvline(p_time, color='orange', linestyle='--', alpha=0.2, linewidth=0.8)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f"{dataset}_global_cumulative_metrics.png"
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  ✅ Saved: {output_file.name}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("="*80)
    print("CUMULATIVE CORRELATION & SNR ANALYSIS ON CLEAN TIME SERIES")
    print("="*80)

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}\n")

    # Load earthquake data
    print("📊 Loading earthquake P-wave data...")
    earthquake_df = pd.read_csv(EARTHQUAKE_CSV)
    earthquake_df['p_wave_eta'] = pd.to_datetime(earthquake_df['p_wave_eta'], errors='coerce')
    earthquake_df = earthquake_df.dropna(subset=['p_wave_eta'])
    print(f"  ✅ Loaded {len(earthquake_df)} earthquakes with P-wave arrivals\n")

    # Load all peaks (clean only - excludes P-wave contaminated)
    print("📊 Loading SWCC peaks (clean only)...")
    all_peaks_df = pd.read_csv(PEAKS_COMBINED_CSV)
    all_peaks_df['peak_time_dt'] = pd.to_datetime(all_peaks_df['peak_time_dt'])

    # Filter out any peaks marked as contaminated
    if 'p_wave_contaminated' in all_peaks_df.columns:
        clean_peaks = all_peaks_df[~all_peaks_df['p_wave_contaminated']].copy()
    else:
        clean_peaks = all_peaks_df.copy()

    print(f"  ✅ Loaded {len(clean_peaks)} clean peaks (total peaks: {len(all_peaks_df)})\n")

    # Process each dataset
    for dataset in ['ingv', 'experiment']:
        print(f"\n{'='*80}")
        print(f"Processing {dataset.upper()} Dataset")
        print(f"{'='*80}\n")

        # Filter peaks for this dataset
        dataset_peaks = clean_peaks[clean_peaks['dataset'] == dataset].copy()

        if len(dataset_peaks) == 0:
            print(f"  ⚠️  No clean peaks found for {dataset}")
            continue

        print(f"  📈 {len(dataset_peaks)} clean peaks in {dataset}")

        # Get unique stations
        stations = sorted(dataset_peaks['station'].unique())
        print(f"  🏠 Stations: {', '.join(stations)}\n")

        # Plot global cumulative metrics
        print(f"  🎨 Plotting global cumulative metrics for {dataset}...")
        plot_global_cumulative_metrics(dataset_peaks, earthquake_df, output_dir, dataset)

        # Plot per-station metrics
        for station in stations:
            print(f"\n  🎨 Plotting {station}...")
            plot_cumulative_metrics_per_station(station, dataset, dataset_peaks,
                                               earthquake_df, output_dir)

    # Create summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    summary_stats = []
    for dataset in ['ingv', 'experiment']:
        dataset_peaks = clean_peaks[clean_peaks['dataset'] == dataset]
        if len(dataset_peaks) == 0:
            continue

        for station in sorted(dataset_peaks['station'].unique()):
            station_peaks = dataset_peaks[dataset_peaks['station'] == station]

            summary_stats.append({
                'dataset': dataset,
                'station': station,
                'total_clean_peaks': len(station_peaks),
                'mean_correlation': station_peaks['peak_corr'].mean(),
                'std_correlation': station_peaks['peak_corr'].std(),
                'mean_snr_db': station_peaks['snr_db'].mean(),
                'std_snr_db': station_peaks['snr_db'].std(),
                'time_span_days': (station_peaks['peak_time_dt'].max() -
                                  station_peaks['peak_time_dt'].min()).total_seconds() / 86400
            })

    summary_df = pd.DataFrame(summary_stats)
    summary_file = output_dir / "summary_statistics.csv"
    summary_df.to_csv(summary_file, index=False)
    print(summary_df.to_string(index=False))
    print(f"\n  ✅ Saved summary statistics: {summary_file.name}")

    print(f"\n{'='*80}")
    print(f"✅ COMPLETE! All plots saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
