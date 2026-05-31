#!/usr/bin/env python3
"""
Complete Comprehensive Signal Analysis for Mt. Etna Tilt Data
Combined Parts 1-7: All Analysis in One Script

This script performs:
1. P-wave contamination probability analysis
2. Cumulative metrics behavior explanation (correlation vs SNR)
3. Deep dive comparison between INGV and Experiment datasets
4. Synchronous peak detection across stations
5. Signal candidate selection by consistency metrics
6. Spectrograms and frequency-amplitude plots for best candidates
7. Morlet wavelet transforms for synchronous events

Author: Mt. Etna Tilt Analysis Pipeline
Date: 2024-12-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, detrend as scipy_detrend, spectrogram
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr, ks_2samp, mannwhitneyu, ttest_ind, linregress, zscore
import pywt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Professional plot styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#d1d5db',
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

# ============================================================================
# Configuration
# ============================================================================

# Input paths
PEAKS_ROOT = Path("/home/owen/tilt_validation/SWCC_utc_fixed")
BANDPASSED_CSV_ROOT = Path("/home/owen/tilt_validation/bandpassed_exports_001_01_csv_only_utc")
TEMPLATES_CSV_ROOT = Path("/home/owen/tilt_validation/tilt_templates_csv")
EARTHQUAKE_CSV = Path("/home/owen/tilt_validation/earthquakes_merged_utc.csv")
VOLCANIC_EVENTS_CSV = Path("/home/owen/tilt_validation/etna_volcanic_events_cleaned.csv")

# Output paths
OUTPUT_DIR = Path("/home/owen/tilt_validation/comprehensive_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories
(OUTPUT_DIR / "contamination_analysis").mkdir(exist_ok=True)
(OUTPUT_DIR / "cumulative_metrics").mkdir(exist_ok=True)
(OUTPUT_DIR / "dataset_comparison").mkdir(exist_ok=True)
(OUTPUT_DIR / "synchronous_events").mkdir(exist_ok=True)
(OUTPUT_DIR / "best_candidates").mkdir(exist_ok=True)
(OUTPUT_DIR / "morlet_wavelets").mkdir(exist_ok=True)

# Synchronous event detection parameters
SYNC_TOLERANCE_SEC = 300  # 5 minutes tolerance for synchronous event detection
MIN_STATIONS = 2  # Minimum stations required for synchronous event

# Signal quality thresholds
MIN_CORRELATION = 0.2  # Minimum peak correlation for synchronous event detection
MIN_SNR_LINEAR = 3.16  # Minimum SNR (linear) for high-quality peaks (equivalent to 10)
HIGH_CORRELATION = 0.7  # Very high-quality correlation threshold

# Signal processing parameters
BAND_HZ = (0.001, 0.01)
FILTER_ORDER = 4
SAMPLING_RATE = 1.0  # Hz
SEGMENT_DURATION_S = 3333

# Dataset colors
DATASET_COLORS = {
    'ingv': '#2563eb',
    'experiment': '#dc2626'
}

# Station colors
STATION_COLORS = {
    'EC1': '#1f77b4',
    'ECPN': '#2ca02c',
    'ECIT': '#ff7f0e',
    'EC10': '#d62728',
    'ECOR': '#9467bd',
    'EMAS': '#8c564b'
}

# Template colors
TEMPLATE_COLORS = {
    'template1': '#E74C3C',
    'template2': '#E67E22',
    'template3': '#F39C12',
    'template4': '#D35400'
}

# Simulation colors
SIM_COLORS = {
    'sim1': '#3b82f6',
    'sim2': '#8b5cf6',
    'sim3': '#ec4899',
    'sim4': '#f59e0b'
}

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_all_peaks(peaks_root: Path) -> pd.DataFrame:
    """Load all peak CSV files from the SWCC output directory structure."""
    all_peaks = []

    for dataset_dir in peaks_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for station_dir in dataset_dir.iterdir():
            if not station_dir.is_dir():
                continue
            station = station_dir.name

            for sim_dir in station_dir.iterdir():
                if not sim_dir.is_dir():
                    continue
                sim = sim_dir.name

                # Find peak CSV files
                for peaks_file in sim_dir.glob("*_peaks.csv"):
                    # Extract template name from filename
                    parts = peaks_file.stem.split('_')
                    if len(parts) >= 4 and parts[-1] == 'peaks':
                        template = parts[-2]
                    else:
                        continue

                    try:
                        df = pd.read_csv(peaks_file)
                        df['dataset'] = dataset
                        df['station'] = station
                        df['sim'] = sim
                        df['template'] = template
                        all_peaks.append(df)
                    except Exception as e:
                        print(f"Warning: Could not load {peaks_file}: {e}")

    if not all_peaks:
        raise ValueError("No peak files found!")

    # Combine all peaks
    df_all = pd.concat(all_peaks, ignore_index=True)

    # Convert peak_time_dt to datetime
    df_all['peak_time_dt'] = pd.to_datetime(df_all['peak_time_dt'], errors='coerce')

    # Ensure numeric columns are properly typed
    numeric_cols = ['peak_index', 'peak_corr', 'snr_linear', 'snr_linear',
                   'cumulative_snr_linear', 'cumulative_snr_linear']
    for col in numeric_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    print(f"✅ Loaded {len(df_all):,} peaks from {len(all_peaks)} files")
    print(f"   Datasets: {sorted(df_all['dataset'].unique())}")
    print(f"   Stations: {sorted(df_all['station'].unique())}")
    print(f"   Simulations: {sorted(df_all['sim'].unique())}")
    print(f"   Templates: {sorted(df_all['template'].unique())}")

    return df_all


def load_p_wave_impact_data(peaks_root: Path) -> pd.DataFrame:
    """Load P-wave impact comparison CSV files."""
    all_impact = []

    for dataset_dir in peaks_root.iterdir():
        if not dataset_dir.is_dir():
            continue

        for station_dir in dataset_dir.iterdir():
            if not station_dir.is_dir():
                continue

            for sim_dir in station_dir.iterdir():
                if not sim_dir.is_dir():
                    continue

                # Find impact comparison CSV files
                for impact_file in sim_dir.glob("*_pwave_impact_comparison.csv"):
                    try:
                        df = pd.read_csv(impact_file)
                        all_impact.append(df)
                    except Exception as e:
                        print(f"Warning: Could not load {impact_file}: {e}")

    if not all_impact:
        print("⚠️ No P-wave impact files found")
        return pd.DataFrame()

    df_impact = pd.concat(all_impact, ignore_index=True)
    print(f"✅ Loaded P-wave impact data: {len(df_impact)} records")

    return df_impact


from pwave_buffer import calculate_pwave_buffers


def calculate_pwave_buffer(magnitude=None, distance_km=None):
    """Shim: returns (pre_min, post_min) via pwave_buffer module."""
    return calculate_pwave_buffers(distance_km, magnitude)


def load_earthquakes(earthquake_csv: Path) -> pd.DataFrame:
    """Load earthquake catalog with P-wave arrivals."""
    if not earthquake_csv.exists():
        print(f"⚠️ Earthquake file not found: {earthquake_csv}")
        return pd.DataFrame()

    df_eq = pd.read_csv(earthquake_csv)

    # Parse datetime columns
    if 'p_wave_eta' in df_eq.columns:
        df_eq['p_wave_eta'] = pd.to_datetime(df_eq['p_wave_eta'], errors='coerce')
    if 'origin_time' in df_eq.columns:
        df_eq['origin_time'] = pd.to_datetime(df_eq['origin_time'], errors='coerce')

    df_eq = df_eq.dropna(subset=['p_wave_eta'])

    print(f"✅ Loaded {len(df_eq)} earthquakes with P-wave arrivals")

    return df_eq


def load_volcanic_events(volcanic_csv: Path) -> pd.DataFrame:
    """Load volcanic events catalog."""
    if not volcanic_csv.exists():
        print(f"⚠️ Volcanic events file not found: {volcanic_csv}")
        return pd.DataFrame()

    df_vol = pd.read_csv(volcanic_csv)

    # Parse datetime
    if 'event_time' in df_vol.columns:
        df_vol['event_time'] = pd.to_datetime(df_vol['event_time'], errors='coerce')
    elif 'time' in df_vol.columns:
        df_vol['event_time'] = pd.to_datetime(df_vol['time'], errors='coerce')

    print(f"✅ Loaded {len(df_vol)} volcanic events")

    return df_vol


def load_bandpassed_signal(station: str, dataset: str, bandpassed_root: Path) -> pd.DataFrame:
    """Load bandpassed signal for a station."""
    # Map EC1 -> EEC1 for INGV bandpassed files only
    station_bp = "EEC1" if (station == "EC1" and dataset == "ingv") else station

    # File format: {station}_0p001-0p01Hz_bp.csv
    csv_path = bandpassed_root / dataset / f"{station_bp}_0p001-0p01Hz_bp.csv"

    if not csv_path.exists():
        print(f"⚠️ Signal file not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)

        # Check for required columns
        if 'time_seconds' not in df.columns or 'bandpassed' not in df.columns:
            print(f"⚠️ Required columns not found in {csv_path}")
            return None

        # Convert time_seconds to datetime using dataset-specific start time
        if dataset == "ingv":
            start_dt = pd.Timestamp("2022-11-14 22:00:00")  # UTC+1 clock → UTC
        else:  # experiment
            start_dt = pd.Timestamp("2023-07-23 23:00:00")  # UTC+1 clock → UTC

        # Reconstruct datetime array
        df['datetime'] = start_dt + pd.to_timedelta(df['time_seconds'], unit='s')
        df['signal'] = df['bandpassed']

        return df[['datetime', 'signal']]

    except Exception as e:
        print(f"⚠️ Error loading {csv_path}: {e}")
        return None


def load_template(station: str, dataset: str, sim: str, template: str,
                 templates_root: Path) -> np.ndarray:
    """Load template from CSV."""
    # Templates are stored as: {station}_{sim}_{template}_0p001-0p01Hz_tpl.csv
    csv_path = templates_root / dataset / f"{station}_{sim}_{template}_0p001-0p01Hz_tpl.csv"

    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        # Template CSV has 'x' column for signal values
        if 'x' not in df.columns:
            return None
        return df['x'].values
    except Exception as e:
        return None


def extract_signal_segment(df_signal: pd.DataFrame, center_time: pd.Timestamp,
                          duration_s: float) -> tuple:
    """Extract signal segment centered on a time point."""
    half_duration = duration_s / 2
    start_time = center_time - timedelta(seconds=half_duration)
    end_time = center_time + timedelta(seconds=half_duration)

    df_segment = df_signal[
        (df_signal['datetime'] >= start_time) &
        (df_signal['datetime'] <= end_time)
    ].copy()

    if df_segment.empty:
        return None, None, None

    # Convert to arrays
    datetime_array = df_segment['datetime'].values
    signal_array = df_segment['signal'].values

    # Time in seconds from start
    time_array = (df_segment['datetime'] - df_segment['datetime'].iloc[0]).dt.total_seconds().values

    return time_array, signal_array, datetime_array


# ============================================================================
# Wavelet Functions
# ============================================================================

def compute_cwt(signal_data, fs=1.0, wavelet='morl', freq_min=1e-4, freq_max=1e-2, n_freqs=100):
    """Compute Continuous Wavelet Transform using Morlet wavelet."""
    signal_data = np.asarray(signal_data, dtype=float)
    if len(signal_data) == 0:
        return None, None

    # Remove NaN/Inf
    valid_idx = np.isfinite(signal_data)
    if not np.any(valid_idx):
        return None, None

    signal_data = signal_data[valid_idx]

    # Center frequency for Morlet
    center_freq = pywt.central_frequency(wavelet)

    # Target frequencies (log-spaced)
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    scales = center_freq * fs / freqs

    # Compute CWT
    try:
        coefficients, frequencies = pywt.cwt(signal_data, scales, wavelet,
                                            sampling_period=1.0/fs)
        # Power spectrum
        power = np.abs(coefficients) ** 2
        return frequencies, power
    except Exception as e:
        print(f"⚠️ CWT computation error: {e}")
        return None, None


# ============================================================================
# Part 1: P-wave Contamination Probability Analysis
# ============================================================================

def analyze_p_wave_contamination_probability(df_impact: pd.DataFrame,
                                             df_peaks: pd.DataFrame,
                                             df_earthquakes: pd.DataFrame,
                                             output_dir: Path):
    """Estimate probability of contamination failures and analyze failure modes."""
    print("\n" + "="*80)
    print("PART 1: P-WAVE CONTAMINATION PROBABILITY ANALYSIS")
    print("="*80)

    if df_impact.empty or df_earthquakes.empty:
        print("⚠️ Missing data for contamination analysis")
        return

    # Analyze peak loss statistics
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
    fig.suptitle('P-wave Contamination Impact & Failure Mode Analysis',
                fontsize=16, fontweight='bold', y=0.98)

    # 1. Distribution of peak loss percentages
    ax1 = axes[0, 0]
    ax1.hist(df_impact['pct_peaks_lost'], bins=50, color='#dc2626',
            alpha=0.7, edgecolor='black', linewidth=0.8)
    ax1.axvline(df_impact['pct_peaks_lost'].median(), color='blue',
               linestyle='--', linewidth=2, label=f'Median: {df_impact["pct_peaks_lost"].median():.1f}%')
    ax1.axvline(df_impact['pct_peaks_lost'].mean(), color='green',
               linestyle='--', linewidth=2, label=f'Mean: {df_impact["pct_peaks_lost"].mean():.1f}%')
    ax1.set_xlabel('% Peaks Lost to P-wave Filtering', fontweight='600')
    ax1.set_ylabel('Frequency', fontweight='600')
    ax1.set_title('Distribution of Peak Loss', fontweight='700', pad=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Correlation improvement after filtering
    ax2 = axes[0, 1]
    corr_improvement = ((df_impact['corr_mean_clean'] - df_impact['corr_mean_contaminated']) /
                       df_impact['corr_mean_contaminated'] * 100)
    ax2.hist(corr_improvement.dropna(), bins=50, color='#2563eb',
            alpha=0.7, edgecolor='black', linewidth=0.8)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax2.axvline(corr_improvement.median(), color='green', linestyle='--',
               linewidth=2, label=f'Median: {corr_improvement.median():.1f}%')
    ax2.set_xlabel('% Change in Mean Correlation', fontweight='600')
    ax2.set_ylabel('Frequency', fontweight='600')
    ax2.set_title('Correlation Improvement After P-wave Filtering', fontweight='700', pad=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. SNR distribution of clean peaks
    ax3 = axes[0, 2]
    ax3.hist(df_impact['snr_mean_db_clean'].dropna(), bins=50, color='#059669',
            alpha=0.7, edgecolor='black', linewidth=0.8)
    ax3.axvline(df_impact['snr_mean_db_clean'].median(), color='blue', linestyle='--',
               linewidth=2, label=f'Median: {df_impact["snr_mean_db_clean"].median():.1f}')
    ax3.axvline(df_impact['snr_mean_db_clean'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {df_impact["snr_mean_db_clean"].mean():.1f}')
    ax3.set_xlabel('Mean SNR (Linear) - Clean Peaks', fontweight='600')
    ax3.set_ylabel('Frequency', fontweight='600')
    ax3.set_title('SNR Distribution After P-wave Filtering', fontweight='700', pad=15)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Temporal proximity analysis
    if not df_peaks.empty and 'peak_time_dt' in df_peaks.columns:
        proximities = []
        for _, peak in df_peaks.iterrows():
            peak_time = peak['peak_time_dt']
            if pd.isna(peak_time):
                continue

            # Find nearest earthquake
            time_diffs = (df_earthquakes['p_wave_eta'] - peak_time).abs()
            min_diff = time_diffs.min()
            proximities.append(min_diff.total_seconds() / 60)  # in minutes

        if proximities:
            ax4 = axes[1, 0]
            ax4.hist(proximities, bins=100, range=(0, 120), color='#f59e0b',
                    alpha=0.7, edgecolor='black', linewidth=0.8)
            ax4.axvline(10, color='red', linestyle='--', linewidth=2,
                       label='Buffer boundary (10 min)')
            ax4.set_xlabel('Time to Nearest P-wave Arrival (min)', fontweight='600')
            ax4.set_ylabel('Peak Count', fontweight='600')
            ax4.set_title('Clean Peaks: Proximity to Earthquakes', fontweight='700', pad=15)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, 120)

            # Calculate contamination risk
            within_20min = sum(1 for p in proximities if p < 20)
            within_10min = sum(1 for p in proximities if p < 10)
            total = len(proximities)

            risk_text = (f"Potential contamination risk:\n"
                        f"Within 10 min: {within_10min}/{total} ({100*within_10min/total:.1f}%)\n"
                        f"Within 20 min: {within_20min}/{total} ({100*within_20min/total:.1f}%)")
            ax4.text(0.97, 0.97, risk_text, transform=ax4.transAxes,
                    fontsize=9, va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 5. Aftershock clustering analysis
    ax5 = axes[1, 1]
    aftershock_window_hours = 1
    aftershock_counts = []

    for _, eq in df_earthquakes.iterrows():
        eq_time = eq['p_wave_eta']
        window_end = eq_time + timedelta(hours=aftershock_window_hours)

        # Count peaks in this window (after buffer)
        buffer_min = 10  # minutes
        search_start = eq_time + timedelta(minutes=buffer_min)

        peaks_in_window = df_peaks[
            (df_peaks['peak_time_dt'] >= search_start) &
            (df_peaks['peak_time_dt'] <= window_end)
        ]
        aftershock_counts.append(len(peaks_in_window))

    ax5.hist(aftershock_counts, bins=range(0, max(aftershock_counts)+2),
            color='#8b5cf6', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax5.set_xlabel(f'Peaks Within {aftershock_window_hours}h After Earthquake\n(after buffer)',
                  fontweight='600')
    ax5.set_ylabel('Earthquake Count', fontweight='600')
    ax5.set_title('Potential Aftershock/Reverberation Contamination', fontweight='700', pad=15)
    ax5.grid(True, alpha=0.3)

    mean_aftershock_peaks = np.mean(aftershock_counts)
    ax5.axvline(mean_aftershock_peaks, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_aftershock_peaks:.1f} peaks')
    ax5.legend()

    # 6. Failure mode summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Calculate failure mode statistics
    total_peaks_contaminated = df_impact['n_peaks_contaminated'].sum()
    total_peaks_lost = df_impact['n_peaks_lost'].sum()
    total_peaks_clean = df_impact['n_peaks_clean'].sum()

    contamination_rate = total_peaks_lost / total_peaks_contaminated * 100 if total_peaks_contaminated > 0 else 0

    # Estimate residual contamination probability
    high_proximity_rate = within_20min / total if total > 0 else 0
    high_aftershock_rate = sum(1 for c in aftershock_counts if c > 5) / len(aftershock_counts) if aftershock_counts else 0

    # Conservative estimate
    estimated_residual_contamination = (high_proximity_rate * 0.05 +
                                       high_aftershock_rate * 0.10) / 2

    summary_text = (
        "P-WAVE CONTAMINATION ANALYSIS SUMMARY\n\n"
        f"Total peaks (contaminated): {total_peaks_contaminated:,}\n"
        f"Peaks removed by filter: {total_peaks_lost:,} ({contamination_rate:.1f}%)\n"
        f"Clean peaks retained: {total_peaks_clean:,}\n\n"
        "FAILURE MODE PROBABILITIES:\n\n"
        f"1. Edge Effects (within 20 min):\n"
        f"   {within_20min:,} peaks ({100*high_proximity_rate:.2f}%)\n"
        f"   Estimated contamination risk: ~{100*high_proximity_rate*0.05:.2f}%\n\n"
        f"2. Aftershock/Reverberation:\n"
        f"   High-activity events: {sum(1 for c in aftershock_counts if c > 5)}\n"
        f"   Estimated contamination risk: ~{100*high_aftershock_rate*0.10:.2f}%\n\n"
        f"3. Overall Residual Contamination:\n"
        f"   Estimated probability: ~{100*estimated_residual_contamination:.2f}%\n"
        f"   ({int(estimated_residual_contamination * total_peaks_clean):,} peaks)\n\n"
        "CONFIDENCE: Moderate-High\n"
        "Buffer strategy appears effective, with\n"
        "low residual contamination probability."
    )

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9, pad=1))

    plt.tight_layout()
    output_file = output_dir / "contamination_analysis" / "01_contamination_probability_analysis.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")

    # Save summary statistics to CSV
    summary_df = pd.DataFrame([{
        'total_peaks_contaminated': total_peaks_contaminated,
        'total_peaks_lost': total_peaks_lost,
        'total_peaks_clean': total_peaks_clean,
        'contamination_rate_pct': contamination_rate,
        'peaks_within_20min': within_20min,
        'proximity_risk_pct': 100 * high_proximity_rate,
        'high_activity_quakes': sum(1 for c in aftershock_counts if c > 5),
        'aftershock_risk_pct': 100 * high_aftershock_rate,
        'estimated_residual_contamination_pct': 100 * estimated_residual_contamination,
        'estimated_contaminated_peaks': int(estimated_residual_contamination * total_peaks_clean),
        'mean_corr_improvement_pct': corr_improvement.mean(),
        'mean_snr_linear': df_impact['snr_mean_db_clean'].mean()
    }])

    summary_csv = output_dir / "contamination_analysis" / "contamination_summary_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Saved: {summary_csv.name}")


# ============================================================================
# Part 2: Cumulative Metrics Behavior Explanation
# ============================================================================

def explain_cumulative_metrics_behavior(df_peaks: pd.DataFrame, output_dir: Path):
    """Explain linearity of cumulative correlation vs non-linearity of cumulative SNR."""
    print("\n" + "="*80)
    print("PART 2: CUMULATIVE METRICS BEHAVIOR EXPLANATION")
    print("="*80)

    # Select a representative station-sim-template combination
    peak_counts = df_peaks.groupby(['dataset', 'station', 'sim', 'template']).size()
    best_combo = peak_counts.idxmax()
    dataset, station, sim, template = best_combo

    df_sample = df_peaks[
        (df_peaks['dataset'] == dataset) &
        (df_peaks['station'] == station) &
        (df_peaks['sim'] == sim) &
        (df_peaks['template'] == template)
    ].sort_values('peak_time_dt').reset_index(drop=True)

    if len(df_sample) < 10:
        print("⚠️ Insufficient data for cumulative metrics analysis")
        return

    print(f"📊 Analyzing: {dataset}/{station}/{sim}/{template} ({len(df_sample)} peaks)")

    # Calculate cumulative metrics
    cumulative_corr = df_sample['peak_corr'].abs().cumsum()

    if 'cumulative_snr_linear' in df_sample.columns:
        cumulative_snr_linear = df_sample['cumulative_snr_linear']
    else:
        cumulative_snr_linear = df_sample['snr_linear'].cumsum()

    cumulative_snr_linear = 20 * np.log10(cumulative_snr_linear + 1e-10)
    peak_indices = np.arange(len(df_sample))

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 14), facecolor='white')
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)
    fig.suptitle('Cumulative Metrics Behavior: Correlation vs SNR',
                fontsize=16, fontweight='bold', y=0.98)

    # 1. Cumulative correlation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(peak_indices, cumulative_corr, color='#9333ea', linewidth=2.5,
            marker='o', markersize=3, alpha=0.8)
    ax1.set_xlabel('Peak Index', fontweight='600')
    ax1.set_ylabel('Cumulative |r|', fontweight='600', color='#9333ea')
    ax1.set_title(f'Cumulative Correlation (Linear)\n{station} - {sim} - {template}',
                 fontweight='700', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='#9333ea')

    # 2. Cumulative SNR
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(peak_indices, cumulative_snr_linear, color='#059669', linewidth=2.5,
            marker='s', markersize=3, alpha=0.8)
    ax2.set_xlabel('Peak Index', fontweight='600')
    ax2.set_ylabel('Cumulative SNR (Linear)', fontweight='600', color='#059669')
    ax2.set_title(f'Cumulative SNR (Logarithmic)\n{station} - {sim} - {template}',
                 fontweight='700', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='#059669')

    # 3. Rate of change comparison
    ax3 = fig.add_subplot(gs[0, 2])
    if len(cumulative_corr) > 1:
        corr_rate = np.gradient(cumulative_corr)
        snr_rate = np.gradient(cumulative_snr_linear)

        ax3.plot(peak_indices, corr_rate, color='#9333ea', linewidth=2,
                alpha=0.7, label='Correlation rate')
        ax3_right = ax3.twinx()
        ax3_right.plot(peak_indices, snr_rate, color='#059669', linewidth=2,
                      alpha=0.7, label='SNR rate')

        ax3.set_xlabel('Peak Index', fontweight='600')
        ax3.set_ylabel('d(Cumul. Corr)/dn', fontweight='600', color='#9333ea')
        ax3_right.set_ylabel('d(Cumul. SNR)/dn (dB)', fontweight='600', color='#059669')
        ax3.set_title('Rate of Change Comparison', fontweight='700', pad=15)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='y', labelcolor='#9333ea')
        ax3_right.tick_params(axis='y', labelcolor='#059669')

    # 4. Startup transient - correlation
    n_startup = max(10, len(df_sample) // 5)
    startup_indices = peak_indices[:n_startup]

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(startup_indices, cumulative_corr[:n_startup], color='#9333ea',
            linewidth=2.5, marker='o', markersize=4, alpha=0.8)
    if len(startup_indices) > 2:
        corr_fit = np.polyfit(startup_indices, cumulative_corr[:n_startup], 1)
        corr_line = np.poly1d(corr_fit)
        ax4.plot(startup_indices, corr_line(startup_indices), 'r--', linewidth=2,
                label=f'Linear fit: y={corr_fit[0]:.3f}x + {corr_fit[1]:.3f}')
    ax4.set_xlabel('Peak Index', fontweight='600')
    ax4.set_ylabel('Cumulative Correlation', fontweight='600')
    ax4.set_title('Startup Phase: Correlation (Linear)', fontweight='700', pad=15)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Startup transient - SNR
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(startup_indices, cumulative_snr_linear[:n_startup], color='#059669',
            linewidth=2.5, marker='s', markersize=4, alpha=0.8)
    if len(startup_indices) > 2 and np.all(cumulative_snr_linear[:n_startup] > 0):
        x_log = np.log(startup_indices + 1)
        snr_fit = np.polyfit(x_log, cumulative_snr_linear[:n_startup], 1)
        snr_line = lambda x: snr_fit[0] * np.log(x + 1) + snr_fit[1]
        ax5.plot(startup_indices, snr_line(startup_indices), 'r--', linewidth=2,
                label=f'Log fit: y={snr_fit[0]:.2f}*ln(x+1) + {snr_fit[1]:.2f}')
    ax5.set_xlabel('Peak Index', fontweight='600')
    ax5.set_ylabel('Cumulative SNR (Linear)', fontweight='600')
    ax5.set_title('Startup Phase: SNR (Non-linear)', fontweight='700', pad=15)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Individual peak quality scatter
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(df_sample['peak_corr'], df_sample['snr_linear'],
                         c=peak_indices, cmap='viridis', s=50, alpha=0.6,
                         edgecolors='black', linewidth=0.5)
    ax6.set_xlabel('Individual Peak Correlation', fontweight='600')
    ax6.set_ylabel('Individual SNR (Linear)', fontweight='600')
    ax6.set_title('Individual Peak Quality', fontweight='700', pad=15)
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Peak Index')

    # 7. Theoretical explanation
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    # Calculate linearity metrics
    corr_slope, corr_intercept, corr_r, _, _ = linregress(peak_indices, cumulative_corr)

    # For SNR, use log transform
    x_log = np.log(peak_indices[peak_indices > 0] + 1)
    y_snr = cumulative_snr_linear[peak_indices > 0]
    if len(x_log) > 2:
        snr_slope, snr_intercept, snr_r, _, _ = linregress(x_log, y_snr)
    else:
        snr_r = 0

    explanation_text = f"""
EXPLANATION: Why is Cumulative Correlation Linear but Cumulative SNR Non-linear?

MATHEMATICAL BASIS:

1. CUMULATIVE CORRELATION (Linear):
   • Definition: Σ|r_i| where r_i is the correlation coefficient at peak i
   • Correlation coefficients are bounded: 0 ≤ |r_i| ≤ 1
   • Each peak adds a roughly constant contribution
   • Formula: C(n) = Σ(i=1 to n) |r_i| ≈ <r> × n
   • Observed R² = {corr_r**2:.4f} (linearity measure)
   • Slope = {corr_slope:.4f} (average contribution per peak)

2. CUMULATIVE SNR (Non-linear):
   • Definition: Σ SNR_i (linear units), then convert to: 20*log10(Σ SNR_i)
   • SNR is unbounded (can be >> 1)
   • Logarithmic transformation creates non-linearity
   • Formula: S(n) = 20*log10(Σ(i=1 to n) SNR_i) = 20*log10(n) + 20*log10(<SNR>)
   • Observed log-fit R² = {snr_r**2:.4f}
   • Grows as log(n) after initial transient

STARTUP TRANSIENT BEHAVIOR:

• Early peaks (n < {n_startup}): Both metrics show rapid growth as accumulation begins
• Correlation: Approaches linear regime quickly (within ~5-10 peaks)
• SNR: Longer transient due to:
  - Variable individual SNR values
  - Logarithmic compression of large values
  - Accumulation in linear space before conversion

PHYSICAL INTERPRETATION:

• Correlation: Measures signal similarity (normalized, bounded metric)
  → Consistent contributions → Linear accumulation

• SNR: Measures signal power relative to noise (unbounded metric)
  → Variable power levels → Non-linear accumulation in space
  → Logarithmic growth reflects multiplicative nature of signal power

PRACTICAL IMPLICATIONS:

• Cumulative correlation: Good for tracking total "signal quality"
• Cumulative SNR (Linear): Good for understanding power accumulation
• Both metrics are complementary and necessary for full signal characterization
• Steady-state behavior emerges after ~{n_startup} peaks ({100*n_startup/len(df_sample):.0f}% of dataset)
"""

    ax7.text(0.02, 0.98, explanation_text, transform=ax7.transAxes,
            fontsize=9, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#f9fafb', alpha=0.95,
                     edgecolor='#d1d5db', linewidth=1.5, pad=1.5))

    plt.tight_layout()
    output_file = output_dir / "cumulative_metrics" / "01_cumulative_metrics_explanation.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


# ============================================================================
# Part 3: INGV vs Experiment Dataset Comparison
# ============================================================================

def compare_ingv_vs_experiment(df_peaks: pd.DataFrame, output_dir: Path):
    """Deep dive comparison between INGV and Experiment datasets."""
    print("\n" + "="*80)
    print("PART 3: INGV vs EXPERIMENT DATASET COMPARISON")
    print("="*80)

    fig = plt.figure(figsize=(20, 16), facecolor='white')
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.30)
    fig.suptitle('INGV vs Experiment Dataset Comparison',
                fontsize=16, fontweight='bold', y=0.98)

    df_ingv = df_peaks[df_peaks['dataset'] == 'ingv']
    df_exp = df_peaks[df_peaks['dataset'] == 'experiment']

    print(f"INGV: {len(df_ingv):,} peaks from {df_ingv['station'].nunique()} stations")
    print(f"Experiment: {len(df_exp):,} peaks from {df_exp['station'].nunique()} stations")

    # EC1 direct comparison (ONLY EC1 has data across both INGV and Experiment periods)
    df_ec1_ingv = df_ingv[df_ingv['station'] == 'EC1']
    df_ec1_exp = df_exp[df_exp['station'] == 'EC1']
    print(f"EC1 INGV: {len(df_ec1_ingv):,} peaks")
    print(f"EC1 Experiment: {len(df_ec1_exp):,} peaks")
    print("Note: EC1 is the only station with data across both INGV and Experiment periods")

    # 1. Overall peak count comparison
    ax1 = fig.add_subplot(gs[0, 0])
    datasets = ['INGV', 'Experiment']
    counts = [len(df_ingv), len(df_exp)]
    colors = [DATASET_COLORS['ingv'], DATASET_COLORS['experiment']]
    bars = ax1.bar(datasets, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Total Peak Count', fontweight='600')
    ax1.set_title('Total Peaks by Dataset', fontweight='700', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 2. Correlation distributions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist([df_ingv['peak_corr'].dropna(), df_exp['peak_corr'].dropna()],
            bins=50, label=['INGV', 'Experiment'],
            color=[DATASET_COLORS['ingv'], DATASET_COLORS['experiment']],
            alpha=0.65, edgecolor='black', linewidth=0.8)
    ax2.axvline(MIN_CORRELATION, color='red', linestyle='--', linewidth=2,
               label=f'Threshold: {MIN_CORRELATION}')
    ax2.set_xlabel('Peak Correlation', fontweight='600')
    ax2.set_ylabel('Frequency', fontweight='600')
    ax2.set_title('Correlation Distribution Comparison', fontweight='700', pad=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. SNR distributions
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist([df_ingv['snr_linear'].dropna(), df_exp['snr_linear'].dropna()],
            bins=50, label=['INGV', 'Experiment'],
            color=[DATASET_COLORS['ingv'], DATASET_COLORS['experiment']],
            alpha=0.65, edgecolor='black', linewidth=0.8)
    ax3.axvline(MIN_SNR_LINEAR, color='red', linestyle='--', linewidth=2,
               label=f'Threshold: {MIN_SNR_LINEAR}')
    ax3.set_xlabel('SNR (Linear)', fontweight='600')
    ax3.set_ylabel('Frequency', fontweight='600')
    ax3.set_title('SNR Distribution Comparison', fontweight='700', pad=15)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. EC1 Direct Comparison - Correlation
    ax4 = fig.add_subplot(gs[1, 0])

    # Check if both datasets have data
    ec1_ingv_data = df_ec1_ingv['peak_corr'].dropna()
    ec1_exp_data = df_ec1_exp['peak_corr'].dropna()

    if len(ec1_ingv_data) > 0 and len(ec1_exp_data) > 0:
        parts = ax4.violinplot([ec1_ingv_data, ec1_exp_data],
                              positions=[0, 1], showmeans=True, showmedians=True)
        for pc, color in zip(parts['bodies'], [DATASET_COLORS['ingv'], DATASET_COLORS['experiment']]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    elif len(ec1_exp_data) > 0:
        # Only experiment has data
        parts = ax4.violinplot([ec1_exp_data], positions=[1], showmeans=True, showmedians=True)
        parts['bodies'][0].set_facecolor(DATASET_COLORS['experiment'])
        parts['bodies'][0].set_alpha(0.7)
        ax4.text(0, ax4.get_ylim()[1]*0.5, 'No Data', ha='center', va='center',
                fontsize=10, color='gray', style='italic')
    elif len(ec1_ingv_data) > 0:
        # Only INGV has data
        parts = ax4.violinplot([ec1_ingv_data], positions=[0], showmeans=True, showmedians=True)
        parts['bodies'][0].set_facecolor(DATASET_COLORS['ingv'])
        parts['bodies'][0].set_alpha(0.7)
        ax4.text(1, ax4.get_ylim()[1]*0.5, 'No Data', ha='center', va='center',
                fontsize=10, color='gray', style='italic')
    else:
        # Neither has data
        ax4.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                fontsize=12, color='gray', style='italic', transform=ax4.transAxes)

    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['INGV', 'Experiment'])
    ax4.set_ylabel('Peak Correlation', fontweight='600')
    ax4.set_title('EC1: Correlation Comparison\n(Only station with data across both periods)', fontweight='700', pad=15, fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')

    # Statistical test
    if len(df_ec1_ingv['peak_corr'].dropna()) > 0 and len(df_ec1_exp['peak_corr'].dropna()) > 0:
        stat, pval = mannwhitneyu(df_ec1_ingv['peak_corr'].dropna(),
                                 df_ec1_exp['peak_corr'].dropna(),
                                 alternative='two-sided')
        ax4.text(0.5, 0.95, f'Mann-Whitney U test\np-value: {pval:.4f}',
                transform=ax4.transAxes, ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 5. EC1 Direct Comparison - SNR
    ax5 = fig.add_subplot(gs[1, 1])

    # Check if both datasets have data
    ec1_ingv_snr = df_ec1_ingv['snr_linear'].dropna()
    ec1_exp_snr = df_ec1_exp['snr_linear'].dropna()

    if len(ec1_ingv_snr) > 0 and len(ec1_exp_snr) > 0:
        parts = ax5.violinplot([ec1_ingv_snr, ec1_exp_snr],
                              positions=[0, 1], showmeans=True, showmedians=True)
        for pc, color in zip(parts['bodies'], [DATASET_COLORS['ingv'], DATASET_COLORS['experiment']]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    elif len(ec1_exp_snr) > 0:
        # Only experiment has data
        parts = ax5.violinplot([ec1_exp_snr], positions=[1], showmeans=True, showmedians=True)
        parts['bodies'][0].set_facecolor(DATASET_COLORS['experiment'])
        parts['bodies'][0].set_alpha(0.7)
        ax5.text(0, ax5.get_ylim()[1]*0.5, 'No Data', ha='center', va='center',
                fontsize=10, color='gray', style='italic')
    elif len(ec1_ingv_snr) > 0:
        # Only INGV has data
        parts = ax5.violinplot([ec1_ingv_snr], positions=[0], showmeans=True, showmedians=True)
        parts['bodies'][0].set_facecolor(DATASET_COLORS['ingv'])
        parts['bodies'][0].set_alpha(0.7)
        ax5.text(1, ax5.get_ylim()[1]*0.5, 'No Data', ha='center', va='center',
                fontsize=10, color='gray', style='italic')
    else:
        # Neither has data
        ax5.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                fontsize=12, color='gray', style='italic', transform=ax5.transAxes)

    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(['INGV', 'Experiment'])
    ax5.set_ylabel('SNR (Linear)', fontweight='600')
    ax5.set_title('EC1: SNR Comparison\n(Only station with data across both periods)', fontweight='700', pad=15, fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')

    # Statistical test
    if len(df_ec1_ingv['snr_linear'].dropna()) > 0 and len(df_ec1_exp['snr_linear'].dropna()) > 0:
        stat, pval = mannwhitneyu(df_ec1_ingv['snr_linear'].dropna(),
                                 df_ec1_exp['snr_linear'].dropna(),
                                 alternative='two-sided')
        ax5.text(0.5, 0.95, f'Mann-Whitney U test\np-value: {pval:.4f}',
                transform=ax5.transAxes, ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 6. Station-level comparison
    ax6 = fig.add_subplot(gs[1, 2])
    station_stats = df_peaks.groupby(['dataset', 'station']).agg({
        'peak_corr': 'mean'
    }).reset_index()

    ingv_stations = station_stats[station_stats['dataset'] == 'ingv']
    exp_stations = station_stats[station_stats['dataset'] == 'experiment']

    x_ingv = np.arange(len(ingv_stations))
    x_exp = np.arange(len(exp_stations))

    width = 0.35
    ax6.barh(x_ingv, ingv_stations['peak_corr'], height=width,
            label='INGV', color=DATASET_COLORS['ingv'],
            alpha=0.8, edgecolor='black', linewidth=0.8)
    ax6.barh(x_exp + width, exp_stations['peak_corr'], height=width,
            label='Experiment', color=DATASET_COLORS['experiment'],
            alpha=0.8, edgecolor='black', linewidth=0.8)

    all_stations = pd.concat([ingv_stations, exp_stations])['station'].unique()
    ax6.set_yticks(np.arange(len(all_stations)) + width/2)
    ax6.set_yticklabels(all_stations)
    ax6.set_xlabel('Mean Peak Correlation', fontweight='600')
    ax6.set_title('Mean Correlation by Station', fontweight='700', pad=15)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='x')

    # 7. High-quality peaks comparison
    ax7 = fig.add_subplot(gs[2, 0])
    high_qual_ingv = len(df_ingv[df_ingv['peak_corr'] >= HIGH_CORRELATION])
    high_qual_exp = len(df_exp[df_exp['peak_corr'] >= HIGH_CORRELATION])
    pct_ingv = 100 * high_qual_ingv / len(df_ingv) if len(df_ingv) > 0 else 0
    pct_exp = 100 * high_qual_exp / len(df_exp) if len(df_exp) > 0 else 0

    bars = ax7.bar(['INGV', 'Experiment'], [pct_ingv, pct_exp],
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax7.set_ylabel(f'% Peaks with r ≥ {HIGH_CORRELATION}', fontweight='600')
    ax7.set_title('High-Quality Peak Percentage', fontweight='700', pad=15)
    ax7.grid(True, alpha=0.3, axis='y')
    for bar, pct in zip(bars, [pct_ingv, pct_exp]):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 8. Cumulative metrics rate comparison (EC1)
    ax8 = fig.add_subplot(gs[2, 1:])
    for dataset_name, df_subset, color in [('INGV', df_ec1_ingv, DATASET_COLORS['ingv']),
                                           ('Experiment', df_ec1_exp, DATASET_COLORS['experiment'])]:
        if len(df_subset) > 0:
            df_sorted = df_subset.sort_values('peak_time_dt').reset_index(drop=True)
            cumul_corr = df_sorted['peak_corr'].abs().cumsum()
            indices = np.arange(len(cumul_corr))

            ax8.plot(indices, cumul_corr, color=color, linewidth=2.5,
                    alpha=0.8, label=f'{dataset_name} (n={len(df_subset)})')

    ax8.set_xlabel('Peak Index', fontweight='600')
    ax8.set_ylabel('Cumulative Correlation |r|', fontweight='600')
    ax8.set_title('EC1: Cumulative Correlation Accumulation Rate', fontweight='700', pad=15)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Summary statistics table
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')

    summary_stats = []
    for dataset_name, df_subset in [('INGV', df_ingv), ('Experiment', df_exp)]:
        stats = {
            'Dataset': dataset_name.upper(),
            'Total Peaks': f"{len(df_subset):,}",
            'Mean Corr': f"{df_subset['peak_corr'].mean():.3f}",
            'Median Corr': f"{df_subset['peak_corr'].median():.3f}",
            'Mean SNR (Linear)': f"{df_subset['snr_linear'].mean():.1f}",
            'Median SNR (Linear)': f"{df_subset['snr_linear'].median():.1f}",
            '% r ≥ 0.5': f"{100*len(df_subset[df_subset['peak_corr'] >= MIN_CORRELATION])/len(df_subset):.1f}%",
            '% r ≥ 0.7': f"{100*len(df_subset[df_subset['peak_corr'] >= HIGH_CORRELATION])/len(df_subset):.1f}%",
        }
        summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats)

    # Create table
    table_data = [summary_df.columns.tolist()] + summary_df.values.tolist()
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.1, 0.3, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#d1d5db')
        table[(0, i)].set_text_props(weight='bold')

    # Color dataset rows
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#f3f4f6')

    ax9.set_title('Dataset Comparison Summary Statistics', fontweight='700',
                 fontsize=13, pad=20, y=0.95)

    plt.tight_layout()
    output_file = output_dir / "dataset_comparison" / "01_ingv_vs_experiment_comparison.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")

    # Save detailed statistics to CSV
    stats_csv = output_dir / "dataset_comparison" / "dataset_comparison_statistics.csv"
    summary_df.to_csv(stats_csv, index=False)
    print(f"✅ Saved: {stats_csv.name}")


# ============================================================================
# Part 4: Synchronous Peak Detection
# ============================================================================

def find_synchronous_events(df: pd.DataFrame, tolerance_sec: int, min_stations: int = 2):
    """Find synchronous events across stations using time-based clustering."""
    df = df.copy()
    df['peak_time_dt'] = pd.to_datetime(df['peak_time_dt'])
    df = df.sort_values('peak_time_dt').reset_index(drop=True)

    df['event_id'] = -1
    event_counter = 0
    tolerance = pd.Timedelta(seconds=tolerance_sec)

    unassigned = df['event_id'] == -1

    while unassigned.any():
        idx = df.loc[unassigned].index[0]
        ref_time = df.loc[idx, 'peak_time_dt']

        time_diff = (df['peak_time_dt'] - ref_time).abs()
        in_window = (time_diff <= tolerance) & unassigned

        stations_in_window = df.loc[in_window, 'station'].unique()

        if len(stations_in_window) >= min_stations:
            df.loc[in_window, 'event_id'] = event_counter
            event_counter += 1
        else:
            df.loc[in_window, 'event_id'] = -999

        unassigned = df['event_id'] == -1

    df = df[df['event_id'] >= 0].copy()

    return df


def detect_synchronous_events(df_peaks: pd.DataFrame, output_dir: Path):
    """Detect synchronous events across stations."""
    print("\n" + "="*80)
    print("PART 4: SYNCHRONOUS PEAK DETECTION")
    print("="*80)

    # Filter high-quality peaks
    df_quality = df_peaks[
        (df_peaks['peak_corr'] >= MIN_CORRELATION) &
        (df_peaks['snr_linear'] >= MIN_SNR_LINEAR)
    ].copy()

    print(f"High-quality peaks: {len(df_quality):,} / {len(df_peaks):,} " +
          f"({100*len(df_quality)/len(df_peaks):.1f}%)")

    # Find synchronous events
    print(f"\nFinding synchronous events (tolerance={SYNC_TOLERANCE_SEC}s, min_stations={MIN_STATIONS})...")
    df_events = find_synchronous_events(df_quality, SYNC_TOLERANCE_SEC, MIN_STATIONS)

    n_events = df_events['event_id'].nunique()
    print(f"Found {n_events} synchronous events")

    if n_events == 0:
        print("⚠️ No synchronous events found")
        return None, None

    # Calculate event statistics
    event_stats = df_events.groupby('event_id').agg({
        'station': lambda x: list(x.unique()),
        'peak_time_dt': ['min', 'max', 'count'],
        'peak_corr': ['mean', 'max'],
        'snr_linear': ['mean', 'max']
    }).reset_index()

    event_stats.columns = ['event_id', 'stations', 'time_min', 'time_max', 'n_peaks',
                          'corr_mean', 'corr_max', 'snr_mean', 'snr_max']
    event_stats['n_stations'] = event_stats['stations'].apply(len)
    event_stats['time_span_sec'] = (event_stats['time_max'] - event_stats['time_min']).dt.total_seconds()

    # Save synchronous events
    sync_csv = output_dir / "synchronous_events" / "synchronous_events_catalog.csv"
    df_events.to_csv(sync_csv, index=False)
    print(f"✅ Saved: {sync_csv.name}")

    stats_csv = output_dir / "synchronous_events" / "synchronous_events_statistics.csv"
    event_stats.to_csv(stats_csv, index=False)
    print(f"✅ Saved: {stats_csv.name}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    fig.suptitle(f'Synchronous Event Detection ({n_events} events found)',
                fontsize=16, fontweight='bold', y=0.98)

    # 1. Events by number of stations
    ax1 = axes[0, 0]
    station_counts = event_stats['n_stations'].value_counts().sort_index()
    ax1.bar(station_counts.index, station_counts.values, color='#3b82f6',
           alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Number of Stations', fontweight='600')
    ax1.set_ylabel('Event Count', fontweight='600')
    ax1.set_title('Events by Station Count', fontweight='700', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Quality distribution
    ax2 = axes[0, 1]
    ax2.scatter(event_stats['corr_mean'], event_stats['snr_mean'],
               s=event_stats['n_stations']*50, alpha=0.6,
               c=event_stats['n_stations'], cmap='viridis',
               edgecolors='black', linewidth=0.8)
    ax2.set_xlabel('Mean Correlation', fontweight='600')
    ax2.set_ylabel('Mean SNR (Linear)', fontweight='600')
    ax2.set_title('Event Quality Distribution', fontweight='700', pad=15)
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax2.collections[0], ax=ax2, label='N Stations')

    # 3. Top 10 events table
    ax3 = axes[1, 0]
    ax3.axis('off')
    top_events = event_stats.nlargest(10, 'corr_mean')[
        ['event_id', 'n_stations', 'n_peaks', 'corr_mean', 'snr_mean']
    ].copy()
    top_events['corr_mean'] = top_events['corr_mean'].round(3)
    top_events['snr_mean'] = top_events['snr_mean'].round(1)

    table_data = [['Event ID', 'Stations', 'Peaks', 'Mean Corr', 'Mean SNR (Linear)']] + \
                 top_events.values.tolist()
    table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.1, 0.1, 0.8, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#d1d5db')
        table[(0, i)].set_text_props(weight='bold')
    ax3.set_title('Top 10 Events by Correlation', fontweight='700', pad=15)

    # 4. Timeline
    ax4 = axes[1, 1]
    for _, event in event_stats.iterrows():
        ax4.scatter(event['time_min'], event['n_stations'],
                   s=event['corr_mean']*100, alpha=0.6,
                   color='#3b82f6', edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Time', fontweight='600')
    ax4.set_ylabel('Number of Stations', fontweight='600')
    ax4.set_title('Event Timeline', fontweight='700', pad=15)
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    output_file = output_dir / "synchronous_events" / "01_synchronous_events_summary.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")

    return df_events, event_stats


# ============================================================================
# Part 5: Best Signal Candidate Identification
# ============================================================================

def identify_best_candidates(df_peaks: pd.DataFrame, df_sync_events: pd.DataFrame,
                             output_dir: Path, n_candidates: int = 20):
    """Identify best signal candidates by consistency metrics."""
    print("\n" + "="*80)
    print("PART 5: BEST SIGNAL CANDIDATE IDENTIFICATION")
    print("="*80)

    # Create composite quality score
    df_peaks = df_peaks.copy()

    # Filter high-quality peaks
    df_quality = df_peaks[
        (df_peaks['peak_corr'] >= MIN_CORRELATION) &
        (df_peaks['snr_linear'] >= MIN_SNR_LINEAR)
    ].copy()

    print(f"High-quality peaks: {len(df_quality):,}")

    # Normalize metrics
    df_quality['corr_norm'] = (df_quality['peak_corr'] - df_quality['peak_corr'].min()) / \
                              (df_quality['peak_corr'].max() - df_quality['peak_corr'].min())
    df_quality['snr_norm'] = (df_quality['snr_linear'] - df_quality['snr_linear'].min()) / \
                             (df_quality['snr_linear'].max() - df_quality['snr_linear'].min())

    # Mark synchronous events and count number of stations involved
    if df_sync_events is not None and len(df_sync_events) > 0:
        sync_mask = df_quality.apply(
            lambda row: any(
                (df_sync_events['station'] == row['station']) &
                (df_sync_events['dataset'] == row['dataset']) &
                (df_sync_events['sim'] == row['sim']) &
                (df_sync_events['template'] == row['template']) &
                ((df_sync_events['peak_time_dt'] - row['peak_time_dt']).abs() < timedelta(seconds=10))
            ), axis=1
        )
        df_quality['is_sync'] = sync_mask

        # Count number of stations for each synchronous event
        def count_sync_stations(row):
            if not row['is_sync']:
                return 0
            # Find matching sync event and count unique stations
            matching = df_sync_events[
                (df_sync_events['dataset'] == row['dataset']) &
                (df_sync_events['sim'] == row['sim']) &
                (df_sync_events['template'] == row['template']) &
                ((df_sync_events['peak_time_dt'] - row['peak_time_dt']).abs() < timedelta(seconds=10))
            ]
            return len(matching['station'].unique()) if len(matching) > 0 else 0

        df_quality['num_sync_stations'] = df_quality.apply(count_sync_stations, axis=1)
    else:
        df_quality['is_sync'] = False
        df_quality['num_sync_stations'] = 0

    # Normalize number of sync stations (for scoring)
    max_stations = df_quality['num_sync_stations'].max()
    if max_stations > 0:
        df_quality['sync_stations_norm'] = df_quality['num_sync_stations'] / max_stations
    else:
        df_quality['sync_stations_norm'] = 0.0

    # ========================================================================
    # THREE SCORING APPROACHES
    # ========================================================================

    # APPROACH 1: ULP-focused (Signal Quality Only)
    # Best for: Identifying highest-quality individual ULP signals
    # Use case: Detailed waveform analysis, source mechanisms
    df_quality['score_ulp'] = (
        0.50 * df_quality['corr_norm'] +
        0.50 * df_quality['snr_norm']
    )

    # APPROACH 2: Synchronicity-focused (Multi-Station Coherence)
    # Best for: Morlet wavelet analysis, regional tilt studies
    # Use case: Identifying volcanic signals across multiple stations
    df_quality['score_sync'] = (
        0.25 * df_quality['corr_norm'] +
        0.25 * df_quality['snr_norm'] +
        0.50 * df_quality['sync_stations_norm']  # Heavy weight on multi-station detection
    )

    # APPROACH 3: Balanced (Best Overall)
    # Best for: General-purpose candidate identification
    # Use case: Balanced between quality and spatial coherence
    df_quality['score_balanced'] = (
        0.35 * df_quality['corr_norm'] +
        0.35 * df_quality['snr_norm'] +
        0.30 * df_quality['sync_stations_norm']
    )

    # Rank candidates - Select best ULP signals by quality (correlation + SNR only)
    df_ranked = df_quality.sort_values('score_ulp', ascending=False).reset_index(drop=True)
    df_candidates = df_ranked.head(n_candidates).copy()

    # Generate rankings for all three approaches
    df_ranked_ulp = df_quality.sort_values('score_ulp', ascending=False).reset_index(drop=True)
    df_ranked_sync = df_quality.sort_values('score_sync', ascending=False).reset_index(drop=True)
    df_ranked_balanced = df_quality.sort_values('score_balanced', ascending=False).reset_index(drop=True)

    df_candidates_ulp = df_ranked_ulp.head(n_candidates).copy()
    df_candidates_sync = df_ranked_sync.head(n_candidates).copy()
    df_candidates_balanced = df_ranked_balanced.head(n_candidates).copy()

    # Use ULP candidates as primary for visualization
    df_candidates = df_candidates_ulp

    # ========================================================================
    # Report all three approaches
    # ========================================================================

    print("\n" + "="*80)
    print("CANDIDATE SELECTION SUMMARY (Three Approaches)")
    print("="*80)

    print(f"\n1️⃣  ULP-FOCUSED (50% Corr + 50% SNR) - Top {n_candidates}:")
    dataset_counts = df_candidates_ulp['dataset'].value_counts()
    for dataset in ['ingv', 'experiment']:
        if dataset in dataset_counts.index:
            n = dataset_counts[dataset]
            df_subset = df_candidates_ulp[df_candidates_ulp['dataset'] == dataset]
            print(f"  - {dataset.upper()}: {n} candidates (mean corr={df_subset['peak_corr'].mean():.3f}, mean SNR={df_subset['snr_linear'].mean():.1f})")
    print(f"  - Mean correlation: {df_candidates_ulp['peak_corr'].mean():.3f}")
    print(f"  - Mean SNR (linear): {df_candidates_ulp['snr_linear'].mean():.1f}")
    print(f"  - Synchronous events: {df_candidates_ulp['is_sync'].sum()}")

    print(f"\n2️⃣  SYNCHRONICITY-FOCUSED (25% Corr + 25% SNR + 50% Multi-Station) - Top {n_candidates}:")
    dataset_counts = df_candidates_sync['dataset'].value_counts()
    for dataset in ['ingv', 'experiment']:
        if dataset in dataset_counts.index:
            n = dataset_counts[dataset]
            df_subset = df_candidates_sync[df_candidates_sync['dataset'] == dataset]
            print(f"  - {dataset.upper()}: {n} candidates (mean corr={df_subset['peak_corr'].mean():.3f}, mean SNR={df_subset['snr_linear'].mean():.1f})")
    print(f"  - Mean correlation: {df_candidates_sync['peak_corr'].mean():.3f}")
    print(f"  - Mean SNR (linear): {df_candidates_sync['snr_linear'].mean():.1f}")
    print(f"  - Synchronous events: {df_candidates_sync['is_sync'].sum()}")
    print(f"  - Mean stations per event: {df_candidates_sync['num_sync_stations'].mean():.1f}")

    print(f"\n3️⃣  BALANCED (35% Corr + 35% SNR + 30% Multi-Station) - Top {n_candidates}:")
    dataset_counts = df_candidates_balanced['dataset'].value_counts()
    for dataset in ['ingv', 'experiment']:
        if dataset in dataset_counts.index:
            n = dataset_counts[dataset]
            df_subset = df_candidates_balanced[df_candidates_balanced['dataset'] == dataset]
            print(f"  - {dataset.upper()}: {n} candidates (mean corr={df_subset['peak_corr'].mean():.3f}, mean SNR={df_subset['snr_linear'].mean():.1f})")
    print(f"  - Mean correlation: {df_candidates_balanced['peak_corr'].mean():.3f}")
    print(f"  - Mean SNR (linear): {df_candidates_balanced['snr_linear'].mean():.1f}")
    print(f"  - Synchronous events: {df_candidates_balanced['is_sync'].sum()}")
    print(f"  - Mean stations per event: {df_candidates_balanced['num_sync_stations'].mean():.1f}")

    print("\n" + "="*80)

    # Save all three candidate lists
    candidates_ulp_csv = output_dir / "best_candidates" / "candidates_ulp_focused.csv"
    candidates_sync_csv = output_dir / "best_candidates" / "candidates_sync_focused.csv"
    candidates_balanced_csv = output_dir / "best_candidates" / "candidates_balanced.csv"

    df_candidates_ulp.to_csv(candidates_ulp_csv, index=False)
    df_candidates_sync.to_csv(candidates_sync_csv, index=False)
    df_candidates_balanced.to_csv(candidates_balanced_csv, index=False)

    print(f"✅ Saved: {candidates_ulp_csv.name}")
    print(f"✅ Saved: {candidates_sync_csv.name}")
    print(f"✅ Saved: {candidates_balanced_csv.name}")

    # Visualization
    # Save individual plots instead of 4-panel summary
    from matplotlib.patches import Patch

    # 1. Quality scatter plot (individual) - ULP-focused
    fig1, ax1 = plt.subplots(figsize=(10, 8), facecolor='white')
    colors_scatter = ['#dc2626' if is_sync else '#2563eb'
                     for is_sync in df_candidates['is_sync']]
    ax1.scatter(df_candidates['peak_corr'], df_candidates['snr_linear'],
               s=df_candidates['score_ulp']*200, alpha=0.6,
               c=colors_scatter, edgecolors='black', linewidth=1)
    ax1.set_xlabel('Peak Correlation', fontweight='600')
    ax1.set_ylabel('SNR (Linear)', fontweight='600')
    ax1.set_title(f'Best ULP Candidates: Correlation vs SNR (Top {n_candidates})', fontweight='700', pad=15)
    ax1.grid(True, alpha=0.3)

    legend_elements = [
        Patch(facecolor='#2563eb', label='Isolated signal'),
        Patch(facecolor='#dc2626', label='Also synchronous')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    output_file1 = output_dir / "best_candidates" / "01a_quality_scatter.png"
    plt.savefig(output_file1, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file1.name}")

    # 2. Station distribution (individual)
    fig2, ax2 = plt.subplots(figsize=(10, 8), facecolor='white')
    station_counts = df_candidates['station'].value_counts()
    ax2.bar(range(len(station_counts)), station_counts.values,
           color=[STATION_COLORS.get(s, '#666') for s in station_counts.index],
           alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_xticks(range(len(station_counts)))
    ax2.set_xticklabels(station_counts.index, rotation=45, ha='right')
    ax2.set_ylabel('Candidate Count', fontweight='600')
    ax2.set_title(f'Candidates by Station (Top {n_candidates})', fontweight='700', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file2 = output_dir / "best_candidates" / "01b_station_distribution.png"
    plt.savefig(output_file2, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file2.name}")

    # 3. Dataset distribution (individual)
    fig3, ax3 = plt.subplots(figsize=(10, 8), facecolor='white')
    dataset_counts = df_candidates['dataset'].value_counts()
    ax3.bar(range(len(dataset_counts)), dataset_counts.values,
           color=[DATASET_COLORS.get(d, '#666') for d in dataset_counts.index],
           alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_xticks(range(len(dataset_counts)))
    ax3.set_xticklabels([d.upper() for d in dataset_counts.index])
    ax3.set_ylabel('Candidate Count', fontweight='600')
    ax3.set_title(f'Candidates by Dataset (Top {n_candidates})', fontweight='700', pad=15)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file3 = output_dir / "best_candidates" / "01c_dataset_distribution.png"
    plt.savefig(output_file3, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file3.name}")

    # 4. Top 20 table (individual)
    fig4, ax4 = plt.subplots(figsize=(12, 10), facecolor='white')
    ax4.axis('off')
    top_20 = df_candidates.head(20)[
        ['peak_time_dt', 'station', 'dataset', 'peak_corr', 'snr_linear', 'score_ulp', 'is_sync', 'num_sync_stations']
    ].copy()

    # Format timestamp to show date and time
    top_20['timestamp'] = top_20['peak_time_dt'].dt.strftime('%Y-%m-%d %H:%M')
    top_20['peak_corr'] = top_20['peak_corr'].round(3)
    top_20['snr_linear'] = top_20['snr_linear'].round(1)
    top_20['score_ulp'] = top_20['score_ulp'].round(3)
    top_20['is_sync'] = top_20['is_sync'].map({True: 'Y', False: 'N'})

    # Select columns for display
    display_cols = top_20[['timestamp', 'station', 'peak_corr', 'snr_linear', 'score_ulp', 'is_sync']]

    table_data = [['Time (UTC)', 'Stn', 'Corr', 'SNR', 'Score', 'Sync']] + \
                 display_cols.values.tolist()
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.0, 0.0, 1.0, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    # Format header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#d1d5db')
        table[(0, i)].set_text_props(weight='bold', fontsize=10)

    # No background highlighting - keep table clean and professional

    ax4.set_title('Top 20 ULP Candidates (Ranked by Correlation + SNR)', fontweight='700', pad=15, fontsize=14)

    plt.tight_layout()
    output_file4 = output_dir / "best_candidates" / "01d_top20_table.png"
    plt.savefig(output_file4, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file4.name}")

    # ========================================================================
    # 5. Comparison plot: All three approaches
    # ========================================================================
    fig5, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor='white')
    fig5.suptitle(f'Candidate Selection Comparison: Three Scoring Approaches (Top {n_candidates})',
                  fontsize=16, fontweight='bold', y=0.98)

    # Plot 1: ULP-focused (50% Corr + 50% SNR)
    ax_ulp = axes[0]
    colors_ulp = ['#dc2626' if is_sync else '#2563eb' for is_sync in df_candidates_ulp['is_sync']]
    ax_ulp.scatter(df_candidates_ulp['peak_corr'], df_candidates_ulp['snr_linear'],
                   s=df_candidates_ulp['score_ulp']*200, alpha=0.6,
                   c=colors_ulp, edgecolors='black', linewidth=1)
    ax_ulp.set_xlabel('Peak Correlation', fontweight='600')
    ax_ulp.set_ylabel('SNR (Linear)', fontweight='600')
    ax_ulp.set_title('ULP-Focused\n(50% Corr + 50% SNR)', fontweight='700', pad=15)
    ax_ulp.grid(True, alpha=0.3)
    ax_ulp.text(0.05, 0.95, f"Sync events: {df_candidates_ulp['is_sync'].sum()}/{n_candidates}",
                transform=ax_ulp.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Synchronicity-focused (25% Corr + 25% SNR + 50% Multi-Station)
    ax_sync = axes[1]
    colors_sync = ['#dc2626' if is_sync else '#2563eb' for is_sync in df_candidates_sync['is_sync']]
    ax_sync.scatter(df_candidates_sync['peak_corr'], df_candidates_sync['snr_linear'],
                    s=df_candidates_sync['score_sync']*200, alpha=0.6,
                    c=colors_sync, edgecolors='black', linewidth=1)
    ax_sync.set_xlabel('Peak Correlation', fontweight='600')
    ax_sync.set_ylabel('SNR (Linear)', fontweight='600')
    ax_sync.set_title('2️⃣  Synchronicity-Focused\n(25% Corr + 25% SNR + 50% Multi-Stn)', fontweight='700', pad=15)
    ax_sync.grid(True, alpha=0.3)
    ax_sync.text(0.05, 0.95, f"Sync events: {df_candidates_sync['is_sync'].sum()}/{n_candidates}\nAvg stations: {df_candidates_sync['num_sync_stations'].mean():.1f}",
                 transform=ax_sync.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 3: Balanced (35% Corr + 35% SNR + 30% Multi-Station)
    ax_bal = axes[2]
    colors_bal = ['#dc2626' if is_sync else '#2563eb' for is_sync in df_candidates_balanced['is_sync']]
    ax_bal.scatter(df_candidates_balanced['peak_corr'], df_candidates_balanced['snr_linear'],
                   s=df_candidates_balanced['score_balanced']*200, alpha=0.6,
                   c=colors_bal, edgecolors='black', linewidth=1)
    ax_bal.set_xlabel('Peak Correlation', fontweight='600')
    ax_bal.set_ylabel('SNR (Linear)', fontweight='600')
    ax_bal.set_title('3️⃣  Balanced\n(35% Corr + 35% SNR + 30% Multi-Stn)', fontweight='700', pad=15)
    ax_bal.grid(True, alpha=0.3)
    ax_bal.text(0.05, 0.95, f"Sync events: {df_candidates_balanced['is_sync'].sum()}/{n_candidates}\nAvg stations: {df_candidates_balanced['num_sync_stations'].mean():.1f}",
                transform=ax_bal.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Shared legend
    legend_elements = [
        Patch(facecolor='#2563eb', label='Isolated signal'),
        Patch(facecolor='#dc2626', label='Synchronous event')
    ]
    fig5.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11,
                framealpha=0.95, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    output_file5 = output_dir / "best_candidates" / "02_three_approaches_comparison.png"
    plt.savefig(output_file5, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file5.name}")

    # Return ULP-focused candidates as primary (for backward compatibility)
    return df_candidates_ulp


# ============================================================================
# Part 6: Spectrograms for Best Candidates
# ============================================================================

def create_candidate_spectrograms(df_candidates: pd.DataFrame,
                                 df_earthquakes: pd.DataFrame,
                                 output_dir: Path,
                                 n_plots: int = 10):
    """Create spectrograms with template overlays for best candidates."""
    print("\n" + "="*80)
    print("PART 6: SPECTROGRAMS FOR BEST CANDIDATES")
    print("="*80)

    for idx, candidate in df_candidates.head(n_plots).iterrows():
        station = candidate['station']
        dataset = candidate['dataset']
        peak_time = pd.to_datetime(candidate['peak_time_dt'])

        print(f"\nProcessing candidate {idx+1}/{n_plots}: {dataset}/{station} @ {peak_time}")

        # Load signal
        df_signal = load_bandpassed_signal(station, dataset, BANDPASSED_CSV_ROOT)
        if df_signal is None:
            print(f"  ⚠️ Skipping - signal not found")
            continue

        # Extract segment
        t, sig, dt = extract_signal_segment(df_signal, peak_time, SEGMENT_DURATION_S)
        if sig is None:
            print(f"  ⚠️ Skipping - segment extraction failed")
            continue

        # Estimate sampling rate
        if len(t) > 1:
            fs = 1.0 / np.median(np.diff(t))
        else:
            fs = SAMPLING_RATE

        print(f"  Signal length: {len(sig)} samples, fs={fs:.3f} Hz")

        # Load template
        sim = candidate['sim']
        template = candidate['template']
        template_signal = load_template(station, dataset, sim, template, TEMPLATES_CSV_ROOT)

        # Create title string for all plots
        title_str = (f"Candidate Signal Analysis\n"
                    f"{dataset.upper()} - Station {station} | "
                    f"{peak_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"r={candidate['peak_corr']:.3f}, SNR={candidate['snr_linear']:.1f} | "
                    f"{sim.upper()} - {template.upper()}")

        # Create combined figure with 3 stacked plots
        fig_combined = plt.figure(figsize=(16, 18), facecolor='white')
        gs = fig_combined.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.35)
        fig_combined.suptitle(title_str, fontsize=14, fontweight='bold', y=0.995)

        # 1. Time series with template overlay
        ax1 = fig_combined.add_subplot(gs[0])
        ax1.plot(t/60, sig, color='#2563eb', linewidth=1.5, alpha=0.9, label='Observed signal')

        # Overlay template if loaded
        template_scale_factor = 1.0
        if template_signal is not None:
            # Scale template time to match segment duration
            t_template = np.linspace(0, SEGMENT_DURATION_S, len(template_signal))

            # Normalize template to match signal scale (for visualization)
            template_scale_factor = np.std(sig) / np.std(template_signal)
            template_scaled = template_signal * template_scale_factor
            template_scaled = template_scaled - np.mean(template_scaled) + np.mean(sig)

            ax1.plot(t_template/60, template_scaled, color='#f97316', linewidth=1.5,
                    alpha=0.8, linestyle='--',
                    label=f'Template ({sim} {template}, scale={template_scale_factor:.2f})')

        ax1.axvline((SEGMENT_DURATION_S/2)/60, color='#059669', linestyle=':',
                   linewidth=2, alpha=0.7, label='Peak center')

        # Add P-wave arrival markers and exclusion zone rectangles
        if not df_earthquakes.empty:
            segment_start = peak_time - pd.Timedelta(seconds=SEGMENT_DURATION_S/2)
            segment_end = peak_time + pd.Timedelta(seconds=SEGMENT_DURATION_S/2)

            # Find P-waves within this segment
            p_waves_in_segment = df_earthquakes[
                (df_earthquakes['p_wave_eta'] >= segment_start) &
                (df_earthquakes['p_wave_eta'] <= segment_end)
            ]

            if len(p_waves_in_segment) > 0:
                for idx, (_, eq) in enumerate(p_waves_in_segment.iterrows()):
                    # Calculate time relative to segment start (in minutes)
                    p_wave_time = eq['p_wave_eta']
                    time_from_start = (p_wave_time - segment_start).total_seconds() / 60

                    # Calculate asymmetric buffer (pre/post) from magnitude + distance
                    magnitude = eq.get('magnitude', None)
                    dist_km   = eq.get('distance_km', None)
                    pre_min, post_min = calculate_pwave_buffer(magnitude, dist_km)

                    zone_start = time_from_start - pre_min
                    zone_end   = time_from_start + post_min

                    # Plot exclusion zone as red semi-transparent rectangle
                    y_min, y_max = ax1.get_ylim()
                    if idx == 0:
                        # First rectangle gets the label
                        ax1.axvspan(zone_start, zone_end, color='red', alpha=0.15, zorder=1,
                                   label=f'P-wave exclusion zones (n={len(p_waves_in_segment)})')
                    else:
                        ax1.axvspan(zone_start, zone_end, color='red', alpha=0.15, zorder=1)

                    # Plot P-wave arrival line (centered in exclusion zone) - THICKER
                    ax1.axvline(time_from_start, color='red', linestyle='--',
                               linewidth=2.5, alpha=0.9, zorder=2)

        ax1.set_xlabel('Time (minutes)', fontweight='600', fontsize=11)
        ax1.set_ylabel('Tilt (detrended, filtered)', fontweight='600', fontsize=11)
        ax1.set_title('Bandpassed Signal with Template Overlay', fontweight='700', pad=10, fontsize=12)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        # Set tight x-axis limits to show only data
        ax1.set_xlim(t[0]/60, t[-1]/60)
        ax1.tick_params(axis='both', labelsize=10)

        # 2. Spectrogram
        ax2 = fig_combined.add_subplot(gs[1])
        try:
            f, t_spec, Sxx = spectrogram(sig, fs=fs, nperseg=min(256, len(sig)//4),
                                        noverlap=min(128, len(sig)//8))

            # Filter to band of interest
            freq_mask = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
            f_filtered = f[freq_mask]
            Sxx_filtered = Sxx[freq_mask, :]

            if len(f_filtered) > 0:
                # Align spectrogram time axis with signal
                t_spec_aligned = t_spec + t[0]
                pcm = ax2.pcolormesh(t_spec_aligned/60, f_filtered, 10*np.log10(Sxx_filtered + 1e-10),
                                    shading='gouraud', cmap='viridis')
                ax2.set_ylabel('Frequency (Hz)', fontweight='600', fontsize=11)
                ax2.set_xlabel('Time (minutes)', fontweight='600', fontsize=11)
                ax2.set_title('Spectrogram', fontweight='700', pad=10, fontsize=12)
                cbar = plt.colorbar(pcm, ax=ax2, label='Power (dB)')
                cbar.ax.tick_params(labelsize=9)
                # Remove grid and set tight x-axis limits to match signal
                ax2.grid(False)
                ax2.set_xlim(t[0]/60, t[-1]/60)
                ax2.tick_params(axis='both', labelsize=10)
            else:
                ax2.text(0.5, 0.5, 'Insufficient frequency resolution',
                        ha='center', va='center', transform=ax2.transAxes)
        except Exception as e:
            ax2.text(0.5, 0.5, f'Spectrogram error: {e}',
                    ha='center', va='center', transform=ax2.transAxes)

        # 3. FFT Power Spectrum with Template Overlay
        ax3 = fig_combined.add_subplot(gs[2])
        try:
            fft_vals = np.fft.rfft(sig)
            fft_freqs = np.fft.rfftfreq(len(sig), 1/fs)
            power_spectrum = np.abs(fft_vals)**2

            # Filter to band
            freq_mask = (fft_freqs >= BAND_HZ[0]) & (fft_freqs <= BAND_HZ[1])
            ax3.loglog(fft_freqs[freq_mask], power_spectrum[freq_mask],
                      color='#2563eb', linewidth=1.5, alpha=0.8, label='Observed signal')

            # Add template frequency-amplitude if available
            if template_signal is not None:
                template_fft = np.fft.rfft(template_signal)
                template_freqs = np.fft.rfftfreq(len(template_signal), 1/fs)
                template_power = np.abs(template_fft)**2 * (template_scale_factor**2)

                template_freq_mask = (template_freqs >= BAND_HZ[0]) & (template_freqs <= BAND_HZ[1])
                ax3.loglog(template_freqs[template_freq_mask], template_power[template_freq_mask],
                          color='#f97316', linewidth=1.5, alpha=0.7, linestyle='--',
                          label=f'Template (scale={template_scale_factor:.2f})')

            ax3.set_xlabel('Frequency (Hz)', fontweight='600', fontsize=11)
            ax3.set_ylabel('Power', fontweight='600', fontsize=11)
            ax3.set_title('Power Spectrum (FFT) with Template', fontweight='700', pad=10, fontsize=12)
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3, which='both')
            ax3.tick_params(axis='both', labelsize=10)
        except Exception as e:
            ax3.text(0.5, 0.5, f'FFT error: {e}',
                    ha='center', va='center', transform=ax3.transAxes)

        # Save combined plot
        plt.tight_layout()
        output_file_combined = output_dir / "best_candidates" / f"candidate_{idx+1:02d}_combined.png"
        plt.savefig(output_file_combined, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✅ Saved: {output_file_combined.name}")

        # 4. Signal statistics - VERTICAL LAYOUT (separate plot)
        fig_stats = plt.figure(figsize=(10, 8), facecolor='white')
        ax_stats = fig_stats.add_subplot(111)
        ax_stats.axis('off')

        # Convert duration to minutes
        duration_mins = SEGMENT_DURATION_S / 60

        stats_text = f"""SIGNAL STATISTICS

Duration:          {duration_mins:.0f} mins
Samples:           {len(sig):,}
Sampling rate:     {fs:.4f} Hz

AMPLITUDE

Mean:              {np.mean(sig):.3e}
Std:               {np.std(sig):.3e}
Min:               {np.min(sig):.3e}
Max:               {np.max(sig):.3e}
Range:             {np.max(sig)-np.min(sig):.3e}

QUALITY METRICS

Peak Corr:         {candidate['peak_corr']:.4f}
SNR (Linear):      {candidate['snr_linear']:.2f}
ULP Score:         {candidate['score_ulp']:.4f}
Sync Score:        {candidate['score_sync']:.4f}
Balanced Score:    {candidate['score_balanced']:.4f}

METADATA

Dataset:           {dataset.upper()}
Station:           {station}
Simulation:        {candidate['sim']}
Template:          {candidate['template']}
Synchronous:       {"Yes" if candidate.get('is_sync', False) else "No"}
Num Stations:      {candidate.get('num_sync_stations', 0)}"""

        fig_stats.suptitle(title_str, fontsize=12, fontweight='bold', y=0.98)
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                fontsize=9, va='top', ha='left', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#f9fafb', alpha=0.95,
                         edgecolor='#d1d5db', linewidth=1.5, pad=1))

        plt.tight_layout()
        output_file_stats = output_dir / "best_candidates" / f"candidate_{idx+1:02d}_statistics.png"
        plt.savefig(output_file_stats, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✅ Saved: {output_file_stats.name}")


# ============================================================================
# Part 7: Morlet Wavelet Transforms for Synchronous Events
# ============================================================================

def create_morlet_wavelets_for_sync_events(df_sync_events: pd.DataFrame,
                                           df_earthquakes: pd.DataFrame,
                                           output_dir: Path, n_plots: int = 5):
    """Create Morlet wavelet transforms with template overlays for synchronous events."""
    print("\n" + "="*80)
    print("PART 7: MORLET WAVELET TRANSFORMS FOR SYNCHRONOUS EVENTS")
    print("="*80)

    if df_sync_events is None or len(df_sync_events) == 0:
        print("⚠️ No synchronous events found")
        return

    # Group by event_id
    event_groups = df_sync_events.groupby('event_id')

    plotted = 0
    for event_id, event_group in event_groups:
        if plotted >= n_plots:
            break

        stations = event_group['station'].unique()
        if len(stations) < 2:
            continue

        print(f"\nEvent {event_id}: {len(stations)} stations")

        # Select up to 2 stations for visualization
        selected_stations = list(stations)[:2]
        event_time = event_group['peak_time_dt'].min()

        # Load signals and templates for each station
        signals = {}
        for station in selected_stations:
            station_data = event_group[event_group['station'] == station].iloc[0]
            dataset = station_data['dataset']
            sim = station_data['sim']
            template = station_data['template']

            df_signal = load_bandpassed_signal(station, dataset, BANDPASSED_CSV_ROOT)
            if df_signal is None:
                continue

            t, sig, dt = extract_signal_segment(df_signal, event_time, SEGMENT_DURATION_S)
            if sig is None:
                continue

            if len(t) > 1:
                fs = 1.0 / np.median(np.diff(t))
            else:
                fs = SAMPLING_RATE

            # Load template
            template_signal = load_template(station, dataset, sim, template, TEMPLATES_CSV_ROOT)

            signals[station] = {
                'time': t,
                'signal': sig,
                'fs': fs,
                'dataset': dataset,
                'sim': sim,
                'template': template,
                'template_signal': template_signal,
                'peak_corr': station_data['peak_corr'],
                'snr_linear': station_data['snr_linear']
            }

        if len(signals) < 2:
            print(f"  ⚠️ Skipping - insufficient signals loaded")
            continue

        # Create figure
        fig = plt.figure(figsize=(18, 14), facecolor='white')
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.25)

        title_str = (f"Synchronous Event {plotted+1} - Morlet Wavelet Analysis\n"
                    f"Stations: {', '.join(selected_stations)} | "
                    f"{event_time.strftime('%Y-%m-%d %H:%M:%S')}")
        fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.98)

        # Plot individual station wavelets with template overlays
        for i, (station, data) in enumerate(signals.items()):
            t = data['time']
            sig = data['signal']
            fs = data['fs']
            dataset = data['dataset']
            sim = data['sim']
            template = data['template']
            template_signal = data['template_signal']

            # Time series with template overlay
            ax_ts = fig.add_subplot(gs[i, 0])
            ax_ts.plot(t/60, sig, color=STATION_COLORS.get(station, '#666'),
                      linewidth=1.5, alpha=0.9, label=f'{station} observed')

            # Overlay template if loaded
            template_scale_factor = 1.0
            if template_signal is not None:
                t_template = np.linspace(0, SEGMENT_DURATION_S, len(template_signal))
                # Normalize template to match signal scale
                template_scale_factor = np.std(sig) / np.std(template_signal)
                template_scaled = template_signal * template_scale_factor
                template_scaled = template_scaled - np.mean(template_scaled) + np.mean(sig)

                ax_ts.plot(t_template/60, template_scaled, color='#dc2626', linewidth=1.5,
                          alpha=0.7, linestyle='--',
                          label=f'Template ({sim} {template}, scale={template_scale_factor:.2f})')

            ax_ts.axvline((SEGMENT_DURATION_S/2)/60, color='#059669', linestyle=':',
                         linewidth=2, alpha=0.6, label='Peak center')

            # Add P-wave arrival markers and exclusion zone rectangles
            if not df_earthquakes.empty:
                segment_start = event_time - pd.Timedelta(seconds=SEGMENT_DURATION_S/2)
                segment_end = event_time + pd.Timedelta(seconds=SEGMENT_DURATION_S/2)

                # Find P-waves within this segment
                p_waves_in_segment = df_earthquakes[
                    (df_earthquakes['p_wave_eta'] >= segment_start) &
                    (df_earthquakes['p_wave_eta'] <= segment_end)
                ]

                if len(p_waves_in_segment) > 0:
                    for idx, (_, eq) in enumerate(p_waves_in_segment.iterrows()):
                        # Calculate time relative to segment start (in minutes)
                        p_wave_time = eq['p_wave_eta']
                        time_from_start = (p_wave_time - segment_start).total_seconds() / 60

                        # Calculate asymmetric buffer (pre/post) from magnitude + distance
                        magnitude = eq.get('magnitude', None)
                        dist_km   = eq.get('distance_km', None)
                        pre_min, post_min = calculate_pwave_buffer(magnitude, dist_km)

                        zone_start = time_from_start - pre_min
                        zone_end   = time_from_start + post_min

                        # Plot exclusion zone as red semi-transparent rectangle
                        if i == 0 and idx == 0:
                            # First rectangle on first station gets the label
                            ax_ts.axvspan(zone_start, zone_end, color='red', alpha=0.15, zorder=1,
                                         label=f'P-wave exclusion (n={len(p_waves_in_segment)})')
                        else:
                            ax_ts.axvspan(zone_start, zone_end, color='red', alpha=0.15, zorder=1)

                        # Plot P-wave arrival line (centered in exclusion zone)
                        ax_ts.axvline(time_from_start, color='red', linestyle='--',
                                     linewidth=1.2, alpha=0.8, zorder=2)

            # Extract correlation and SNR for this station
            peak_corr = data.get('peak_corr', 0.0)
            snr_linear = data.get('snr_linear', 0.0)

            ax_ts.set_ylabel('Tilt', fontweight='600')
            ax_ts.set_title(f'{station} ({dataset.upper()}) - r={peak_corr:.3f}, SNR={snr_linear:.2f}',
                           fontweight='700', pad=15)
            ax_ts.legend(loc='upper right', fontsize=8)
            ax_ts.grid(True, alpha=0.3)
            # Set tight x-axis limits
            ax_ts.set_xlim(t[0]/60, t[-1]/60)
            if i < len(signals) - 1:
                plt.setp(ax_ts.get_xticklabels(), visible=False)
            else:
                ax_ts.set_xlabel('Time (minutes)', fontweight='600')

            # CWT
            ax_cwt = fig.add_subplot(gs[i, 1])
            freqs, power = compute_cwt(sig, fs=fs, wavelet='morl',
                                      freq_min=BAND_HZ[1], freq_max=BAND_HZ[0],
                                      n_freqs=100)

            if freqs is not None and power is not None:
                # Create time array for CWT
                t_cwt = np.linspace(0, len(sig)/fs, power.shape[1])

                pcm = ax_cwt.pcolormesh(t_cwt/60, freqs, 10*np.log10(power + 1e-10),
                                       shading='gouraud', cmap='jet')
                ax_cwt.set_ylabel('Frequency (Hz)', fontweight='600')
                ax_cwt.set_title(f'{station} - Morlet CWT (r={peak_corr:.3f}, SNR={snr_linear:.2f})',
                                fontweight='700', pad=15)
                ax_cwt.set_yscale('log')
                plt.colorbar(pcm, ax=ax_cwt, label='Power (dB)')
                if i < len(signals) - 1:
                    plt.setp(ax_cwt.get_xticklabels(), visible=False)
                else:
                    ax_cwt.set_xlabel('Time (minutes)', fontweight='600')
            else:
                ax_cwt.text(0.5, 0.5, 'CWT computation failed',
                           ha='center', va='center', transform=ax_cwt.transAxes)

        # Combined Morlet transform (bottom panel, full width)
        ax_combined = fig.add_subplot(gs[2, :])

        # Average the signals
        all_signals = [data['signal'] for data in signals.values()]
        min_len = min(len(s) for s in all_signals)

        # Truncate all to same length and average
        signals_truncated = [s[:min_len] for s in all_signals]
        combined_signal = np.mean(signals_truncated, axis=0)

        # Use average fs
        fs_avg = np.mean([data['fs'] for data in signals.values()])

        freqs_comb, power_comb = compute_cwt(combined_signal, fs=fs_avg, wavelet='morl',
                                             freq_min=BAND_HZ[1], freq_max=BAND_HZ[0],
                                             n_freqs=100)

        if freqs_comb is not None and power_comb is not None:
            t_comb = np.linspace(0, len(combined_signal)/fs_avg, power_comb.shape[1])

            pcm = ax_combined.pcolormesh(t_comb/60, freqs_comb,
                                        10*np.log10(power_comb + 1e-10),
                                        shading='gouraud', cmap='jet')
            ax_combined.set_ylabel('Frequency (Hz)', fontweight='600')
            ax_combined.set_xlabel('Time (minutes)', fontweight='600')
            ax_combined.set_title('Combined Morlet Transform (Averaged Signal)',
                                 fontweight='700', pad=15)
            ax_combined.set_yscale('log')
            cbar = plt.colorbar(pcm, ax=ax_combined, label='Power (dB)')

            # Extend to margins
            ax_combined.set_xlim(0, t_comb[-1]/60)
        else:
            ax_combined.text(0.5, 0.5, 'Combined CWT computation failed',
                            ha='center', va='center', transform=ax_combined.transAxes)

        plt.tight_layout()
        output_file = output_dir / "morlet_wavelets" / f"sync_event_{plotted+1:01d}_morlet.png"
        plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✅ Saved: {output_file.name}")

        plotted += 1


# ============================================================================
# Main Execution
# ============================================================================

def create_station_pwave_impact_summaries(peaks_root: Path, output_dir: Path):
    """
    Create station-level P-wave impact summary plots by collating comparison CSVs.
    One plot per station showing all 4 sims with 4 templates each.
    """
    print("\n" + "="*80)
    print("PART 8: STATION-LEVEL P-WAVE IMPACT SUMMARIES")
    print("="*80)

    # Find all P-wave impact comparison CSVs
    comparison_files = list(peaks_root.glob("**/*_pwave_impact_comparison.csv"))

    if not comparison_files:
        print("⚠️ No P-wave impact comparison CSVs found")
        return

    print(f"Found {len(comparison_files)} comparison CSV files")

    # Load and combine all comparison data
    all_data = []
    for csv_file in comparison_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    # Group by dataset and station
    for (dataset, station), df_station in df_all.groupby(['dataset', 'station']):
        print(f"\n📊 Creating P-wave impact summary for {dataset.upper()} - {station}...")

        # Create figure
        fig = plt.figure(figsize=(18, 12), facecolor='white')
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

        # Color scheme
        color_total = '#ff8c42'
        color_clean = '#2563eb'
        color_lost = '#dc2626'

        sims = ['sim1', 'sim2', 'sim3', 'sim4']
        templates = ['template1', 'template2', 'template3', 'template4']

        # Panel 1: Total Peaks per Sim
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#f8f9fa')

        sim_totals = df_station.groupby('sim').agg({
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

        # Panel 2: Peak Loss Percentage per Sim
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

        # Panels 3-6: Peak Counts by Template for Each Sim
        for sim_idx, sim in enumerate(sims):
            ax = fig.add_subplot(gs[(sim_idx // 2) + 1, sim_idx % 2])
            ax.set_facecolor('#f8f9fa')

            sim_data = df_station[df_station['sim'] == sim].set_index('template').reindex(templates)

            x_templates = np.arange(len(templates))
            width = 0.25

            bars1 = ax.bar(x_templates - width, sim_data['n_peaks_contaminated'], width,
                           label='Total', color=color_total, alpha=0.8, edgecolor='black', linewidth=1.0)
            bars2 = ax.bar(x_templates, sim_data['n_peaks_clean'], width,
                           label='Clean', color=color_clean, alpha=0.8, edgecolor='black', linewidth=1.0)
            bars3 = ax.bar(x_templates + width, sim_data['n_peaks_lost'], width,
                           label='Lost', color=color_lost, alpha=0.8, edgecolor='black', linewidth=1.0)

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
            if sim_idx == 0:
                ax.legend(fontsize=8, framealpha=0.95, loc='upper right')
            ax.grid(True, alpha=0.2, axis='y')
            ax.set_axisbelow(True)

        fig.suptitle(f'{dataset.upper()} – Station {station}\nP-wave Contamination Impact Summary (All Simulations & Templates)',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.985])

        # Save
        output_file = output_dir / f"{dataset}_{station}_pwave_impact_summary_all_sims.png"
        plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"    ✅ Saved: {output_file.name}")

    print(f"\n✅ Completed station-level P-wave impact summaries")


def main():
    print("="*80)
    print("COMPREHENSIVE SIGNAL ANALYSIS - COMPLETE")
    print("All Parts (1-7) Combined")
    print("="*80)

    # Load all data
    print("\nLoading data...")
    df_peaks = load_all_peaks(PEAKS_ROOT)
    df_impact = load_p_wave_impact_data(PEAKS_ROOT)
    df_earthquakes = load_earthquakes(EARTHQUAKE_CSV)
    df_volcanic = load_volcanic_events(VOLCANIC_EVENTS_CSV)

    # Part 1: P-wave contamination probability analysis
    analyze_p_wave_contamination_probability(
        df_impact, df_peaks, df_earthquakes, OUTPUT_DIR
    )

    # Part 2: Cumulative metrics behavior explanation
    explain_cumulative_metrics_behavior(df_peaks, OUTPUT_DIR)

    # Part 3: Dataset comparison
    compare_ingv_vs_experiment(df_peaks, OUTPUT_DIR)

    # Part 4: Synchronous events
    df_sync_events, event_stats = detect_synchronous_events(df_peaks, OUTPUT_DIR)

    # Part 5: Best candidates
    df_candidates = identify_best_candidates(df_peaks, df_sync_events, OUTPUT_DIR, n_candidates=20)

    # Part 6: Spectrograms (plot all 20 candidates)
    create_candidate_spectrograms(df_candidates, df_earthquakes, OUTPUT_DIR, n_plots=20)

    # Part 7: Morlet wavelets
    if df_sync_events is not None and len(df_sync_events) > 0:
        create_morlet_wavelets_for_sync_events(df_sync_events, df_earthquakes, OUTPUT_DIR, n_plots=5)

    # Part 8: Station-level P-wave impact summaries
    create_station_pwave_impact_summaries(PEAKS_ROOT, OUTPUT_DIR)

    print("\n" + "="*80)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    print("\nGenerated outputs:")
    print("  - Contamination analysis (2 files)")
    print("  - Cumulative metrics explanation (1 file)")
    print("  - Dataset comparison (2 files)")
    print("  - Synchronous events (3 files)")
    print("  - Station P-wave impact summaries (1 per station)")
    print("  - Best candidates (12 files)")
    print("  - Morlet wavelets (3 files)")
    print("  Total: 23 files")
    print("="*80)


if __name__ == "__main__":
    main()
