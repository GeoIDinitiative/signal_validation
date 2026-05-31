#!/usr/bin/env python3
"""
Wavelet Analysis for Clean Matched Peak Data
==============================================

This script adapts the wavelet analysis to work with clean peak CSVs from SWCC analysis.
It loads peak data, computes distributions, and creates comprehensive visualizations.

Input: Clean peak CSVs from SWCC_p_wave_cleaned output
Output: Wavelet-based visualizations and statistical distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import detrend as scipy_detrend
from scipy.interpolate import interp1d
import pywt  # For wavelet analysis
from scipy.ndimage import gaussian_filter
import seaborn as sns
from datetime import datetime
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

# Input paths - SWCC clean peaks
PEAKS_ROOT = Path("/home/owen/Etna_signals/screening/SWCC_results_eq_fix")

# Output directory
OUTPUT_DIR = Path("/home/owen/tilt_validation/wavelet_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Subdirectories
(OUTPUT_DIR / "distributions").mkdir(exist_ok=True)
(OUTPUT_DIR / "by_station").mkdir(exist_ok=True)
(OUTPUT_DIR / "by_template").mkdir(exist_ok=True)
(OUTPUT_DIR / "temporal").mkdir(exist_ok=True)
(OUTPUT_DIR / "correlations").mkdir(exist_ok=True)

# Analysis parameters
CORRELATION_THRESHOLDS = {
    'moderate': (0.2, 0.5),
    'high': (0.5, 0.7),
    'very_high': (0.7, 1.0)
}

SNR_THRESHOLDS = {
    'low': (-np.inf, 0),
    'moderate': (0, 10),
    'high': (10, 20),
    'very_high': (20, np.inf)
}

# Color palettes
DATASET_COLORS = {
    'ingv': '#2563eb',
    'experiment': '#059669'
}

TEMPLATE_COLORS = {
    'template1': '#8b5cf6',
    'template2': '#3b82f6',
    'template3': '#10b981',
    'template4': '#f59e0b'
}

SIM_COLORS = {
    'sim1': '#ef4444',
    'sim2': '#f97316',
    'sim3': '#eab308',
    'sim4': '#84cc16'
}

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_all_peak_csvs():
    """
    Load all clean peak CSVs from SWCC analysis.
    Returns consolidated DataFrame with metadata.
    """
    all_peaks = []
    peaks_root = Path(PEAKS_ROOT)

    # Find all peaks CSV files
    for csv_file in peaks_root.rglob("*_peaks.csv"):
        try:
            df = pd.read_csv(csv_file)

            # Extract metadata from filepath and filename
            # Format: dataset/station/sim/{station}_{sim}_{template}_peaks.csv
            parts = csv_file.stem.split('_')

            if len(parts) >= 3:
                df['station'] = parts[0]
                df['sim'] = parts[1]
                df['template'] = parts[2]

                # Get dataset from parent directory
                path_parts = csv_file.parts
                if 'ingv' in path_parts:
                    df['dataset'] = 'ingv'
                elif 'experiment' in path_parts:
                    df['dataset'] = 'experiment'
                else:
                    df['dataset'] = 'unknown'

                all_peaks.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")

    if not all_peaks:
        raise FileNotFoundError("No peak CSV files found!")

    combined_df = pd.concat(all_peaks, ignore_index=True)

    # Convert peak_time_dt to datetime if present
    if 'peak_time_dt' in combined_df.columns:
        combined_df['peak_time_dt'] = pd.to_datetime(combined_df['peak_time_dt'], errors='coerce')
        combined_df['peak_hour'] = combined_df['peak_time_dt'].dt.hour
        combined_df['peak_day'] = combined_df['peak_time_dt'].dt.day
        combined_df['peak_month'] = combined_df['peak_time_dt'].dt.month

    # Add categorical labels
    combined_df['corr_category'] = pd.cut(
        combined_df['peak_corr'],
        bins=[0, 0.5, 0.7, 1.0],
        labels=['moderate', 'high', 'very_high']
    )

    combined_df['snr_category'] = pd.cut(
        combined_df['snr_db'],
        bins=[-np.inf, 0, 10, 20, np.inf],
        labels=['low', 'moderate', 'high', 'very_high']
    )

    print(f"✅ Loaded {len(combined_df):,} peak records from {len(all_peaks)} files")
    print(f"   Datasets: {combined_df['dataset'].unique()}")
    print(f"   Stations: {combined_df['station'].unique()}")
    print(f"   Simulations: {combined_df['sim'].unique()}")
    print(f"   Templates: {combined_df['template'].unique()}")

    return combined_df


# ============================================================================
# Distribution Analysis
# ============================================================================

def plot_correlation_distributions(df, output_dir):
    """
    Plot correlation coefficient distributions across different facets.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
    fig.suptitle('Correlation Coefficient Distributions', fontsize=16, fontweight='bold', y=0.98)

    # 1. Overall distribution
    ax1 = axes[0, 0]
    ax1.hist(df['peak_corr'], bins=50, color='#2563eb', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='High (r=0.5)')
    ax1.axvline(0.7, color='darkred', linestyle='--', linewidth=2, label='Very High (r=0.7)')
    ax1.set_xlabel('Peak Correlation', fontweight='600')
    ax1.set_ylabel('Frequency', fontweight='600')
    ax1.set_title('Overall Distribution', fontweight='700', pad=15)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax1.grid(True, alpha=0.3)

    # 2. By dataset (violin plot)
    ax2 = axes[0, 1]
    datasets = sorted(df['dataset'].unique())
    data_to_plot = [df[df['dataset'] == d]['peak_corr'].dropna() for d in datasets]
    parts = ax2.violinplot(data_to_plot, positions=range(len(datasets)), showmeans=True, showmedians=True)
    for pc, dataset in zip(parts['bodies'], datasets):
        pc.set_facecolor(DATASET_COLORS.get(dataset, '#666'))
        pc.set_alpha(0.7)
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels([d.upper() for d in datasets])
    ax2.set_ylabel('Peak Correlation', fontweight='600')
    ax2.set_title('Distribution by Dataset', fontweight='700', pad=15)
    ax2.grid(True, alpha=0.4, axis='y')

    # 3. By station (box plot)
    ax3 = axes[0, 2]
    stations = sorted(df['station'].unique())
    station_data = [df[df['station'] == s]['peak_corr'].dropna() for s in stations]
    bp = ax3.boxplot(station_data, labels=stations, patch_artist=True, showfliers=False, whis=[5, 95])
    for patch in bp['boxes']:
        patch.set_facecolor('#3b82f6')
        patch.set_alpha(0.7)
    ax3.set_ylabel('Peak Correlation', fontweight='600')
    ax3.set_xlabel('Station', fontweight='600')
    ax3.set_title('Distribution by Station', fontweight='700', pad=15)
    ax3.grid(True, alpha=0.4, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. By template
    ax4 = axes[1, 0]
    templates = sorted(df['template'].unique())
    for template in templates:
        data = df[df['template'] == template]['peak_corr']
        ax4.hist(data, bins=30, alpha=0.5, label=template,
                color=TEMPLATE_COLORS.get(template, '#666'), edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Peak Correlation', fontweight='600')
    ax4.set_ylabel('Frequency', fontweight='600')
    ax4.set_title('Distribution by Template', fontweight='700', pad=15)
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax4.grid(True, alpha=0.3)

    # 5. By simulation
    ax5 = axes[1, 1]
    sims = sorted(df['sim'].unique())
    sim_data = [df[df['sim'] == s]['peak_corr'].dropna() for s in sims]
    bp = ax5.boxplot(sim_data, labels=sims, patch_artist=True, showfliers=False, whis=[5, 95])
    for patch, s in zip(bp['boxes'], sims):
        patch.set_facecolor(SIM_COLORS.get(s, '#666'))
        patch.set_alpha(0.7)
    ax5.set_ylabel('Peak Correlation', fontweight='600')
    ax5.set_xlabel('Simulation', fontweight='600')
    ax5.set_title('Distribution by Simulation', fontweight='700', pad=15)
    ax5.grid(True, alpha=0.4, axis='y')

    # 6. Cumulative distribution
    ax6 = axes[1, 2]
    sorted_corr = np.sort(df['peak_corr'].dropna())
    cumulative = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr) * 100
    ax6.plot(sorted_corr, cumulative, color='#2563eb', linewidth=2)
    ax6.axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='r=0.5')
    ax6.axvline(0.7, color='darkred', linestyle='--', linewidth=2, alpha=0.5, label='r=0.7')
    ax6.set_xlabel('Peak Correlation', fontweight='600')
    ax6.set_ylabel('Cumulative %', fontweight='600')
    ax6.set_title('Cumulative Distribution', fontweight='700', pad=15)
    ax6.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "distributions" / "01_correlation_distributions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


def plot_snr_distributions(df, output_dir):
    """
    Plot SNR distributions across different facets.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
    fig.suptitle('SNR Distributions', fontsize=16, fontweight='bold', y=0.98)

    # 1. Overall distribution
    ax1 = axes[0, 0]
    ax1.hist(df['snr_db'].dropna(), bins=50, color='#059669', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='SNR=0 dB')
    ax1.axvline(10, color='orange', linestyle='--', linewidth=2, label='SNR=10 dB')
    ax1.axvline(20, color='darkred', linestyle='--', linewidth=2, label='SNR=20 dB')
    ax1.set_xlabel('SNR (dB)', fontweight='600')
    ax1.set_ylabel('Frequency', fontweight='600')
    ax1.set_title('Overall Distribution', fontweight='700', pad=15)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax1.grid(True, alpha=0.3)

    # 2. By dataset (violin plot)
    ax2 = axes[0, 1]
    datasets = sorted(df['dataset'].unique())
    data_to_plot = [df[df['dataset'] == d]['snr_db'].dropna() for d in datasets]
    parts = ax2.violinplot(data_to_plot, positions=range(len(datasets)), showmeans=True, showmedians=True)
    for pc, dataset in zip(parts['bodies'], datasets):
        pc.set_facecolor(DATASET_COLORS.get(dataset, '#666'))
        pc.set_alpha(0.7)
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels([d.upper() for d in datasets])
    ax2.set_ylabel('SNR (dB)', fontweight='600')
    ax2.set_title('Distribution by Dataset', fontweight='700', pad=15)
    ax2.grid(True, alpha=0.4, axis='y')

    # 3. By station (box plot)
    ax3 = axes[0, 2]
    stations = sorted(df['station'].unique())
    station_data = [df[df['station'] == s]['snr_db'].dropna() for s in stations]
    bp = ax3.boxplot(station_data, labels=stations, patch_artist=True, showfliers=False, whis=[5, 95])
    for patch in bp['boxes']:
        patch.set_facecolor('#10b981')
        patch.set_alpha(0.7)
    ax3.set_ylabel('SNR (dB)', fontweight='600')
    ax3.set_xlabel('Station', fontweight='600')
    ax3.set_title('Distribution by Station', fontweight='700', pad=15)
    ax3.grid(True, alpha=0.4, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. By template
    ax4 = axes[1, 0]
    templates = sorted(df['template'].unique())
    for template in templates:
        data = df[df['template'] == template]['snr_db'].dropna()
        ax4.hist(data, bins=30, alpha=0.5, label=template,
                color=TEMPLATE_COLORS.get(template, '#666'), edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('SNR (dB)', fontweight='600')
    ax4.set_ylabel('Frequency', fontweight='600')
    ax4.set_title('Distribution by Template', fontweight='700', pad=15)
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax4.grid(True, alpha=0.3)

    # 5. By simulation
    ax5 = axes[1, 1]
    sims = sorted(df['sim'].unique())
    sim_data = [df[df['sim'] == s]['snr_db'].dropna() for s in sims]
    bp = ax5.boxplot(sim_data, labels=sims, patch_artist=True, showfliers=False, whis=[5, 95])
    for patch, s in zip(bp['boxes'], sims):
        patch.set_facecolor(SIM_COLORS.get(s, '#666'))
        patch.set_alpha(0.7)
    ax5.set_ylabel('SNR (dB)', fontweight='600')
    ax5.set_xlabel('Simulation', fontweight='600')
    ax5.set_title('Distribution by Simulation', fontweight='700', pad=15)
    ax5.grid(True, alpha=0.4, axis='y')

    # 6. Cumulative distribution
    ax6 = axes[1, 2]
    sorted_snr = np.sort(df['snr_db'].dropna())
    cumulative = np.arange(1, len(sorted_snr) + 1) / len(sorted_snr) * 100
    ax6.plot(sorted_snr, cumulative, color='#059669', linewidth=2)
    ax6.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='0 dB')
    ax6.axvline(10, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='10 dB')
    ax6.axvline(20, color='darkred', linestyle='--', linewidth=2, alpha=0.5, label='20 dB')
    ax6.set_xlabel('SNR (dB)', fontweight='600')
    ax6.set_ylabel('Cumulative %', fontweight='600')
    ax6.set_title('Cumulative Distribution', fontweight='700', pad=15)
    ax6.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "distributions" / "02_snr_distributions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


def plot_joint_distributions(df, output_dir):
    """
    Plot joint distributions of correlation and SNR.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor='white')
    fig.suptitle('Joint Distributions: Correlation vs SNR', fontsize=16, fontweight='bold', y=0.98)

    # 1. Scatter plot with density
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['peak_corr'], df['snr_db'],
                         c=df['peak_corr'], cmap='viridis',
                         alpha=0.4, s=20, edgecolors='none')
    ax1.set_xlabel('Peak Correlation', fontweight='600')
    ax1.set_ylabel('SNR (dB)', fontweight='600')
    ax1.set_title('Correlation vs SNR (All Peaks)', fontweight='700', pad=15)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Correlation')

    # 2. 2D histogram (hexbin)
    ax2 = axes[0, 1]
    hexbin = ax2.hexbin(df['peak_corr'], df['snr_db'], gridsize=30, cmap='YlOrRd', mincnt=1)
    ax2.set_xlabel('Peak Correlation', fontweight='600')
    ax2.set_ylabel('SNR (dB)', fontweight='600')
    ax2.set_title('Density Distribution (Hexbin)', fontweight='700', pad=15)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(hexbin, ax=ax2, label='Count')

    # 3. By dataset
    ax3 = axes[1, 0]
    for dataset in sorted(df['dataset'].unique()):
        data = df[df['dataset'] == dataset]
        ax3.scatter(data['peak_corr'], data['snr_db'],
                   label=dataset.upper(), alpha=0.4, s=20,
                   color=DATASET_COLORS.get(dataset, '#666'))
    ax3.set_xlabel('Peak Correlation', fontweight='600')
    ax3.set_ylabel('SNR (dB)', fontweight='600')
    ax3.set_title('Correlation vs SNR by Dataset', fontweight='700', pad=15)
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax3.grid(True, alpha=0.3)

    # 4. Contour plot
    ax4 = axes[1, 1]
    # Create 2D histogram for contour
    H, xedges, yedges = np.histogram2d(df['peak_corr'].dropna(),
                                       df['snr_db'].dropna(),
                                       bins=50)
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    contour = ax4.contourf(X, Y, H.T, levels=15, cmap='plasma')
    ax4.set_xlabel('Peak Correlation', fontweight='600')
    ax4.set_ylabel('SNR (dB)', fontweight='600')
    ax4.set_title('Density Contour Plot', fontweight='700', pad=15)
    ax4.grid(True, alpha=0.3)
    plt.colorbar(contour, ax=ax4, label='Density')

    plt.tight_layout()
    output_file = output_dir / "correlations" / "01_joint_distributions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


def plot_temporal_patterns(df, output_dir):
    """
    Plot temporal patterns in peak detections.
    """
    if 'peak_time_dt' not in df.columns or df['peak_time_dt'].isna().all():
        print("⚠️  No temporal data available, skipping temporal analysis")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor='white')
    fig.suptitle('Temporal Patterns in Peak Detection', fontsize=16, fontweight='bold', y=0.98)

    # Filter valid temporal data
    temp_df = df[df['peak_time_dt'].notna()].copy()

    # 1. Peaks by hour of day
    ax1 = axes[0, 0]
    if 'peak_hour' in temp_df.columns:
        hourly_counts = temp_df.groupby('peak_hour').size()
        ax1.bar(hourly_counts.index, hourly_counts.values, color='#3b82f6', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Hour of Day (UTC)', fontweight='600')
        ax1.set_ylabel('Peak Count', fontweight='600')
        ax1.set_title('Peak Distribution by Hour', fontweight='700', pad=15)
        ax1.grid(True, alpha=0.4, axis='y')
        ax1.set_xlim(-0.5, 23.5)

    # 2. Peaks over time (time series)
    ax2 = axes[0, 1]
    temp_df_sorted = temp_df.sort_values('peak_time_dt')
    ax2.plot(temp_df_sorted['peak_time_dt'],
            np.arange(len(temp_df_sorted)),
            color='#2563eb', linewidth=2)
    ax2.set_xlabel('Time', fontweight='600')
    ax2.set_ylabel('Cumulative Peak Count', fontweight='600')
    ax2.set_title('Cumulative Peaks Over Time', fontweight='700', pad=15)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Average correlation by hour
    ax3 = axes[1, 0]
    if 'peak_hour' in temp_df.columns:
        hourly_corr = temp_df.groupby('peak_hour')['peak_corr'].agg(['mean', 'std'])
        ax3.errorbar(hourly_corr.index, hourly_corr['mean'], yerr=hourly_corr['std'],
                    marker='o', linestyle='-', linewidth=2, markersize=6,
                    color='#8b5cf6', capsize=5, capthick=2)
        ax3.set_xlabel('Hour of Day (UTC)', fontweight='600')
        ax3.set_ylabel('Mean Peak Correlation ± Std', fontweight='600')
        ax3.set_title('Correlation Variation by Hour', fontweight='700', pad=15)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-0.5, 23.5)

    # 4. Average SNR by hour
    ax4 = axes[1, 1]
    if 'peak_hour' in temp_df.columns:
        hourly_snr = temp_df.groupby('peak_hour')['snr_db'].agg(['mean', 'std'])
        ax4.errorbar(hourly_snr.index, hourly_snr['mean'], yerr=hourly_snr['std'],
                    marker='s', linestyle='-', linewidth=2, markersize=6,
                    color='#059669', capsize=5, capthick=2)
        ax4.set_xlabel('Hour of Day (UTC)', fontweight='600')
        ax4.set_ylabel('Mean SNR (dB) ± Std', fontweight='600')
        ax4.set_title('SNR Variation by Hour', fontweight='700', pad=15)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-0.5, 23.5)

    plt.tight_layout()
    output_file = output_dir / "temporal" / "01_temporal_patterns.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


def plot_categorical_breakdown(df, output_dir):
    """
    Plot categorical breakdown by correlation and SNR categories.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor='white')
    fig.suptitle('Categorical Breakdown', fontsize=16, fontweight='bold', y=0.98)

    # 1. Peaks by correlation category
    ax1 = axes[0, 0]
    corr_counts = df['corr_category'].value_counts().sort_index()
    colors = ['#10b981', '#f59e0b', '#ef4444']
    ax1.bar(range(len(corr_counts)), corr_counts.values,
           color=colors[:len(corr_counts)], alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_xticks(range(len(corr_counts)))
    ax1.set_xticklabels(corr_counts.index, rotation=0)
    ax1.set_ylabel('Peak Count', fontweight='600')
    ax1.set_xlabel('Correlation Category', fontweight='600')
    ax1.set_title('Peaks by Correlation Category', fontweight='700', pad=15)
    ax1.grid(True, alpha=0.4, axis='y')

    # Add percentage labels
    total = corr_counts.sum()
    max_height = corr_counts.max()
    ax1.set_ylim(0, max_height * 1.15)  # 15% padding for annotations
    for i, (cat, count) in enumerate(corr_counts.items()):
        pct = 100 * count / total
        ax1.text(i, count * 1.02, f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 2. Peaks by SNR category
    ax2 = axes[0, 1]
    snr_counts = df['snr_category'].value_counts().sort_index()
    colors_snr = ['#dc2626', '#f59e0b', '#10b981', '#059669']
    ax2.bar(range(len(snr_counts)), snr_counts.values,
           color=colors_snr[:len(snr_counts)], alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_xticks(range(len(snr_counts)))
    ax2.set_xticklabels(snr_counts.index, rotation=0)
    ax2.set_ylabel('Peak Count', fontweight='600')
    ax2.set_xlabel('SNR Category', fontweight='600')
    ax2.set_title('Peaks by SNR Category', fontweight='700', pad=15)
    ax2.grid(True, alpha=0.4, axis='y')

    # Add percentage labels
    total_snr = snr_counts.sum()
    max_height_snr = snr_counts.max()
    ax2.set_ylim(0, max_height_snr * 1.15)  # 15% padding for annotations
    for i, (cat, count) in enumerate(snr_counts.items()):
        pct = 100 * count / total_snr
        ax2.text(i, count * 1.02, f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 3. Cross-tabulation heatmap
    ax3 = axes[1, 0]
    crosstab = pd.crosstab(df['corr_category'], df['snr_category'])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlGnBu', ax=ax3,
               cbar_kws={'label': 'Count'})
    ax3.set_ylabel('Correlation Category', fontweight='600')
    ax3.set_xlabel('SNR Category', fontweight='600')
    ax3.set_title('Cross-tabulation: Correlation × SNR', fontweight='700', pad=15)

    # 4. Stacked bar by dataset
    ax4 = axes[1, 1]
    dataset_corr = pd.crosstab(df['dataset'], df['corr_category'])
    dataset_corr.plot(kind='bar', stacked=True, ax=ax4,
                     color=colors[:len(dataset_corr.columns)],
                     alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_ylabel('Peak Count', fontweight='600')
    ax4.set_xlabel('Dataset', fontweight='600')
    ax4.set_title('Correlation Categories by Dataset', fontweight='700', pad=15)
    ax4.legend(title='Correlation', loc='upper right', fontsize=9, framealpha=0.95)
    ax4.grid(True, alpha=0.4, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    output_file = output_dir / "distributions" / "03_categorical_breakdown.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


def export_summary_statistics(df, output_dir):
    """
    Export comprehensive summary statistics.
    """
    # Overall statistics
    overall_stats = pd.DataFrame({
        'Metric': ['Total Peaks', 'Mean Correlation', 'Std Correlation',
                  'Median Correlation', 'Mean SNR (dB)', 'Std SNR (dB)',
                  'Median SNR (dB)', 'Moderate Corr (0.2-0.5)',
                  'High Corr (0.5-0.7)', 'Very High Corr (>0.7)'],
        'Value': [
            len(df),
            df['peak_corr'].mean(),
            df['peak_corr'].std(),
            df['peak_corr'].median(),
            df['snr_db'].mean(),
            df['snr_db'].std(),
            df['snr_db'].median(),
            len(df[(df['peak_corr'] >= 0.2) & (df['peak_corr'] < 0.5)]),
            len(df[(df['peak_corr'] >= 0.5) & (df['peak_corr'] < 0.7)]),
            len(df[df['peak_corr'] >= 0.7])
        ]
    })
    overall_stats.to_csv(output_dir / "overall_statistics.csv", index=False)
    print(f"✅ Exported overall statistics")

    # By station
    station_stats = df.groupby('station').agg({
        'peak_corr': ['count', 'mean', 'std', 'median', 'max'],
        'snr_db': ['mean', 'std', 'median', 'max']
    }).round(4)
    station_stats.to_csv(output_dir / "by_station" / "station_statistics.csv")
    print(f"✅ Exported station statistics")

    # By template
    template_stats = df.groupby('template').agg({
        'peak_corr': ['count', 'mean', 'std', 'median', 'max'],
        'snr_db': ['mean', 'std', 'median', 'max']
    }).round(4)
    template_stats.to_csv(output_dir / "by_template" / "template_statistics.csv")
    print(f"✅ Exported template statistics")

    # By simulation
    sim_stats = df.groupby('sim').agg({
        'peak_corr': ['count', 'mean', 'std', 'median', 'max'],
        'snr_db': ['mean', 'std', 'median', 'max']
    }).round(4)
    sim_stats.to_csv(output_dir / "by_template" / "simulation_statistics.csv")
    print(f"✅ Exported simulation statistics")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("="*80)
    print("WAVELET ANALYSIS FOR CLEAN MATCHED PEAK DATA")
    print("="*80)
    print()

    # Load peak data
    print("📊 Loading clean peak CSVs...")
    df = load_all_peak_csvs()
    print()

    # Distribution analysis
    print("📈 Analyzing correlation distributions...")
    plot_correlation_distributions(df, OUTPUT_DIR)
    print()

    print("📈 Analyzing SNR distributions...")
    plot_snr_distributions(df, OUTPUT_DIR)
    print()

    print("📈 Analyzing joint distributions...")
    plot_joint_distributions(df, OUTPUT_DIR)
    print()

    print("🕒 Analyzing temporal patterns...")
    plot_temporal_patterns(df, OUTPUT_DIR)
    print()

    print("📊 Creating categorical breakdowns...")
    plot_categorical_breakdown(df, OUTPUT_DIR)
    print()

    print("💾 Exporting summary statistics...")
    export_summary_statistics(df, OUTPUT_DIR)
    print()

    print("="*80)
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"   Total peaks analyzed: {len(df):,}")
    print(f"   Output directory: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
