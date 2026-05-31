#!/usr/bin/env python3
"""
Comprehensive Analysis of SWCC Results
======================================

This script analyzes all outputs from swcc_p_wave_cleaned_with_cumulative.py
to provide insights into:
1. P-wave contamination impact across stations and datasets
2. Station-level performance comparisons
3. Template effectiveness analysis
4. Simulation consistency evaluation
5. Correlation and SNR statistics

Handles 112 variant combinations (2 datasets × 7 stations × 4 sims × 4 templates)
with multi-faceted visualizations and statistical summaries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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

INPUT_ROOT = "/home/owen/Etna_signals/screening/SWCC_p_wave_cleaned"
OUTPUT_DIR = "/home/owen/tilt_validation/SWCC_analysis"

# Create output directory structure
OUTPUT_DIR = Path(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Subdirectories for different analysis types
(OUTPUT_DIR / "by_station").mkdir(exist_ok=True)
(OUTPUT_DIR / "by_dataset").mkdir(exist_ok=True)
(OUTPUT_DIR / "by_template").mkdir(exist_ok=True)
(OUTPUT_DIR / "by_simulation").mkdir(exist_ok=True)
(OUTPUT_DIR / "p_wave_impact").mkdir(exist_ok=True)
(OUTPUT_DIR / "comprehensive").mkdir(exist_ok=True)
(OUTPUT_DIR / "rankings").mkdir(exist_ok=True)

# Professional color palettes
DATASET_COLORS = {
    'ingv': '#2563eb',      # Professional blue
    'experiment': '#059669'  # Forest green
}

TEMPLATE_COLORS = {
    'template1': '#8b5cf6',  # Purple
    'template2': '#3b82f6',  # Blue
    'template3': '#10b981',  # Green
    'template4': '#f59e0b'   # Amber
}

SIM_COLORS = {
    'sim1': '#ef4444',  # Red
    'sim2': '#f97316',  # Orange
    'sim3': '#eab308',  # Yellow
    'sim4': '#84cc16'   # Lime
}

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_all_pwave_impact_data():
    """
    Load all P-wave impact comparison CSVs from all stations.
    Returns a single consolidated DataFrame.
    """
    all_data = []
    input_root = Path(INPUT_ROOT)

    # Find all pwave_impact_comparison.csv files
    for csv_file in input_root.rglob("*_pwave_impact_comparison.csv"):
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")

    if not all_data:
        raise FileNotFoundError("No P-wave impact comparison files found!")

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Loaded {len(combined_df)} P-wave impact records from {len(all_data)} files")
    return combined_df


def load_all_peaks_data():
    """
    Load all peaks CSVs from all stations.
    Returns a single consolidated DataFrame with metadata.
    """
    all_peaks = []
    input_root = Path(INPUT_ROOT)

    # Find all peaks CSV files
    for csv_file in input_root.rglob("*_peaks.csv"):
        try:
            df = pd.read_csv(csv_file)

            # Extract metadata from filename
            # Format: {station}_{sim}_{template}_peaks.csv
            parts = csv_file.stem.split('_')
            if len(parts) >= 3:
                df['station'] = parts[0]
                df['sim'] = parts[1]
                df['template'] = parts[2]

                # Infer dataset from parent directory
                if 'ingv' in str(csv_file.parent):
                    df['dataset'] = 'ingv'
                elif 'experiment' in str(csv_file.parent):
                    df['dataset'] = 'experiment'
                else:
                    df['dataset'] = 'unknown'

                all_peaks.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")

    if not all_peaks:
        print("Warning: No peaks files found!")
        return pd.DataFrame()

    combined_df = pd.concat(all_peaks, ignore_index=True)
    print(f"✅ Loaded {len(combined_df)} peak records from {len(all_peaks)} files")
    return combined_df


# ============================================================================
# P-wave Contamination Impact Analysis
# ============================================================================

def plot_pwave_contamination_overview(df, output_dir):
    """
    Comprehensive overview of P-wave contamination impact.
    """
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25)
    fig.suptitle('P-wave Contamination Impact Analysis', fontsize=16, fontweight='bold', y=0.98)

    # 1. Peaks lost by dataset and station
    ax1 = fig.add_subplot(gs[0, 0])
    pivot = df.pivot_table(values='pct_peaks_lost', index='station', columns='dataset', aggfunc='mean')
    pivot.plot(kind='bar', ax=ax1, color=[DATASET_COLORS.get(c, '#666') for c in pivot.columns], width=0.7, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('% Peaks Lost to P-wave Contamination', fontweight='600')
    ax1.set_xlabel('Station', fontweight='600')
    ax1.set_title('Average Peak Loss by Station & Dataset', fontweight='700', pad=15)
    ax1.legend(title='Dataset', title_fontsize=10, loc='best', framealpha=0.95)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Correlation change (contaminated vs clean)
    ax2 = fig.add_subplot(gs[0, 1])
    df['corr_mean_change_pct'] = 100 * (df['corr_mean_clean'] - df['corr_mean_contaminated']) / df['corr_mean_contaminated']

    # Box plot by dataset
    data_to_plot = [df[df['dataset'] == ds]['corr_mean_change_pct'].dropna() for ds in df['dataset'].unique()]
    bp = ax2.boxplot(data_to_plot, labels=df['dataset'].unique(), patch_artist=True, showfliers=False, whis=[5, 95])
    for patch, ds in zip(bp['boxes'], df['dataset'].unique()):
        patch.set_facecolor(DATASET_COLORS.get(ds, '#666'))
        patch.set_alpha(0.85)
    ax2.set_ylabel('% Change in Mean Correlation', fontweight='600')
    ax2.set_xlabel('Dataset', fontweight='600')
    ax2.set_title('Correlation Change After P-wave Removal', fontweight='700', pad=15)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # 3. Peak counts comparison
    ax3 = fig.add_subplot(gs[1, 0])
    station_summary = df.groupby('station').agg({
        'n_peaks_contaminated': 'sum',
        'n_peaks_clean': 'sum',
        'n_peaks_lost': 'sum'
    }).reset_index()

    x = np.arange(len(station_summary))
    width = 0.35
    ax3.bar(x - width, station_summary['n_peaks_contaminated'], width, label='Contaminated',
            color='#ff8c42', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax3.bar(x, station_summary['n_peaks_clean'], width, label='Clean',
            color='#2563eb', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax3.bar(x + width, station_summary['n_peaks_lost'], width, label='Lost',
            color='#dc2626', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax3.set_ylabel('Total Peak Count', fontweight='600')
    ax3.set_xlabel('Station', fontweight='600')
    ax3.set_title('Peak Counts by Station (All Sims & Templates)', fontweight='700', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(station_summary['station'], rotation=45, ha='right')
    ax3.legend(loc='best', framealpha=0.95)

    # 4. High correlation peaks (>0.5) - contaminated vs clean
    ax4 = fig.add_subplot(gs[1, 1])
    df['pct_high_corr_change'] = df['n_windows_above_0.5_clean'] - df['n_windows_above_0.5_contaminated']

    template_summary = df.groupby('template').agg({
        'n_windows_above_0.5_contaminated': 'mean',
        'n_windows_above_0.5_clean': 'mean'
    }).reset_index()

    x = np.arange(len(template_summary))
    width = 0.4
    ax4.bar(x - width/2, template_summary['n_windows_above_0.5_contaminated'], width,
            label='Contaminated', color='#ff8c42', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax4.bar(x + width/2, template_summary['n_windows_above_0.5_clean'], width,
            label='Clean', color='#2563eb', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax4.set_ylabel('Avg. Windows with r > 0.5', fontweight='600')
    ax4.set_xlabel('Template', fontweight='600')
    ax4.set_title('High-Correlation Windows by Template', fontweight='700', pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(template_summary['template'])
    ax4.legend(loc='best', framealpha=0.95)
    output_file = output_dir / "p_wave_impact" / "01_contamination_overview.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


def plot_pwave_impact_heatmaps(df, output_dir):
    """
    Heatmap visualizations of P-wave impact across different dimensions.
    """
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25)
    fig.suptitle('P-wave Impact Heatmaps', fontsize=16, fontweight='bold', y=0.98)

    # 1. % Peaks lost: Station × Template
    ax1 = fig.add_subplot(gs[0, 0])
    pivot1 = df.pivot_table(values='pct_peaks_lost', index='station', columns='template', aggfunc='mean')
    sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': '% Peaks Lost'}, annot_kws={'fontsize': 9})
    ax1.set_title('Peak Loss: Station × Template', fontweight='700', pad=15)
    ax1.set_ylabel('Station', fontweight='600')
    ax1.set_xlabel('Template', fontweight='600')

    # 2. % Peaks lost: Station × Sim
    ax2 = fig.add_subplot(gs[0, 1])
    pivot2 = df.pivot_table(values='pct_peaks_lost', index='station', columns='sim', aggfunc='mean')
    sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2, cbar_kws={'label': '% Peaks Lost'}, annot_kws={'fontsize': 9})
    ax2.set_title('Peak Loss: Station × Simulation', fontweight='700', pad=15)
    ax2.set_ylabel('Station', fontweight='600')
    ax2.set_xlabel('Simulation', fontweight='600')

    # 3. Mean correlation (clean): Station × Template
    ax3 = fig.add_subplot(gs[1, 0])
    pivot3 = df.pivot_table(values='corr_mean_clean', index='station', columns='template', aggfunc='mean')
    sns.heatmap(pivot3, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax3, cbar_kws={'label': 'Mean Correlation'}, annot_kws={'fontsize': 9})
    ax3.set_title('Mean Correlation (Clean): Station × Template', fontweight='700', pad=15)
    ax3.set_ylabel('Station', fontweight='600')
    ax3.set_xlabel('Template', fontweight='600')

    # 4. SNR (clean): Station × Template
    ax4 = fig.add_subplot(gs[1, 1])
    pivot4 = df.pivot_table(values='snr_mean_db_clean', index='station', columns='template', aggfunc='mean')
    sns.heatmap(pivot4, annot=True, fmt='.1f', cmap='viridis', ax=ax4, cbar_kws={'label': 'Mean SNR (dB)'}, annot_kws={'fontsize': 9})
    ax4.set_title('Mean SNR (Clean): Station × Template', fontweight='700', pad=15)
    ax4.set_ylabel('Station', fontweight='600')
    ax4.set_xlabel('Template', fontweight='600')

    output_file = output_dir / "p_wave_impact" / "02_impact_heatmaps.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


# ============================================================================
# Station-Level Analysis
# ============================================================================

def plot_station_performance_comparison(df, output_dir):
    """
    Comprehensive station performance comparison.
    """
    fig = plt.figure(figsize=(20, 12), facecolor='white')
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle('Station Performance Comparison', fontsize=16, fontweight='bold', y=0.98)

    # Prepare data aggregated by station
    station_stats = df.groupby(['dataset', 'station']).agg({
        'corr_mean_clean': ['mean', 'std'],
        'corr_max_clean': 'max',
        'peak_corr_mean_clean': ['mean', 'std'],
        'peak_corr_max_clean': 'max',
        'snr_mean_db_clean': ['mean', 'std'],
        'snr_max_db_clean': 'max',
        'n_peaks_clean': 'sum',
        'pct_peaks_lost': 'mean',
        'n_peaks_above_0.7_clean': 'sum'
    }).reset_index()

    station_stats.columns = ['_'.join(col).strip('_') for col in station_stats.columns]
    station_stats['station_label'] = station_stats['dataset'] + '_' + station_stats['station']

    # 1. Mean correlation by station
    ax1 = fig.add_subplot(gs[0, 0])
    for dataset in station_stats['dataset'].unique():
        data = station_stats[station_stats['dataset'] == dataset]
        ax1.errorbar(data['station'], data['corr_mean_clean_mean'],
                    yerr=data['corr_mean_clean_std'],
                    marker='o', linestyle='-', linewidth=2, markersize=8,
                    label=dataset.upper(), color=DATASET_COLORS.get(dataset, '#666'),
                    capsize=5, capthick=2, alpha=0.85)
    ax1.set_ylabel('Mean Correlation ± Std', fontweight='600')
    ax1.set_xlabel('Station', fontweight='600')
    ax1.set_title('Mean Correlation by Station', fontweight='700', pad=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Max correlation by station
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(station_stats))
    colors = [DATASET_COLORS.get(ds, '#666') for ds in station_stats['dataset']]
    bars = ax2.bar(x, station_stats['corr_max_clean_max'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Max Correlation', fontweight='600')
    ax2.set_xlabel('Station', fontweight='600')
    ax2.set_title('Maximum Correlation by Station', fontweight='700', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(station_stats['station'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Mean SNR by station
    ax3 = fig.add_subplot(gs[0, 2])
    for dataset in station_stats['dataset'].unique():
        data = station_stats[station_stats['dataset'] == dataset]
        ax3.errorbar(data['station'], data['snr_mean_db_clean_mean'],
                    yerr=data['snr_mean_db_clean_std'],
                    marker='s', linestyle='-', linewidth=2, markersize=8,
                    label=dataset.upper(), color=DATASET_COLORS.get(dataset, '#666'),
                    capsize=5, capthick=2, alpha=0.85)
    ax3.set_ylabel('Mean SNR (dB) ± Std', fontweight='600')
    ax3.set_xlabel('Station', fontweight='600')
    ax3.set_title('Mean SNR by Station', fontweight='700', pad=15)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Max SNR by station
    ax4 = fig.add_subplot(gs[1, 0])
    bars = ax4.bar(x, station_stats['snr_max_db_clean_max'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_ylabel('Max SNR (dB)', fontweight='600')
    ax4.set_xlabel('Station', fontweight='600')
    ax4.set_title('Maximum SNR by Station', fontweight='700', pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(station_stats['station'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Total clean peaks by station
    ax5 = fig.add_subplot(gs[1, 1])
    bars = ax5.bar(x, station_stats['n_peaks_clean_sum'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax5.set_ylabel('Total Clean Peaks', fontweight='600')
    ax5.set_xlabel('Station', fontweight='600')
    ax5.set_title('Total Clean Peaks by Station', fontweight='700', pad=15)
    ax5.set_xticks(x)
    ax5.set_xticklabels(station_stats['station'], rotation=45, ha='right')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. High correlation peaks (>0.7)
    ax6 = fig.add_subplot(gs[1, 2])
    bars = ax6.bar(x, station_stats['n_peaks_above_0.7_clean_sum'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax6.set_ylabel('Peaks with r > 0.7', fontweight='600')
    ax6.set_xlabel('Station', fontweight='600')
    ax6.set_title('Very High Correlation Peaks by Station', fontweight='700', pad=15)
    ax6.set_xticks(x)
    ax6.set_xticklabels(station_stats['station'], rotation=45, ha='right')
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. % Peaks lost by station
    ax7 = fig.add_subplot(gs[2, 0])
    bars = ax7.bar(x, station_stats['pct_peaks_lost_mean'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax7.set_ylabel('% Peaks Lost', fontweight='600')
    ax7.set_xlabel('Station', fontweight='600')
    ax7.set_title('Average Peak Loss by Station', fontweight='700', pad=15)
    ax7.set_xticks(x)
    ax7.set_xticklabels(station_stats['station'], rotation=45, ha='right')
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Peak correlation by station
    ax8 = fig.add_subplot(gs[2, 1])
    for dataset in station_stats['dataset'].unique():
        data = station_stats[station_stats['dataset'] == dataset]
        ax8.errorbar(data['station'], data['peak_corr_mean_clean_mean'],
                    yerr=data['peak_corr_mean_clean_std'],
                    marker='D', linestyle='-', linewidth=2, markersize=8,
                    label=dataset.upper(), color=DATASET_COLORS.get(dataset, '#666'),
                    capsize=5, capthick=2, alpha=0.85)
    ax8.set_ylabel('Mean Peak Correlation ± Std', fontweight='600')
    ax8.set_xlabel('Station', fontweight='600')
    ax8.set_title('Mean Peak Correlation by Station', fontweight='700', pad=15)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 9. Max peak correlation by station
    ax9 = fig.add_subplot(gs[2, 2])
    bars = ax9.bar(x, station_stats['peak_corr_max_clean_max'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax9.set_ylabel('Max Peak Correlation', fontweight='600')
    ax9.set_xlabel('Station', fontweight='600')
    ax9.set_title('Maximum Peak Correlation by Station', fontweight='700', pad=15)
    ax9.set_xticks(x)
    ax9.set_xticklabels(station_stats['station'], rotation=45, ha='right')
    ax9.grid(True, alpha=0.3, axis='y')

    output_file = output_dir / "by_station" / "01_station_performance_overview.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


def plot_individual_station_details(df, output_dir):
    """
    Create detailed plots for each station showing template and sim performance.
    """
    for station in df['station'].unique():
        station_data = df[df['station'] == station]
        dataset = station_data['dataset'].iloc[0]

        fig = plt.figure(figsize=(16, 12), facecolor='white')
        gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25)
        fig.suptitle(f'Station {station} ({dataset.upper()}) - Detailed Analysis',
                    fontsize=14, fontweight='bold', y=0.98)

        # 1. Correlation by template and sim
        ax1 = fig.add_subplot(gs[0, 0])
        pivot = station_data.pivot_table(values='corr_mean_clean', index='template', columns='sim')
        pivot.plot(kind='bar', ax=ax1, color=[SIM_COLORS.get(c, '#666') for c in pivot.columns], width=0.7, alpha=0.85, edgecolor='black', linewidth=0.8)
        ax1.set_ylabel('Mean Correlation', fontweight='600')
        ax1.set_xlabel('Template', fontweight='600')
        ax1.set_title('Mean Correlation by Template & Simulation', fontweight='700', pad=15)
        ax1.legend(title='Simulation', ncol=2, loc='upper right', framealpha=0.95)
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

        # 2. SNR by template and sim
        ax2 = fig.add_subplot(gs[0, 1])
        pivot = station_data.pivot_table(values='snr_mean_db_clean', index='template', columns='sim')
        pivot.plot(kind='bar', ax=ax2, color=[SIM_COLORS.get(c, '#666') for c in pivot.columns], width=0.7, alpha=0.85, edgecolor='black', linewidth=0.8)
        ax2.set_ylabel('Mean SNR (dB)', fontweight='600')
        ax2.set_xlabel('Template', fontweight='600')
        ax2.set_title('Mean SNR by Template & Simulation', fontweight='700', pad=15)
        ax2.legend(title='Simulation', ncol=2, loc='upper right', framealpha=0.95)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

        # 3. Peaks lost by template and sim
        ax3 = fig.add_subplot(gs[1, 0])
        pivot = station_data.pivot_table(values='pct_peaks_lost', index='template', columns='sim')
        pivot.plot(kind='bar', ax=ax3, color=[SIM_COLORS.get(c, '#666') for c in pivot.columns], width=0.7, alpha=0.85, edgecolor='black', linewidth=0.8)
        ax3.set_ylabel('% Peaks Lost', fontweight='600')
        ax3.set_xlabel('Template', fontweight='600')
        ax3.set_title('Peak Loss by Template & Simulation', fontweight='700', pad=15)
        ax3.legend(title='Simulation', ncol=2, loc='upper right', framealpha=0.95)
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

        # 4. High correlation peaks (>0.5)
        ax4 = fig.add_subplot(gs[1, 1])
        pivot = station_data.pivot_table(values='n_peaks_above_0.7_clean', index='template', columns='sim')
        pivot.plot(kind='bar', ax=ax4, color=[SIM_COLORS.get(c, '#666') for c in pivot.columns], width=0.7, alpha=0.85, edgecolor='black', linewidth=0.8)
        ax4.set_ylabel('Peaks with r > 0.7', fontweight='600')
        ax4.set_xlabel('Template', fontweight='600')
        ax4.set_title('Very High Correlation Peaks by Template & Simulation', fontweight='700', pad=15)
        ax4.legend(title='Simulation', ncol=2, loc='upper right', framealpha=0.95)
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)

        output_file = output_dir / "by_station" / f"station_{station}_{dataset}_detail.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Saved: {output_file.name}")


# ============================================================================
# Template Effectiveness Analysis
# ============================================================================

def plot_template_effectiveness(df, output_dir):
    """
    Analyze template effectiveness across all stations and simulations.
    """
    fig = plt.figure(figsize=(20, 12), facecolor='white')
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.25)
    fig.suptitle('Template Effectiveness Analysis', fontsize=16, fontweight='bold', y=0.98)

    # Prepare template statistics
    template_stats = df.groupby('template').agg({
        'corr_mean_clean': ['mean', 'std', 'max'],
        'peak_corr_mean_clean': ['mean', 'std', 'max'],
        'snr_mean_db_clean': ['mean', 'std', 'max'],
        'n_peaks_clean': 'sum',
        'n_peaks_above_0.7_clean': 'sum',
        'pct_peaks_lost': 'mean'
    }).reset_index()

    template_stats.columns = ['_'.join(col).strip('_') for col in template_stats.columns]

    # 1. Mean correlation by template
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(template_stats))
    ax1.bar(x, template_stats['corr_mean_clean_mean'],
           yerr=template_stats['corr_mean_clean_std'],
           color=[TEMPLATE_COLORS.get(t, '#666') for t in template_stats['template']],
           alpha=0.85, capsize=5, edgecolor='black', linewidth=0.8, width=0.6)
    ax1.set_ylabel('Mean Correlation ± Std', fontweight='600')
    ax1.set_xlabel('Template', fontweight='600')
    ax1.set_title('Mean Correlation by Template', fontweight='700', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(template_stats['template'])
    ax1.grid(True, alpha=0.3, axis='y')
    # Set y-limit with padding for error bars
    max_val = (template_stats['corr_mean_clean_mean'] + template_stats['corr_mean_clean_std']).max()
    ax1.set_ylim(bottom=0, top=max_val * 1.10)

    # 2. Max correlation by template
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x, template_stats['corr_mean_clean_max'],
           color=[TEMPLATE_COLORS.get(t, '#666') for t in template_stats['template']],
           alpha=0.85, edgecolor='black', linewidth=0.8, width=0.6)
    ax2.set_ylabel('Max Correlation', fontweight='600')
    ax2.set_xlabel('Template', fontweight='600')
    ax2.set_title('Maximum Correlation by Template', fontweight='700', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(template_stats['template'])
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Mean SNR by template
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(x, template_stats['snr_mean_db_clean_mean'],
           yerr=template_stats['snr_mean_db_clean_std'],
           color=[TEMPLATE_COLORS.get(t, '#666') for t in template_stats['template']],
           alpha=0.85, capsize=5, edgecolor='black', linewidth=0.8, width=0.6)
    ax3.set_ylabel('Mean SNR (dB) ± Std', fontweight='600')
    ax3.set_xlabel('Template', fontweight='600')
    ax3.set_title('Mean SNR by Template', fontweight='700', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(template_stats['template'])
    ax3.grid(True, alpha=0.3, axis='y')
    # Set y-limit with padding for error bars
    max_val = (template_stats['snr_mean_db_clean_mean'] + template_stats['snr_mean_db_clean_std']).max()
    ax3.set_ylim(bottom=0, top=max_val * 1.10)

    # 4. Total clean peaks by template
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(x, template_stats['n_peaks_clean_sum'],
           color=[TEMPLATE_COLORS.get(t, '#666') for t in template_stats['template']],
           alpha=0.85, edgecolor='black', linewidth=0.8, width=0.6)
    ax4.set_ylabel('Total Clean Peaks', fontweight='600')
    ax4.set_xlabel('Template', fontweight='600')
    ax4.set_title('Total Clean Peaks by Template', fontweight='700', pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(template_stats['template'])
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Very high correlation peaks (>0.7)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(x, template_stats['n_peaks_above_0.7_clean_sum'],
           color=[TEMPLATE_COLORS.get(t, '#666') for t in template_stats['template']],
           alpha=0.85, edgecolor='black', linewidth=0.8, width=0.6)
    ax5.set_ylabel('Peaks with r > 0.7', fontweight='600')
    ax5.set_xlabel('Template', fontweight='600')
    ax5.set_title('Very High Correlation Peaks by Template', fontweight='700', pad=15)
    ax5.set_xticks(x)
    ax5.set_xticklabels(template_stats['template'])
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Peak loss by template
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(x, template_stats['pct_peaks_lost_mean'],
           color=[TEMPLATE_COLORS.get(t, '#666') for t in template_stats['template']],
           alpha=0.85, edgecolor='black', linewidth=0.8, width=0.6)
    ax6.set_ylabel('% Peaks Lost', fontweight='600')
    ax6.set_xlabel('Template', fontweight='600')
    ax6.set_title('Average Peak Loss by Template', fontweight='700', pad=15)
    ax6.set_xticks(x)
    ax6.set_xticklabels(template_stats['template'])
    ax6.grid(True, alpha=0.3, axis='y')

    output_file = output_dir / "by_template" / "01_template_effectiveness.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


def plot_template_consistency(df, output_dir):
    """
    Analyze template consistency across datasets and stations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor='white')
    fig.suptitle('Template Consistency Analysis', fontsize=16, fontweight='bold', y=0.98)

    # 1. Correlation variability by template (box plot)
    ax1 = axes[0, 0]
    data_to_plot = [df[df['template'] == t]['corr_mean_clean'].dropna() for t in sorted(df['template'].unique())]
    bp = ax1.boxplot(data_to_plot, labels=sorted(df['template'].unique()), patch_artist=True, showfliers=False, whis=[5, 95])
    for patch, t in zip(bp['boxes'], sorted(df['template'].unique())):
        patch.set_facecolor(TEMPLATE_COLORS.get(t, '#666'))
        patch.set_alpha(0.7)
    ax1.set_ylabel('Mean Correlation', fontweight='600')
    ax1.set_xlabel('Template', fontweight='600')
    ax1.set_title('Correlation Distribution by Template', fontweight='700', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. SNR variability by template
    ax2 = axes[0, 1]
    data_to_plot = [df[df['template'] == t]['snr_mean_db_clean'].dropna() for t in sorted(df['template'].unique())]
    bp = ax2.boxplot(data_to_plot, labels=sorted(df['template'].unique()), patch_artist=True, showfliers=False, whis=[5, 95])
    for patch, t in zip(bp['boxes'], sorted(df['template'].unique())):
        patch.set_facecolor(TEMPLATE_COLORS.get(t, '#666'))
        patch.set_alpha(0.7)
    ax2.set_ylabel('Mean SNR (dB)', fontweight='600')
    ax2.set_xlabel('Template', fontweight='600')
    ax2.set_title('SNR Distribution by Template', fontweight='700', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Template performance by dataset
    ax3 = axes[1, 0]
    pivot = df.pivot_table(values='corr_mean_clean', index='template', columns='dataset', aggfunc='mean')
    pivot.plot(kind='bar', ax=ax3, color=[DATASET_COLORS.get(c, '#666') for c in pivot.columns])
    ax3.set_ylabel('Mean Correlation', fontweight='600')
    ax3.set_xlabel('Template', fontweight='600')
    ax3.set_title('Template Performance by Dataset', fontweight='700', pad=15)
    ax3.legend(title='Dataset')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

    # 4. Template coefficient of variation (consistency metric)
    ax4 = axes[1, 1]
    template_cv = df.groupby('template').agg({
        'corr_mean_clean': lambda x: x.std() / x.mean() if x.mean() != 0 else 0
    }).reset_index()
    template_cv.columns = ['template', 'cv']

    x = np.arange(len(template_cv))
    ax4.bar(x, template_cv['cv'],
           color=[TEMPLATE_COLORS.get(t, '#666') for t in template_cv['template']],
           alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_ylabel('Coefficient of Variation', fontweight='600')
    ax4.set_xlabel('Template', fontweight='600')
    ax4.set_title('Template Consistency (Lower = More Consistent)', fontweight='700', pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(template_cv['template'])
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = output_dir / "by_template" / "02_template_consistency.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


# ============================================================================
# Simulation Consistency Analysis
# ============================================================================

def plot_simulation_consistency(df, output_dir):
    """
    Analyze consistency across simulations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
    fig.suptitle('Simulation Consistency Analysis', fontsize=16, fontweight='bold', y=0.98)

    # Prepare simulation statistics
    sim_stats = df.groupby('sim').agg({
        'corr_mean_clean': ['mean', 'std', 'max'],
        'snr_mean_db_clean': ['mean', 'std', 'max'],
        'n_peaks_clean': 'sum',
        'n_peaks_above_0.7_clean': 'sum',
        'pct_peaks_lost': 'mean'
    }).reset_index()

    sim_stats.columns = ['_'.join(col).strip('_') for col in sim_stats.columns]

    # 1. Mean correlation by simulation
    ax1 = axes[0, 0]
    x = np.arange(len(sim_stats))
    ax1.bar(x, sim_stats['corr_mean_clean_mean'],
           yerr=sim_stats['corr_mean_clean_std'],
           color=[SIM_COLORS.get(s, '#666') for s in sim_stats['sim']],
           alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Mean Correlation ± Std', fontweight='600')
    ax1.set_xlabel('Simulation', fontweight='600')
    ax1.set_title('Mean Correlation by Simulation', fontweight='700', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sim_stats['sim'])
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Max correlation by simulation
    ax2 = axes[0, 1]
    ax2.bar(x, sim_stats['corr_mean_clean_max'],
           color=[SIM_COLORS.get(s, '#666') for s in sim_stats['sim']],
           alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Max Correlation', fontweight='600')
    ax2.set_xlabel('Simulation', fontweight='600')
    ax2.set_title('Maximum Correlation by Simulation', fontweight='700', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sim_stats['sim'])
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Mean SNR by simulation
    ax3 = axes[0, 2]
    ax3.bar(x, sim_stats['snr_mean_db_clean_mean'],
           yerr=sim_stats['snr_mean_db_clean_std'],
           color=[SIM_COLORS.get(s, '#666') for s in sim_stats['sim']],
           alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Mean SNR (dB) ± Std', fontweight='600')
    ax3.set_xlabel('Simulation', fontweight='600')
    ax3.set_title('Mean SNR by Simulation', fontweight='700', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(sim_stats['sim'])
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Total clean peaks by simulation
    ax4 = axes[1, 0]
    ax4.bar(x, sim_stats['n_peaks_clean_sum'],
           color=[SIM_COLORS.get(s, '#666') for s in sim_stats['sim']],
           alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_ylabel('Total Clean Peaks', fontweight='600')
    ax4.set_xlabel('Simulation', fontweight='600')
    ax4.set_title('Total Clean Peaks by Simulation', fontweight='700', pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(sim_stats['sim'])
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Very high correlation peaks
    ax5 = axes[1, 1]
    ax5.bar(x, sim_stats['n_peaks_above_0.7_clean_sum'],
           color=[SIM_COLORS.get(s, '#666') for s in sim_stats['sim']],
           alpha=0.8, edgecolor='black', linewidth=0.5)
    ax5.set_ylabel('Peaks with r > 0.7', fontweight='600')
    ax5.set_xlabel('Simulation', fontweight='600')
    ax5.set_title('Very High Correlation Peaks by Simulation', fontweight='700', pad=15)
    ax5.set_xticks(x)
    ax5.set_xticklabels(sim_stats['sim'])
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Correlation distribution by simulation (box plot)
    ax6 = axes[1, 2]
    data_to_plot = [df[df['sim'] == s]['corr_mean_clean'].dropna() for s in sorted(df['sim'].unique())]
    bp = ax6.boxplot(data_to_plot, labels=sorted(df['sim'].unique()), patch_artist=True, showfliers=False, whis=[5, 95])
    for patch, s in zip(bp['boxes'], sorted(df['sim'].unique())):
        patch.set_facecolor(SIM_COLORS.get(s, '#666'))
        patch.set_alpha(0.7)
    ax6.set_ylabel('Mean Correlation', fontweight='600')
    ax6.set_xlabel('Simulation', fontweight='600')
    ax6.set_title('Correlation Distribution by Simulation', fontweight='700', pad=15)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = output_dir / "by_simulation" / "01_simulation_consistency.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


# ============================================================================
# Comprehensive Dataset Comparison
# ============================================================================

def plot_dataset_comparison(df, output_dir):
    """
    Comprehensive comparison between INGV and Experiment datasets.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
    fig.suptitle('Dataset Comparison: INGV vs Experiment', fontsize=16, fontweight='bold', y=0.98)

    # 1. Mean correlation comparison
    ax1 = axes[0, 0]
    dataset_corr = df.groupby('dataset')['corr_mean_clean'].agg(['mean', 'std']).reset_index()
    x = np.arange(len(dataset_corr))
    ax1.bar(x, dataset_corr['mean'], yerr=dataset_corr['std'],
           color=[DATASET_COLORS.get(d, '#666') for d in dataset_corr['dataset']],
           alpha=0.8, capsize=8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Mean Correlation ± Std', fontweight='600')
    ax1.set_xlabel('Dataset', fontweight='600')
    ax1.set_title('Mean Correlation Comparison', fontweight='700', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.upper() for d in dataset_corr['dataset']])
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Mean SNR comparison
    ax2 = axes[0, 1]
    dataset_snr = df.groupby('dataset')['snr_mean_db_clean'].agg(['mean', 'std']).reset_index()
    ax2.bar(x, dataset_snr['mean'], yerr=dataset_snr['std'],
           color=[DATASET_COLORS.get(d, '#666') for d in dataset_snr['dataset']],
           alpha=0.8, capsize=8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Mean SNR (dB) ± Std', fontweight='600')
    ax2.set_xlabel('Dataset', fontweight='600')
    ax2.set_title('Mean SNR Comparison', fontweight='700', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.upper() for d in dataset_snr['dataset']])
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Peak loss comparison
    ax3 = axes[0, 2]
    dataset_loss = df.groupby('dataset')['pct_peaks_lost'].mean().reset_index()
    ax3.bar(x, dataset_loss['pct_peaks_lost'],
           color=[DATASET_COLORS.get(d, '#666') for d in dataset_loss['dataset']],
           alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_ylabel('% Peaks Lost', fontweight='600')
    ax3.set_xlabel('Dataset', fontweight='600')
    ax3.set_title('Average Peak Loss Comparison', fontweight='700', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels([d.upper() for d in dataset_loss['dataset']])
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Correlation distribution comparison (violin plot)
    ax4 = axes[1, 0]
    data_ingv = df[df['dataset'] == 'ingv']['corr_mean_clean'].dropna()
    data_exp = df[df['dataset'] == 'experiment']['corr_mean_clean'].dropna()
    parts = ax4.violinplot([data_ingv, data_exp], positions=[0, 1], showmeans=True, showmedians=True)
    for pc, dataset in zip(parts['bodies'], ['ingv', 'experiment']):
        pc.set_facecolor(DATASET_COLORS.get(dataset, '#666'))
        pc.set_alpha(0.7)
    ax4.set_ylabel('Mean Correlation', fontweight='600')
    ax4.set_xlabel('Dataset', fontweight='600')
    ax4.set_title('Correlation Distribution Comparison', fontweight='700', pad=15)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['INGV', 'EXPERIMENT'])
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. SNR distribution comparison (violin plot)
    ax5 = axes[1, 1]
    data_ingv_snr = df[df['dataset'] == 'ingv']['snr_mean_db_clean'].dropna()
    data_exp_snr = df[df['dataset'] == 'experiment']['snr_mean_db_clean'].dropna()
    parts = ax5.violinplot([data_ingv_snr, data_exp_snr], positions=[0, 1], showmeans=True, showmedians=True)
    for pc, dataset in zip(parts['bodies'], ['ingv', 'experiment']):
        pc.set_facecolor(DATASET_COLORS.get(dataset, '#666'))
        pc.set_alpha(0.7)
    ax5.set_ylabel('Mean SNR (dB)', fontweight='600')
    ax5.set_xlabel('Dataset', fontweight='600')
    ax5.set_title('SNR Distribution Comparison', fontweight='700', pad=15)
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(['INGV', 'EXPERIMENT'])
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Total peaks comparison
    ax6 = axes[1, 2]
    dataset_peaks = df.groupby('dataset').agg({
        'n_peaks_contaminated': 'sum',
        'n_peaks_clean': 'sum',
        'n_peaks_lost': 'sum'
    }).reset_index()

    x = np.arange(len(dataset_peaks))
    width = 0.25
    ax6.bar(x - width, dataset_peaks['n_peaks_contaminated'], width,
           label='Contaminated', color='#ff8c42', alpha=0.85)
    ax6.bar(x, dataset_peaks['n_peaks_clean'], width,
           label='Clean', color='#2563eb', alpha=0.85)
    ax6.bar(x + width, dataset_peaks['n_peaks_lost'], width,
           label='Lost', color='#dc2626', alpha=0.85)
    ax6.set_ylabel('Total Peak Count', fontweight='600')
    ax6.set_xlabel('Dataset', fontweight='600')
    ax6.set_title('Total Peaks Comparison', fontweight='700', pad=15)
    ax6.set_xticks(x)
    ax6.set_xticklabels([d.upper() for d in dataset_peaks['dataset']])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = output_dir / "by_dataset" / "01_dataset_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


# ============================================================================
# Best Performers Ranking
# ============================================================================

def generate_rankings(df, output_dir):
    """
    Generate comprehensive rankings of best performing combinations.
    """
    # Create a combined identifier
    df['combo'] = df['dataset'] + '_' + df['station'] + '_' + df['sim'] + '_' + df['template']

    # Compute composite scores
    # Normalize metrics to 0-1 scale
    df['corr_score'] = (df['corr_mean_clean'] - df['corr_mean_clean'].min()) / (df['corr_mean_clean'].max() - df['corr_mean_clean'].min())
    df['snr_score'] = (df['snr_mean_db_clean'] - df['snr_mean_db_clean'].min()) / (df['snr_mean_db_clean'].max() - df['snr_mean_db_clean'].min())
    df['peak_corr_score'] = (df['peak_corr_mean_clean'] - df['peak_corr_mean_clean'].min()) / (df['peak_corr_mean_clean'].max() - df['peak_corr_mean_clean'].min())

    # Composite score (weighted average)
    df['composite_score'] = 0.4 * df['corr_score'] + 0.4 * df['snr_score'] + 0.2 * df['peak_corr_score']

    # Top 20 by composite score
    top_20 = df.nlargest(20, 'composite_score')[['combo', 'dataset', 'station', 'sim', 'template',
                                                   'corr_mean_clean', 'snr_mean_db_clean',
                                                   'peak_corr_mean_clean', 'composite_score']]

    # Save to CSV
    rankings_csv = output_dir / "rankings" / "top_20_combinations.csv"
    top_20.to_csv(rankings_csv, index=False)
    print(f"✅ Saved top 20 rankings: {rankings_csv.name}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), facecolor='white')
    fig.suptitle('Top Performers Ranking', fontsize=16, fontweight='bold', y=0.98)

    # 1. Top 15 by correlation
    ax1 = axes[0, 0]
    top_corr = df.nlargest(15, 'corr_mean_clean')
    y_pos = np.arange(len(top_corr))
    colors = [DATASET_COLORS.get(d, '#666') for d in top_corr['dataset']]
    ax1.barh(y_pos, top_corr['corr_mean_clean'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{row['station']}_{row['sim']}_{row['template']}"
                         for _, row in top_corr.iterrows()], fontsize=8)
    ax1.set_xlabel('Mean Correlation', fontweight='600')
    ax1.set_title('Top 15: Highest Mean Correlation', fontweight='700', pad=15)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    # 2. Top 15 by SNR
    ax2 = axes[0, 1]
    top_snr = df.nlargest(15, 'snr_mean_db_clean')
    y_pos = np.arange(len(top_snr))
    colors = [DATASET_COLORS.get(d, '#666') for d in top_snr['dataset']]
    ax2.barh(y_pos, top_snr['snr_mean_db_clean'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{row['station']}_{row['sim']}_{row['template']}"
                         for _, row in top_snr.iterrows()], fontsize=8)
    ax2.set_xlabel('Mean SNR (dB)', fontweight='600')
    ax2.set_title('Top 15: Highest Mean SNR', fontweight='700', pad=15)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    # 3. Top 15 by peak correlation
    ax3 = axes[1, 0]
    top_peak = df.nlargest(15, 'peak_corr_mean_clean')
    y_pos = np.arange(len(top_peak))
    colors = [DATASET_COLORS.get(d, '#666') for d in top_peak['dataset']]
    ax3.barh(y_pos, top_peak['peak_corr_mean_clean'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f"{row['station']}_{row['sim']}_{row['template']}"
                         for _, row in top_peak.iterrows()], fontsize=8)
    ax3.set_xlabel('Mean Peak Correlation', fontweight='600')
    ax3.set_title('Top 15: Highest Peak Correlation', fontweight='700', pad=15)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()

    # 4. Top 15 by composite score
    ax4 = axes[1, 1]
    top_composite = df.nlargest(15, 'composite_score')
    y_pos = np.arange(len(top_composite))
    colors = [DATASET_COLORS.get(d, '#666') for d in top_composite['dataset']]
    ax4.barh(y_pos, top_composite['composite_score'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([f"{row['station']}_{row['sim']}_{row['template']}"
                         for _, row in top_composite.iterrows()], fontsize=8)
    ax4.set_xlabel('Composite Score', fontweight='600')
    ax4.set_title('Top 15: Composite Score (40% Corr + 40% SNR + 20% Peak)', fontweight='700', pad=15)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()

    plt.tight_layout()
    output_file = output_dir / "rankings" / "01_top_performers.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")

    return top_20


def plot_comprehensive_correlation_matrix(df, output_dir):
    """
    Create correlation matrices showing relationships between key metrics.
    """
    # Select key metrics
    metrics = ['corr_mean_clean', 'corr_max_clean', 'peak_corr_mean_clean',
               'snr_mean_db_clean', 'snr_max_db_clean', 'n_peaks_clean',
               'n_peaks_above_0.7_clean', 'pct_peaks_lost']

    metric_labels = ['Mean Corr', 'Max Corr', 'Mean Peak Corr',
                    'Mean SNR', 'Max SNR', '# Clean Peaks',
                    '# High Peaks', '% Peaks Lost']

    # Compute correlation matrix
    corr_matrix = df[metrics].corr()
    corr_matrix.index = metric_labels
    corr_matrix.columns = metric_labels

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='white')

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
               vmin=-1, vmax=1, square=True, linewidths=0.5,
               cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)

    ax.set_title('Metric Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_file = output_dir / "comprehensive" / "01_metric_correlations.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_file.name}")


# ============================================================================
# Summary Statistics Export
# ============================================================================

def export_summary_statistics(df, output_dir):
    """
    Export comprehensive summary statistics to CSV files.
    """
    # Overall summary
    overall_summary = df.agg({
        'corr_mean_clean': ['count', 'mean', 'std', 'min', 'max'],
        'snr_mean_db_clean': ['mean', 'std', 'min', 'max'],
        'n_peaks_clean': 'sum',
        'n_peaks_contaminated': 'sum',
        'n_peaks_lost': 'sum',
        'pct_peaks_lost': 'mean'
    }).transpose()
    overall_summary.to_csv(output_dir / "comprehensive" / "overall_summary.csv")
    print(f"✅ Exported overall summary statistics")

    # By station
    station_summary = df.groupby(['dataset', 'station']).agg({
        'corr_mean_clean': ['mean', 'std', 'max'],
        'snr_mean_db_clean': ['mean', 'std', 'max'],
        'n_peaks_clean': 'sum',
        'pct_peaks_lost': 'mean',
        'n_peaks_above_0.7_clean': 'sum'
    }).round(4)
    station_summary.to_csv(output_dir / "by_station" / "station_summary_stats.csv")
    print(f"✅ Exported station summary statistics")

    # By template
    template_summary = df.groupby('template').agg({
        'corr_mean_clean': ['mean', 'std', 'max'],
        'snr_mean_db_clean': ['mean', 'std', 'max'],
        'n_peaks_clean': 'sum',
        'pct_peaks_lost': 'mean'
    }).round(4)
    template_summary.to_csv(output_dir / "by_template" / "template_summary_stats.csv")
    print(f"✅ Exported template summary statistics")

    # By simulation
    sim_summary = df.groupby('sim').agg({
        'corr_mean_clean': ['mean', 'std', 'max'],
        'snr_mean_db_clean': ['mean', 'std', 'max'],
        'n_peaks_clean': 'sum',
        'pct_peaks_lost': 'mean'
    }).round(4)
    sim_summary.to_csv(output_dir / "by_simulation" / "simulation_summary_stats.csv")
    print(f"✅ Exported simulation summary statistics")

    # By dataset
    dataset_summary = df.groupby('dataset').agg({
        'corr_mean_clean': ['mean', 'std', 'max'],
        'snr_mean_db_clean': ['mean', 'std', 'max'],
        'n_peaks_clean': 'sum',
        'n_peaks_contaminated': 'sum',
        'pct_peaks_lost': 'mean'
    }).round(4)
    dataset_summary.to_csv(output_dir / "by_dataset" / "dataset_summary_stats.csv")
    print(f"✅ Exported dataset summary statistics")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("="*80)
    print("COMPREHENSIVE SWCC RESULTS ANALYSIS")
    print("="*80)
    print()

    # Load data
    print("📊 Loading P-wave impact comparison data...")
    df = load_all_pwave_impact_data()
    print()

    # P-wave contamination analysis
    print("🔬 Analyzing P-wave contamination impact...")
    plot_pwave_contamination_overview(df, OUTPUT_DIR)
    plot_pwave_impact_heatmaps(df, OUTPUT_DIR)
    print()

    # Station analysis
    print("📍 Analyzing station performance...")
    plot_station_performance_comparison(df, OUTPUT_DIR)
    plot_individual_station_details(df, OUTPUT_DIR)
    print()

    # Template analysis
    print("📋 Analyzing template effectiveness...")
    plot_template_effectiveness(df, OUTPUT_DIR)
    plot_template_consistency(df, OUTPUT_DIR)
    print()

    # Simulation analysis
    print("🔄 Analyzing simulation consistency...")
    plot_simulation_consistency(df, OUTPUT_DIR)
    print()

    # Dataset comparison
    print("📊 Comparing datasets...")
    plot_dataset_comparison(df, OUTPUT_DIR)
    print()

    # Rankings
    print("🏆 Generating rankings...")
    top_20 = generate_rankings(df, OUTPUT_DIR)
    print()
    print("Top 5 Combinations:")
    print(top_20.head()[['station', 'sim', 'template', 'corr_mean_clean', 'snr_mean_db_clean']].to_string(index=False))
    print()

    # Comprehensive analyses
    print("🔗 Creating comprehensive visualizations...")
    plot_comprehensive_correlation_matrix(df, OUTPUT_DIR)
    print()

    # Export statistics
    print("💾 Exporting summary statistics...")
    export_summary_statistics(df, OUTPUT_DIR)
    print()

    print("="*80)
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"   Output directory: {OUTPUT_DIR}")
    print(f"   Total combinations analyzed: {len(df)}")
    print("="*80)


if __name__ == "__main__":
    main()
