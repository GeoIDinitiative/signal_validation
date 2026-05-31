#!/usr/bin/env python3
"""
Create bar plots showing top 40 station-sim-template combinations
across different correlation thresholds (All, 0.2-0.5, ≥0.5)

Replicates the plots from PhD thesis page 145
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Color schemes
DATASET_COLORS = {
    'ingv': '#2563eb',      # Blue
    'experiment': '#dc2626'  # Red
}

STATION_COLORS = {
    'ECPN': '#ef4444',  # Red
    'EC1': '#f97316',   # Orange
    'EEC1': '#f59e0b',  # Amber
    'EC10': '#eab308',  # Yellow
    'ECIT': '#84cc16',  # Lime
    'ECOR': '#22c55e',  # Green
    'EMAS': '#06b6d4',  # Cyan
}

def load_all_data():
    """Load all SWCC peak data from both datasets."""
    base_path = Path("/home/owen/tilt_validation/SWCC_utc_fixed")

    dfs = []

    # Load INGV clean peaks from all sim-template combinations
    ingv_stations = ['EC1', 'ECPN']
    sims = ['sim1', 'sim2', 'sim3', 'sim4']
    templates = ['template1', 'template2', 'template3', 'template4']

    for station in ingv_stations:
        for sim in sims:
            for template in templates:
                peak_file = base_path / "ingv" / station / sim / f"{station}_{sim}_{template}_peaks.csv"
                if peak_file.exists():
                    df = pd.read_csv(peak_file)
                    # Add metadata
                    df['dataset'] = 'ingv'
                    df['station'] = station
                    df['sim'] = sim
                    df['template'] = template
                    dfs.append(df)

    ingv_count = sum(len(df) for df in dfs)
    print(f"✅ Loaded INGV: {ingv_count:,} peaks from {len(dfs)} sim-template files")

    # Load EXPERIMENT clean peaks
    exp_stations = ['EC1', 'EC10', 'ECIT', 'ECOR', 'EMAS']
    for station in exp_stations:
        for sim in sims:
            for template in templates:
                peak_file = base_path / "experiment" / station / sim / f"{station}_{sim}_{template}_peaks.csv"
                if peak_file.exists():
                    df = pd.read_csv(peak_file)
                    # Add metadata
                    df['dataset'] = 'experiment'
                    df['station'] = station
                    df['sim'] = sim
                    df['template'] = template
                    dfs.append(df)

    exp_count = sum(len(df) for df in dfs if df['dataset'].iloc[0] == 'experiment')
    print(f"✅ Loaded IMPROVE: {exp_count:,} peaks from experiment files")

    if not dfs:
        raise FileNotFoundError("No peak files found!")

    # Combine datasets
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"✅ Combined dataset: {len(df_all):,} total peaks")

    return df_all

def create_sst_label(row):
    """Create station-sim-template label."""
    return f"{row['station']}-{row['sim']}-{row['template']}"

def filter_by_correlation(df, corr_min=None, corr_max=None):
    """
    Filter data by correlation value range.

    Args:
        df: DataFrame with peak_corr column
        corr_min: Minimum correlation (inclusive)
        corr_max: Maximum correlation (exclusive)

    Returns:
        Filtered DataFrame
    """
    if corr_min is not None and corr_max is not None:
        return df[(df['peak_corr'] >= corr_min) & (df['peak_corr'] < corr_max)]
    elif corr_min is not None:
        return df[df['peak_corr'] >= corr_min]
    else:
        return df

def aggregate_sst_peaks(df):
    """
    Aggregate peaks by station-sim-template combination.

    Returns DataFrame with total peaks per SST combination.
    """
    # Count peaks by dataset, station, sim, template
    grouped = df.groupby(['dataset', 'station', 'sim', 'template']).agg({
        'peak_index': 'count',  # Count number of peaks
        'peak_corr': 'mean',    # Average correlation
        'snr_linear': 'mean'     # Average SNR
    }).reset_index()

    # Rename columns
    grouped.rename(columns={
        'peak_index': 'n_peaks',
        'peak_corr': 'avg_corr',
        'snr_linear': 'avg_snr_linear'
    }, inplace=True)

    # Create SST label
    grouped['sst_label'] = grouped.apply(create_sst_label, axis=1)

    # Sort by peak count descending
    grouped = grouped.sort_values('n_peaks', ascending=False)

    return grouped

def create_top40_barplot(df_agg, title, output_file, top_n=40):
    """
    Create bar plot of top N station-sim-template combinations.

    Args:
        df_agg: Aggregated DataFrame
        title: Plot title
        output_file: Output file path
        top_n: Number of top combinations to show
    """
    # Take top N
    df_top = df_agg.head(top_n).copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')

    # Create bars with colors based on dataset
    colors = [DATASET_COLORS.get(dataset, '#666666') for dataset in df_top['dataset']]

    x_pos = np.arange(len(df_top))
    bars = ax.bar(x_pos, df_top['n_peaks'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)

    # Customize plot
    ax.set_ylabel('Number of Peaks', fontweight='700', fontsize=13)
    ax.set_title(title, fontweight='700', fontsize=16, pad=20)

    # Set x-axis labels (no xlabel title, just the station-sim-template labels)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_top['sst_label'], rotation=90, ha='right', fontsize=9, fontweight='bold')

    # Add grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)

    # Add legend (top right)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=DATASET_COLORS['ingv'], label='INGV', alpha=0.8),
        Patch(facecolor=DATASET_COLORS['experiment'], label='IMPROVE', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)

    # Add statistics text box (right side, below legend)
    total_peaks = df_top['n_peaks'].sum()
    ingv_peaks = df_top[df_top['dataset'] == 'ingv']['n_peaks'].sum()
    exp_peaks = df_top[df_top['dataset'] == 'experiment']['n_peaks'].sum()

    stats_text = f"Top {min(top_n, len(df_top))} Combinations\n"
    stats_text += f"Total peaks: {total_peaks:,}\n"
    if total_peaks > 0:
        stats_text += f"INGV: {ingv_peaks:,} ({100*ingv_peaks/total_peaks:.1f}%)\n"
        stats_text += f"IMPROVE: {exp_peaks:,} ({100*exp_peaks/total_peaks:.1f}%)"
    else:
        stats_text += "No peaks in this range"

    # Position box on right side, below legend
    ax.text(0.98, 0.78, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                     edgecolor='black', linewidth=1))

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ Saved: {output_file}")
    if len(df_top) > 0:
        print(f"   Top combination: {df_top.iloc[0]['sst_label']} ({df_top.iloc[0]['n_peaks']:,} peaks)")
    print(f"   Total peaks (top {min(top_n, len(df_top))}): {total_peaks:,}")

def main():
    """Main execution."""
    print("="*80)
    print("CREATING TOP 40 STATION-SIM-TEMPLATE BAR PLOTS")
    print("="*80)

    # Load data
    print("\n1. Loading SWCC summary data...")
    df_all = load_all_data()

    # Output directory
    output_dir = Path("/home/owen/tilt_validation/top40_sst_plots")
    output_dir.mkdir(exist_ok=True)
    print(f"\n2. Output directory: {output_dir}")

    print("\n" + "="*80)
    print("PLOT 1: ALL THRESHOLDS (≥0.2)")
    print("="*80)

    # Filter for all thresholds (≥0.2)
    df_all_thresh = filter_by_correlation(df_all, corr_min=0.2)
    df_agg_all = aggregate_sst_peaks(df_all_thresh)

    print(f"Total SST combinations: {len(df_agg_all)}")
    print(f"Creating bar plot...")

    create_top40_barplot(
        df_agg_all,
        title="Top 40 Station-Sim-Template Combinations (All Thresholds ≥0.2)",
        output_file=output_dir / "01_top40_all_thresholds.png",
        top_n=40
    )

    print("\n" + "="*80)
    print("PLOT 2: MODERATE CORRELATION (0.2 ≤ r < 0.5)")
    print("="*80)

    # Filter for 0.2-0.5 range
    df_02_05 = filter_by_correlation(df_all, corr_min=0.2, corr_max=0.5)
    df_agg_02_05 = aggregate_sst_peaks(df_02_05)

    print(f"Total SST combinations: {len(df_agg_02_05)}")
    print(f"Creating bar plot...")

    create_top40_barplot(
        df_agg_02_05,
        title="Top 40 Station-Sim-Template Combinations (Moderate: 0.2 ≤ r < 0.5)",
        output_file=output_dir / "02_top40_moderate_02_05.png",
        top_n=40
    )

    print("\n" + "="*80)
    print("PLOT 3: HIGH CORRELATION (≥0.5)")
    print("="*80)

    # Filter for ≥0.5
    df_ge_05 = filter_by_correlation(df_all, corr_min=0.5)
    df_agg_ge_05 = aggregate_sst_peaks(df_ge_05)

    print(f"Total SST combinations: {len(df_agg_ge_05)}")
    print(f"Creating bar plot...")

    create_top40_barplot(
        df_agg_ge_05,
        title="Top 40 Station-Sim-Template Combinations (High Quality: r ≥ 0.5)",
        output_file=output_dir / "03_top40_high_quality_ge_05.png",
        top_n=40
    )

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Overall stats
    print(f"\nAll Thresholds (≥0.2):")
    print(f"  Unique SST combinations: {len(df_agg_all)}")
    print(f"  Total peaks: {df_agg_all['n_peaks'].sum():,}")
    print(f"  INGV peaks: {df_agg_all[df_agg_all['dataset']=='ingv']['n_peaks'].sum():,}")
    print(f"  IMPROVE peaks: {df_agg_all[df_agg_all['dataset']=='experiment']['n_peaks'].sum():,}")

    print(f"\nModerate Correlation (0.2-0.5):")
    print(f"  Unique SST combinations: {len(df_agg_02_05)}")
    print(f"  Total peaks: {df_agg_02_05['n_peaks'].sum():,}")

    print(f"\nHigh Correlation (≥0.5):")
    print(f"  Unique SST combinations: {len(df_agg_ge_05)}")
    print(f"  Total peaks: {df_agg_ge_05['n_peaks'].sum():,}")

    # Save summary tables
    print("\n" + "="*80)
    print("SAVING SUMMARY TABLES")
    print("="*80)

    df_agg_all.to_csv(output_dir / "top40_all_thresholds_data.csv", index=False)
    df_agg_02_05.to_csv(output_dir / "top40_moderate_02_05_data.csv", index=False)
    df_agg_ge_05.to_csv(output_dir / "top40_high_quality_ge_05_data.csv", index=False)

    print(f"✅ Summary tables saved to {output_dir}")

    print("\n" + "="*80)
    print("✅ ALL PLOTS CREATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nPlots saved to: {output_dir}")
    print("  - 01_top40_all_thresholds.png")
    print("  - 02_top40_moderate_02_05.png")
    print("  - 03_top40_high_quality_ge_05.png")

if __name__ == "__main__":
    main()
