#!/usr/bin/env python3
"""
Analyze which templates yield the most clean peaks for ECPN and EC1 stations.
Separate analysis for INGV (eruptive) and IMPROVE (experiment, quiescence) datasets.
"""

import pandas as pd
from pathlib import Path

def load_station_data(station, dataset):
    """
    Load all clean peaks for a given station and dataset.

    Args:
        station: Station name (e.g., 'ECPN', 'EC1')
        dataset: 'ingv' or 'experiment'

    Returns:
        DataFrame with all peaks for that station-dataset combination
    """
    base_path = Path("/home/owen/tilt_validation/SWCC_utc_fixed")

    dfs = []
    sims = ['sim1', 'sim2', 'sim3', 'sim4']
    templates = ['template1', 'template2', 'template3', 'template4']

    for sim in sims:
        for template in templates:
            peak_file = base_path / dataset / station / sim / f"{station}_{sim}_{template}_peaks.csv"
            if peak_file.exists():
                df = pd.read_csv(peak_file)
                df['sim'] = sim
                df['template'] = template
                dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def analyze_templates(df, station, dataset_name):
    """
    Analyze template performance for a station-dataset combination.

    Args:
        df: DataFrame with peak data
        station: Station name
        dataset_name: 'INGV' or 'IMPROVE'

    Returns:
        DataFrame with template statistics
    """
    if df.empty:
        print(f"\n❌ No data found for {station} ({dataset_name})")
        return pd.DataFrame()

    # Group by template and count peaks
    template_stats = df.groupby('template').agg({
        'peak_index': 'count',
        'peak_corr': ['mean', 'std'],
        'snr_linear': 'mean'
    }).reset_index()

    # Flatten column names
    template_stats.columns = ['template', 'n_peaks', 'mean_corr', 'std_corr', 'mean_snr']

    # Sort by peak count descending
    template_stats = template_stats.sort_values('n_peaks', ascending=False)

    # Calculate percentage
    total_peaks = template_stats['n_peaks'].sum()
    template_stats['pct_of_total'] = 100 * template_stats['n_peaks'] / total_peaks

    return template_stats

def print_top_templates(stats, station, dataset_name, top_n=5):
    """
    Print top N templates for a station-dataset combination.
    """
    print(f"\n{'='*80}")
    print(f"TOP {min(top_n, len(stats))} TEMPLATES: {station} ({dataset_name})")
    print(f"{'='*80}")

    if stats.empty:
        print("No data available")
        return

    total_peaks = stats['n_peaks'].sum()
    print(f"Total clean peaks: {total_peaks:,}\n")

    for i, row in stats.head(top_n).iterrows():
        print(f"{row['template']:12s}  {row['n_peaks']:6,} peaks  "
              f"({row['pct_of_total']:5.1f}%)  "
              f"r̄={row['mean_corr']:.3f}±{row['std_corr']:.3f}  "
              f"SNR̄={row['mean_snr']:.2f}")

def main():
    """Main execution."""
    print("="*80)
    print("TEMPLATE PERFORMANCE ANALYSIS: ECPN & EC1")
    print("="*80)

    stations = ['ECPN', 'EC1']
    datasets = [('ingv', 'INGV'), ('experiment', 'IMPROVE')]

    all_results = {}

    for station in stations:
        for dataset_code, dataset_name in datasets:
            print(f"\n📊 Loading data: {station} ({dataset_name})...")

            # Load data
            df = load_station_data(station, dataset_code)

            if not df.empty:
                print(f"   Loaded {len(df):,} peaks from {dataset_code}/{station}")

            # Analyze templates
            stats = analyze_templates(df, station, dataset_name)

            # Store results
            key = f"{station}_{dataset_name}"
            all_results[key] = stats

            # Print top templates
            print_top_templates(stats, station, dataset_name, top_n=5)

    # Save summary tables
    print(f"\n{'='*80}")
    print("SAVING SUMMARY TABLES")
    print(f"{'='*80}")

    output_dir = Path("/home/owen/tilt_validation")

    for key, stats in all_results.items():
        if not stats.empty:
            output_file = output_dir / f"top_templates_{key}.csv"
            stats.to_csv(output_file, index=False)
            print(f"✅ Saved: {output_file}")

    # Create combined summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    for station in stations:
        print(f"\n{station}:")
        for dataset_code, dataset_name in datasets:
            key = f"{station}_{dataset_name}"
            stats = all_results[key]
            if not stats.empty:
                total = stats['n_peaks'].sum()
                top_template = stats.iloc[0]['template']
                top_count = stats.iloc[0]['n_peaks']
                print(f"  {dataset_name:8s}: {total:6,} peaks total  "
                      f"(best: {top_template} with {top_count:,} peaks)")
            else:
                print(f"  {dataset_name:8s}: No data")

    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
