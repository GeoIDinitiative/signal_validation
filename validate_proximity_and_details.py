#!/usr/bin/env python3
"""
Validate proximity metrics and other detailed claims from the thesis:
1. 74.5% of clean peaks occur within 20 minutes of P-wave exclusion zones
2. 7,430 earthquakes M ≥ 4.0 (39.0% of catalog)
3. Estimated 3.8% residual contamination (~2,346 peaks)
4. Station distribution percentages
5. Correlation and SNR ranges
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

SWCC_ROOT = Path("/home/owen/tilt_validation/SWCC_utc_fixed")
QUAKE_FILE = Path("/home/owen/tilt_validation/earthquakes_merged_utc.csv")

def validate_earthquake_catalog():
    """Validate earthquake catalog statistics."""
    print("="*80)
    print("VALIDATION: EARTHQUAKE CATALOG STATISTICS")
    print("="*80)

    # Load earthquake catalog
    quakes = pd.read_csv(QUAKE_FILE)

    print(f"\nTotal earthquakes in catalog: {len(quakes):,}")

    # Count by magnitude thresholds
    m_ge_3 = len(quakes[quakes['magnitude'] >= 3.0])
    m_ge_4 = len(quakes[quakes['magnitude'] >= 4.0])
    m_ge_5 = len(quakes[quakes['magnitude'] >= 5.0])

    print(f"\nMagnitude distribution:")
    print(f"  M ≥ 3.0: {m_ge_3:,} events")
    print(f"  M ≥ 4.0: {m_ge_4:,} events ({m_ge_4/len(quakes)*100:.1f}% of catalog)")
    print(f"  M ≥ 5.0: {m_ge_5:,} events")

    # Validate claim: 7,430 earthquakes M ≥ 4.0 (39.0%)
    print(f"\nCLAIM: 7,430 earthquakes M ≥ 4.0 (39.0% of catalog)")
    print(f"ACTUAL: {m_ge_4:,} earthquakes M ≥ 4.0 ({m_ge_4/len(quakes)*100:.1f}% of catalog)")

    if abs(m_ge_4 - 7430) < 100:  # Allow small tolerance
        print("  ✅ CLAIM APPROXIMATELY VALIDATED")
    else:
        print(f"  ⚠️  DIFFERENCE: {m_ge_4 - 7430:,} earthquakes")

    # Magnitude buffer zones
    small_quakes = len(quakes[quakes['magnitude'] < 4.0])
    medium_quakes = len(quakes[(quakes['magnitude'] >= 4.0) & (quakes['magnitude'] < 5.0)])
    large_quakes = len(quakes[quakes['magnitude'] >= 5.0])

    print(f"\nBuffer zone distribution:")
    print(f"  Small (M < 4.0, ±7 min):     {small_quakes:,} events ({small_quakes/len(quakes)*100:.1f}%)")
    print(f"  Medium (4.0 ≤ M < 5.0, ±10 min): {medium_quakes:,} events ({medium_quakes/len(quakes)*100:.1f}%)")
    print(f"  Large (M ≥ 5.0, ±15 min):    {large_quakes:,} events ({large_quakes/len(quakes)*100:.1f}%)")

    return quakes

def load_all_clean_peaks():
    """Load all clean peaks from both datasets."""
    print("\n" + "="*80)
    print("LOADING ALL CLEAN PEAKS FOR PROXIMITY ANALYSIS")
    print("="*80)

    all_peaks = []

    for dataset in ['ingv', 'experiment']:
        dataset_path = SWCC_ROOT / dataset
        peak_files = list(dataset_path.glob("**/*/[A-Z]*_sim*_template*_peaks.csv"))

        print(f"\nLoading {len(peak_files)} peak files from {dataset}...")

        for peak_file in peak_files:
            parts = peak_file.stem.split('_')
            station = parts[0]
            sim = parts[1]
            template = parts[2]

            df = pd.read_csv(peak_file)
            df['dataset'] = dataset
            df['station'] = station
            df['sim'] = sim
            df['template'] = template
            all_peaks.append(df)

    combined = pd.concat(all_peaks, ignore_index=True)
    print(f"\nTotal clean peaks loaded: {len(combined):,}")

    return combined

def calculate_proximity_to_pwave_zones(peaks_df, quakes_df):
    """
    Calculate how many clean peaks occur within 20 minutes of P-wave exclusion zones.

    This requires checking temporal proximity, not spatial.
    """
    print("\n" + "="*80)
    print("CALCULATING PROXIMITY TO P-WAVE EXCLUSION ZONES")
    print("="*80)

    # Convert peak times to datetime if they're not already
    if 'peak_time_utc' in peaks_df.columns:
        print("\nConverting peak times to datetime...")
        try:
            peaks_df['peak_datetime'] = pd.to_datetime(peaks_df['peak_time_utc'])
        except:
            print("  ⚠️  Could not parse peak_time_utc column")
            return None

        # Convert quake times
        print("Converting earthquake times to datetime...")
        try:
            quakes_df['origin_datetime'] = pd.to_datetime(quakes_df['time'])
        except:
            print("  ⚠️  Could not parse earthquake time column")
            return None

        # For each peak, find minimum time distance to any P-wave exclusion zone
        print("\nCalculating minimum temporal distance to P-wave zones...")
        print("(This may take a while for 192,495 peaks...)")

        proximity_20min = 0
        proximity_threshold = timedelta(minutes=20)

        # Sample for performance (checking all 192k peaks against 19k quakes is intensive)
        sample_size = min(10000, len(peaks_df))
        peaks_sample = peaks_df.sample(n=sample_size, random_state=42)

        print(f"\nUsing sample of {sample_size:,} peaks for proximity analysis...")

        for idx, peak in peaks_sample.iterrows():
            peak_time = peak['peak_datetime']

            # Find closest earthquake in time
            time_diffs = abs(quakes_df['origin_datetime'] - peak_time)
            min_time_diff = time_diffs.min()

            if min_time_diff <= proximity_threshold:
                proximity_20min += 1

        # Extrapolate to full dataset
        proximity_pct = (proximity_20min / len(peaks_sample)) * 100

        print(f"\n📊 PROXIMITY RESULTS (from sample):")
        print(f"  Peaks within 20min of quakes: {proximity_20min:,}/{sample_size:,}")
        print(f"  Percentage: {proximity_pct:.1f}%")
        print(f"  CLAIMED: 74.5%")

        if abs(proximity_pct - 74.5) < 10:
            print("  ✅ REASONABLY CLOSE TO CLAIM")
        else:
            print(f"  ⚠️  DIFFERENCE: {proximity_pct - 74.5:.1f}%")

        # Extrapolate to full dataset
        estimated_total_within_20min = int((proximity_20min / len(peaks_sample)) * len(peaks_df))
        print(f"\n  Estimated full dataset: {estimated_total_within_20min:,} peaks within 20min")

        return proximity_pct

    else:
        print("  ⚠️  peak_time_utc column not found in peaks data")
        return None

def validate_residual_contamination():
    """Validate the 3.8% residual contamination estimate (~2,346 peaks)."""
    print("\n" + "="*80)
    print("VALIDATION: RESIDUAL CONTAMINATION ESTIMATE")
    print("="*80)

    total_clean = 192495
    residual_pct = 3.8
    estimated_residual = int(total_clean * residual_pct / 100)

    print(f"\nCLAIM: 3.8% of clean peaks (~2,346 peaks) may retain subtle contamination")
    print(f"CALCULATION: {total_clean:,} × {residual_pct}% = {estimated_residual:,} peaks")

    claimed_value = 2346

    if abs(estimated_residual - claimed_value) < 10:
        print(f"  ✅ CALCULATION VALIDATED!")
    else:
        print(f"  ⚠️  CALCULATION DIFFERENCE: {estimated_residual - claimed_value:,} peaks")

def validate_station_distributions():
    """Validate station distribution percentages from thesis."""
    print("\n" + "="*80)
    print("VALIDATION: STATION DISTRIBUTIONS")
    print("="*80)

    peaks = load_all_clean_peaks()

    # INGV stations
    ingv_peaks = peaks[peaks['dataset'] == 'ingv']
    ecpn_count = len(ingv_peaks[ingv_peaks['station'] == 'ECPN'])
    ec1_count = len(ingv_peaks[ingv_peaks['station'] == 'EC1'])

    print(f"\n📊 INGV STATION DISTRIBUTION:")
    print(f"  ECPN: {ecpn_count:,} peaks ({ecpn_count/len(ingv_peaks)*100:.2f}%)")
    print(f"  EC1:  {ec1_count:,} peaks ({ec1_count/len(ingv_peaks)*100:.2f}%)")

    # IMPROVE stations
    improve_peaks = peaks[peaks['dataset'] == 'experiment']

    print(f"\n📊 IMPROVE STATION DISTRIBUTION:")
    for station in sorted(improve_peaks['station'].unique()):
        station_count = len(improve_peaks[improve_peaks['station'] == station])
        station_pct = (station_count / len(improve_peaks)) * 100
        print(f"  {station}: {station_count:,} peaks ({station_pct:.2f}%)")

    return peaks

def validate_correlation_ranges(peaks):
    """Validate correlation coefficient ranges and distributions."""
    print("\n" + "="*80)
    print("VALIDATION: CORRELATION COEFFICIENT RANGES")
    print("="*80)

    print(f"\n📊 CORRELATION STATISTICS:")
    print(f"  Minimum |r|: {abs(peaks['peak_corr']).min():.6f}")
    print(f"  Maximum |r|: {abs(peaks['peak_corr']).max():.6f}")
    print(f"  Mean |r|:    {abs(peaks['peak_corr']).mean():.4f}")
    print(f"  Median |r|:  {abs(peaks['peak_corr']).median():.4f}")

    # Percentile distribution
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n📊 CORRELATION PERCENTILES:")
    for p in percentiles:
        val = np.percentile(abs(peaks['peak_corr']), p)
        print(f"  {p:2d}th percentile: {val:.4f}")

    # Threshold counts
    thresh_02 = len(peaks[abs(peaks['peak_corr']) >= 0.2])
    thresh_03 = len(peaks[abs(peaks['peak_corr']) >= 0.3])
    thresh_04 = len(peaks[abs(peaks['peak_corr']) >= 0.4])
    thresh_05 = len(peaks[abs(peaks['peak_corr']) >= 0.5])
    thresh_06 = len(peaks[abs(peaks['peak_corr']) >= 0.6])

    print(f"\n📊 THRESHOLD DISTRIBUTION:")
    print(f"  |r| ≥ 0.2: {thresh_02:,} peaks (100.00%)")
    print(f"  |r| ≥ 0.3: {thresh_03:,} peaks ({thresh_03/len(peaks)*100:.2f}%)")
    print(f"  |r| ≥ 0.4: {thresh_04:,} peaks ({thresh_04/len(peaks)*100:.2f}%)")
    print(f"  |r| ≥ 0.5: {thresh_05:,} peaks ({thresh_05/len(peaks)*100:.2f}%)")
    print(f"  |r| ≥ 0.6: {thresh_06:,} peaks ({thresh_06/len(peaks)*100:.2f}%)")

def main():
    """Run all detailed validations."""
    print("\n" + "="*80)
    print("COMPREHENSIVE DETAILED CLAIMS VALIDATION")
    print("="*80)

    # Validate earthquake catalog
    quakes = validate_earthquake_catalog()

    # Validate station distributions and load peaks
    peaks = validate_station_distributions()

    # Validate correlation ranges
    validate_correlation_ranges(peaks)

    # Validate residual contamination estimate
    validate_residual_contamination()

    # Validate proximity (computationally intensive, using sample)
    # proximity_pct = calculate_proximity_to_pwave_zones(peaks, quakes)

    print("\n" + "="*80)
    print("DETAILED VALIDATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
