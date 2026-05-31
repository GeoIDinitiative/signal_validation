#!/usr/bin/env python3
"""
Validate specific claims about top-performing SST combinations from the thesis.

Key claims to validate:
1. ECPN-sim1-template3 leads with 2,361 peaks
2. Top 15 SSTs all from INGV dataset
3. EC1-sim1-template3 is top INGV EC1 performer (2,084 peaks)
4. Template4 combinations have higher SNR but fewer peaks
5. Specific statistics for top performers
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the top40 data file
TOP40_FILE = Path("/home/owen/tilt_validation/top40_sst_plots/top40_all_thresholds_data.csv")
SWCC_ROOT = Path("/home/owen/tilt_validation/SWCC_utc_fixed")

def validate_top40_rankings():
    """Validate the top 40 SST rankings from thesis."""
    print("="*80)
    print("VALIDATION: TOP 40 SST RANKINGS")
    print("="*80)

    # Load top40 data
    top40 = pd.read_csv(TOP40_FILE)

    print(f"\nLoaded {len(top40)} SST combinations from top40 data file")
    print(f"\nColumns: {list(top40.columns)}")

    # Validate #1: ECPN-sim1-template3
    print("\n" + "-"*80)
    print("CLAIM 1: ECPN-sim1-template3 leads with 2,361 peaks")
    print("-"*80)

    top_sst = top40.iloc[0]
    print(f"\nTop SST from data:")
    print(f"  Station: {top_sst['station']}")
    print(f"  Sim: {top_sst['sim']}")
    print(f"  Template: {top_sst['template']}")
    print(f"  Peaks: {top_sst['n_peaks']:,}")
    print(f"  Avg Corr: {top_sst['avg_corr']:.4f}")
    print(f"  Avg SNR: {top_sst['avg_snr_linear']:.4f}")

    if (top_sst['station'] == 'ECPN' and
        top_sst['sim'] == 'sim1' and
        top_sst['template'] == 'template3' and
        top_sst['n_peaks'] == 2361):
        print("\n  ✅ CLAIM VALIDATED!")
    else:
        print("\n  ❌ CLAIM MISMATCH!")

    # Validate #2: Top 15 all from INGV
    print("\n" + "-"*80)
    print("CLAIM 2: Top 15 SSTs are all from INGV dataset")
    print("-"*80)

    top15 = top40.head(15)
    ingv_count = (top15['dataset'] == 'ingv').sum()

    print(f"\nTop 15 SSTs:")
    for i, row in top15.iterrows():
        print(f"  {i+1:2d}. {row['sst_label']:20s} - {row['dataset']:10s} - {row['n_peaks']:,} peaks")

    print(f"\nINGV count in top 15: {ingv_count}/15")

    if ingv_count == 15:
        print("  ✅ CLAIM VALIDATED! All top 15 are INGV")
    else:
        print(f"  ❌ CLAIM MISMATCH! Only {ingv_count}/15 are INGV")

    # Validate #3: Top EC1 INGV performer
    print("\n" + "-"*80)
    print("CLAIM 3: EC1-sim1-template3 is top INGV EC1 performer (2,084 peaks)")
    print("-"*80)

    ec1_ingv = top40[(top40['station'] == 'EC1') & (top40['dataset'] == 'ingv')]
    top_ec1 = ec1_ingv.iloc[0]

    print(f"\nTop EC1 INGV SST:")
    print(f"  SST: {top_ec1['sst_label']}")
    print(f"  Peaks: {top_ec1['n_peaks']:,}")
    print(f"  Avg Corr: {top_ec1['avg_corr']:.4f}")
    print(f"  Avg SNR: {top_ec1['avg_snr_linear']:.4f}")

    if (top_ec1['sim'] == 'sim1' and
        top_ec1['template'] == 'template3' and
        top_ec1['n_peaks'] == 2084):
        print("\n  ✅ CLAIM VALIDATED!")
    else:
        print("\n  ❌ CLAIM MISMATCH!")

    # Validate #4: Template4 characteristics
    print("\n" + "-"*80)
    print("CLAIM 4: Template4 combinations have higher SNR but fewer peaks")
    print("-"*80)

    template4 = top40[top40['template'] == 'template4']
    other_templates = top40[top40['template'] != 'template4']

    print(f"\nTemplate4 statistics:")
    print(f"  Count: {len(template4)} SSTs")
    print(f"  Avg peaks: {template4['n_peaks'].mean():.1f}")
    print(f"  Avg SNR: {template4['avg_snr_linear'].mean():.4f}")

    print(f"\nOther templates statistics:")
    print(f"  Count: {len(other_templates)} SSTs")
    print(f"  Avg peaks: {other_templates['n_peaks'].mean():.1f}")
    print(f"  Avg SNR: {other_templates['avg_snr_linear'].mean():.4f}")

    if (template4['n_peaks'].mean() < other_templates['n_peaks'].mean() and
        template4['avg_snr_linear'].mean() > other_templates['avg_snr_linear'].mean()):
        print("\n  ✅ CLAIM VALIDATED! Template4 has fewer peaks but higher SNR")
    else:
        print("\n  ❌ CLAIM NEEDS REVIEW")

    return top40

def validate_specific_sst_statistics(top40):
    """Validate specific SST statistics mentioned in thesis."""
    print("\n" + "="*80)
    print("VALIDATION: SPECIFIC SST STATISTICS FROM THESIS")
    print("="*80)

    # Key SSTs mentioned in thesis
    key_ssts = [
        ("ECPN", "sim1", "template3", 2361),
        ("ECPN", "sim4", "template1", 2119),
        ("ECPN", "sim3", "template1", 2096),
        ("EC1", "sim1", "template3", 2084),
        ("EC1", "sim1", "template1", 1910),
    ]

    print("\nValidating specific SST peak counts mentioned in thesis:")
    all_valid = True

    for station, sim, template, claimed_peaks in key_ssts:
        # Find this SST in top40
        sst_row = top40[(top40['station'] == station) &
                        (top40['sim'] == sim) &
                        (top40['template'] == template)]

        if len(sst_row) > 0:
            actual_peaks = sst_row.iloc[0]['n_peaks']
            match = "✅" if actual_peaks == claimed_peaks else "❌"
            print(f"  {match} {station}-{sim}-{template}: {actual_peaks:,} peaks (claimed: {claimed_peaks:,})")

            if actual_peaks != claimed_peaks:
                all_valid = False
        else:
            print(f"  ❌ {station}-{sim}-{template}: NOT FOUND in data")
            all_valid = False

    if all_valid:
        print("\n  ✅ ALL SPECIFIC SST CLAIMS VALIDATED!")
    else:
        print("\n  ⚠️  SOME DISCREPANCIES FOUND")

def validate_improve_top_performers(top40):
    """Validate IMPROVE dataset top performers."""
    print("\n" + "="*80)
    print("VALIDATION: IMPROVE (EXPERIMENT) TOP PERFORMERS")
    print("="*80)

    improve_ssts = top40[top40['dataset'] == 'experiment'].head(20)

    print(f"\nTop 20 IMPROVE SSTs:")
    for i, row in improve_ssts.iterrows():
        print(f"  {row['sst_label']:20s} - {row['n_peaks']:4,} peaks - "
              f"r̄={row['avg_corr']:.3f}, SNR={row['avg_snr_linear']:.3f}")

    # Validate first IMPROVE SST
    first_improve = improve_ssts.iloc[0]
    print(f"\nTop IMPROVE SST:")
    print(f"  SST: {first_improve['sst_label']}")
    print(f"  Peaks: {first_improve['n_peaks']:,}")
    print(f"  Position in overall top40: {improve_ssts.index[0] + 1}")

def cross_validate_with_raw_files():
    """Cross-validate top40 data with raw peak CSV files."""
    print("\n" + "="*80)
    print("CROSS-VALIDATION: TOP40 vs RAW PEAK FILES")
    print("="*80)

    # Check a few specific files to ensure data integrity
    test_cases = [
        ("ingv", "ECPN", "sim1", "template3", 2361),
        ("ingv", "EC1", "sim1", "template3", 2084),
        ("experiment", "EMAS", "sim1", "template2", 291),
    ]

    print("\nCross-checking peak counts with raw CSV files:")

    for dataset, station, sim, template, expected_count in test_cases:
        peak_file = SWCC_ROOT / dataset / station / sim / f"{station}_{sim}_{template}_peaks.csv"

        if peak_file.exists():
            df = pd.read_csv(peak_file)
            actual_count = len(df)
            match = "✅" if actual_count == expected_count else "❌"

            print(f"  {match} {dataset}/{station}/{sim}/{template}:")
            print(f"      Raw file: {actual_count:,} peaks")
            print(f"      Expected: {expected_count:,} peaks")

            if actual_count != expected_count:
                print(f"      ⚠️  MISMATCH: Difference of {actual_count - expected_count:,} peaks")
        else:
            print(f"  ❌ {dataset}/{station}/{sim}/{template}: FILE NOT FOUND")

def main():
    """Run all validations."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TOP40 SST VALIDATION")
    print("="*80)

    top40 = validate_top40_rankings()
    validate_specific_sst_statistics(top40)
    validate_improve_top_performers(top40)
    cross_validate_with_raw_files()

    print("\n" + "="*80)
    print("TOP40 VALIDATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
