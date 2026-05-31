#!/usr/bin/env python3
"""
REDESIGNED SWCC PIPELINE - Quality-Weighted Multi-Station Detection

This script implements a fundamentally improved SWCC approach:
- Phase 1: Intelligent pre-segmentation by signal quality
- Phase 3: SNR-weighted peak detection with quality scores
- Phase 4: Multi-station coherence detection
- Phase 5: Adaptive thresholds based on background
- Phase 6: Correlation with uncertainty quantification
- Phase 7: Hierarchical detection strategy

Key improvements over swcc_edit.py:
1. Detects peaks on PRE-CLEANED signal (not post-filtering)
2. Quality scores (0-100) instead of binary clean/contaminated
3. Multi-station event detection (volcanic signals appear on multiple stations)
4. Adaptive thresholds per station
5. Hierarchical 3-pass detection (fast → validate → characterize)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

# Input paths
BANDPASSED_CSV_ROOT = "/home/owen/bandpassed_exports_001_01_csv_only"
TEMPLATES_CSV_ROOT = "/home/owen/tilt_validation/tilt_templates_csv"
EARTHQUAKE_CSV = "/home/owen/tilt_validation/earthquakes_merged.csv"

# Output path
OUTPUT_ROOT = "/home/owen/tilt_validation/SWCC_redesigned"

# Detection parameters
BASE_THRESHOLD = 0.2  # Base threshold, will be adapted per station
SAMPLING_RATE = 1.0  # Hz

# Quality score weights
WEIGHT_CORRELATION = 40.0  # Max points from correlation
WEIGHT_SNR = 30.0          # Max points from SNR
WEIGHT_PWAVE_DIST = 20.0   # Max points from P-wave distance
WEIGHT_MULTI_STATION = 10.0  # Max points from multi-station coherence

# Quality tiers for segmentation
TIER1_MIN_DISTANCE = 30.0  # minutes - clean segments
TIER2_MIN_DISTANCE = 15.0  # minutes - marginal segments
# < 15 min = Tier 3 (contaminated, skip)

# Multi-station detection
MULTI_STATION_TIME_WINDOW = 5.0  # minutes - time window for coherence
MIN_STATIONS_FOR_EVENT = 3  # Minimum stations to confirm an event

# P-wave buffering (from swcc_edit.py)
USE_DISTANCE_BASED_BUFFER = True
DEFAULT_P_WAVE_BUFFER_MIN = 10.0

# Volcanic events overlay
VOLCANIC_EVENTS_CSV = "/home/owen/tilt_validation/etna_volcanic_events_cleaned.csv"

# SNR computation parameters
SNR_GUARD_S = 3000.0
SNR_HALFWIN_S = 10000.0

# Datasets and stations to process
DATASETS_STATIONS = {
    "experiment": ["EC1", "ECIT", "EMAS", "EC10", "ECOR"],
    "ingv": ["ECPN", "EC1"]  # EC1 uses EEC1 bandpassed files (mapped in load function)
}

# Hierarchical detection parameters
PASS1_THRESHOLD_MULTIPLIER = 0.75  # Lower threshold for initial scan
PASS2_THRESHOLD_MULTIPLIER = 1.25  # Higher threshold for validation
PASS3_MIN_QUALITY = 50.0  # Minimum quality score for characterization

# Uncertainty quantification
N_BOOTSTRAP_SAMPLES = 50  # Number of bootstrap iterations for uncertainty
BOOTSTRAP_NOISE_LEVEL = 0.1  # Noise level as fraction of template std

# ============================================================================
# Utility Functions (from swcc_edit.py)
# ============================================================================

def sliding_window_cross_correlation(template, signal):
    """Compute sliding window cross-correlation between template and signal."""
    if len(template) > len(signal):
        return np.array([])

    template_normalized = (template - np.mean(template)) / (np.std(template) + 1e-10)
    correlations = correlate(signal, template_normalized, mode='valid', method='fft')

    window_size = len(template)
    correlations_normalized = np.zeros_like(correlations)
    for i in range(len(correlations)):
        window = signal[i:i+window_size]
        window_normalized = (window - np.mean(window)) / (np.std(window) + 1e-10)
        correlations_normalized[i] = correlations[i] / (np.linalg.norm(template_normalized) *
                                                         np.linalg.norm(window_normalized) + 1e-10)

    return correlations_normalized


def compute_template_snr_against_signal(signal, peak_idx, template_length, fs,
                                       noise_guard_s=3000.0, noise_halfwin_s=10000.0):
    """
    Compute SNR at a peak location by comparing signal power to nearby noise power.
    Returns SNR in linear scale (not dB).
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

    noise_concat = np.concatenate(noise_segments)
    noise_power = np.mean(noise_concat**2) if len(noise_concat) > 0 else 0.0

    # Return linear SNR (not dB)
    snr_linear = np.sqrt(signal_power / noise_power) if noise_power > 0 else 0.0

    return snr_linear


def calculate_distance_based_buffer(distance_km, magnitude=None):
    """Calculate P-wave buffer time based on magnitude (distance kept for API compatibility)."""
    if magnitude is None or pd.isna(magnitude):
        return DEFAULT_P_WAVE_BUFFER_MIN
    elif magnitude >= 5.0:
        return 15.0  # Large quakes: ±15 min
    elif magnitude >= 4.0:
        return 10.0  # Medium quakes: ±10 min
    else:
        return 7.0   # Small quakes: ±7 min


def load_volcanic_events(csv_path):
    """Load volcanic events from CSV."""
    df = pd.read_csv(csv_path)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['stop_time'] = pd.to_datetime(df['stop_time'])
    return df


# ============================================================================
# Phase 1: Signal Quality Segmentation
# ============================================================================

def compute_pwave_distance_timeseries(time_dt, earthquake_df):
    """
    For each sample, compute minimum distance (in minutes) to any P-wave.
    Returns array of distances in minutes.

    OPTIMIZED: Uses vectorized operations instead of nested loops.
    """
    distances = np.full(len(time_dt), np.inf)

    # Convert time_dt to seconds since epoch for faster computation
    time_seconds = np.array([t.timestamp() for t in time_dt])

    for _, quake in earthquake_df.iterrows():
        p_time = pd.to_datetime(quake['p_wave_eta'])
        magnitude = quake.get('magnitude', None)
        buffer_min = calculate_distance_based_buffer(quake.get('distance_km', None), magnitude=magnitude)
        buffer_sec = buffer_min * 60.0

        # Compute distance from this P-wave exclusion zone (vectorized)
        exclusion_start_sec = p_time.timestamp() - buffer_sec
        exclusion_end_sec = p_time.timestamp() + buffer_sec

        # Distance to exclusion zone (0 if inside, positive if outside)
        # For each sample, compute distance to nearest edge of zone
        dist_to_start = np.abs(time_seconds - exclusion_start_sec) / 60.0
        dist_to_end = np.abs(time_seconds - exclusion_end_sec) / 60.0
        dist_to_zone = np.minimum(dist_to_start, dist_to_end)

        # If inside zone, distance is 0
        inside_zone = (time_seconds >= exclusion_start_sec) & (time_seconds <= exclusion_end_sec)
        dist_to_zone[inside_zone] = 0.0

        # Update minimum distance
        distances = np.minimum(distances, dist_to_zone)

    return distances


def segment_signal_by_quality(time_dt, pwave_distances):
    """
    Segment signal into quality tiers:
    - Tier 1 (clean): >30 min from any P-wave
    - Tier 2 (marginal): 15-30 min from P-wave
    - Tier 3 (contaminated): <15 min from P-wave

    Returns list of segments: [(start_idx, end_idx, tier), ...]
    """
    segments = []
    current_tier = None
    segment_start = 0

    for i, dist in enumerate(pwave_distances):
        if dist >= TIER1_MIN_DISTANCE:
            tier = 1
        elif dist >= TIER2_MIN_DISTANCE:
            tier = 2
        else:
            tier = 3

        # Start new segment when tier changes
        if tier != current_tier:
            if current_tier is not None:
                segments.append((segment_start, i, current_tier))
            segment_start = i
            current_tier = tier

    # Add final segment
    if current_tier is not None:
        segments.append((segment_start, len(pwave_distances), current_tier))

    return segments


# ============================================================================
# Phase 3: Quality Score Computation
# ============================================================================

def compute_peak_quality_score(corr_value, snr_linear, pwave_distance_min,
                               multi_station_coherence=0):
    """
    Compute quality score (0-100) for a detected peak.

    Components:
    - Correlation: 0-40 points (normalized from 0.2-1.0)
    - SNR: 0-30 points (normalized from 0.56 to 3.16 linear scale, equivalent to -5 to +10 dB)
    - P-wave distance: 0-20 points (normalized from 0-30 min)
    - Multi-station coherence: 0-10 points (from multi-station detection)
    """
    # Correlation component
    corr_score = np.clip((corr_value - 0.2) / 0.8 * WEIGHT_CORRELATION, 0, WEIGHT_CORRELATION)

    # SNR component (linear scale)
    # Linear range: 0.56 (= 10^(-5/20)) to 3.16 (= 10^(10/20))
    MIN_SNR_LINEAR = 0.56  # Equivalent to -5 dB
    MAX_SNR_LINEAR = 3.16  # Equivalent to +10 dB
    if np.isfinite(snr_linear):
        snr_score = np.clip((snr_linear - MIN_SNR_LINEAR) / (MAX_SNR_LINEAR - MIN_SNR_LINEAR) * WEIGHT_SNR, 0, WEIGHT_SNR)
    else:
        snr_score = 0

    # P-wave distance component
    pwave_score = np.clip(pwave_distance_min / 30 * WEIGHT_PWAVE_DIST, 0, WEIGHT_PWAVE_DIST)

    # Multi-station coherence component
    coherence_score = multi_station_coherence * WEIGHT_MULTI_STATION

    total_score = corr_score + snr_score + pwave_score + coherence_score
    return total_score


# ============================================================================
# Phase 5: Adaptive Threshold
# ============================================================================

def compute_adaptive_threshold(correlations, pwave_distances, base_threshold=0.2):
    """
    Compute adaptive threshold based on background correlation in clean periods.
    Threshold = mean + 3*std of background, clipped to reasonable bounds.
    """
    # Find clean background periods (>30 min from P-waves)
    clean_mask = pwave_distances > TIER1_MIN_DISTANCE

    if np.sum(clean_mask) < 100:  # Not enough clean samples
        return base_threshold

    background_corr = correlations[clean_mask]
    background_mean = np.mean(np.abs(background_corr))
    background_std = np.std(np.abs(background_corr))

    # Threshold = mean + 3*sigma
    adaptive_threshold = background_mean + 3 * background_std

    # Clip to reasonable bounds [0.15, 0.35]
    adaptive_threshold = np.clip(adaptive_threshold, 0.15, 0.35)

    return adaptive_threshold


# ============================================================================
# Phase 6: Correlation with Uncertainty
# ============================================================================

def swcc_with_uncertainty(template, signal, n_bootstrap=N_BOOTSTRAP_SAMPLES,
                         noise_level=BOOTSTRAP_NOISE_LEVEL):
    """
    Compute correlation with uncertainty via bootstrap.
    Returns: correlation, lower CI, upper CI, std
    """
    # Main correlation
    correlations = sliding_window_cross_correlation(template, signal)

    if n_bootstrap == 0:
        return correlations, correlations, correlations, np.zeros_like(correlations)

    # Bootstrap: resample template with noise
    bootstrap_corrs = []
    template_std = np.std(template)

    for _ in range(n_bootstrap):
        noise = np.random.normal(0, template_std * noise_level, len(template))
        template_noisy = template + noise
        corr_boot = sliding_window_cross_correlation(template_noisy, signal)

        if len(corr_boot) == len(correlations):
            bootstrap_corrs.append(corr_boot)

    if len(bootstrap_corrs) == 0:
        return correlations, correlations, correlations, np.zeros_like(correlations)

    bootstrap_corrs = np.array(bootstrap_corrs)
    corr_std = np.std(bootstrap_corrs, axis=0)
    corr_ci_lower = correlations - 1.96 * corr_std
    corr_ci_upper = correlations + 1.96 * corr_std

    return correlations, corr_ci_lower, corr_ci_upper, corr_std


# ============================================================================
# Phase 7: Hierarchical Detection
# ============================================================================

def hierarchical_detection(signal, time_dt, templates, earthquake_df, station,
                          pwave_distances, adaptive_threshold):
    """
    Three-pass hierarchical detection:
    Pass 1: Quick scan with all templates (find candidates)
    Pass 2: Validate candidates with quality scores
    Pass 3: Characterize validated peaks with full analysis
    """
    all_peaks = []

    for template_name, template in templates.items():
        print(f"    🔍 {template_name}:")

        # ===== PASS 1: Quick scan =====
        print(f"      Pass 1: Scanning with threshold {adaptive_threshold * PASS1_THRESHOLD_MULTIPLIER:.3f}...")
        correlations = sliding_window_cross_correlation(template, signal)

        if len(correlations) == 0:
            print(f"      ⚠️  No correlation results")
            continue

        corr_time_dt = time_dt[:len(correlations)]
        pwave_dist_corr = pwave_distances[:len(correlations)]

        # Detect peaks with lower threshold
        distance = 1000  # 1000 samples = 1000s at 1 Hz
        pass1_threshold = adaptive_threshold * PASS1_THRESHOLD_MULTIPLIER
        peaks_pass1, props = find_peaks(np.abs(correlations), height=pass1_threshold, distance=int(distance))
        peak_heights_pass1 = props.get("peak_heights", np.array([]))

        print(f"      Pass 1: Found {len(peaks_pass1)} candidates")

        if len(peaks_pass1) == 0:
            continue

        # ===== PASS 2: Validate with quality scores =====
        print(f"      Pass 2: Validating with quality scores...")
        validated_peaks = []

        for peak_idx, peak_corr in zip(peaks_pass1, peak_heights_pass1):
            # Compute SNR (linear scale)
            snr_linear = compute_template_snr_against_signal(
                signal, peak_idx, len(template), SAMPLING_RATE,
                SNR_GUARD_S, SNR_HALFWIN_S
            )

            # Get P-wave distance at this peak
            pwave_dist = pwave_dist_corr[peak_idx]

            # Compute quality score (no multi-station info yet)
            quality = compute_peak_quality_score(peak_corr, snr_linear, pwave_dist,
                                                multi_station_coherence=0)

            # Validate: must exceed higher threshold OR have good quality
            if peak_corr >= adaptive_threshold * PASS2_THRESHOLD_MULTIPLIER or quality >= PASS3_MIN_QUALITY:
                validated_peaks.append({
                    'index': peak_idx,
                    'time': corr_time_dt[peak_idx],
                    'correlation': peak_corr,
                    'snr_linear': snr_linear,
                    'pwave_distance_min': pwave_dist,
                    'quality_score': quality,
                    'template': template_name,
                    'station': station,
                    'tier': 1 if pwave_dist >= TIER1_MIN_DISTANCE else (2 if pwave_dist >= TIER2_MIN_DISTANCE else 3)
                })

        print(f"      Pass 2: Validated {len(validated_peaks)} peaks")

        # ===== PASS 3: Characterize (already done with quality scores) =====
        all_peaks.extend(validated_peaks)

    return all_peaks


# ============================================================================
# Phase 4: Multi-Station Coherence Detection
# ============================================================================

def detect_multi_station_events(all_station_peaks, time_window_min=MULTI_STATION_TIME_WINDOW):
    """
    Find events that appear on multiple stations within time_window.
    Returns list of multi-station events with enhanced quality scores.
    """
    time_window = pd.Timedelta(minutes=time_window_min)
    candidate_events = []
    processed_peaks = set()

    # Group peaks by station
    peaks_by_station = {}
    for peak in all_station_peaks:
        station = peak['station']
        if station not in peaks_by_station:
            peaks_by_station[station] = []
        peaks_by_station[station].append(peak)

    # For each peak, find coherent peaks on other stations
    for station1, peaks1 in peaks_by_station.items():
        for peak1 in peaks1:
            peak1_id = (station1, peak1['time'])
            if peak1_id in processed_peaks:
                continue

            coherent_stations = [station1]
            coherent_peaks = [peak1]

            # Check other stations
            for station2, peaks2 in peaks_by_station.items():
                if station2 == station1:
                    continue

                for peak2 in peaks2:
                    time_diff = abs((peak1['time'] - peak2['time']).total_seconds() / 60.0)
                    if time_diff < time_window_min:
                        coherent_stations.append(station2)
                        coherent_peaks.append(peak2)

            # If appears on multiple stations, it's a strong candidate
            if len(coherent_stations) >= MIN_STATIONS_FOR_EVENT:
                # Mark all peaks as processed
                for p in coherent_peaks:
                    processed_peaks.add((p['station'], p['time']))

                # Compute multi-station coherence score (0-1)
                coherence = (len(coherent_stations) - 1) / (len(peaks_by_station) - 1)

                # Update quality scores for all peaks in this event
                for p in coherent_peaks:
                    # Recompute quality with multi-station bonus
                    p['multi_station_coherence'] = coherence
                    p['quality_score'] = compute_peak_quality_score(
                        p['correlation'], p['snr_linear'], p['pwave_distance_min'],
                        multi_station_coherence=coherence
                    )

                event = {
                    'time': np.median([p['time'] for p in coherent_peaks]),
                    'stations': coherent_stations,
                    'n_stations': len(coherent_stations),
                    'coherence': coherence,
                    'avg_correlation': np.mean([p['correlation'] for p in coherent_peaks]),
                    'avg_quality': np.mean([p['quality_score'] for p in coherent_peaks]),
                    'max_quality': np.max([p['quality_score'] for p in coherent_peaks]),
                    'peaks': coherent_peaks
                }
                candidate_events.append(event)

    # Sort by quality
    candidate_events.sort(key=lambda e: e['avg_quality'], reverse=True)

    return candidate_events


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_quality_segmentation(time_dt, pwave_distances, segments, output_path, station):
    """Plot signal quality segmentation showing tiers."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), facecolor='white')

    # Top: P-wave distance timeseries
    ax1.plot(time_dt, pwave_distances, color='#3b82f6', linewidth=0.5, alpha=0.7)
    ax1.axhline(TIER1_MIN_DISTANCE, color='#22c55e', linestyle='--', linewidth=2,
                label=f'Tier 1 threshold ({TIER1_MIN_DISTANCE} min)')
    ax1.axhline(TIER2_MIN_DISTANCE, color='#f59e0b', linestyle='--', linewidth=2,
                label=f'Tier 2 threshold ({TIER2_MIN_DISTANCE} min)')
    ax1.fill_between(time_dt, 0, TIER2_MIN_DISTANCE, color='#dc2626', alpha=0.2, label='Tier 3 (contaminated)')
    ax1.fill_between(time_dt, TIER2_MIN_DISTANCE, TIER1_MIN_DISTANCE, color='#f59e0b', alpha=0.2, label='Tier 2 (marginal)')
    ax1.fill_between(time_dt, TIER1_MIN_DISTANCE, pwave_distances.max(), color='#22c55e', alpha=0.2, label='Tier 1 (clean)')

    ax1.set_ylabel('Distance to P-wave (min)', fontsize=12, fontweight='600')
    ax1.set_title(f'Signal Quality Segmentation - {station}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, min(60, pwave_distances.max()))

    # Bottom: Quality tiers
    tier_colors = {1: '#22c55e', 2: '#f59e0b', 3: '#dc2626'}
    tier_labels = {1: 'Clean', 2: 'Marginal', 3: 'Contaminated'}

    for start_idx, end_idx, tier in segments:
        ax2.axvspan(time_dt[start_idx], time_dt[end_idx-1],
                   color=tier_colors[tier], alpha=0.6)

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=tier_colors[t], alpha=0.6, label=f'Tier {t}: {tier_labels[t]}')
                      for t in [1, 2, 3]]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax2.set_ylabel('Quality Tier', fontsize=12, fontweight='600')
    ax2.set_xlabel('Time (UTC)', fontsize=12, fontweight='600')
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    import matplotlib.dates as mdates
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_multi_station_events(events, output_path, dataset):
    """Plot multi-station events timeline."""
    if len(events) == 0:
        return

    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')

    # Plot each event as a point
    for i, event in enumerate(events):
        color = '#22c55e' if event['avg_quality'] >= 75 else '#f59e0b' if event['avg_quality'] >= 50 else '#dc2626'
        size = event['n_stations'] * 100

        ax.scatter(event['time'], event['avg_quality'], s=size, c=color, alpha=0.7,
                  edgecolors='black', linewidths=1.5, zorder=10)

        # Add station labels
        station_text = '+'.join(event['stations'])
        ax.annotate(f"{i+1}: {station_text}", (event['time'], event['avg_quality']),
                   xytext=(5, 5), textcoords='offset points', fontsize=7, alpha=0.7)

    ax.axhline(75, color='#22c55e', linestyle='--', linewidth=2, alpha=0.5, label='High quality (>75)')
    ax.axhline(50, color='#f59e0b', linestyle='--', linewidth=2, alpha=0.5, label='Medium quality (50-75)')

    ax.set_xlabel('Time (UTC)', fontsize=12, fontweight='600')
    ax.set_ylabel('Average Quality Score', fontsize=12, fontweight='600')
    ax.set_title(f'Multi-Station Events - {dataset.upper()}\n(Size = number of stations)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_quality_score_scatter(peaks, output_path, station):
    """Plot quality scores vs correlation and SNR."""
    if len(peaks) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    correlations = [p['correlation'] for p in peaks]
    snrs_linear = [p['snr_linear'] for p in peaks if np.isfinite(p['snr_linear'])]
    qualities = [p['quality_score'] for p in peaks]
    qualities_with_snr = [p['quality_score'] for p in peaks if np.isfinite(p['snr_linear'])]

    # Left: Quality vs Correlation
    colors = ['#22c55e' if q >= 75 else '#f59e0b' if q >= 50 else '#dc2626' for q in qualities]
    ax1.scatter(correlations, qualities, c=colors, s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
    ax1.set_xlabel('Correlation Coefficient |r|', fontsize=12, fontweight='600')
    ax1.set_ylabel('Quality Score', fontsize=12, fontweight='600')
    ax1.set_title(f'Quality vs Correlation - {station}', fontsize=13, fontweight='bold')
    ax1.axhline(75, color='#22c55e', linestyle='--', alpha=0.5)
    ax1.axhline(50, color='#f59e0b', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Right: Quality vs SNR (Linear)
    if len(snrs_linear) > 0:
        colors_snr = ['#22c55e' if q >= 75 else '#f59e0b' if q >= 50 else '#dc2626' for q in qualities_with_snr]
        ax2.scatter(snrs_linear, qualities_with_snr, c=colors_snr, s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
        ax2.set_xlabel('SNR (Linear)', fontsize=12, fontweight='600')
        ax2.set_ylabel('Quality Score', fontsize=12, fontweight='600')
        ax2.set_title(f'Quality vs SNR - {station}', fontsize=13, fontweight='bold')
        ax2.axhline(75, color='#22c55e', linestyle='--', alpha=0.5)
        ax2.axhline(50, color='#f59e0b', linestyle='--', alpha=0.5)
        # Add reference line at SNR = 3.16 (equivalent to 10 dB)
        ax2.axvline(3.16, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='SNR = 3.16 (10 dB)')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================================
# Main Processing
# ============================================================================

def main():
    print("="*80)
    print("REDESIGNED SWCC PIPELINE - Quality-Weighted Multi-Station Detection")
    print("="*80)
    print("\nKey features:")
    print("  ✓ Phase 1: Signal quality segmentation (Tier 1/2/3)")
    print("  ✓ Phase 3: Quality scores (0-100) combining correlation, SNR, P-wave distance")
    print("  ✓ Phase 4: Multi-station coherence detection")
    print("  ✓ Phase 5: Adaptive thresholds per station")
    print("  ✓ Phase 7: Hierarchical 3-pass detection (scan → validate → characterize)")
    print()

    # Load earthquake data
    print(f"📊 Loading earthquake data...")
    earthquake_df = pd.read_csv(EARTHQUAKE_CSV)
    earthquake_df['p_wave_eta'] = pd.to_datetime(earthquake_df['p_wave_eta'])
    print(f"  ✅ Loaded {len(earthquake_df):,} earthquakes\n")

    # Load volcanic events
    volcanic_events_df = None
    try:
        volcanic_events_df = load_volcanic_events(VOLCANIC_EVENTS_CSV)
        print(f"  ✅ Loaded {len(volcanic_events_df):,} volcanic events\n")
    except Exception as e:
        print(f"  ⚠️  Could not load volcanic events: {e}\n")

    # Process each dataset
    for DATASET, STATIONS in DATASETS_STATIONS.items():
        print(f"{'='*80}")
        print(f"PROCESSING DATASET: {DATASET.upper()}")
        print(f"{'='*80}")
        print(f"Stations: {', '.join(STATIONS)}\n")

        dataset_output_dir = Path(OUTPUT_ROOT) / DATASET
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all peaks from all stations for multi-station detection
        all_station_peaks = []

        for STATION in STATIONS:
            print(f"{'='*80}")
            print(f"PROCESSING STATION: {STATION}")
            print(f"{'='*80}\n")

            station_output_dir = dataset_output_dir / STATION
            station_output_dir.mkdir(parents=True, exist_ok=True)

            # Load signal
            print(f"📊 Loading bandpassed signal for {STATION}...")
            csv_path = Path(BANDPASSED_CSV_ROOT) / DATASET / f"{STATION}_0p001-0p01Hz_bp.csv"

            if not csv_path.exists():
                print(f"  ⚠️  Signal file not found: {csv_path}\n")
                continue

            df_signal = pd.read_csv(csv_path)
            # Bandpassed CSV uses time_seconds (relative) - need to convert to absolute datetime
            # For now, use a dummy start time (Aug 24, 2023) - same as data collection
            start_time = pd.Timestamp('2023-08-24 00:00:00')
            time_dt_full = start_time + pd.to_timedelta(df_signal['time_seconds'], unit='s')
            signal_full = df_signal['bandpassed'].values

            print(f"  ✅ Loaded {len(signal_full):,} samples\n")

            # ===== PHASE 1: Signal Quality Segmentation =====
            print(f"{'='*80}")
            print(f"Phase 1: Segmenting signal by quality...")
            print(f"{'='*80}\n")

            pwave_distances = compute_pwave_distance_timeseries(time_dt_full, earthquake_df)
            segments = segment_signal_by_quality(time_dt_full, pwave_distances)

            # Count segment statistics
            tier_stats = {1: 0, 2: 0, 3: 0}
            for start, end, tier in segments:
                tier_stats[tier] += (end - start)

            total_samples = len(signal_full)
            print(f"  ✅ Signal segmented into {len(segments)} segments:")
            print(f"     Tier 1 (clean, >30 min): {tier_stats[1]:,} samples ({100*tier_stats[1]/total_samples:.1f}%)")
            print(f"     Tier 2 (marginal, 15-30 min): {tier_stats[2]:,} samples ({100*tier_stats[2]/total_samples:.1f}%)")
            print(f"     Tier 3 (contaminated, <15 min): {tier_stats[3]:,} samples ({100*tier_stats[3]/total_samples:.1f}%)\n")

            # Save segmentation plot
            seg_plot_path = station_output_dir / f"{STATION}_quality_segmentation.png"
            plot_quality_segmentation(time_dt_full, pwave_distances, segments, seg_plot_path, STATION)
            print(f"  ✅ Saved quality segmentation plot: {seg_plot_path.name}\n")

            # ===== PHASE 5: Adaptive Threshold =====
            # We'll compute this per template during detection

            # ===== PHASE 7: Hierarchical Detection =====
            print(f"{'='*80}")
            print(f"Phase 7: Hierarchical detection (3-pass)...")
            print(f"{'='*80}\n")

            # Load templates (using sim1 for demonstration)
            templates = {}
            for i in range(1, 5):
                template_name = f"template{i}"
                template_csv = Path(TEMPLATES_CSV_ROOT) / DATASET / f"{STATION}_sim1_{template_name}_0p001-0p01Hz_tpl.csv"
                if template_csv.exists():
                    df_template = pd.read_csv(template_csv)
                    templates[template_name] = df_template['x'].values  # Column is 'x' not 'tilt'

            print(f"  Loaded {len(templates)} templates: {', '.join(templates.keys())}\n")

            if len(templates) == 0:
                print(f"  ⚠️  No templates found for {STATION}\n")
                continue

            # Run hierarchical detection
            station_peaks = []

            # Compute adaptive threshold (use first template as reference)
            first_template = list(templates.values())[0]
            correlations_temp = sliding_window_cross_correlation(first_template, signal_full)
            pwave_dist_temp = pwave_distances[:len(correlations_temp)]
            adaptive_threshold = compute_adaptive_threshold(correlations_temp, pwave_dist_temp, BASE_THRESHOLD)
            print(f"  📏 Adaptive threshold: {adaptive_threshold:.3f} (base: {BASE_THRESHOLD:.3f})\n")

            # Detect peaks with hierarchical approach
            peaks = hierarchical_detection(signal_full, time_dt_full, templates, earthquake_df,
                                          STATION, pwave_distances, adaptive_threshold)

            station_peaks.extend(peaks)
            all_station_peaks.extend(peaks)

            print(f"\n  ✅ Station {STATION}: Detected {len(station_peaks)} peaks total\n")

            # Save station peaks
            if len(station_peaks) > 0:
                df_peaks = pd.DataFrame(station_peaks)
                df_peaks['time'] = pd.to_datetime(df_peaks['time'])
                peaks_csv = station_output_dir / f"{STATION}_peaks_with_quality.csv"
                df_peaks.to_csv(peaks_csv, index=False)
                print(f"  ✅ Saved peaks with quality scores: {peaks_csv.name}\n")

                # Plot quality scores
                quality_plot_path = station_output_dir / f"{STATION}_quality_scores.png"
                plot_quality_score_scatter(station_peaks, quality_plot_path, STATION)
                print(f"  ✅ Saved quality score plot: {quality_plot_path.name}\n")

        # ===== PHASE 4: Multi-Station Coherence Detection =====
        print(f"{'='*80}")
        print(f"Phase 4: Multi-station coherence detection...")
        print(f"{'='*80}\n")

        events = detect_multi_station_events(all_station_peaks, MULTI_STATION_TIME_WINDOW)

        print(f"  ✅ Detected {len(events)} multi-station events (≥{MIN_STATIONS_FOR_EVENT} stations)\n")

        if len(events) > 0:
            # Save events
            events_data = []
            for i, event in enumerate(events):
                events_data.append({
                    'event_id': i + 1,
                    'time': event['time'],
                    'n_stations': event['n_stations'],
                    'stations': '+'.join(event['stations']),
                    'coherence': event['coherence'],
                    'avg_correlation': event['avg_correlation'],
                    'avg_quality': event['avg_quality'],
                    'max_quality': event['max_quality']
                })

            df_events = pd.DataFrame(events_data)
            events_csv = dataset_output_dir / f"{DATASET}_multi_station_events.csv"
            df_events.to_csv(events_csv, index=False)
            print(f"  ✅ Saved multi-station events: {events_csv.name}\n")

            # Plot events timeline
            events_plot_path = dataset_output_dir / f"{DATASET}_multi_station_events.png"
            plot_multi_station_events(events, events_plot_path, DATASET)
            print(f"  ✅ Saved multi-station events plot: {events_plot_path.name}\n")

            # Print top 10 events
            print(f"  Top 10 multi-station events by quality:\n")
            for i, event in enumerate(events[:10]):
                print(f"    {i+1}. {event['time']}: {event['n_stations']} stations "
                      f"({'+'.join(event['stations'])}), quality={event['avg_quality']:.1f}")
            print()

    print(f"{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"   Output directory: {OUTPUT_ROOT}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    print("=== REDESIGNED SWCC STARTING ===", flush=True, file=sys.stderr)
    main()
