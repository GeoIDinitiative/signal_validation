"""
swcc_oldstyle_plots.py
======================
Reproduces the ORIGINAL swcc_edit.py SWCC figure look (blue correlation line,
red shaded volcanic periods, 0.2/0.5/0.7 threshold lines, cyan-circle / red-star
peak markers) for the new segment-aware pipeline.

The volcanic-overlay and standalone-plot styling are copied verbatim from
swcc_edit.py (plot_volcanic_events_on_swcc / plot_swcc_standalone), with only two
changes required by the new design:
  · the obsolete "contaminated (orange)" overlay is dropped — P-waves are excised
    upfront, so there is a single clean blue |r| line;
  · the phase-randomised null-floor line is added (the modern significance level).

One figure per (dataset, station, component, sim, template), written to
  SWCC_comprehensive/<dataset>/<station>/<sim>/<station>_<comp>_<sim>_<template>_swcc.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from swcc_comprehensive import (swcc_segment, load_clean, load_template,
                                SIMS, THRESHOLD)

# plot all four templates here (incl. the long template4 handled separately in SWCC);
# build_r_series naturally yields nothing where no segment is long enough.
TEMPLATES = ["template1", "template2", "template3", "template4"]

BASE = Path("/home/owen/tilt_validation")
SWCC = BASE / "SWCC_comprehensive"
VOLCANIC_EVENTS_CSV = BASE / "etna_volcanic_events_cleaned.csv"


# ── verbatim helpers from swcc_edit.py ───────────────────────────────────────
def load_volcanic_events(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['start_datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['end_datetime'] = pd.to_datetime(df['End Date'] + ' ' + df['End Time'])
    df['datetime'] = df['start_datetime']
    df['duration_hours'] = (df['end_datetime'] - df['start_datetime']).dt.total_seconds() / 3600
    df['is_time_window'] = df['duration_hours'] > 1.0
    return df


def get_event_type_abbreviation(event_type: str) -> str:
    abbrev_map = {
        'Vent Opening': 'VO', 'Lava Flow': 'LF', 'Weather Obscuration': 'WO',
        'Hornito Formation': 'HF', 'Hornito Activity': 'HA', 'Lava Flow Advance': 'LFA',
        'Volume Estimate': 'VE', 'Effusion Decrease': 'ED', 'Effusion Stop': 'ES',
        'Effusion Renewal': 'ER', 'Variable Effusion': 'VEf', 'Ash Emission': 'AE',
        'Degassing': 'DG', 'Incandescence': 'IN', 'Effusion Increase': 'EI',
        'Effusion Restart': 'ERs', 'Cooling Begins': 'CB', 'Effusion End': 'EE',
        'Aviation Code Change': 'ACC'}
    return abbrev_map.get(event_type, 'EFF')


def plot_volcanic_events_on_swcc(ax, events_df, time_range=None):
    """Verbatim from swcc_edit.py: red (start) / green (stop) shaded windows + lines + labels."""
    if events_df is None or len(events_df) == 0:
        return 0
    if time_range is not None:
        start_time, end_time = time_range
        if not isinstance(start_time, pd.Timestamp): start_time = pd.Timestamp(start_time)
        if not isinstance(end_time, pd.Timestamp): end_time = pd.Timestamp(end_time)
        mask = (events_df['datetime'] >= start_time) & (events_df['datetime'] <= end_time)
        events_to_plot = events_df[mask].copy()
    else:
        events_to_plot = events_df.copy()
    if len(events_to_plot) == 0:
        return 0
    ymin, ymax = ax.get_ylim(); y_range = ymax - ymin
    events_to_plot = events_to_plot.sort_values('datetime').reset_index(drop=True)
    annotation_positions = []; min_time_spacing = pd.Timedelta(hours=12)

    def find_non_overlapping_y(event_time, used_positions):
        y_levels = [0.95, 0.88, 0.81, 0.74, 0.67, 0.60]
        for y_frac in y_levels:
            y_pos = ymin + y_range * y_frac
            is_clear = True
            for used_time, used_y in used_positions:
                time_diff = abs((event_time - used_time).total_seconds())
                if time_diff < min_time_spacing.total_seconds() and abs(used_y - y_pos) < y_range * 0.05:
                    is_clear = False; break
            if is_clear:
                return y_pos
        return ymin + y_range * 0.95

    phase_colors = {'start': 'red', 'stop': 'green'}
    for idx, row in events_to_plot.iterrows():
        event_time = row['datetime']; event_type = row['Event_Type']
        phase = row.get('Phase', 'start').lower(); abbrev = get_event_type_abbreviation(event_type)
        color = phase_colors.get(phase, 'purple')
        is_time_window = row.get('is_time_window', False)
        if is_time_window and 'end_datetime' in row:
            end_time = row['end_datetime']
            ax.axvspan(event_time, end_time, color=color, alpha=0.2, zorder=1)
            ax.axvline(event_time, color=color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
            ax.axvline(end_time, color=color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
        else:
            ax.axvline(event_time, color=color, linestyle=':', linewidth=1.5, alpha=0.6, zorder=2)
        y_pos = find_non_overlapping_y(event_time, annotation_positions)
        annotation_positions.append((event_time, y_pos))
        ax.text(event_time, y_pos, abbrev, rotation=90, verticalalignment='bottom',
                horizontalalignment='right', fontsize=7, color=color, alpha=0.9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.7, linewidth=0.5))
    return len(events_to_plot)


def plot_swcc_standalone(time_dt, correlation, peak_times, peak_corrs, peak_sig,
                         template_name, sim, output_dir, dataset, station, comp,
                         null_floor, volcanic_events_df=None):
    """Old swcc_edit.py styling, verbatim, clean-only single blue |r| line + null floor."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(1, 1, figsize=(16, 5), facecolor='white')
    ax.set_facecolor('#f8f9fa')

    color_clean = '#2563eb'          # professional blue
    color_peaks_low = '#00FFFF'      # cyan
    color_peaks_high = '#ff0000'     # red

    # clean correlation line (segments separated by NaN so the line breaks at gaps)
    ax.plot(time_dt, correlation, color=color_clean, linewidth=1.2, alpha=0.95,
            label='Cleaned (P-waves removed)', zorder=10)

    # threshold lines (verbatim)
    threshold_color = '#1f2937' if dataset.lower() == 'ingv' else '#dc2626'
    ax.axhline(THRESHOLD, color=threshold_color, linestyle='--', linewidth=2.0, alpha=0.8,
               label=f'Detection Threshold (r = {THRESHOLD})', zorder=14)
    ax.axhline(0.5, color='#1e40af', linestyle='--', linewidth=2.5, alpha=0.85,
               label='High Correlation (r = 0.5)', zorder=14)
    ax.axhline(0.7, color='#7c3aed', linestyle='--', linewidth=2.5, alpha=0.85,
               label='Very High Correlation (r = 0.7)', zorder=14)
    # null floor (new, modern significance level)
    ax.axhline(null_floor, color='#16a34a', linestyle='--', linewidth=2.5, alpha=0.9,
               label=f'Null floor (r = {null_floor:.2f})', zorder=14)

    # peaks: moderate (cyan circle) vs high (red star), verbatim markers
    peak_times = np.asarray(peak_times); peak_corrs = np.asarray(peak_corrs)
    if len(peak_corrs) > 0:
        mask_low = (peak_corrs >= THRESHOLD) & (peak_corrs < 0.5)
        mask_high = peak_corrs >= 0.5
        if np.any(mask_low):
            ax.scatter(peak_times[mask_low], peak_corrs[mask_low], s=40, c=color_peaks_low,
                       marker='o', zorder=12, edgecolors='black', linewidths=1.5, alpha=1.0,
                       label=f'Clean Peaks: Moderate (r = 0.2–0.5, n={int(np.sum(mask_low))})')
        if np.any(mask_high):
            ax.scatter(peak_times[mask_high], peak_corrs[mask_high], s=150, c=color_peaks_high,
                       marker='*', zorder=15, edgecolors='#8b0000', linewidths=1.2, alpha=1.0,
                       label=f'Clean Peaks: High (r > 0.5, n={int(np.sum(mask_high))})')

    # valid (non-gap) time extent, used for volcanic range and x-limits
    valid = pd.Series(time_dt).dropna()

    # volcanic overlay (INGV only) — verbatim
    num_volcanic_events = 0
    if dataset.lower() == "ingv" and volcanic_events_df is not None and len(volcanic_events_df) > 0:
        tr = (pd.Timestamp(valid.iloc[0]), pd.Timestamp(valid.iloc[-1]))
        num_volcanic_events = plot_volcanic_events_on_swcc(ax, volcanic_events_df, tr)

    ax.set_ylabel('Correlation Coefficient |r|', fontsize=13, fontweight='600', color='#1f2937')
    ax.set_xlabel('Time (UTC)', fontsize=13, fontweight='600', color='#1f2937')
    title_parts = [f'{dataset.upper()} – Station {station} ({comp})',
                   f'{sim.upper().replace("SIM", "Simulation ")} – {template_name.replace("template", "Template ")}']
    if num_volcanic_events > 0:
        title_parts.append(f'{num_volcanic_events} Volcanic Events')
    ax.set_title(' | '.join(title_parts), fontsize=14, fontweight='700', pad=15, color='#111827')
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='#9ca3af'); ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=8.5, ncol=2, frameon=True, fancybox=True,
              shadow=True, framealpha=0.95, edgecolor='#d1d5db')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(valid.iloc[0], valid.iloc[-1])
    ax.tick_params(axis='both', labelsize=10, colors='#374151')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{station}_{comp}_{sim}_{template_name}_swcc.png"
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(); plt.style.use('default')


# ── driver ────────────────────────────────────────────────────────────────────
def build_r_series(clean_df, tpl):
    """Full |r|(t) across segments, NaN-separated so the line breaks at gaps."""
    times, rs = [], []
    for _, seg in clean_df.groupby("segment_id"):
        sig = seg["bandpassed"].to_numpy(float)
        if len(sig) < len(tpl):
            continue
        r = np.abs(swcc_segment(tpl, sig))
        dt = seg["datetime"].to_numpy()[:len(r)]
        times.append(dt); rs.append(r)
        times.append(np.array([dt[-1] + np.timedelta64(1, "s")])); rs.append(np.array([np.nan]))
    if not times:
        return None, None
    return np.concatenate(times), np.concatenate(rs)


STATIONS = {"ingv": ["ECPN", "EEC1"], "experiment": ["EC1", "EC10", "ECIT", "ECOR", "EMAS"]}
COMPONENTS = ["dir", "mag"]


def main():
    volc = load_volcanic_events(VOLCANIC_EVENTS_CSV)
    peaks = pd.read_csv(SWCC / "all_peaks_flagged.csv", parse_dates=["peak_time"])
    floors = peaks.groupby("dataset")["null_floor"].first().to_dict()
    n = 0
    for dataset, stations in STATIONS.items():
        for station in stations:
            for comp in COMPONENTS:
                clean = load_clean(dataset, station, comp)
                if clean is None:
                    continue
                pk_sc = peaks[(peaks.dataset == dataset) & (peaks.station == station)
                              & (peaks.component == comp)]
                for sim in SIMS:
                    for tname in TEMPLATES:
                        tpl = load_template(dataset, station, sim, tname)
                        if tpl is None:
                            continue
                        t, r = build_r_series(clean, tpl)
                        if t is None:
                            continue
                        pk = pk_sc[(pk_sc.sim == sim) & (pk_sc.template == tname)]
                        out_dir = SWCC / dataset / station / sim
                        plot_swcc_standalone(
                            t, r, pk["peak_time"].to_numpy(), pk["abs_r"].to_numpy(),
                            pk["significant"].to_numpy(), tname, sim, out_dir,
                            dataset, station, comp, floors.get(dataset, 0.6),
                            volcanic_events_df=volc if dataset == "ingv" else None)
                        n += 1
                print(f"  {dataset}/{station}/{comp}: 16 templates plotted")
    print(f"\nTotal old-style SWCC figures: {n}")


if __name__ == "__main__":
    main()
