"""
build_clean_bandpassed_continuous.py  —  Design B: analyse-the-full-signal pipeline
==================================================================================
Instead of cutting every earthquake out of the raw signal (which shreds it into
~24-min fragments → only 28% assessable), this conditions and bandpasses the FULL
continuous record and marks contaminated regions for a post-SWCC peak veto.

Per station / component:
  1. Load the FULL raw signal (all earthquakes present); 1 Hz grid t0..t1.
  2. Split into BLOCKS only at genuine recording breaks (gap > GAP_SPLIT_S).
  3. Within each block:
       · set the few DETECTABLE earthquake windows to NaN and linearly interpolate
         them (so the big teleseismic transients can't ring the 0.001-0.01 Hz filter);
         undetectable quakes are sub-noise and left untouched;
       · interpolate any short recording gaps; despike (Hampel); detrend; bandpass
         as ONE continuous series (no per-segment edge transients).
  4. Emit columns: datetime, time_seconds, bandpassed, block_id,
       interp (sample was filled, not real) and veto (within a detectable-quake
       buffer → any SWCC peak overlapping it is rejected downstream).

Output: continuous_bandpassed/<dataset>/<station>_<dir|mag>_0p001-0p01Hz_cont_bp.feather
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, detrend as scipy_detrend, hilbert, find_peaks, welch, csd
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from numpy.fft import rfft, irfft, rfftfreq

from build_clean_bandpassed import hampel_despike
from pwave_buffer import calculate_pwave_buffers

BASE = Path("/home/owen/tilt_validation")
OUT  = BASE / "continuous_bandpassed"
EQ_CSV = BASE / "earthquakes_merged_utc.csv"

FS, BAND, ORDER = 1.0, (0.001, 0.01), 4
GAP_SPLIT_S   = 3600      # split blocks only at recording breaks longer than this
MAX_INTERP_S  = 7200      # interpolate gaps/events up to this long; longer → block split
DET_MARGIN    = 0.5       # interpolate/veto events with mag >= Mmin(d) - margin
EDGE_SETTLE_S = 1716

# station -> (dataset, raw_path, fmt, directional_col)
SRC = {
    "ECPN": ("ingv", "/home/owen/Signals/experiment/INGV/ECPN.feather", "feather", "east"),
    "EEC1": ("ingv", "/home/owen/Signals/experiment/INGV/EEC1.feather", "feather", "east"),
    "EC1":  ("experiment", "/home/owen/Signals/experiment/EC1.csv", "csv", "x"),
    "EC10": ("experiment", "/home/owen/Signals/experiment/school-data/INGV_feather/EC10.feather", "feather", "x"),
    "ECIT": ("experiment", "/home/owen/Signals/experiment/school-data/INGV_feather/ECIT.feather", "feather", "x"),
    "ECOR": ("experiment", "/home/owen/Signals/experiment/school-data/INGV_feather/ECOR.feather", "feather", "x"),
    "EMAS": ("experiment", "/home/owen/Signals/experiment/school-data/INGV_feather/EMAS.feather", "feather", "x"),
}

def make_sos():
    nyq = 0.5 * FS
    return butter(ORDER, [BAND[0]/nyq, BAND[1]/nyq], btype="bandpass", output="sos")
SOS = make_sos()


def Mmin(d):
    return 5.5 + 1.1 * np.log10(d / 1000.0)


def veto_window(mag, dist):
    """
    Magnitude-tiered exclusion window (minutes pre/post ETA). Large teleseisms shake
    the tiltmeter for hours (coda far outlasts the P-wave buffer), so they get a wide
    window; only small detectable events use the short physics buffer. Windows wider
    than MAX_INTERP_S are treated as recording-gap-like splits (never filtered through);
    narrower ones are interpolated and veto-flagged.
    """
    if mag is not None and not pd.isna(mag):
        if mag >= 7.0:
            return 12*60, 12*60      # ±12 h  (Turkey M7.9, Jan-9 M7.7 …)
        if mag >= 6.0:
            return 60, 4*60          # ±~4 h
    pre, post = calculate_pwave_buffers(dist, mag)
    return pre, post


# ── data-driven contamination veto (in-band STA/LTA, validated against the catalogue) ──
DD_STA, DD_LTA, DD_TRIG, DD_TOL = 600, 6000, 4.0, 1800   # s ; STA/LTA windows, trigger, EQ-match tol
DD_PRE, DD_POST = 600, 1800                              # s ; veto buffer around a confirmed anomaly


def data_driven_veto(x_grid, gi, eq, t0, t1):
    """Veto windows where the IN-BAND (0.001-0.01 Hz) signal is anomalous (STA/LTA trigger)
    AND a catalogued earthquake is nearby — i.e. earthquakes the detectability model misses
    but that demonstrably reach the template band. Data-only anomalies (no nearby EQ) are
    left alone (they are the real-signal candidates the search is for)."""
    bp = sosfiltfilt(SOS, np.nan_to_num(x_grid - np.nanmean(x_grid)))
    cf = np.abs(hilbert(bp)) ** 2
    ratio = uniform_filter1d(cf, DD_STA, mode="nearest") / (uniform_filter1d(cf, DD_LTA, mode="nearest") + 1e-30)
    pk, _ = find_peaks(ratio, height=DD_TRIG, distance=DD_TOL)
    etas = eq[(eq.p_wave_eta >= t0) & (eq.p_wave_eta <= t1)]["p_wave_eta"].to_numpy()
    if len(pk) == 0 or len(etas) == 0:
        return []
    iv = []
    for i in pk:
        ti = gi[i]
        if np.min(np.abs((ti - etas) / np.timedelta64(1, "s"))) <= DD_TOL:   # earthquake-coincident
            iv.append((pd.Timestamp(ti) - pd.Timedelta(seconds=DD_PRE),
                       pd.Timestamp(ti) + pd.Timedelta(seconds=DD_POST)))
    return iv


def detectable_intervals(eq, t0, t1):
    """Exclusion windows for earthquakes above the detectability threshold (margin DET_MARGIN)."""
    iv = []
    for _, q in eq.iterrows():
        eta = q["p_wave_eta"]
        if not (t0 <= eta <= t1):
            continue
        m, d = q.get("magnitude"), q.get("distance_km")
        det = (not pd.isna(m)) and (not pd.isna(d)) and d > 0 and m >= Mmin(d) - DET_MARGIN
        if not det:
            continue
        pre, post = veto_window(m, d)
        iv.append((eta - pd.Timedelta(minutes=pre), eta + pd.Timedelta(minutes=post)))
    iv.sort()
    return iv


def load_raw(station):
    ds, path, fmt, dcol = SRC[station]
    df = pd.read_csv(path) if fmt == "csv" else pd.read_feather(path)
    df = df.rename(columns={" x": "x"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    return df, ds, dcol


DENOISE_THERMAL = True   # remove the 'na' (temperature/level) coupling from the in-band tilt
COH_THR = 0.3            # only subtract the admittance where na→tilt coherence exceeds this


def estimate_admittance(tilt_bp, na_bp):
    """Complex transfer function H(f)=S_tilt,na / S_na,na (gain+phase), gated to the band and
    to frequencies where the na→tilt coherence is high. Best-practice coherent noise removal."""
    nper = int(min(len(tilt_bp)//8, 16384))
    if nper < 256:
        return None
    f, Stt = welch(tilt_bp, fs=FS, nperseg=nper)
    _, Snn = welch(na_bp,   fs=FS, nperseg=nper)
    _, Stn = csd(tilt_bp, na_bp, fs=FS, nperseg=nper)
    H = Stn / (Snn + 1e-30)
    coh = np.abs(Stn)**2 / (Stt*Snn + 1e-30)
    H[(coh < COH_THR) | (f < BAND[0]) | (f > BAND[1])] = 0.0    # only remove coherent in-band part
    return f, H


def apply_admittance(na_block, adm):
    """Predict the thermal tilt from na via H(f) (subtract this from the tilt)."""
    f, H = adm
    N = rfft(na_block); fr = rfftfreq(len(na_block), 1.0/FS)
    Hr = interp1d(f, H.real, bounds_error=False, fill_value=0.0)(fr)
    Hi = interp1d(f, H.imag, bounds_error=False, fill_value=0.0)(fr)
    return irfft(N * (Hr + 1j*Hi), n=len(na_block))


def component(df, dcol, which):
    if which == "dir":
        return df[dcol].to_numpy(float), dcol
    if which == "dir2":                                   # 2nd recorded axis, for the VECTOR filter
        c = "north" if "north" in df.columns else "y"     # INGV: east/north ; experiment: x/y
        return df[c].to_numpy(float), c
    if "mag" in df.columns:
        return df["mag"].to_numpy(float), "mag"
    return np.sqrt(df["x"].to_numpy(float)**2 + df["y"].to_numpy(float)**2), "sqrt(x^2+y^2)"


def process(station, which, eq):
    df, ds, dcol = load_raw(station)
    vals, src = component(df, dcol, which)
    dt = df["datetime"]
    t0, t1 = dt.iloc[0], dt.iloc[-1]

    # 1 Hz grid
    grid = pd.date_range(t0, t1, freq="1s")
    s = pd.Series(np.nan, index=grid, dtype=float)
    s.loc[dt.values] = vals
    real = s.notna().to_numpy()                      # where actual samples exist

    # thermal regressor: the 'na' (temperature/level) channel on the same grid
    na_grid = None
    if DENOISE_THERMAL and "na" in df.columns:
        na_s = pd.Series(np.nan, index=grid, dtype=float)
        na_s.loc[dt.values] = df["na"].to_numpy(float)
        na_grid = na_s.to_numpy()

    # detectable-quake mask on the grid
    det_iv = detectable_intervals(eq, t0, t1)
    veto = np.zeros(len(grid), bool)
    gi = grid.values
    for a, b in det_iv:
        lo = np.searchsorted(gi, np.datetime64(a)); hi = np.searchsorted(gi, np.datetime64(b), "right")
        veto[lo:hi] = True

    x = s.to_numpy()
    # data-driven veto: in-band anomalies coincident with catalogued earthquakes (model misses these)
    n_model = int(veto.sum())
    for a, b in data_driven_veto(x, gi, eq, t0, t1):
        lo = np.searchsorted(gi, np.datetime64(a)); hi = np.searchsorted(gi, np.datetime64(b), "right")
        veto[lo:hi] = True
    n_dd = int(veto.sum()) - n_model

    interp = (~real) | veto                          # samples that will be filled
    x[veto] = np.nan                                 # drop detectable-quake transients before fill

    # blocks split only at long missing stretches
    miss = ~np.isfinite(x)
    # find runs of missing longer than MAX_INTERP_S → block boundaries
    idx = np.arange(len(x))
    boundaries = []
    i = 0
    while i < len(x):
        if miss[i]:
            j = i
            while j < len(x) and miss[j]:
                j += 1
            if (j - i) > MAX_INTERP_S:
                boundaries.append((i, j))           # un-fillable gap → split
            i = j
        else:
            i += 1
    # block edges
    edges = [0]
    for a, b in boundaries:
        edges += [a, b]
    edges.append(len(x))
    blocks = [(edges[k], edges[k+1]) for k in range(0, len(edges)-1, 2)]

    out_bp = np.full(len(x), np.nan)
    na_bp = np.full(len(x), np.nan)                  # bandpassed thermal regressor
    block_id = np.full(len(x), -1, int)
    edge = np.zeros(len(x), bool)
    kept = []
    bid = 0
    for (a, b) in blocks:
        seg = x[a:b].copy()
        if np.all(np.isnan(seg)) or (b - a) < 4 * EDGE_SETTLE_S:
            continue
        seg = pd.Series(seg).interpolate(limit_direction="both").to_numpy()
        if not np.all(np.isfinite(seg)):
            continue
        seg, _ = hampel_despike(seg)
        seg = scipy_detrend(seg, type="linear")
        padlen = min(EDGE_SETTLE_S, len(seg) - 1)
        out_bp[a:b] = sosfiltfilt(SOS, seg, padtype="even", padlen=padlen)
        # condition the thermal regressor identically (subtraction happens after H is estimated)
        if na_grid is not None:
            nseg = na_grid[a:b]
            if np.isfinite(nseg).mean() > 0.9:
                nseg = pd.Series(nseg).interpolate(limit_direction="both").to_numpy()
                if np.all(np.isfinite(nseg)) and np.std(nseg) > 0:
                    na_bp[a:b] = sosfiltfilt(SOS, scipy_detrend(nseg, type="linear"),
                                             padtype="even", padlen=padlen)
        block_id[a:b] = bid
        e = min(EDGE_SETTLE_S, b - a)
        edge[a:a+e] = True; edge[b-e:b] = True       # settling zones at block boundaries
        kept.append((a, b)); bid += 1

    # ── coherent thermal correction: estimate admittance H(f) once (longest block), subtract ──
    if DENOISE_THERMAL and na_grid is not None and kept:
        a, b = max(kept, key=lambda r: r[1]-r[0])
        if np.all(np.isfinite(na_bp[a:b])):
            adm = estimate_admittance(out_bp[a:b], na_bp[a:b])
            if adm is not None:
                for (a, b) in kept:
                    if np.all(np.isfinite(na_bp[a:b])):
                        out_bp[a:b] = out_bp[a:b] - apply_admittance(na_bp[a:b], adm)

    keep = block_id >= 0
    res = pd.DataFrame({
        "datetime": grid[keep],
        "time_seconds": (grid[keep] - grid[0]).total_seconds(),
        "bandpassed": out_bp[keep],
        "block_id": block_id[keep],
        "interp": interp[keep],
        "veto": veto[keep] | edge[keep],             # veto = detectable-quake OR block-edge settling
    })
    stats = dict(src=src, ds=ds, n_grid=len(grid), n_real=int(real.sum()),
                 n_out=int(keep.sum()), n_blocks=bid, n_det=len(det_iv), dd_samples=n_dd,
                 pct_veto=100*veto[keep].sum()/max(keep.sum(),1),
                 pct_interp=100*interp[keep].sum()/max(keep.sum(),1))
    return res, stats


def main():
    eq = pd.read_csv(EQ_CSV); eq["p_wave_eta"] = pd.to_datetime(eq["p_wave_eta"])
    summary = ["CONTINUOUS BANDPASS (Design B) — summary", "="*60,
               f"band {BAND} Hz order {ORDER} | detect margin {DET_MARGIN} | "
               f"block split > {GAP_SPLIT_S}s", "="*60, ""]
    for station, (ds, *_ ) in SRC.items():
        outdir = OUT / ds; outdir.mkdir(parents=True, exist_ok=True)
        line = [f"{station} ({ds})"]
        for which in ("dir", "mag", "dir2"):
            res, st = process(station, which, eq)
            tag = which                                   # dir | mag | dir2
            base = outdir / f"{station}_{tag}_0p001-0p01Hz_cont_bp"
            res.to_feather(base.with_suffix(".feather"))
            base.with_suffix(".meta.json").write_text(json.dumps({
                "dataset": ds, "station": station, "component": which, "source_column": st["src"],
                "fs": FS, "band": list(BAND), "order": ORDER, "detect_margin": DET_MARGIN,
                "n_detectable_events": st["n_det"], "n_blocks": st["n_blocks"],
                "n_output": st["n_out"], "pct_veto": round(st["pct_veto"],2),
                "columns": ["datetime","time_seconds","bandpassed","block_id","interp","veto"],
            }, indent=2))
            cover = 100*st["n_out"]/max(st["n_real"],1)
            line.append(f"  {which}: out={st['n_out']:,} blocks={st['n_blocks']} "
                        f"coverage={cover:.0f}% of real  veto={st['pct_veto']:.1f}% "
                        f"det_events={st['n_det']}")
        print("\n".join(line)); summary += line + [""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "continuous_summary.txt").write_text("\n".join(summary))
    print(f"\nSummary → {OUT/'continuous_summary.txt'}")


if __name__ == "__main__":
    main()
