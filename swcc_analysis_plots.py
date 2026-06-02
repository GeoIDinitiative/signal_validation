"""
swcc_analysis_plots.py  —  stage 7: clean-peak analysis plots (Design B)
=======================================================================
Consumes the per-detection list written by swcc_continuous.py
(SWCC_comprehensive/continuous/all_detections_continuous.csv) and regenerates the
analysis plots that the SWCC per-template figures don't cover:

  · detection_overview_<dataset>.png   score vs time per station (MAX & STACK),
        detection/significance floors, significant points highlighted, volcanic
        overlay for INGV
  · detections_by_station.png          bar chart: detections vs significant per station/method
  · candidates/<...>.png               ULP-AWARE shortlist: raw broadband + bandpassed + window
        PSD (+ Morlet CWT) for the detections that pass glitch-rejection gates (veto, above-band
        leakage, impulsivity) ranked by a combined ULP figure of merit — NOT score-only
  · ulp_candidate_ranking.csv          every detection scored on r/SNR/leakage/kurtosis/econc/veto

Output: SWCC_comprehensive/analysis/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_labels import slab, slabs, STATION_LABEL
from pathlib import Path
from scipy.signal import spectrogram, butter, sosfiltfilt, welch
from scipy.stats import kurtosis as excess_kurt

from swcc_oldstyle_plots import load_volcanic_events, plot_volcanic_events_on_swcc
from swcc_comprehensive import load_template, SIMS
from swcc_accumulated import TEMPLATES

try:
    import pywt
    HAVE_PYWT = True
except Exception:
    HAVE_PYWT = False

import phd_env                                          # branch-aware OUT / detections / components
warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
CONT = BASE / "continuous_bandpassed"
SWCC = phd_env.out(BASE / "SWCC_comprehensive")
OUT  = SWCC / "analysis"
DETS = phd_env.dets_dir() / "all_detections_continuous.csv"
VOLC = load_volcanic_events(BASE / "etna_volcanic_events_cleaned.csv")
PERIODS = {"ingv": ("2022-11-14", "2023-03-01"), "experiment": ("2023-07-23", "2023-08-03")}
COL = {"max": "#2563eb", "stack": "#f59e0b"}


def load_cont(ds, st, comp):
    f = CONT / ds / f"{st}_{comp}_0p001-0p01Hz_cont_bp.feather"
    if not f.exists():
        return None
    d = pd.read_feather(f); d["datetime"] = pd.to_datetime(d["datetime"])
    return d


# ── 1. per-dataset detection overview ─────────────────────────────────────────
def overview(dets):
    for ds, g0 in dets.groupby("dataset"):
        stations = sorted(g0.station.unique())
        fig, axes = plt.subplots(len(stations), 1, figsize=(14, 2.4*len(stations)),
                                 squeeze=False, sharex=True)
        for ax, st in zip(axes[:, 0], stations):
            gst = g0[g0.station == st]
            if ds == "ingv":
                plot_volcanic_events_on_swcc(ax, VOLC, (pd.Timestamp(PERIODS[ds][0]), pd.Timestamp(PERIODS[ds][1])))
            for method in ["max", "stack"]:
                gm = gst[gst.method == method]
                if gm.empty:
                    continue
                ax.scatter(gm.peak_time, gm.score, s=14, c=COL[method], alpha=0.5, label=f"{method} detect")
                sig = gm[gm.significant]
                if len(sig):
                    ax.scatter(sig.peak_time, sig.score, s=70, marker="*", c=COL[method],
                               edgecolors="k", linewidths=0.6, label=f"{method} above-floor (screening)")
                ax.axhline(gm.floor_detect.iloc[0], ls="--", c=COL[method], alpha=0.5, lw=1)
                ax.axhline(gm.floor_signif.iloc[0], ls="-", c=COL[method], alpha=0.7, lw=1.2)
            ax.set_ylabel(f"{slab(st)}\nscore"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2, loc="upper right")
            ax.set_xlim(pd.Timestamp(PERIODS[ds][0]), pd.Timestamp(PERIODS[ds][1]))
        axes[0, 0].set_title(f"{ds}: detections over time (dashed=detect floor, solid=significance floor"
                             + ("; green=volcanic events)" if ds == "ingv" else ")"))
        fig.tight_layout(); fig.savefig(OUT / f"detection_overview_{ds}.png", dpi=300); plt.close(fig)
    print(f"  overviews → {OUT}/detection_overview_*.png")


# ── 2. detections-by-station bar chart ────────────────────────────────────────
def by_station(dets):
    g = (dets.groupby(["dataset", "station", "method"])
         .agg(detect=("significant", "size"), signif=("significant", "sum")).reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ds in zip(axes, ["ingv", "experiment"]):
        sub = g[g.dataset == ds]
        stations = sorted(sub.station.unique())
        x = np.arange(len(stations)); w = 0.2
        for j, (method, hatch) in enumerate([("max", None), ("stack", None)]):
            sm = sub[sub.method == method].set_index("station").reindex(stations).fillna(0)
            ax.bar(x + (j-0.5)*w*2, sm.detect, w*2, color=COL[method], alpha=0.4, label=f"{method} detect")
            ax.bar(x + (j-0.5)*w*2, sm.signif, w*2, color=COL[method], label=f"{method} above-floor (screening)")
        ax.set_xticks(x); ax.set_xticklabels(slabs(stations), rotation=30, ha="right")
        ax.set_ylabel("count"); ax.set_title(f"{ds}: detections vs above-floor (screening) by station")
        ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(top=max(1, sub.detect.max())*1.3)
    fig.tight_layout(); fig.savefig(OUT / "detections_by_station.png", dpi=300); plt.close(fig)
    print(f"  bar chart → {OUT}/detections_by_station.png")


# ── 3. ULP-aware candidate selection ──────────────────────────────────────────
# A magma-mixing ULP is a smooth, sustained, IN-BAND tilt transient — not an impulsive P-wave
# step or instrument glitch. Ranking detections by SWCC score ALONE surfaces glitches (which can
# correlate r≈0.7 with a template) and ignores the contamination veto. So we score each detection
# on the WHOLE workflow and reject glitch morphology before ranking:
#   r       SWCC score                                            [reward]
#   SNR     peak|bandpassed| / clean in-band noise RMS            [reward]
#   leakage RMS(0.01–0.1 Hz)/RMS(0.001–0.01 Hz) in the RAW window [penalise — glitch leaks above band]
#   kurt    excess kurtosis of the bandpassed window              [penalise — glitch is spiky]
#   econc   energy in ±2 min / energy in ±15 min                  [penalise — glitch dumps energy at one instant]
#   veto    contaminated/earthquake window?                       [hard reject]
ULP_PANELS = 12                       # ULP-consistent panels per dataset
G_LEAK, G_KURT, G_ECONC = 1.0, 4.0, 0.45
W, CORE = pd.Timedelta(minutes=15), pd.Timedelta(minutes=2)
SOS_IN = butter(4, [0.001/0.5, 0.01/0.5], btype="bandpass", output="sos")
SOS_HI = butter(4, [0.01/0.5, 0.1/0.5],  btype="bandpass", output="sos")
RAW_SRC = {
    "ECPN": ("east", "/home/owen/Signals/experiment/INGV/ECPN.feather", "f"),
    "EEC1": ("east", "/home/owen/Signals/experiment/INGV/EEC1.feather", "f"),
    "EC1":  ("x", "/home/owen/Signals/experiment/EC1.csv", "c"),
    "EC10": ("x", "/home/owen/Signals/experiment/school-data/INGV_feather/EC10.feather", "f"),
    "ECIT": ("x", "/home/owen/Signals/experiment/school-data/INGV_feather/ECIT.feather", "f"),
    "ECOR": ("x", "/home/owen/Signals/experiment/school-data/INGV_feather/ECOR.feather", "f"),
    "EMAS": ("x", "/home/owen/Signals/experiment/school-data/INGV_feather/EMAS.feather", "f"),
}
SIM_DIR = Path("/home/owen/Signal_Validation/solid_dofs/tilt")
DOF_DIR = BASE / "tilt_templates_dofs"          # full-resolution DOF templates (proj_x / √(px²+py²))
DOF_TEMPLATES = ["template1", "template2", "template3", "template4"]
_cont_idx, _raw_cache, _noise, _simraw, _dof = {}, {}, {}, {}, {}


def _load_dof(ds, st, sim, tn):
    k = (ds, st, sim, tn)
    if k not in _dof:
        f = DOF_DIR / ds / f"{st}_{sim}_{tn}_dof.csv"
        _dof[k] = pd.read_csv(f) if f.exists() else None
    return _dof[k]


def _tstation(st):
    return "EC1" if st == "EEC1" else st        # template/sim station mapping (EEC1 site = sim node EC1)


def _raw_sim_bp(st, sim):
    """The RAW simulated tilt for a station/sim and its band-passed version (for offset matching)."""
    k = (st, sim)
    if k not in _simraw:
        f = SIM_DIR / sim / "tilt" / f"{_tstation(st)}.txt"
        if not f.exists():
            _simraw[k] = (None, None)
        else:
            a = np.loadtxt(f); v = a[:, 1] if a.ndim == 2 else np.ravel(a)
            _simraw[k] = (v, sosfiltfilt(SOS_IN, v - v.mean()))
    return _simraw[k]


def _match_offset(rb, tpl):
    """Offset of the template inside the band-passed raw sim (templates are windows of the sim)."""
    Ln = len(tpl); best = (0.0, 0)
    step = max(1, (len(rb) - Ln) // 400) if len(rb) > Ln else 1
    for off in range(0, max(1, len(rb) - Ln + 1), step):
        w = rb[off:off+Ln]
        if w.std() < 1e-30:
            continue
        r = float(np.corrcoef(tpl, w)[0, 1])
        if abs(r) > abs(best[0]):
            best = (r, off)
    return best[1]


def _cont(ds, st, comp):
    k = (ds, st, comp)
    if k not in _cont_idx:
        d = load_cont(ds, st, comp)
        _cont_idx[k] = d.set_index("datetime") if d is not None else None
    return _cont_idx[k]


def _noise_rms(ds, st, comp):
    k = (ds, st, comp)
    if k not in _noise:
        d = _cont(ds, st, comp)
        m = (~d.veto.astype(bool)) & (~d.interp.astype(bool)) & np.isfinite(d.bandpassed)
        _noise[k] = float(np.sqrt(np.mean(d.bandpassed[m]**2)))
    return _noise[k]


def _raw_trace(st, comp):
    k = (st, comp)
    if k not in _raw_cache:
        col, path, fmt = RAW_SRC[st]
        d = pd.read_csv(path) if fmt == "c" else pd.read_feather(path)
        d.columns = [c.strip() for c in d.columns]
        if comp == "mag":
            v = (pd.to_numeric(d["mag"], errors="coerce").to_numpy() if "mag" in d.columns
                 else np.hypot(pd.to_numeric(d["x"], errors="coerce"), pd.to_numeric(d["y"], errors="coerce")))
        elif comp == "dir2":                              # Y axis: north (INGV) / y (experiment)
            ycol = "north" if "north" in d.columns else "y"
            v = pd.to_numeric(d[ycol], errors="coerce").to_numpy()
        else:
            v = pd.to_numeric(d[col], errors="coerce").to_numpy()
        _raw_cache[k] = pd.Series(v, index=pd.to_datetime(d["datetime"])).sort_index()
    return _raw_cache[k]


def _indicators(p):
    d = _cont(p.dataset, p.station, p.component)
    if d is None:
        return None
    t0 = pd.Timestamp(p.peak_time)
    bp = d.loc[t0-W:t0+W].bandpassed.to_numpy(); bp = bp[np.isfinite(bp)]
    if len(bp) < 200:
        return None
    core = d.loc[t0-CORE:t0+CORE].bandpassed.to_numpy(); core = core[np.isfinite(core)]
    rs = _raw_trace(p.station, p.component).loc[t0-W:t0+W].to_numpy(); rs = rs[np.isfinite(rs)]
    leak = (float(np.std(sosfiltfilt(SOS_HI, rs-rs.mean()))/(np.std(sosfiltfilt(SOS_IN, rs-rs.mean()))+1e-30))
            if len(rs) > 100 else np.nan)
    return dict(snr=float(np.max(np.abs(bp))/(_noise_rms(p.dataset, p.station, p.component)+1e-30)),
                kurt=float(excess_kurt(bp)),
                econc=float(np.sum(core**2)/(np.sum(bp**2)+1e-30)),
                leakage=leak,
                vetoed=bool(d.loc[t0-CORE:t0+CORE].veto.astype(bool).any()))


def _best_template(p):
    """Find the bank template that best matches the REAL signal in the MATCHED window
    [peak_time, peak_time+template_len] (peak_time = SWCC window-start). Returns (r, sim, tname, tpl, L)."""
    d = _cont(p.dataset, p.station, p.component); t0 = pd.Timestamp(p.peak_time)
    comp = {"dir": "dir", "mag": "mag", "dir2": "ortho"}.get(p.component, "dir")   # DOF column prefix
    best = None
    for sim in SIMS:
        for tn in DOF_TEMPLATES:
            dof = _load_dof(p.dataset, p.station, sim, tn)       # full-res DOF template, component-matched
            if dof is None:
                continue
            tpl = dof[f"{comp}_bp"].to_numpy(); raw = dof[f"{comp}_raw"].to_numpy()
            Ln = len(tpl)
            win = d.loc[t0:t0+pd.Timedelta(seconds=Ln-1)].bandpassed.to_numpy()
            n = min(len(win), Ln)
            a, b = win[:n], tpl[:n]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 0.8*n or a[m].std() < 1e-30:
                continue
            r = float(np.corrcoef(a[m], b[m])[0, 1])
            if best is None or abs(r) > abs(best[0]):
                best = (r, sim, tn, tpl, raw, Ln)
    return best


def _spectro(ax, sig):
    bb = np.nan_to_num(sig - np.nanmean(sig))
    f, ts, Sxx = spectrogram(bb, fs=1.0, nperseg=min(256, max(16, len(bb)//4)))
    ax.pcolormesh(ts/60, f, 10*np.log10(Sxx+1e-20), shading="gouraud", cmap="viridis")
    ax.axhline(0.001, color="w", ls=":", lw=0.6); ax.axhline(0.01, color="w", ls=":", lw=0.6)
    ax.set_ylim(0, 0.02)


def _morlet(ax, sig):
    bb = np.nan_to_num(sig - np.nanmean(sig))
    freqs = np.logspace(np.log10(1e-4), np.log10(2e-2), 80)
    coef, _ = pywt.cwt(bb, pywt.central_frequency("morl")/freqs, "morl", sampling_period=1.0)
    ax.pcolormesh(np.arange(len(bb))/60, freqs, np.abs(coef)**2, shading="gouraud", cmap="magma")
    ax.set_ylim(1e-3, 1e-2)


def _panel(p, cdir, i):
    """4×2 mega-panel: col1 = REAL candidate, col2 = the TEMPLATE that detected it; rows =
    original / denoised (0.001–0.01 Hz) / spectrogram / Morlet CWT. Anchored on the matched window."""
    bt = _best_template(p)
    if bt is None:
        return False
    r, sim, tn, tpl, tpl_raw, L = bt                       # tpl = DOF band-passed, tpl_raw = full-res DOF raw
    d = _cont(p.dataset, p.station, p.component); t0 = pd.Timestamp(p.peak_time)
    real_raw = _raw_trace(p.station, p.component).loc[t0:t0+pd.Timedelta(seconds=L-1)].to_numpy()
    real_bp = d.loc[t0:t0+pd.Timedelta(seconds=L-1)].bandpassed.to_numpy()
    nrow = 4 if HAVE_PYWT else 3
    fig, ax = plt.subplots(nrow, 2, figsize=(13, 2.5*nrow))
    cols = [(f"REAL  {p.station}/{p.component} ({p.method})", real_raw, real_bp, COL[p.method]),
            (f"TEMPLATE  {sim}/{tn}", tpl_raw, tpl, "#b45309")]
    for c, (name, raw, bp, color) in enumerate(cols):
        if raw is not None and len(raw):
            ax[0, c].plot(np.arange(len(raw))/60, np.nan_to_num(raw - np.nanmean(raw)), lw=0.7, color=color)
        else:
            ax[0, c].text(0.5, 0.5, "raw n/a", ha="center", transform=ax[0, c].transAxes)
        ax[0, c].set_title(name, fontsize=10); ax[0, c].grid(alpha=0.3)
        ax[1, c].plot(np.arange(len(bp))/60, bp, lw=1.0, color=color); ax[1, c].grid(alpha=0.3)
        _spectro(ax[2, c], bp)
        if HAVE_PYWT:
            _morlet(ax[3, c], bp)
            ax[3, c].set_xlabel("minutes from window start")
        else:
            ax[2, c].set_xlabel("minutes from window start")
    for rr, lab in enumerate(["original (raw)", "denoised\n0.001–0.01 Hz", "spectrogram\nfreq (Hz)",
                              "Morlet CWT\nfreq (Hz)"][:nrow]):
        ax[rr, 0].set_ylabel(lab)
    fig.suptitle(f"{p.dataset}/{p.station}/{p.component} · {p.method} · candidate {t0:%Y-%m-%d %H:%M}  "
                 f"vs best template {sim}/{tn}  (r={r:+.2f}, SNR={p.snr:.1f}, {L/60:.0f} min window)",
                 fontweight="600")
    fig.tight_layout()
    fig.savefig(cdir / f"{p.dataset}_{i:02d}_{p.station}_{p.component}_{p.method}_{t0:%Y%m%d_%H%M}.png", dpi=300)
    plt.close(fig); return True


def candidates(dets):
    cdir = OUT / "candidates"; cdir.mkdir(parents=True, exist_ok=True)
    for old in cdir.glob("*.png"):     # clear stale (score-only) panels
        old.unlink()
    rows = []
    for p in dets.itertuples():
        ind = _indicators(p)
        if ind is None:
            continue
        rows.append(dict(dataset=p.dataset, station=p.station, component=p.component, method=p.method,
                         peak_time=p.peak_time, score=p.score, significant=bool(p.significant), **ind))
    df = pd.DataFrame(rows)
    if df.empty:
        print("  no candidates with computable indicators."); return

    def z(c):
        v = df[c]; return (v - v.mean())/(v.std()+1e-30)
    df["ulp_score"] = z("score") + z("snr") - z("kurt") - z("econc") - z("leakage").fillna(0)
    df["ulp_pass"] = ((~df["vetoed"]) & (df["leakage"] < G_LEAK) & (df["kurt"] < G_KURT) & (df["econc"] < G_ECONC))
    df.sort_values("ulp_score", ascending=False).to_csv(OUT / "ulp_candidate_ranking.csv", index=False)

    n = 0
    for ds in df.dataset.unique():
        sel = (df[(df.dataset == ds) & df.ulp_pass].sort_values("ulp_score", ascending=False).head(ULP_PANELS))
        for i, p in enumerate(sel.itertuples()):
            n += _panel(p, cdir, i)
    n_pass = int(df.ulp_pass.sum())
    n_glitch_in_old = int(df[df.significant].sort_values("score", ascending=False)
                          .groupby("dataset").head(CAP_OLD := 20).pipe(lambda x: (~x.ulp_pass).sum()))
    print(f"  ULP-aware candidates: {n} panels (top {ULP_PANELS}/dataset of {n_pass}/{len(df)} that pass glitch gates)")
    print(f"  (of the former score-only top-20/dataset, {n_glitch_in_old} were glitch-like and are now rejected)")
    print(f"  ranking + indicators → {OUT}/ulp_candidate_ranking.csv")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not DETS.exists():
        print(f"  {DETS} not found — run swcc_continuous.py (stage 3) first."); return
    dets = pd.read_csv(DETS, parse_dates=["peak_time"])
    if "component" in dets.columns:          # real components only (branch-aware); vec → vector_orientation.py
        dets = dets[dets.component.isin(phd_env.components(["dir", "mag"]))].reset_index(drop=True)
    if dets.empty:
        print("  no detections to plot."); return
    OUT.mkdir(parents=True, exist_ok=True)
    overview(dets)
    by_station(dets)
    candidates(dets)
    n_sig = int(dets.significant.sum())
    print(f"\nclean-peak analysis: {len(dets)} detections, {n_sig} significant → {OUT}")


if __name__ == "__main__":
    main()
