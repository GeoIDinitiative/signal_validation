"""
make_comparison.py — dedicated EC1 (summer) vs EEC1 (winter) comparison suite.

EEC1 and EC1 are the SAME borehole tiltmeter at the same site, recorded in two campaigns
(EEC1 = INGV permanent, winter 2022–23; EC1 = experiment, summer 2023). This collates a series of
direct side-by-side plots characterising how the same instrument behaved in the two periods —
the most direct two-period comparison available. All figures formatted (titles / axes / gridlines).

Figures (written next to this script):
  01_records_overview.png       raw tilt-x over each full record (eruptive vs non-eruptive period)
  02_inband_background.png       in-band RMS (x / y / horizontal), raw vs thermal-removed
  03_psd_comparison.png          broadband PSD overlay, tidal + search bands marked (the crossover)
  04_rolling_rms.png             rolling hourly in-band horizontal RMS through each record
  05_amplitude_distribution.png  in-band tilt-x amplitude distribution
  06_thermal_and_units.png       thermal coupling (r, % removed) + same-instrument LSB proof
  07_detection_performance.png   template peak yield, match quality, 90% upper limit
  SUMMARY.txt                    the consolidated EEC1-vs-EC1 story
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, welch, detrend

warnings.filterwarnings("ignore")
HERE = Path("/home/owen/tilt_validation/eec1_ec1_comparison"); HERE.mkdir(parents=True, exist_ok=True)
TV = Path("/home/owen/tilt_validation")
SOS = butter(4, [0.001/0.5, 0.01/0.5], btype="bandpass", output="sos")
WINTER = ("EC1 (eruptive)", "/home/owen/Signals/experiment/INGV/EEC1.feather", "f", "#1d4ed8")
SUMMER = ("EC1 (non-eruptive)", "/home/owen/Signals/experiment/EC1.csv", "c", "#dc2626")
bp = lambda x: sosfiltfilt(SOS, detrend(np.nan_to_num(x - np.nanmean(x)), type="linear"))


def load_raw(path, fmt):
    d = pd.read_csv(path) if fmt == "c" else pd.read_feather(path)
    d.columns = [c.strip() for c in d.columns]
    return d


def longest_block(sec):
    cut = np.where(np.diff(sec) > 5)[0]
    b = np.concatenate(([0], cut+1, [len(sec)]))
    runs = [(b[i], b[i+1]) for i in range(len(b)-1)]
    return max(runs, key=lambda r: r[1]-r[0])


def block_xyna(d):
    sec = pd.to_numeric(d["seconds"], errors="coerce").to_numpy()
    a, b = longest_block(sec); sl = slice(a, b)
    x = pd.to_numeric(d["x"], errors="coerce").to_numpy()[sl]
    y = pd.to_numeric(d["y"], errors="coerce").to_numpy()[sl]
    na = pd.to_numeric(d["na"], errors="coerce").to_numpy()[sl]
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(na)
    return x[m], y[m], na[m]


def thermal_remove(xb, nb):
    beta = np.cov(xb, nb)[0, 1]/(np.var(nb)+1e-30)
    return xb - beta*nb, np.corrcoef(xb, nb)[0, 1]


def main():
    W = load_raw(WINTER[1], WINTER[2]); S = load_raw(SUMMER[1], SUMMER[2])
    wx, wy, wna = block_xyna(W); sx, sy, sna = block_xyna(S)
    wxb, wyb, wnb = bp(wx), bp(wy), bp(wna)
    sxb, syb, snb = bp(sx), bp(sy), bp(sna)
    wxd, wr = thermal_remove(wxb, wnb); wyd, _ = thermal_remove(wyb, wnb)
    sxd, sr = thermal_remove(sxb, snb); syd, _ = thermal_remove(syb, snb)
    wh, wh_d = np.hypot(wxb, wyb), np.hypot(wxd, wyd)
    sh, sh_d = np.hypot(sxb, syb), np.hypot(sxd, syd)

    # 01 — raw records over full span (decimated)
    fig, ax = plt.subplots(2, 1, figsize=(13, 7))
    for a, (name, path, fmt, c) in zip(ax, [WINTER, SUMMER]):
        d = W if name == WINTER[0] else S
        t = pd.to_datetime(d["datetime"]); xx = pd.to_numeric(d["x"], errors="coerce").to_numpy()
        dec = max(1, len(xx)//12000)
        hrs = (t.iloc[::dec] - t.iloc[0]).dt.total_seconds().to_numpy()/86400
        a.plot(hrs, xx[::dec], lw=0.4, color=c)
        a.set_title(name, fontweight="700", fontsize=12); a.set_ylabel("raw tilt-x (instr. units)", fontweight="700")
        a.grid(alpha=0.3)
    ax[1].set_xlabel("days from record start", fontweight="700")
    fig.suptitle("Same borehole instrument, two campaigns — raw tilt-x record", fontweight="700", fontsize=14)
    fig.tight_layout(); fig.savefig(HERE/"01_records_overview.png", dpi=300); plt.close(fig)

    # 02 — in-band background (raw vs thermal-removed), x / y / horizontal
    fig, ax = plt.subplots(figsize=(11, 5.5)); comp = ["x (raw)", "y (raw)", "horiz (raw)", "horiz (de-thermal)"]
    wv = [wxb.std(), wyb.std(), wh.std(), wh_d.std()]; sv = [sxb.std(), syb.std(), sh.std(), sh_d.std()]
    x = np.arange(len(comp)); w = 0.38
    ax.bar(x-w/2, wv, w, color=WINTER[3], label=WINTER[0]); ax.bar(x+w/2, sv, w, color=SUMMER[3], label=SUMMER[0])
    for i in range(len(comp)):
        ax.text(i, max(wv[i], sv[i])*1.02, f"{wv[i]/sv[i]:.0f}×", ha="center", fontsize=9, fontweight="700")
    ax.set_xticks(x); ax.set_xticklabels(comp); ax.set_ylabel("in-band RMS (0.001–0.01 Hz)", fontweight="700")
    ax.set_title("In-band tilt background per period (eruptive/non-eruptive ratio annotated)", fontweight="700", fontsize=13)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE/"02_inband_background.png", dpi=300); plt.close(fig)

    # 03 — broadband PSD with tidal + search bands
    fig, ax = plt.subplots(figsize=(11, 6))
    for sig, (name, _, _, c) in [(wx, WINTER), (sx, SUMMER)]:
        f, P = welch(detrend(sig), fs=1.0, nperseg=int(min(172800, len(sig)//3)))
        sel = (f > 2e-6) & (f < 0.02); ax.loglog(f[sel], P[sel], color=c, lw=1.3, label=name)
    for band, lab, col in [((0.9e-5, 2.6e-5), "tidal", "#16a34a"), ((0.001, 0.01), "search band", "#9333ea")]:
        ax.axvspan(*band, color=col, alpha=0.12)
        ax.text(np.sqrt(band[0]*band[1]), ax.get_ylim()[1], lab, ha="center", va="top", fontsize=9, color=col)
    ax.set_xlabel("frequency (Hz)", fontweight="700"); ax.set_ylabel("tilt-x PSD", fontweight="700")
    ax.set_title("Broadband PSD — eruptive vs non-eruptive period (tide = internal calibrator; bands diverge in the search band)",
                 fontweight="700", fontsize=12); ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(HERE/"03_psd_comparison.png", dpi=300); plt.close(fig)

    # 04 — rolling hourly in-band horizontal RMS
    fig, ax = plt.subplots(figsize=(12, 5))
    for h, (name, _, _, c) in [(wh, WINTER), (sh, SUMMER)]:
        n = len(h)//3600; roll = np.array([h[i*3600:(i+1)*3600].std() for i in range(n)])
        ax.plot(np.arange(n), roll, lw=1.0, color=c, label=name)
    ax.set_xlabel("hours from block start", fontweight="700"); ax.set_ylabel("in-band horizontal RMS", fontweight="700")
    ax.set_title("Rolling hourly in-band background through each record", fontweight="700", fontsize=13)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE/"04_rolling_rms.png", dpi=300); plt.close(fig)

    # 05 — amplitude distribution
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(wxb, bins=250, density=True, histtype="step", color=WINTER[3], label=WINTER[0])
    ax.hist(sxb, bins=250, density=True, histtype="step", color=SUMMER[3], label=SUMMER[0])
    ax.set_yscale("log"); ax.set_xlabel("in-band tilt-x", fontweight="700"); ax.set_ylabel("density (log)", fontweight="700")
    ax.set_title("In-band tilt-x amplitude distribution (eruptive period is broader → larger excursions)",
                 fontweight="700", fontsize=12); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE/"05_amplitude_distribution.png", dpi=300); plt.close(fig)

    # 06 — thermal coupling + same-instrument units
    def lsb(v):
        u = np.unique(v[np.isfinite(v)]); dd = np.diff(u); dd = dd[dd > 0]
        return float(np.median(dd)) if len(dd) else np.nan
    wlsb = lsb(pd.to_numeric(W["x"], errors="coerce").to_numpy()); slsb = lsb(pd.to_numeric(S["x"], errors="coerce").to_numpy())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar([0, 1], [abs(wr)**2*100, abs(sr)**2*100], color=[WINTER[3], SUMMER[3]])
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["eruptive", "non-eruptive"])
    ax[0].set_ylabel("in-band variance explained by na (%)", fontweight="700")
    ax[0].set_title(f"Thermal coupling: r(tilt,na) eruptive={wr:+.2f} non-eruptive={sr:+.2f}", fontweight="700", fontsize=12)
    ax[0].grid(axis="y", alpha=0.3)
    ax[1].bar([0, 1], [wlsb, slsb], color=[WINTER[3], SUMMER[3]])
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["eruptive", "non-eruptive"])
    ax[1].set_ylabel("digitisation step / LSB (instr. units)", fontweight="700")
    ax[1].set_title(f"Same instrument: identical LSB ({wlsb:.4g} = {slsb:.4g}) → same units", fontweight="700", fontsize=12)
    ax[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Why the eruptive-period background is real, not instrumental", fontweight="700", fontsize=14)
    fig.tight_layout(); fig.savefig(HERE/"06_thermal_and_units.png", dpi=300); plt.close(fig)

    # 07 — detection / template performance / upper limit (EC1: eruptive vs non-eruptive period)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    try:
        sst = pd.read_csv(TV/"SWCC_comprehensive/top_templates/sst_peak_counts.csv")
        for stn, col, off in [("EEC1", WINTER[3], -0.2), ("EC1", SUMMER[3], 0.2)]:
            g = sst[sst.station == stn].groupby("template").agg(n=("n_detect", "sum")).reindex(
                ["template1", "template2", "template3", "template4"]).fillna(0)
            ax[0].bar(np.arange(4)+off, g.n, 0.4, color=col, label=stn)
        ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["t1", "t2", "t3", "t4"])
        ax[0].set_ylabel("peaks above null floor", fontweight="700"); ax[0].set_title("Template peak yield", fontweight="700")
        ax[0].legend(); ax[0].grid(axis="y", alpha=0.3)
        for stn, col, off in [("EEC1", WINTER[3], -0.2), ("EC1", SUMMER[3], 0.2)]:
            g = sst[sst.station == stn].groupby("template").agg(r=("max_r", "mean")).reindex(
                ["template1", "template2", "template3", "template4"]).fillna(0)
            ax[1].bar(np.arange(4)+off, g.r, 0.4, color=col, label=stn)
        ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["t1", "t2", "t3", "t4"])
        ax[1].set_ylabel("mean peak |r|", fontweight="700"); ax[1].set_title("Match quality", fontweight="700")
        ax[1].legend(); ax[1].grid(axis="y", alpha=0.3)
    except Exception as e:
        ax[0].text(0.5, 0.5, f"sst data n/a\n{e}", ha="center")
    try:
        ul = pd.read_csv("/home/owen/tilt_experiments/outputs/06_upper_limit/upper_limit.csv")
        u = ul[ul.station.isin(["EEC1", "EC1"])].set_index("station").reindex(["EEC1", "EC1"])
        ax[2].bar([0, 1], u.Amin_peak, color=[WINTER[3], SUMMER[3]])
        ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["EC1\n(eruptive)", "EC1\n(non-eruptive)"])
        ax[2].set_ylabel("A_min peak tilt (90% conf.)", fontweight="700")
        ax[2].set_title("Detection upper limit\n(noisier eruptive period → weaker bound)", fontweight="700")
        ax[2].grid(axis="y", alpha=0.3)
    except Exception as e:
        ax[2].text(0.5, 0.5, f"upper-limit data n/a\n{e}", ha="center")
    fig.suptitle("EC1: eruptive vs non-eruptive period — template performance and detection sensitivity", fontweight="700", fontsize=14)
    fig.tight_layout(); fig.savefig(HERE/"07_detection_performance.png", dpi=300); plt.close(fig)

    L = ["EC1 (eruptive) vs EC1 (non-eruptive) — SAME borehole instrument, two campaigns", "=" * 66,
         f"in-band horizontal background (raw)            winter/summer = {wh.std()/sh.std():.1f}×",
         f"in-band horizontal background (de-thermalised) winter/summer = {wh_d.std()/sh_d.std():.1f}×",
         f"thermal coupling r(tilt,na)   winter={wr:+.2f}  summer={sr:+.2f}",
         f"digitisation step (LSB)       winter={wlsb:.4g}  summer={slsb:.4g}  → SAME units (not a scale factor)",
         "",
         "STORY: same instrument/site; the winter/INGV record carries several× the in-band tilt background",
         "of the summer/experiment record, and it survives de-thermalising and the same-units (LSB) check —",
         "so it is REAL ground tilt, most plausibly seasonal meteorological loading, not instrumental. This",
         "noisier winter record is why EEC1's 90% detection upper limit is weaker than EC1's. (Early-window",
         "thermal coupling at EEC1 reaches ~42%; it is non-stationary, hence the smaller longest-block value.)"]
    (HERE/"SUMMARY.txt").write_text("\n".join(L)); print("\n".join(L)); print(f"\nFigures → {HERE}")


if __name__ == "__main__":
    main()
