"""
credibility_checks.py
=====================
Validation / credibility figures for the rebuilt pipeline (PIPELINE_AUDIT.md, Part F).

Produces, in SWCC_comprehensive/credibility/:
  1. filter_response.png  — Butterworth 0.001-0.01 Hz magnitude + step response
                            (confirms -3 dB corners and settling length).
  2. before_after.png     — per-segment filtering vs naive across-gap filtering on the
                            SAME raw stretch (demonstrates the A3/A5 de-ringing fix).
  3. null_test.png/.txt   — phase-randomised surrogate SWCC -> false-peak rate at r=0.2
                            and an empirical significance threshold for |r|.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, sosfreqz, find_peaks

from swcc_comprehensive import (swcc_segment, load_clean, load_template,
                                SIMS, TEMPLATES, THRESHOLD, PEAK_DISTANCE)

BASE = Path("/home/owen/tilt_validation")
RAW  = BASE / "tilt_raw_pwave_removed"
OUT  = BASE / "SWCC_comprehensive" / "credibility"
FS, BAND, ORDER = 1.0, (0.001, 0.01), 4


def sos():
    nyq = 0.5 * FS
    return butter(ORDER, [BAND[0] / nyq, BAND[1] / nyq], btype="bandpass", output="sos")


# ── 1. filter response ───────────────────────────────────────────────────────
def filter_response():
    s = sos()
    w, h = sosfreqz(s, worN=8192, fs=FS)
    mag = 20 * np.log10(np.maximum(np.abs(h), 1e-12))

    # step response settling
    step = np.ones(20000)
    sr = sosfiltfilt(s, step)            # zero-phase: transient at both ends
    imp = np.zeros(20000); imp[10000] = 1.0
    ir = sosfiltfilt(s, imp)
    settle = np.where(np.abs(ir) > 0.01 * np.abs(ir).max())[0]
    settle_len = (settle.max() - settle.min()) if len(settle) else 0

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].semilogx(w, mag, color="#2563eb")
    for fc in BAND:
        ax[0].axvline(fc, ls="--", c="k", alpha=0.5)
    ax[0].axhline(-3, ls=":", c="r", alpha=0.7, label="-3 dB")
    ax[0].set(xlim=(1e-4, 0.5), ylim=(-80, 5), xlabel="Frequency (Hz)",
              ylabel="Magnitude (dB)", title=f"Butterworth order {ORDER} bandpass {BAND} Hz")
    ax[0].legend(); ax[0].grid(which="both", alpha=0.3)

    ax[1].plot(ir, color="#dc2626")
    ax[1].set(xlabel="Sample (s)", ylabel="Impulse response",
              title=f"Impulse response — settling ≈ {settle_len} s")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "filter_response.png", dpi=140); plt.close(fig)
    print(f"  filter_response.png  (settling ≈ {settle_len} s)")
    return settle_len


# ── 2. before/after de-ringing on identical raw input ────────────────────────
def before_after():
    df = pd.read_feather(RAW / "ECPN_tilt_raw_pwave_removed.feather")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    sec = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds().to_numpy(float)
    x = df["east"].to_numpy(float)

    # find a spot with a gap flanked by two decent runs
    gaps = np.where(np.diff(sec) > 1.5)[0]
    pick = None
    for g in gaps:
        a = max(0, g - 5000); b = min(len(x), g + 5000)
        if (g - a) > 4500 and (b - g) > 4500:
            pick = (a, g, b); break
    if pick is None:
        print("  before_after: no suitable gap found"); return
    a, g, b = pick
    seg_sec, seg_x = sec[a:b], x[a:b]

    s = sos()
    # NAIVE: filter straight across the gap (the old A3 defect)
    naive = sosfiltfilt(s, np.nan_to_num(seg_x - seg_x.mean()))
    # CORRECT: per-segment
    correct = np.full(len(seg_x), np.nan)
    left = slice(0, g - a); right = slice(g - a, b - a)
    for sl in (left, right):
        xx = seg_x[sl] - seg_x[sl].mean()
        if len(xx) > 100:
            correct[sl] = sosfiltfilt(s, xx)

    t = (seg_sec - seg_sec[0])
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(t, naive, color="#dc2626", lw=0.8, label="naive: one sosfiltfilt across gap (rings)")
    ax.plot(t, correct, color="#2563eb", lw=0.8, label="per-segment (this pipeline)")
    ax.axvline(t[g - a], ls="--", c="k", alpha=0.6, label="gap (excised window)")
    ax.set(xlabel="Time within stretch (s)", ylabel="Bandpassed (east)",
           title="De-ringing: per-segment filtering removes the across-gap transient")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "before_after.png", dpi=140); plt.close(fig)
    print("  before_after.png")


# ── 3. phase-randomised null test ────────────────────────────────────────────
def phase_randomize(x, rng):
    X = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0
    if len(x) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x))


def null_test(n_surrogates=200, seed=0):
    rng = np.random.default_rng(seed)
    df = load_clean("ingv", "ECPN", "dir")
    # use the longest segment as the host
    sid = df.groupby("segment_id").size().idxmax()
    host = df[df.segment_id == sid]["bandpassed"].to_numpy(float)
    tpls = [load_template("ingv", "ECPN", s, t) for s in SIMS for t in TEMPLATES]
    tpls = [t for t in tpls if t is not None]

    surr_max_list = []  # peak |r| per surrogate (max across templates)
    surr_peaks = []
    for _ in range(n_surrogates):
        xs = phase_randomize(host, rng)
        best = 0.0; npk = 0
        for tpl in tpls:
            if len(tpl) > len(xs):
                continue
            r = np.abs(swcc_segment(tpl, xs))
            if r.size:
                pk, _ = find_peaks(r, height=THRESHOLD, distance=PEAK_DISTANCE)
                npk += len(pk)
                if r.max() > best:
                    best = r.max()
        surr_max_list.append(best); surr_peaks.append(npk)

    surr_max = np.array(surr_max_list)
    p95, p99 = np.percentile(surr_max, [95, 99])
    mean_falsepk = np.mean(surr_peaks)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(surr_max, bins=30, color="#6b7280", alpha=0.8)
    ax[0].axvline(THRESHOLD, ls="--", c="r", label=f"current threshold {THRESHOLD}")
    ax[0].axvline(p95, ls="--", c="#2563eb", label=f"95th pct {p95:.3f}")
    ax[0].axvline(p99, ls="--", c="#16a34a", label=f"99th pct {p99:.3f}")
    ax[0].set(xlabel="max |r| per surrogate", ylabel="count",
              title="Null distribution of peak |r| (phase-randomised)")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    ax[1].hist(surr_peaks, bins=20, color="#9333ea", alpha=0.7)
    ax[1].set(xlabel="false peaks > threshold per surrogate", ylabel="count",
              title=f"Mean false peaks/surrogate = {mean_falsepk:.1f}")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "null_test.png", dpi=140); plt.close(fig)

    txt = (
        "PHASE-RANDOMISED NULL TEST (ECPN dir, longest segment host)\n"
        f"surrogates                 : {n_surrogates}\n"
        f"host segment length        : {len(host)} samples\n"
        f"templates                  : {len(tpls)}\n"
        f"current threshold          : {THRESHOLD}\n"
        f"surrogate max|r| 95th pct  : {p95:.3f}\n"
        f"surrogate max|r| 99th pct  : {p99:.3f}\n"
        f"mean false peaks/surrogate : {mean_falsepk:.2f}\n\n"
        f"INTERPRETATION: |r| below ~{p99:.2f} is indistinguishable from a signal with the\n"
        f"same power spectrum but random phase. Detections should be judged against this\n"
        f"empirical floor rather than the nominal {THRESHOLD} threshold.\n"
    )
    (OUT / "null_test.txt").write_text(txt)
    print(f"  null_test.png/.txt  (95th={p95:.3f}, 99th={p99:.3f}, "
          f"mean false peaks/surrogate={mean_falsepk:.1f})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Credibility checks →", OUT)
    filter_response()
    before_after()
    null_test()


if __name__ == "__main__":
    main()
