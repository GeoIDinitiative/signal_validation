"""
make_thesis_figures.py — thesis-ready figures for the Chapter 5 overhaul (honest null + upper limits)
=====================================================================================================
Renders publication-formatted PNG+PDF from the existing phd_output / SWCC_comprehensive CSVs, with
"Configuration N" labels (Puglisi: simulation→configuration) and DATA-DRIVEN floor lines instead of
the fixed 0.2/0.5 thresholds. Each output is recorded in figure_map.csv against the thesis figure /
table number(s) it replaces or augments.

Outputs → phd_output/thesis_figures/{png,pdf}/  +  figure_map.csv
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.labelsize": 11, "axes.grid": True,
    "grid.alpha": 0.3, "legend.fontsize": 9,
})

BASE = Path("/home/owen/tilt_validation")
SWCC = BASE / "SWCC_comprehensive"
PHD = BASE / "phd_output"
OUT = PHD / "thesis_figures"
(OUT / "png").mkdir(parents=True, exist_ok=True)
(OUT / "pdf").mkdir(parents=True, exist_ok=True)

CFG = {"sim1": "Configuration 1", "sim2": "Configuration 2",
       "sim3": "Configuration 3", "sim4": "Configuration 4"}
TPL = {"template1": "Template 1\n(0–3333 s)", "template2": "Template 2\n(3333–6666 s)",
       "template3": "Template 3\n(6666–9999 s)", "template4": "Template 4\n(0–10000 s)"}
DS = {"ingv": "INGV (eruptive)", "experiment": "IMPROVE (quiescent)"}
DSC = {"ingv": "#b91c1c", "experiment": "#1d4ed8"}
MAP = []


def save(fig, name, replaces, caption):
    fig.savefig(OUT / "png" / f"{name}.png"); fig.savefig(OUT / "pdf" / f"{name}.pdf")
    plt.close(fig)
    MAP.append({"new_figure": name, "replaces_thesis": replaces, "caption": caption})
    print(f"  ✓ {name}  (→ {replaces})")


# ── T1: data-driven floors vs the old fixed thresholds ────────────────────────
def fig_floors():
    d = pd.read_csv(SWCC / "continuous" / "detect_counts.csv")
    d = d[d.component == "dir"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for ds, g in d.groupby("dataset"):
        g = g.sort_values("station")
        ax.plot(g.station, g.max_f95, "o-", color=DSC[ds], label=f"{DS[ds]} — detection floor (95th)")
        ax.plot(g.station, g.max_f99, "s--", color=DSC[ds], alpha=0.7, label=f"{DS[ds]} — significance floor (99th)")
    ax.set(xlabel="Station", ylabel="Correlation coefficient |r|",
           title="Per-station data-driven detection and significance floors")
    ax.legend(fontsize=8, ncol=2, loc="upper center"); ax.set_ylim(0, 0.75)
    save(fig, "T1_data_driven_floors", "Figs 36, 47a/b (threshold lines); Tables 8–11",
         "Per-station detection (95th-percentile) and significance (99th-percentile) floors for the "
         "directional channel in both monitoring periods, derived from phase-randomised surrogates of each "
         "station's own noise. A matched-filter peak is counted as a detection only where it exceeds the "
         "station/template floor; the floors range ≈0.45–0.66.")


# ── T2/T3: floor-based performance by template & configuration ────────────────
def _agg(sst, key):
    return (sst.groupby([key, "dataset"]).agg(n_detect=("n_detect", "sum"),
            n_signif=("n_signif", "sum"), mean_r=("mean_r", "mean")).reset_index())


def fig_performance():
    sst = pd.read_csv(SWCC / "top_templates" / "sst_peak_counts.csv")
    for key, lab, name, repl, order in [
        ("template", TPL, "T2_performance_by_template", "Fig 44 + Table 9 (by template)",
         ["template1", "template2", "template3", "template4"]),
        ("sim", CFG, "T3_performance_by_configuration", "Fig 41 + Table 8 (by configuration)",
         ["sim1", "sim2", "sim3", "sim4"])]:
        g = _agg(sst, key)
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(order)); w = 0.38
        for i, ds in enumerate(["ingv", "experiment"]):
            gd = g[g.dataset == ds].set_index(key).reindex(order)
            ax.bar(x + (i - 0.5) * w, gd.n_detect.fillna(0), w, color=DSC[ds], alpha=0.85, label=DS[ds])
        ax.set_xticks(x); ax.set_xticklabels([lab[o].replace("\n", " ") for o in order], fontsize=8)
        ax.set(ylabel="detections above the data-driven floor",
               title=f"Detections above the null floor, by {'template' if key=='template' else 'configuration'}")
        ax.legend()
        save(fig, name, repl,
             f"Floor-based detection counts per {'template (time window)' if key=='template' else 'configuration'}, "
             "aggregated over all stations and the complementary axis. Counts are tens–hundreds (above the "
             "data-driven floor), not the tens of thousands previously reported above the 0.2 threshold; they "
             "are per-window screening crossings and do not survive the synchrony/FAR significance tests.")


# ── T4: observed candidate-event correlations relative to the detection floor ──
def fig_corr_vs_floor():
    cd = pd.read_csv(PHD / "synthesis" / "codetection.csv")
    fl = pd.read_csv(SWCC / "continuous" / "template_floors.csv")
    fl_lo, fl_hi = fl.floor_detect.quantile(0.1), fl.floor_detect.quantile(0.9)
    series = [("X axis", cd.rX, "#1d4ed8"), ("Y axis", cd.rY, "#0891b2"),
              ("magnitude", cd.rMag, "#16a34a"), ("vector |R|", cd.rVec, "#dc2626")]
    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot([s.dropna().values for _, s, _ in series], showmedians=True, widths=0.8)
    for pc, (_, _, c) in zip(parts["bodies"], series):
        pc.set_facecolor(c); pc.set_alpha(0.45)
    ax.axhspan(fl_lo, fl_hi, color="#6b7280", alpha=0.18)
    ax.axhline(fl_lo, color="#374151", lw=1.3)
    ax.axhline(fl_hi, color="#374151", lw=1.3, label=f"detection-floor range ({fl_lo:.2f}–{fl_hi:.2f})")
    ax.set_xticks(range(1, len(series) + 1)); ax.set_xticklabels([s[0] for s in series])
    ax.set(ylabel="best-matching-template correlation |r|", ylim=(0, 0.75),
           title="Correlations of the strongest candidate events sit below the detection floor")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "T4_correlations_below_floor", "Figs 42, 43, 45, 46 (peak-corr / SNR distributions)",
         "Distribution of the best-matching-template correlation at each of the 288 strongest candidate events, "
         "on the two axes, the magnitude and the 2-component vector filter. Even the strongest candidates cluster "
         "below the per-station/template detection floor (grey band): the correlation population is consistent "
         "with noise, with no configuration- or template-diagnostic excess.")


# ── T5: injection-recovery SNR50/90 (upper limits) ────────────────────────────
def fig_injection():
    f = SWCC / "injection" / "recovery.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    tcol = "track" if "track" in d.columns else None
    fig, ax = plt.subplots(figsize=(9, 5))
    for ds in ["ingv", "experiment"]:
        g = d[(d.dataset == ds)]
        if tcol:
            g = g[g[tcol] == (g[tcol].iloc[0] if len(g) else "")]
        g = g.sort_values("snr")
        if g.empty:
            continue
        ax.plot(g.snr, g.p_detect_max, "o-", color=DSC[ds], label=DS[ds])
        y, x = g.p_detect_max.to_numpy(), g.snr.to_numpy()
        for p, ls in [(0.5, ":"), (0.9, "--")]:
            s = float(np.interp(p, y, x)) if y.max() >= p else np.nan
            if np.isfinite(s):
                ax.plot([s, s], [0, p], ls, color=DSC[ds], alpha=0.6)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.6); ax.axhline(0.9, color="gray", ls=":", alpha=0.6)
    ax.set(xlabel="injection SNR (template band-RMS / noise band-RMS)", ylabel="detection probability",
           ylim=(0, 1.02), title="Injection-recovery sensitivity — the 90% detectable-amplitude upper limit")
    ax.legend()
    save(fig, "T5_injection_upper_limits", "new (no thesis equivalent) — quantified upper limits",
         "End-to-end injection-recovery: probability of recovering a synthetic template injected into the real "
         "noise at a given SNR. The SNR90 (90% recovery) sets the upper limit — any genuine ULP tilt matching the "
         "templates must be below this amplitude, since none was detected.")


# ── T6: three-branch (facet) null summary ─────────────────────────────────────
def fig_cross_branch():
    f = PHD / "synthesis" / "cross_branch_summary.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    BL = {"components": "Components (X & Y)", "magnitude": "Magnitude", "vector": "Vector |R|"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(d)); w = 0.38
    for i, (col, ds) in enumerate([("synchrony_minp_ingv", "ingv"), ("synchrony_minp_experiment", "experiment")]):
        ax[0].bar(x + (i - 0.5) * w, d[col].fillna(1.0), w, color=DSC[ds], alpha=0.85, label=DS[ds])
    ax[0].axhline(0.05, color="k", ls="--", label="p = 0.05")
    ax[0].set_xticks(x); ax[0].set_xticklabels([BL[b] for b in d.branch], fontsize=8)
    ax[0].set(ylabel="smallest synchrony p-value", ylim=(0, 1.05),
              title="Cross-station synchrony by facet (all ≫ 0.05 ⇒ null)"); ax[0].legend(fontsize=8)
    for i, col in enumerate(["inj_SNR90_experiment", "net_SNR90_experiment"]):
        ax[1].bar(x + (i - 0.5) * w, d[col], w, color=["#0891b2", "#16a34a"][i], alpha=0.85,
                  label=col.replace("_experiment", "").replace("inj", "single").replace("net", "network").replace("_", " "))
    ax[1].set_xticks(x); ax[1].set_xticklabels([BL[b] for b in d.branch], fontsize=8)
    ax[1].set(ylabel="SNR$_{90}$ (lower = more sensitive)", title="Detection sensitivity by facet (IMPROVE)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Three-facet evidence: X&Y components, magnitude, and the 2-component vector filter — all null",
                 fontweight="bold")
    save(fig, "T6_three_facet_null", "new — multi-facet corroboration",
         "The search run independently on the two axes (X & Y), the magnitude, and the full 2-component vector "
         "matched filter. Cross-station synchrony is non-significant in every facet (left); the vector filter is "
         "the most sensitive (lowest SNR90, right) yet still detects nothing — the null is robust to representation.")


# ── T7: cross-facet co-detection with direction agreement ─────────────────────
def fig_codetection():
    f = PHD / "synthesis" / "codetection.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    joint = (d.rX * d.rY * d.rMag) ** (1 / 3)
    amp = d.passX & d.passY & d.passMag
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc = ax.scatter(d.dAz, joint, c=np.where(amp, "#dc2626", "#9ca3af"), s=22, alpha=0.75,
                    edgecolor="k", linewidth=0.2)
    for dd in [10, 20, 30, 45]:
        ax.axvline(dd, ls=":", color="gray", alpha=0.5); ax.text(dd, ax.get_ylim()[1]*0.98, f"{dd}°", fontsize=7, ha="center")
    ax.set(xlabel="tilt-vector direction disagreement |Δazimuth| (deg)",
           ylabel="joint correlation (rX·rY·rMag)$^{1/3}$",
           title="Cross-facet co-detection: X & Y & magnitude AND direction agreement")
    ax.scatter([], [], c="#dc2626", label="X & Y & magnitude all above floor")
    ax.scatter([], [], c="#9ca3af", label="below amplitude floor")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "T7_codetection_direction", "new — strongest joint evidence",
         "Every detected candidate event (288) scored on all three facets and the tilt-vector direction match. "
         "Zero events clear the strict criterion — X, Y and magnitude all above their floors AND the observed tilt "
         "vector pointing the modelled way (Δazimuth ≤ 10–45°). The joint-and-directional test, the hardest to "
         "satisfy by chance, yields no co-detections.")


# ── T8: time-slide false-alarm rate ───────────────────────────────────────────
def fig_far():
    f = SWCC.parent / "gw_methods" / "far.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    g = d.groupby("dataset").far_per_yr.min() if "dataset" in d.columns else None
    if g is not None:
        ax.bar([DS.get(k, k) for k in g.index], g.values, color=[DSC.get(k, "#444") for k in g.index], alpha=0.85)
        ax.set(ylabel="false-alarm rate (events / year)",
               title="Loudest real coincidence — false-alarm rate vs time-slide background")
        for i, v in enumerate(g.values):
            ax.text(i, v, f"{v:.1f}/yr", ha="center", va="bottom", fontsize=9)
    save(fig, "T8_false_alarm_rate", "new — time-slide significance",
         "False-alarm rate of the loudest real cross-station coincidence, measured against a time-slide background. "
         "A FAR of order 1/year is consistent with background noise — the coincidences are not significant.")


def main():
    print("Rendering thesis figures →", OUT)
    fig_floors(); fig_performance(); fig_corr_vs_floor(); fig_injection()
    fig_cross_branch(); fig_codetection(); fig_far()
    pd.DataFrame(MAP).to_csv(OUT / "figure_map.csv", index=False)
    print(f"\n{len(MAP)} figures + figure_map.csv → {OUT}")


if __name__ == "__main__":
    main()
