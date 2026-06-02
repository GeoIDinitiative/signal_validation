"""
exceedance_rates.py — ⚠ RETRACTED / KNOWN-FLAWED (see template_exceedance/SUMMARY.txt).
The per-template count-enrichment null below is methodologically invalid: a circular shift of one
station's signal leaves the above-floor COUNT near-invariant (degenerate), and the single quiet
"longest clean block" is unrepresentative of the non-stationary record, so expected counts collapse
to ~0 and "enrichment" explodes spuriously. Its results CONTRADICT the FAR / synchrony / battery
(all null) and are NOT to be cited. Retained only as a documented cautionary lesson; not a pipeline
stage. Per-template counts are descriptive only; significance comes from FAR + synchrony + battery.

(Original intent below.)
exceedance_rates.py — SUPPORTIVE calibration of the top-template raw peak counts.

The raw counts (top_templates/sst_peak_counts.csv) are "peaks above each (station,sim,template)'s own
null/significance floor". Those counts are NOT directly comparable across templates: under a calibrated
null every template exceeds its own floor at some baseline rate, and a low-floor template (e.g. the long,
smooth template4) racks up exceedances without matching better. This procedure asks the calibrated
question per combination:

    is the OBSERVED number of above-floor peaks MORE than expected by chance,
    using a STRUCTURE-PRESERVING (circular-shift) null of the real signal?

For each (dataset, station, sim, template) on the directional tilt:
  • observed n_detect / n_signif are read from the raw-count table (above the data-driven floors);
  • the null count is built by CIRCULARLY SHIFTING a representative clean segment of the REAL signal
    (preserves its PSD, non-Gaussianity and waveform structure; destroys only template alignment),
    counting peaks above the same floor, and scaling to the analysed record length;
  • enrichment = observed / expected-null   (≈1 → fully explained by chance; ≫1 → genuine excess);
  • Monte-Carlo p-value (Bonferroni-corrected for the 112 combinations).

This runs ALONGSIDE the raw counts — it does not replace them. Phase-randomised floors stay as the
thresholds; this only re-bases the COUNTS against a fair, structure-preserving expectation.

Outputs (template_exceedance/):
  exceedance_table.csv   per-SST observed / expected / enrichment / p
  01_enrichment_by_sst.png   enrichment per combination (chance line + significance markers)
  02_observed_vs_expected.png  observed vs expected null counts (excess above the y=x line)
  03_enrichment_by_template.png  does the template4 raw-count dominance survive calibration?
  SUMMARY.txt
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

sys.path.insert(0, "/home/owen/tilt_validation")
from swcc_gapaware import swcc_gapaware
from swcc_comprehensive import load_template
from swcc_continuous import load_cont, to_grid

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
OUT = BASE/"template_exceedance"; OUT.mkdir(parents=True, exist_ok=True)
RAW = pd.read_csv(BASE/"SWCC_comprehensive"/"top_templates"/"sst_peak_counts.csv")
COMP, DIST = "dir", 1000
SEG_MAX, M_SHIFT, RESAMPLE = 500000, 200, 4000
DSCOL = {"ingv": "#1f2937", "experiment": "#dc2626"}
rng = np.random.default_rng(11)


def clean_segment(gx):
    """longest finite run of the grid signal, capped — the structure-preserving null source."""
    fin = np.isfinite(gx).astype(int)
    d = np.diff(np.concatenate(([0], fin, [0])))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    i = np.argmax(ends-starts); a, b = starts[i], ends[i]
    seg = gx[a:b]
    return seg[:SEG_MAX]


def null_counts(tpl, seg, fdet, fsig):
    cd, cs = [], []
    L = len(seg)
    for _ in range(M_SHIFT):
        rr = np.abs(swcc_gapaware(tpl, np.roll(seg, int(rng.integers(L*0.05, L*0.95)))))
        rr = np.nan_to_num(rr)
        pk, _ = find_peaks(rr, height=fdet, distance=DIST)
        v = rr[pk]; cd.append(len(v)); cs.append(int((v >= fsig).sum()))
    return np.array(cd), np.array(cs)


def record_null(per_seg, K):
    """sum of K i.i.d. per-segment null counts → null distribution of the record-level count."""
    return np.array([per_seg[rng.integers(0, len(per_seg), K)].sum() for _ in range(RESAMPLE)])


def main():
    rows = []
    for (ds, st), grp in RAW.groupby(["dataset", "station"]):
        d = load_cont(ds, st, COMP)
        if d is None:
            continue
        _, gx, _ = to_grid(d)
        seg = clean_segment(gx)
        analysed = int(np.isfinite(gx).sum())
        K = max(1, round(analysed/len(seg)))
        for r in grp.itertuples():
            tpl = load_template(ds, st, r.sim, r.template)
            if tpl is None:
                continue
            nd, ns = null_counts(tpl, seg, r.floor_detect, r.floor_signif)
            rec_d, rec_s = record_null(nd, K), record_null(ns, K)
            exp_d, exp_s = rec_d.mean(), rec_s.mean()
            enr_d = r.n_detect/exp_d if exp_d > 0 else np.nan
            enr_s = r.n_signif/exp_s if exp_s > 0 else np.nan
            p_d = float((rec_d >= r.n_detect).mean()); p_s = float((rec_s >= r.n_signif).mean())
            rows.append({"dataset": ds, "station": st, "sim": r.sim, "template": r.template,
                         "floor_signif": r.floor_signif,
                         "obs_detect": r.n_detect, "exp_detect": round(exp_d, 1), "enrich_detect": round(enr_d, 2), "p_detect": round(p_d, 4),
                         "obs_signif": r.n_signif, "exp_signif": round(exp_s, 1), "enrich_signif": round(enr_s, 2), "p_signif": round(p_s, 4)})
        print(f"  calibrated {ds}/{st} (K={K}, seg={len(seg)})")
    df = pd.DataFrame(rows); df.to_csv(OUT/"exceedance_table.csv", index=False)
    bonf = 0.05/len(df)

    def sst(d):
        return d.dataset.str[:3]+"·"+d.station+"·"+d.sim.str.replace("sim", "s")+"·"+d.template.str.replace("template", "t")
    # 01 enrichment by SST
    dd = df.copy(); dd["sst"] = sst(dd); dd = dd.sort_values("enrich_signif", ascending=False)
    fig, ax = plt.subplots(figsize=(16, 7))
    bars = ax.bar(np.arange(len(dd)), dd.enrich_signif, color=[DSCOL[x] for x in dd.dataset])
    for i, (e, p) in enumerate(zip(dd.enrich_signif, dd.p_signif)):
        if p < bonf:
            ax.text(i, e, "*", ha="center", va="bottom", fontsize=12, color="green")
    ax.axhline(1.0, ls="--", c="k", label="enrichment = 1 (chance)")
    ax.set_xticks(np.arange(len(dd))); ax.set_xticklabels(dd.sst, rotation=90, fontsize=6)
    ax.set_xlabel("Station · Sim · Template combination"); ax.set_ylabel("enrichment = observed / expected-null (above significance floor)")
    ax.set_title("Calibrated significance-peak enrichment per combination (structure-preserving null; * = p<Bonferroni)")
    ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout(); fig.savefig(OUT/"01_enrichment_by_sst.png", dpi=300); plt.close(fig)

    # 02 observed vs expected
    fig, ax = plt.subplots(figsize=(8, 8))
    for ds, c in DSCOL.items():
        g = df[df.dataset == ds]
        ax.scatter(g.exp_signif, g.obs_signif, s=28, color=c, alpha=0.7, label=ds)
    m = max(df.obs_signif.max(), df.exp_signif.max())*1.1
    ax.plot([0, m], [0, m], "k--", label="observed = expected (chance)")
    ax.set_xlabel("expected null count (above significance floor)"); ax.set_ylabel("observed count")
    ax.set_title("Observed vs structure-preserving expected exceedances\n(points on the line = consistent with chance)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0); ax.set_ylim(0); fig.tight_layout()
    fig.savefig(OUT/"02_observed_vs_expected.png", dpi=300); plt.close(fig)

    # 03 enrichment by template (does template4 dominance survive?)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    keys = ["template1", "template2", "template3", "template4"]; x = np.arange(4); w = 0.38
    for j, ds in enumerate(["ingv", "experiment"]):
        g = df[df.dataset == ds].groupby("template").enrich_signif.mean().reindex(keys)
        ax.bar(x+(j-0.5)*w, g.values, w, color=DSCOL[ds], label=ds)
    ax.axhline(1.0, ls="--", c="k", label="chance")
    ax.set_xticks(x); ax.set_xticklabels(["t1", "t2", "t3", "t4"]); ax.set_xlabel("Template")
    ax.set_ylabel("mean enrichment (above significance floor)")
    ax.set_title("Calibrated enrichment by template — does the raw-count template4 dominance survive?")
    ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout(); fig.savefig(OUT/"03_enrichment_by_template.png", dpi=300); plt.close(fig)

    sig = df[df.p_signif < bonf]
    frac_zero = float((df.exp_signif < 0.5).mean())     # artifact signature: null segment too quiet
    # This procedure is KNOWN-FLAWED (see SUMMARY.txt): the circular-shift count null is degenerate
    # and the single clean-block null source is unrepresentative for non-stationary records. Any
    # apparent "enrichment" is an artefact and is RETRACTED — do not overwrite the retraction note.
    L = ["TEMPLATE EXCEEDANCE / ENRICHMENT — RESULT RETRACTED (methodological artifact)", "=" * 70,
         f"combinations: {len(df)}  | {len(sig)} show raw p<Bonferroni — but these are ARTIFACTS, not detections.",
         f"artifact signature: {100*frac_zero:.0f}% of combinations have an ~zero expected count (quiet null segment).",
         "",
         "WHY FLAWED: (1) a circular shift of one station only permutes window order, so the above-floor",
         "COUNT is near-invariant — the 'null' collapses to the segment's own count; (2) the longest clean",
         "block is a QUIET stretch while the real peaks live in the bursty parts (winter CoV=2.4, kurt=754),",
         "so expected≈0 and enrichment explodes spuriously. This measures 'is the bursty record louder than",
         "one quiet block' (trivially yes), NOT 'does the template match above chance'.",
         "",
         "These artifacts CONTRADICT the robust, confound-free tests — FAR, cross-station SYNCHRONY, and the",
         "four-method correlation BATTERY — all NULL. Per-template counts are DESCRIPTIVE only; per-template",
         "significance is NOT established by counting. Trustworthy significances = FAR + synchrony + battery.",
         "(Note: a full rewrite would need a representative, template-relationship-breaking null; circular-",
         "shift is degenerate and phase-randomisation under-calibrates smooth templates — no cheap clean null.)"]
    if not (OUT/"SUMMARY.txt").exists() or "RETRACTED" not in (OUT/"SUMMARY.txt").read_text():
        (OUT/"SUMMARY.txt").write_text("\n".join(L))
    print("\n".join(L)); print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
