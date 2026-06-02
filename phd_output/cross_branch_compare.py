"""
phd_output/synthesis/cross_branch_compare.py — three-branch headline comparison
===============================================================================
Puts the three evidence branches (components = X&Y, magnitude, vector |R|) side by side on the
metrics that decide the result: cross-station synchrony p, false-alarm rate, injection & network
SNR90 (sensitivity), and the per-window significant-detection counts. All three should be null.

Outputs: phd_output/synthesis/cross_branch_summary.{csv,png}
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/owen/tilt_validation")
PHD = BASE / "phd_output"
OUT = PHD / "synthesis"
BRANCHES = ["components", "magnitude", "vector"]
warnings.filterwarnings("ignore")


def _snr90(csv, pcol):
    if not csv.exists():
        return {}
    d = pd.read_csv(csv)
    out = {}
    for ds, g in d.groupby("dataset"):
        g = g.sort_values("snr")
        y, x = g[pcol].to_numpy(), g.snr.to_numpy()
        out[ds] = float(np.interp(0.9, y, x)) if len(y) and y.max() >= 0.9 else np.nan
    return out


def collect(branch):
    root = PHD / branch / "SWCC_comprehensive"
    row = {"branch": branch}
    syn = root / "continuous" / "synchrony.csv"
    if syn.exists():
        s = pd.read_csv(syn)
        for ds in ("ingv", "experiment"):
            sd = s[(s.dataset == ds)].p.dropna()
            row[f"synchrony_minp_{ds}"] = float(sd.min()) if len(sd) else np.nan
    far = PHD / branch / "gw_methods" / "far.csv"
    if far.exists():
        f = pd.read_csv(far)
        row["FAR_per_yr_ingv"] = float(f[f.dataset == "ingv"].far_per_yr.min()) if len(f) else np.nan
    inj = _snr90(root / "injection" / "recovery.csv", "p_detect_max")
    net = _snr90(root / "network" / "recovery.csv", "p_network")
    row["inj_SNR90_experiment"] = inj.get("experiment", np.nan)
    row["net_SNR90_experiment"] = net.get("experiment", np.nan)
    dc = root / "continuous" / "detect_counts.csv"
    if dc.exists():
        c = pd.read_csv(dc)
        row["signif_total"] = int(c[["max_signif", "stack_signif"]].sum().sum())
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [collect(b) for b in BRANCHES if (PHD / b).exists()]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cross_branch_summary.csv", index=False)
    print("THREE-BRANCH COMPARISON\n" + df.to_string(index=False))

    # figure: synchrony p and SNR90 by branch
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(df)); w = 0.35
    a = ax[0]
    for i, ds in enumerate(("ingv", "experiment")):
        col = f"synchrony_minp_{ds}"
        if col in df:
            a.bar(x + (i - 0.5) * w, df[col].fillna(1.0), w, label=ds,
                  color=["#1f2937", "#dc2626"][i], alpha=0.8)
    a.axhline(0.05, ls="--", c="k", label="p=0.05")
    a.set_xticks(x); a.set_xticklabels(df.branch)
    a.set(ylabel="smallest synchrony p-value", ylim=(0, 1.05),
          title="Cross-station synchrony by branch (all ≫ 0.05 ⇒ null)")
    a.legend(fontsize=9); a.grid(axis="y", alpha=0.3)
    b = ax[1]
    for i, col in enumerate(("inj_SNR90_experiment", "net_SNR90_experiment")):
        if col in df:
            b.bar(x + (i - 0.5) * w, df[col], w,
                  label=col.replace("_experiment", "").replace("_", " "),
                  color=["#0891b2", "#16a34a"][i], alpha=0.85)
    b.set_xticks(x); b.set_xticklabels(df.branch)
    b.set(ylabel="SNR$_{90}$ (lower = more sensitive)",
          title="Detection sensitivity by branch (experiment)")
    b.legend(fontsize=9); b.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "cross_branch_summary.png", dpi=300); plt.close(fig)
    print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
