"""
quant_tables_figs.py — thesis tables + a per-component figure from the quantitative assessment.
Builds the floor-based detection tables (by template / configuration / station, broken out by
X, Y, magnitude, vector) that replace Tables 8–11, plus figure T9. Outputs → quant/.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Q = Path("/home/owen/tilt_validation/phd_output/thesis_figures/quant")
FIG = Path("/home/owen/tilt_validation/phd_output/thesis_figures")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.family": "serif", "font.size": 10, "axes.titleweight": "bold",
                     "axes.grid": True, "grid.alpha": 0.3})
COMPS = ["dir", "dir2", "mag", "vec"]
CLAB = {"dir": "X", "dir2": "Y", "mag": "magnitude", "vec": "vector |R|"}
CCOL = {"dir": "#1d4ed8", "dir2": "#0891b2", "mag": "#16a34a", "vec": "#dc2626"}
ROWLAB = {**{f"sim{i}": f"Config. {i}" for i in range(1, 5)},
          **{f"template{i}": f"Template {i}" for i in range(1, 5)}}


def lab(x):
    return ROWLAB.get(x, x)


def md_table(key, fname, title):
    d = pd.read_csv(Q / f"quant_by_{key}.csv")
    rows_order = (sorted(d[key].unique()))
    lines = [f"**{title}** — detections above the data-driven floor (n_detect); "
             f"significance-floor count in brackets. Floors per component differ (X≈0.45, Y≈0.46, "
             f"magnitude≈0.46, vector≈0.36).", "",
             "| " + key.capitalize() + " | Dataset | X | Y | magnitude | vector |R| |",
             "|---|---|---|---|---|---|"]
    for ds, dl in [("ingv", "INGV"), ("experiment", "IMPROVE")]:
        for rk in rows_order:
            cells = []
            for c in COMPS:
                r = d[(d[key] == rk) & (d.component == c) & (d.dataset == ds)]
                if len(r):
                    cells.append(f"{int(r.n_detect.iloc[0])} ({int(r.n_signif.iloc[0])})")
                else:
                    cells.append("—")
            lines.append(f"| {lab(rk)} | {dl} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def fig_t9():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, ttl) in zip(axes, [("template", "by template (time window)"),
                                     ("configuration", "by configuration"),
                                     ("station", "by station")]):
        d = pd.read_csv(Q / f"quant_by_{key}.csv")
        g = d.groupby([key, "component"]).n_detect.sum().unstack().reindex(columns=COMPS)
        order = sorted(g.index)
        g = g.reindex(order)
        x = np.arange(len(order)); w = 0.2
        for i, c in enumerate(COMPS):
            ax.bar(x + (i - 1.5) * w, g[c].values, w, label=CLAB[c], color=CCOL[c], alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels([lab(o) for o in order], rotation=25, ha="right", fontsize=8)
        ax.set_title(ttl); ax.set_ylabel("detections above floor")
    axes[0].legend(title="component", fontsize=8)
    fig.suptitle("Floor-based detections by component (tilt-x, tilt-y, magnitude, vector |R|) — "
                 "counts scale with floor height and background energy, not configuration-diagnostic signal",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "png" / "T9_detections_by_component.png")
    fig.savefig(FIG / "pdf" / "T9_detections_by_component.pdf")
    plt.close(fig)
    print("  ✓ T9_detections_by_component")


def main():
    fig_t9()
    md = ["# Thesis tables — floor-based performance per component (replaces Tables 8–11)", ""]
    md.append(md_table("template", "by_template", "Table A — Performance by template"))
    md.append(md_table("configuration", "by_configuration", "Table B — Performance by configuration"))
    md.append(md_table("station", "by_station", "Table C — Performance by station"))
    (Q / "thesis_tables.md").write_text("\n".join(md))
    print("  ✓ thesis_tables.md")
    # append T9 to figure_map
    fm = pd.read_csv(FIG / "figure_map.csv")
    if "T9_detections_by_component" not in set(fm.new_figure):
        fm = pd.concat([fm, pd.DataFrame([{"new_figure": "T9_detections_by_component",
            "replaces_thesis": "Figs 41/44 + Tables 8/9/11 (per-component breakdown)",
            "caption": "Floor-based detection counts by component (X, Y, magnitude, vector) for the by-template, "
                       "by-configuration and by-station breakdowns; the vector channel has the most crossings "
                       "because its floor is lowest, not because the matches are stronger."}])], ignore_index=True)
        fm.to_csv(FIG / "figure_map.csv", index=False)
    print("done →", Q / "thesis_tables.md")


if __name__ == "__main__":
    main()
