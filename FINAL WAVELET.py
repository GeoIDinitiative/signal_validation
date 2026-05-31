#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import detrend as scipy_detrend
from scipy.interpolate import interp1d
import pywt  # For wavelet analysis
from scipy.ndimage import gaussian_filter

#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
swcc_consistent_quakefilter_MATCHED_LEGENDS.py

What’s new vs _MATCHED.py:
- Legends are REMOVED from all plots.
- Each plot’s legend is exported as a separate PNG in ./figures/ (same run dir).
  This makes the legends reusable outside the figures.

Exports (examples):
  figures/correlated_peaks_by_station_sim_experiment_ge_0_5_legend.png
  figures/mean_snr_db_station_sim_experiment_ge_0_5_legend.png
  ...

All other functionality remains:
- Build df02 (≥0.2) & df1 (≥0.5).
- Match to earthquake arrivals from: /home/owen/Etna_signals/earthquake_data/earthquakes.csv
- Split into kept vs matched; save matched CSVs.
- Plot both cohorts (kept and matched) with legendless figures + separate legend PNGs.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# Config
# ==========================
PEAKS_ROOT = Path("/home/owen/tilt_validation/SWCC_utc_fixed")
QUAKE_FILE = Path("/home/owen/tilt_validation/earthquakes_merged_utc.csv")

START_COL = "seg_start_time"
END_COL   = "seg_end_time"

TOLERANCE_SEC = 600
LABEL_ROT_DEG = 15
FIG_HEIGHT = 6

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# ==========================
# Helpers
# ==========================
def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    s = re.sub(r"_+", "_", s)
    return s

def save_legend_as_png(handles, labels, filename: Path, title: str | None = None, ncol: int | None = None):
    """
    Save a legend-only PNG. Legend background is transparent; no axes.
    """
    if not handles or not labels:
        return
    fig = plt.figure(figsize=(6, 1.0))
    leg = fig.legend(handles, labels, loc="center", ncol=ncol if ncol else len(labels), frameon=False, title=title)
    if leg is not None and leg.get_title():
        leg.get_title().set_fontsize(10)
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.axis("off")
    fig.savefig(filename, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

def clean_dataset(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def clean_station(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def parse_timedelta_safe(s: pd.Series) -> pd.Series:
    return pd.to_timedelta(s, errors="coerce")

def compute_duration_s(df: pd.DataFrame, start_col: str = START_COL, end_col: str = END_COL,
                       out_td: str = "duration", out_sec: str = "duration_s") -> pd.DataFrame:
    if start_col not in df.columns or end_col not in df.columns:
        df[out_td] = pd.NaT
        df[out_sec] = np.nan
        return df
    start = parse_timedelta_safe(df[start_col])
    end   = parse_timedelta_safe(df[end_col])
    dur = end - start
    dur = dur.mask(dur < pd.Timedelta(0), dur + pd.Timedelta(days=1))
    df[out_td] = dur
    df[out_sec] = df[out_td].dt.total_seconds()
    return df

def build_df1_from_df05(df05: pd.DataFrame) -> pd.DataFrame:
    # Check if we have peak_time_dt or separate peak_date/peak_time
    has_combined = "peak_time_dt" in df05.columns
    has_separate = "peak_date" in df05.columns and "peak_time" in df05.columns
    
    if has_combined:
        # Split peak_time_dt into date and time components
        dt_series = pd.to_datetime(df05["peak_time_dt"], errors="coerce")
        peak_date = dt_series.dt.date.astype(str)
        peak_time = dt_series.dt.time.astype(str)
    elif has_separate:
        peak_date = df05["peak_date"]
        peak_time = df05["peak_time"]
    else:
        raise KeyError("df05 must have either 'peak_time_dt' or both 'peak_date' and 'peak_time'")
    
    # Use seg_duration_s if available, otherwise duration_s
    if "duration_s" in df05.columns:
        duration = df05["duration_s"]
    elif "seg_duration_s" in df05.columns:
        duration = df05["seg_duration_s"]
    else:
        duration = pd.Series([np.nan] * len(df05), index=df05.index)
    
    # Handle threshold_used if it exists (old format), otherwise use peak_corr threshold
    if "threshold_used" in df05.columns:
        threshold = df05["threshold_used"].reset_index(drop=True)
    else:
        # For SWCC clean peaks, infer threshold from peak_corr value
        threshold = pd.Series(["0.5"] * len(df05), index=df05.index).reset_index(drop=True)

    return pd.DataFrame({
        "dataset":        df05["dataset"].reset_index(drop=True),
        "station":        df05["station"].reset_index(drop=True),
        "sim":            df05["sim"].reset_index(drop=True),
        "template":       df05["template"].reset_index(drop=True),
        "peak_date":      peak_date.reset_index(drop=True),
        "peak_time":      peak_time.reset_index(drop=True),
        "peak_corr":      df05["peak_corr"].reset_index(drop=True),
        "snr_linear":     df05["snr_linear"].reset_index(drop=True),
        "snr_db":         df05["snr_db"].reset_index(drop=True),
        "threshold_used": threshold,
        "duration_s":     duration.reset_index(drop=True),
    })

def norm_sim_to_int(sim_series: pd.Series) -> pd.Series:
    return sim_series.astype(str).str.extract(r"(\d+)")[0].astype("Int64")

def _norm_template_to_int(template_series: pd.Series) -> pd.Series:
    return template_series.astype(str).str.extract(r"(\d+)")[0].astype("Int64")

def _to_numeric_series(s):
    return pd.to_numeric(s, errors="coerce")

def coalesce_event_dt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "seg_start_dt" in df.columns:
        dt1 = pd.to_datetime(df["seg_start_dt"], errors="coerce")
    else:
        dt1 = pd.Series(pd.NaT, index=df.index)
    if "seg_end_dt" in df.columns:
        dt2 = pd.to_datetime(df["seg_end_dt"], errors="coerce")
    else:
        dt2 = pd.Series(pd.NaT, index=df.index)
    event_dt = dt1.fillna(dt2)
    if event_dt.isna().any() and {"peak_date","peak_time"}.issubset(df.columns):
        combo = (df["peak_date"].astype(str).str.strip() + " " +
                 df["peak_time"].astype(str).str.strip())
        event_dt = event_dt.fillna(pd.to_datetime(combo, errors="coerce"))
    if event_dt.isna().any():
        cand_cols = [c for c in df.columns if re.search(r"(dt|date|time)$", c, flags=re.I)]
        for c in cand_cols:
            try:
                tmp = pd.to_datetime(df[c], errors="coerce")
                event_dt = event_dt.fillna(tmp)
            except Exception:
                pass
    df["event_dt"] = event_dt
    return df

def load_quake_arrivals_from_file(quake_file: Path) -> pd.DataFrame:
    q = pd.read_csv(quake_file)
    cols = {c.lower(): c for c in q.columns}
    arrival = None
    # Prioritize p_wave_eta if available (P-wave arrival time)
    if "p_wave_eta" in cols:
        arrival = pd.to_datetime(q[cols["p_wave_eta"]], errors="coerce")
    elif "event_time" in cols:
        arrival = pd.to_datetime(q[cols["event_time"]], errors="coerce")
    elif "date" in cols and "time" in cols:
        combo = q[cols["date"]].astype(str).str.strip() + " " + q[cols["time"]].astype(str).str.strip()
        arrival = pd.to_datetime(combo, errors="coerce")
    else:
        for key in ["origin_time","timestamp","datetime","time_utc","time_local","otime"]:
            if key in cols:
                arrival = pd.to_datetime(q[cols[key]], errors="coerce")
                break
    if arrival is None:
        raise ValueError("Could not locate time columns in earthquake file.")
    out = pd.DataFrame({"arrival_dt": arrival})
    for st_key in ["station","sta","station_code","Station","STA"]:
        if st_key in q.columns:
            out["station"] = clean_station(q[st_key])
            break
    out = out.dropna(subset=["arrival_dt"]).sort_values("arrival_dt").reset_index(drop=True)
    return out

def match_split_against_quakes(df: pd.DataFrame, quakes: pd.DataFrame, tolerance_sec: int):
    if df.empty or quakes.empty:
        return df.copy(), df.head(0).copy()
    d = df.copy()
    if "station" in d.columns:
        d["station"] = clean_station(d["station"])
    d = coalesce_event_dt(d)
    d = d.dropna(subset=["event_dt"]).sort_values("event_dt").reset_index(drop=True)
    q = quakes.copy()
    tol = pd.Timedelta(seconds=int(tolerance_sec))

    if "station" in q.columns and "station" in d.columns:
        kept_chunks, match_chunks = [], []
        for st, d_g in d.groupby("station", sort=False):
            q_g = q[q["station"] == st][["arrival_dt"]].sort_values("arrival_dt")
            if q_g.empty:
                kept_chunks.append(d_g)
                continue
            merged = pd.merge_asof(d_g, q_g, left_on="event_dt", right_on="arrival_dt",
                                   direction="nearest", tolerance=tol)
            matched = merged[merged["arrival_dt"].notna()].copy()
            if not matched.empty:
                delta = (matched["event_dt"] - matched["arrival_dt"]).dt.total_seconds()
                matched["matched_arrival_dt"] = matched["arrival_dt"]
                matched["match_delta_s"] = delta
                matched = matched.drop(columns=["arrival_dt"])
                match_chunks.append(matched)
            kept = merged[merged["arrival_dt"].isna()].drop(columns=["arrival_dt"])
            kept_chunks.append(kept)
        kept_df = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else d.head(0)
        matched_df = pd.concat(match_chunks, ignore_index=True) if match_chunks else d.head(0)
    else:
        q_g = q[["arrival_dt"]].sort_values("arrival_dt")
        merged = pd.merge_asof(d, q_g, left_on="event_dt", right_on="arrival_dt",
                               direction="nearest", tolerance=tol)
        matched_df = merged[merged["arrival_dt"].notna()].copy()
        kept_df    = merged[merged["arrival_dt"].isna()].drop(columns=["arrival_dt"])
        if not matched_df.empty:
            matched_df["matched_arrival_dt"] = matched_df["arrival_dt"]
            matched_df["match_delta_s"] = (matched_df["event_dt"] - matched_df["arrival_dt"]).dt.total_seconds()
            matched_df = matched_df.drop(columns=["arrival_dt"])
    return kept_df, matched_df

# ==========================
# Plotting helpers (legendless + legend PNG export)
# ==========================
def _bar_labels(ax, bars, values):
    labels = [f"{int(v)}" if float(v) > 0 else "" for v in values]
    ax.bar_label(bars, labels=labels, padding=2, fontsize=8)

def station_sim_pivot(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if df.empty: 
        return pd.DataFrame()
    mask = clean_dataset(df["dataset"]).eq(dataset_name.lower())
    if "sim" not in df.columns or "station" not in df.columns:
        return pd.DataFrame()
    sim_i = norm_sim_to_int(df.loc[mask, "sim"])
    tmp = (df.loc[mask, ["station"]].assign(sim=sim_i.values).dropna(subset=["sim"]))
    if tmp.empty: 
        return pd.DataFrame()
    counts = (tmp.groupby(["station","sim"], dropna=False).size().rename("n").reset_index())
    piv = counts.pivot(index="station", columns="sim", values="n").fillna(0).astype(int)
    ordered = [c for c in [1,2,3,4] if c in piv.columns]
    return piv.reindex(columns=ordered)

def plot_station_sim_grouped(pivot: pd.DataFrame, title: str):
    if pivot is None or pivot.empty:
        print(f"[info] No data to plot for: {title}")
        return
    totals = pivot.sum(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[totals.index]
    stations = pivot.index.astype(str).tolist()
    sim_cols = list(pivot.columns)
    values   = pivot.values
    n_st, n_sims = values.shape
    x = np.arange(n_st)
    width = 0.8 / max(1, n_sims)
    fig_w = max(10, min(28, 0.25 * n_st))
    fig, ax = plt.subplots(figsize=(fig_w, FIG_HEIGHT))
    cluster_tops = np.zeros(n_st, dtype=float)
    handle_cache = []
    for j, sim in enumerate(sim_cols):
        offs = (j - (n_sims - 1)/2) * width
        bars = ax.bar(x + offs, values[:, j], width=width, label=f"sim{int(sim)}", zorder=2)
        _bar_labels(ax, bars, values[:, j])
        cluster_tops = np.maximum(cluster_tops, values[:, j])
        handle_cache.append(bars[0])
    ax.set_xticks(x)
    ax.set_xticklabels(stations, rotation=LABEL_ROT_DEG, ha="right")
    ax.set_xlabel("Station")
    ax.set_ylabel("No. of correlated peaks")
    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35, zorder=1)

    # Save legend separately (remove from plot)
    labels = [f"sim{int(s)}" for s in sim_cols]
    legend_name = OUT_DIR / f"{slugify(title)}_legend.png"
    save_legend_as_png(handle_cache, labels, legend_name, title="Simulation", ncol=min(4, len(labels)))

    ymax = int(values.max()) if values.size else 1
    step = max(1, int(np.ceil(max(1, ymax)/5)))
    ax.set_yticks(np.arange(0, max(1, ymax)+step, step))
    ax.margins(y=0.1)
    # totals_arr = pivot.sum(axis=1).to_numpy()
    # y_off = max(1, int(np.ceil(ymax/20))) if ymax > 0 else 1
    # for xi, ytop, tot in zip(x, cluster_tops, totals_arr):
    # if tot > 0:
            # ax.text(xi, ytop + y_off, str(int(tot)), ha="center", va="bottom", fontsize=9)
    plt.subplots_adjust(bottom=0.22)
    plt.tight_layout()
    plt.show()

def station_template_pivot(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    mask = clean_dataset(df["dataset"]).eq(dataset_name.lower())
    if "template" not in df.columns or "station" not in df.columns:
        return pd.DataFrame()
    tmpl_i = _norm_template_to_int(df.loc[mask, "template"])
    tmp = (df.loc[mask, ["station"]].assign(template=tmpl_i.values).dropna(subset=["template"]))
    if tmp.empty:
        return pd.DataFrame()
    counts = (tmp.groupby(["station","template"], dropna=False).size().rename("n").reset_index())
    piv = counts.pivot(index="station", columns="template", values="n").fillna(0).astype(int)
    ordered = [c for c in [1,2,3,4] if c in piv.columns]
    return piv.reindex(columns=ordered)

def plot_station_template_grouped(pivot: pd.DataFrame, title: str):
    if pivot is None or pivot.empty:
        print(f"[info] No data to plot for: {title}")
        return
    totals = pivot.sum(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[totals.index]
    stations = pivot.index.astype(str).tolist()
    tmpl_cols = list(pivot.columns)
    values    = pivot.values
    n_st, n_tm = values.shape
    x = np.arange(n_st)
    width = 0.8 / max(1, n_tm)
    fig_w = max(10, min(28, 0.25 * n_st))
    fig, ax = plt.subplots(figsize=(fig_w, FIG_HEIGHT))
    cluster_tops = np.zeros(n_st, dtype=float)
    handle_cache = []
    for j, t in enumerate(tmpl_cols):
        offs = (j - (n_tm - 1)/2) * width
        bars = ax.bar(x + offs, values[:, j], width=width, label=f"T{int(t)}", zorder=2)
        _bar_labels(ax, bars, values[:, j])
        cluster_tops = np.maximum(cluster_tops, values[:, j])
        handle_cache.append(bars[0])
    ax.set_xticks(x)
    ax.set_xticklabels(stations, rotation=LABEL_ROT_DEG, ha="right")
    ax.set_xlabel("Station")
    ax.set_ylabel("No. of matched peaks")
    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35, zorder=1)

    # Save legend separately
    labels = [f"T{int(t)}" for t in tmpl_cols]
    legend_name = OUT_DIR / f"{slugify(title)}_legend.png"
    save_legend_as_png(handle_cache, labels, legend_name, title="Template", ncol=min(4, len(labels)))

    ymax = int(values.max()) if values.size else 1
    step = max(1, int(np.ceil(max(1, ymax)/5)))
    ax.set_yticks(np.arange(0, max(1, ymax)+step, step))
    ax.margins(y=0.1)
    # totals_arr = pivot.sum(axis=1).to_numpy()
    # y_off = max(1, int(np.ceil(ymax/20))) if ymax > 0 else 1
    # for xi, ytop, tot in zip(x, cluster_tops, totals_arr):
    # if tot > 0:
            # ax.text(xi, ytop + y_off, str(int(tot)), ha="center", va="bottom", fontsize=9)
    plt.subplots_adjust(bottom=0.22)
    plt.tight_layout()
    plt.show()

def corr_pivot(df: pd.DataFrame, row: str, col: str, dataset_name: str | None = None):
    d = df.copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    d["peak_corr"] = _to_numeric_series(d.get("peak_corr"))
    if dataset_name is not None and "dataset" in d.columns:
        d = d[clean_dataset(d["dataset"]).eq(dataset_name.lower())]
    if d.empty: return pd.DataFrame(), pd.DataFrame()
    if col == "sim" and "sim" in d.columns:
        cc = norm_sim_to_int(d["sim"])
    elif col == "template" and "template" in d.columns:
        cc = _norm_template_to_int(d["template"])
    else:
        cc = d.get(col)
    rr = d.get(row).astype(str) if row in d.columns else pd.Series([], dtype=str)
    vals = d.get("peak_corr")
    mean_tbl = pd.pivot_table(pd.DataFrame({row: rr, col: cc, "peak_corr": vals}),
                              index=row, columns=col, values="peak_corr", aggfunc="mean")
    count_tbl = pd.pivot_table(pd.DataFrame({row: rr, col: cc, "peak_corr": vals}),
                               index=row, columns=col, values="peak_corr", aggfunc="count")
    if col in {"sim", "template"}:
        order = [1,2,3,4]
        exist = [c for c in order if c in mean_tbl.columns]
        mean_tbl = mean_tbl.reindex(columns=exist)
        count_tbl = count_tbl.reindex(columns=exist)
    row_means = mean_tbl.mean(axis=1, skipna=True).sort_values(ascending=False)
    mean_tbl = mean_tbl.loc[row_means.index]
    count_tbl = count_tbl.loc[row_means.index]
    return mean_tbl, count_tbl

def _annot_matrix(ax, M, C=None, fmt="{:.2f}", cfmt=" ({:d})"):
    nrows, ncols = M.shape
    for i in range(nrows):
        for j in range(ncols):
            val = M[i, j]
            if np.isfinite(val):
                text = fmt.format(val)
                if C is not None and np.isfinite(C[i, j]):
                    text += cfmt.format(int(C[i, j]))
                ax.text(j, i, text, ha="center", va="center", fontsize=8)

def plot_corr_heatmap(mean_tbl: pd.DataFrame, count_tbl: pd.DataFrame, title: str, xlabel: str, ylabel: str):
    if mean_tbl is None or mean_tbl.empty:
        print(f"[info] No data to plot for: {title}")
        return
    M = mean_tbl.to_numpy(dtype=float)
    C = count_tbl.reindex(index=mean_tbl.index, columns=mean_tbl.columns).to_numpy(dtype=float)
    fig_h = max(4.5, min(14, 0.4 * len(mean_tbl.index)))
    fig_w = max(6.5, min(18, 1.6 + 0.8 * len(mean_tbl.columns)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(np.arange(len(mean_tbl.columns)))
    ax.set_xticklabels([f"{xlabel[0].upper()}{int(c)}" if isinstance(c, (int, np.integer)) else str(c) for c in mean_tbl.columns], rotation=0)
    ax.set_yticks(np.arange(len(mean_tbl.index)))
    ax.set_yticklabels(mean_tbl.index.astype(str))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xticks(np.arange(-.5, len(mean_tbl.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(mean_tbl.index), 1), minor=True)
    ax.grid(which="minor", color="w", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    _annot_matrix(ax, M, C=C, fmt="{:.2f}", cfmt=" ({:d})")
    cb = fig.colorbar(im, ax=ax, shrink=0.9); cb.set_label("mean peak_corr")
    plt.tight_layout(); plt.show()

def boxplot_corr_by(df: pd.DataFrame, group_col: str, dataset_name: str, title: str):
    d = df.copy()
    if d.empty:
        print(f"[info] No data for boxplot: {title}"); return
    d["peak_corr"] = _to_numeric_series(d.get("peak_corr"))
    d = d[clean_dataset(d["dataset"]).eq(dataset_name.lower())] if "dataset" in d.columns else d
    if d.empty: print(f"[info] No data for boxplot: {title}"); return
    if group_col == "sim" and "sim" in d.columns:
        g = norm_sim_to_int(d["sim"])
    elif group_col == "template" and "template" in d.columns:
        g = _norm_template_to_int(d["template"])
    else:
        g = d.get(group_col)
    d = d.assign(_grp=g).dropna(subset=["_grp", "peak_corr"])
    if d.empty:
        print(f"[info] No data for boxplot: {title}"); return
    order = [c for c in [1,2,3,4] if c in d["_grp"].unique().tolist()]
    data = [d.loc[d["_grp"] == k, "peak_corr"].to_numpy() for k in order]
    if not data:
        print(f"[info] No data for boxplot: {title}"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=[f"{group_col[0].upper()}{k}" for k in order], showfliers=False)
    ax.set_ylabel("peak_corr"); ax.set_title(title)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35); plt.tight_layout(); plt.show()

def dataset_compare_corr_bars(df: pd.DataFrame, by_col: str, title: str):
    d = df.copy()
    if d.empty:
        print(f"[info] No data to plot for: {title}"); return
    d["peak_corr"] = _to_numeric_series(d.get("peak_corr"))
    d["dataset_lc"] = clean_dataset(d["dataset"]) if "dataset" in d.columns else d.get("dataset")
    if by_col == "sim" and "sim" in d.columns:
        d["_grp"] = norm_sim_to_int(d["sim"])
    elif by_col == "template" and "template" in d.columns:
        d["_grp"] = _norm_template_to_int(d["template"])
    else:
        d["_grp"] = d.get(by_col)
    d = d.dropna(subset=["_grp", "peak_corr"])
    if d.empty:
        print(f"[info] No data to plot for: {title}"); return
    g = (d.groupby(["dataset_lc","_grp"])["peak_corr"].mean().reset_index())
    pv = g.pivot(index="_grp", columns="dataset_lc", values="peak_corr").astype(float)
    pv = pv.reindex([c for c in [1,2,3,4] if c in pv.index])
    x = np.arange(len(pv)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    exp_vals = pv.get("experiment", pd.Series(0, index=pv.index)).to_numpy()
    ingv_vals = pv.get("ingv", pd.Series(0, index=pv.index)).to_numpy()
    b1 = ax.bar(x - w/2, exp_vals, width=w, label="experiment")
    b2 = ax.bar(x + w/2, ingv_vals, width=w, label="ingv")
    ax.set_xticks(x); ax.set_xticklabels([f"{by_col[0].upper()}{int(i)}" for i in pv.index])
    ax.set_ylabel("mean peak_corr"); ax.set_title(title)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)

    # Save legend separately (Experiment vs INGV)
    handles, labels = ax.get_legend_handles_labels()
    legend_name = OUT_DIR / f"{slugify(title)}_legend.png"
    save_legend_as_png(handles, labels, legend_name, title="Dataset", ncol=2)

    for bars in (b1, b2):
        labels = [f"{v:.2f}" for v in [bar.get_height() for bar in bars]]
        ax.bar_label(bars, labels=labels, padding=2, fontsize=8)
    plt.tight_layout(); plt.show()

def metric_pivot(df: pd.DataFrame, metric_col: str, row: str, col: str, dataset_name: str | None = None):
    d = df.copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    d[metric_col] = _to_numeric_series(d.get(metric_col))
    if dataset_name is not None and "dataset" in d.columns:
        d = d[clean_dataset(d["dataset"]).eq(dataset_name.lower())]
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    if col == "sim" and "sim" in d.columns:
        cc = norm_sim_to_int(d["sim"])
    elif col == "template" and "template" in d.columns:
        cc = _norm_template_to_int(d["template"])
    else:
        cc = d.get(col)
    rr = d.get(row).astype(str) if row in d.columns else pd.Series([], dtype=str)
    vals = d.get(metric_col)
    mean_tbl = pd.pivot_table(pd.DataFrame({row: rr, col: cc, metric_col: vals}),
                              index=row, columns=col, values=metric_col, aggfunc="mean")
    count_tbl = pd.pivot_table(pd.DataFrame({row: rr, col: cc, metric_col: vals}),
                               index=row, columns=col, values=metric_col, aggfunc="count")
    if col in {"sim", "template"}:
        order = [1,2,3,4]
        exist = [c for c in order if c in mean_tbl.columns]
        mean_tbl = mean_tbl.reindex(columns=exist)
        count_tbl = count_tbl.reindex(columns=exist)
    row_means = mean_tbl.mean(axis=1, skipna=True).sort_values(ascending=False)
    mean_tbl = mean_tbl.loc[row_means.index]
    count_tbl = count_tbl.loc[row_means.index]
    return mean_tbl, count_tbl

def plot_metric_heatmap(mean_tbl: pd.DataFrame, count_tbl: pd.DataFrame, title: str, xlabel: str, ylabel: str, colorbar_label: str):
    if mean_tbl is None or mean_tbl.empty:
        print(f"[info] No data to plot for: {title}")
        return
    M = mean_tbl.to_numpy(dtype=float)
    C = count_tbl.reindex(index=mean_tbl.index, columns=mean_tbl.columns).to_numpy(dtype=float)
    fig_h = max(4.5, min(14, 0.4 * len(mean_tbl.index)))
    fig_w = max(6.5, min(18, 1.6 + 0.8 * len(mean_tbl.columns)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(np.arange(len(mean_tbl.columns)))
    ax.set_xticklabels([f"{xlabel[0].upper()}{int(c)}" if isinstance(c, (int, np.integer)) else str(c) for c in mean_tbl.columns])
    ax.set_yticks(np.arange(len(mean_tbl.index)))
    ax.set_yticklabels(mean_tbl.index.astype(str))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xticks(np.arange(-.5, len(mean_tbl.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(mean_tbl.index), 1), minor=True)
    ax.grid(which="minor", color="w", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    _annot_matrix(ax, M, C=C, fmt="{:.2f}", cfmt=" ({:d})")
    cb = fig.colorbar(im, ax=ax, shrink=0.9); cb.set_label(colorbar_label)
    plt.tight_layout(); plt.show()

def boxplot_metric_by(df: pd.DataFrame, metric_col: str, group_col: str, dataset_name: str, title: str, ylabel: str):
    d = df.copy()
    if d.empty:
        print(f"[info] No data for boxplot: {title}"); return
    d[metric_col] = _to_numeric_series(d.get(metric_col))
    d = d[clean_dataset(d["dataset"]).eq(dataset_name.lower())] if "dataset" in d.columns else d
    if d.empty: print(f"[info] No data for boxplot: {title}"); return
    if group_col == "sim" and "sim" in d.columns:
        g = norm_sim_to_int(d["sim"])
    elif group_col == "template" and "template" in d.columns:
        g = _norm_template_to_int(d["template"])
    else:
        g = d.get(group_col)
    d = d.assign(_grp=g).dropna(subset=["_grp", metric_col])
    if d.empty:
        print(f"[info] No data for boxplot: {title}"); return
    order = [c for c in [1,2,3,4] if c in d["_grp"].unique().tolist()]
    data = [d.loc[d["_grp"] == k, metric_col].to_numpy() for k in order]
    if not data:
        print(f"[info] No data for boxplot: {title}"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=[f"{group_col[0].upper()}{k}" for k in order], showfliers=False)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)
    plt.tight_layout(); plt.show()

def dataset_compare_metric_bars(df: pd.DataFrame, metric_col: str, by_col: str, title: str, ylabel: str):
    d = df.copy()
    if d.empty:
        print(f"[info] No data to plot for: {title}"); return
    d[metric_col] = _to_numeric_series(d.get(metric_col))
    d["dataset_lc"] = clean_dataset(d["dataset"]) if "dataset" in d.columns else d.get("dataset")
    if by_col == "sim" and "sim" in d.columns:
        d["_grp"] = norm_sim_to_int(d["sim"])
    elif by_col == "template" and "template" in d.columns:
        d["_grp"] = _norm_template_to_int(d["template"])
    else:
        d["_grp"] = d.get(by_col)
    d = d.dropna(subset=["_grp", metric_col])
    if d.empty:
        print(f"[info] No data to plot for: {title}"); return
    g = (d.groupby(["dataset_lc","_grp"])[metric_col].mean().reset_index())
    pv = g.pivot(index="_grp", columns="dataset_lc", values=metric_col).astype(float)
    pv = pv.reindex([c for c in [1,2,3,4] if c in pv.index])
    x = np.arange(len(pv)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    exp_vals = pv.get("experiment", pd.Series(0, index=pv.index)).to_numpy()
    ingv_vals = pv.get("ingv", pd.Series(0, index=pv.index)).to_numpy()
    b1 = ax.bar(x - w/2, exp_vals, width=w, label="experiment")
    b2 = ax.bar(x + w/2, ingv_vals, width=w, label="ingv")
    ax.set_xticks(x); ax.set_xticklabels([f"{by_col[0].upper()}{int(i)}" for i in pv.index])
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)

    # Save legend separately (Experiment vs INGV)
    handles, labels = ax.get_legend_handles_labels()
    legend_name = OUT_DIR / f"{slugify(title)}_legend.png"
    save_legend_as_png(handles, labels, legend_name, title="Dataset", ncol=2)

    for bars in (b1, b2):
        labels = [f"{v:.2f}" for v in [bar.get_height() for bar in bars]]
        ax.bar_label(bars, labels=labels, padding=2, fontsize=8)
    plt.tight_layout(); plt.show()

# ==========================
# Pipelines (kept + matched)
# ==========================
def run_all_plots(df1: pd.DataFrame, df02: pd.DataFrame, title_suffix: str = ""):
    # Counts — Station × Simulation
    plot_station_sim_grouped(station_sim_pivot(df1,  "experiment"),
        title=f"Correlated Peaks by Station × Simulation — experiment (≥ 0.5){title_suffix}")
    plot_station_sim_grouped(station_sim_pivot(df02, "experiment"),
        title=f"Correlated Peaks by Station × Simulation — experiment (0.2–0.5){title_suffix}")
    plot_station_sim_grouped(station_sim_pivot(df1,  "ingv"),
        title=f"Correlated Peaks by Station × Simulation — INGV (≥ 0.5){title_suffix}")
    plot_station_sim_grouped(station_sim_pivot(df02, "ingv"),
        title=f"Correlated Peaks by Station × Simulation — INGV (0.2–0.5){title_suffix}")

    # Counts — Station × Template
    plot_station_template_grouped(station_template_pivot(df1,  "experiment"),
        title=f"Matched Peaks by Station × Template — experiment (≥ 0.5){title_suffix}")
    plot_station_template_grouped(station_template_pivot(df02, "experiment"),
        title=f"Matched Peaks by Station × Template — experiment (0.2–0.5){title_suffix}")
    plot_station_template_grouped(station_template_pivot(df1,  "ingv"),
        title=f"Matched Peaks by Station × Template — INGV (≥ 0.5){title_suffix}")
    plot_station_template_grouped(station_template_pivot(df02, "ingv"),
        title=f"Matched Peaks by Station × Template — INGV (0.2–0.5){title_suffix}")

    # peak_corr heatmaps
    m,c = corr_pivot(df1,  row="station", col="sim",      dataset_name="experiment"); plot_corr_heatmap(m, c, f"Mean peak_corr — Station × Sim — experiment (≥ 0.5){title_suffix}", "sim", "station")
    m,c = corr_pivot(df02, row="station", col="sim",      dataset_name="experiment"); plot_corr_heatmap(m, c, f"Mean peak_corr — Station × Sim — experiment (0.2–0.5){title_suffix}", "sim", "station")
    m,c = corr_pivot(df1,  row="station", col="sim",      dataset_name="ingv");       plot_corr_heatmap(m, c, f"Mean peak_corr — Station × Sim — INGV (≥ 0.5){title_suffix}", "sim", "station")
    m,c = corr_pivot(df02, row="station", col="sim",      dataset_name="ingv");       plot_corr_heatmap(m, c, f"Mean peak_corr — Station × Sim — INGV (0.2–0.5){title_suffix}", "sim", "station")
    m,c = corr_pivot(df1,  row="station", col="template", dataset_name="experiment"); plot_corr_heatmap(m, c, f"Mean peak_corr — Station × Template — experiment (≥ 0.5){title_suffix}", "template", "station")
    m,c = corr_pivot(df02, row="station", col="template", dataset_name="experiment"); plot_corr_heatmap(m, c, f"Mean peak_corr — Station × Template — experiment (0.2–0.5){title_suffix}", "template", "station")
    m,c = corr_pivot(df1,  row="station", col="template", dataset_name="ingv");       plot_corr_heatmap(m, c, f"Mean peak_corr — Station × Template — INGV (≥ 0.5){title_suffix}", "template", "station")
    m,c = corr_pivot(df02, row="station", col="template", dataset_name="ingv");       plot_corr_heatmap(m, c, f"Mean peak_corr — Station × Template — INGV (0.2–0.5){title_suffix}", "template", "station")

    # peak_corr distributions + dataset compare bars
    boxplot_corr_by(df1,  "sim",      "experiment", f"peak_corr distribution by Sim — experiment (≥ 0.5){title_suffix}")
    boxplot_corr_by(df02, "sim",      "experiment", f"peak_corr distribution by Sim — experiment (0.2–0.5){title_suffix}")
    boxplot_corr_by(df1,  "template", "experiment", f"peak_corr distribution by Template — experiment (≥ 0.5){title_suffix}")
    boxplot_corr_by(df02, "template", "experiment", f"peak_corr distribution by Template — experiment (0.2–0.5){title_suffix}")
    boxplot_corr_by(df1,  "sim",      "ingv",       f"peak_corr distribution by Sim — INGV (≥ 0.5){title_suffix}")
    boxplot_corr_by(df02, "sim",      "ingv",       f"peak_corr distribution by Sim — INGV (0.2–0.5){title_suffix}")
    boxplot_corr_by(df1,  "template", "ingv",       f"peak_corr distribution by Template — INGV (≥ 0.5){title_suffix}")
    boxplot_corr_by(df02, "template", "ingv",       f"peak_corr distribution by Template — INGV (0.2–0.5){title_suffix}")

    dataset_compare_corr_bars(df1,  "sim",      f"Mean peak_corr — experiment vs INGV by Sim (≥ 0.5){title_suffix}")
    dataset_compare_corr_bars(df02, "sim",      f"Mean peak_corr — experiment vs INGV by Sim (0.2–0.5){title_suffix}")
    dataset_compare_corr_bars(df1,  "template", f"Mean peak_corr — experiment vs INGV by Template (≥ 0.5){title_suffix}")
    dataset_compare_corr_bars(df02, "template", f"Mean peak_corr — experiment vs INGV by Template (0.2–0.5){title_suffix}")

    # SNR metrics: dB and linear
    for METRIC_COL in ("snr_db", "snr_linear"):
        metric_name = "SNR (dB)" if METRIC_COL == "snr_db" else "SNR (linear)"
        m,c = metric_pivot(df1,  METRIC_COL, row="station", col="sim", dataset_name="experiment")
        plot_metric_heatmap(m, c, f"Mean {metric_name} — Station × Sim — experiment (≥ 0.5){title_suffix}", "sim", "station", f"mean {metric_name}")
        m,c = metric_pivot(df02, METRIC_COL, row="station", col="sim", dataset_name="experiment")
        plot_metric_heatmap(m, c, f"Mean {metric_name} — Station × Sim — experiment (0.2–0.5){title_suffix}", "sim", "station", f"mean {metric_name}")
        m,c = metric_pivot(df1,  METRIC_COL, row="station", col="sim", dataset_name="ingv")
        plot_metric_heatmap(m, c, f"Mean {metric_name} — Station × Sim — INGV (≥ 0.5){title_suffix}", "sim", "station", f"mean {metric_name}")
        m,c = metric_pivot(df02, METRIC_COL, row="station", col="sim", dataset_name="ingv")
        plot_metric_heatmap(m, c, f"Mean {metric_name} — Station × Sim — INGV (0.2–0.5){title_suffix}", "sim", "station", f"mean {metric_name}")

        m,c = metric_pivot(df1,  METRIC_COL, row="station", col="template", dataset_name="experiment")
        plot_metric_heatmap(m, c, f"Mean {metric_name} — Station × Template — experiment (≥ 0.5){title_suffix}", "template", "station", f"mean {metric_name}")
        m,c = metric_pivot(df02, METRIC_COL, row="station", col="template", dataset_name="experiment")
        plot_metric_heatmap(m, c, f"Mean {metric_name} — Station × Template — experiment (0.2–0.5){title_suffix}", "template", "station", f"mean {metric_name}")
        m,c = metric_pivot(df1,  METRIC_COL, row="station", col="template", dataset_name="ingv")
        plot_metric_heatmap(m, c, f"Mean {metric_name} — Station × Template — INGV (≥ 0.5){title_suffix}", "template", "station", f"mean {metric_name}")
        m,c = metric_pivot(df02, METRIC_COL, row="station", col="template", dataset_name="ingv")
        plot_metric_heatmap(m, c, f"Mean {metric_name} — Station × Template — INGV (0.2–0.5){title_suffix}", "template", "station", f"mean {metric_name}")

        boxplot_metric_by(df1,  METRIC_COL, "sim",      "experiment", f"{metric_name} distribution by Sim — experiment (≥ 0.5){title_suffix}", metric_name)
        boxplot_metric_by(df02, METRIC_COL, "sim",      "experiment", f"{metric_name} distribution by Sim — experiment (0.2–0.5){title_suffix}", metric_name)
        boxplot_metric_by(df1,  METRIC_COL, "template", "experiment", f"{metric_name} distribution by Template — experiment (≥ 0.5){title_suffix}", metric_name)
        boxplot_metric_by(df02, METRIC_COL, "template", "experiment", f"{metric_name} distribution by Template — experiment (0.2–0.5){title_suffix}", metric_name)

        boxplot_metric_by(df1,  METRIC_COL, "sim",      "ingv",       f"{metric_name} distribution by Sim — INGV (≥ 0.5){title_suffix}", metric_name)
        boxplot_metric_by(df02, METRIC_COL, "sim",      "ingv",       f"{metric_name} distribution by Sim — INGV (0.2–0.5){title_suffix}", metric_name)
        boxplot_metric_by(df1,  METRIC_COL, "template", "ingv",       f"{metric_name} distribution by Template — INGV (≥ 0.5){title_suffix}", metric_name)
        boxplot_metric_by(df02, METRIC_COL, "template", "ingv",       f"{metric_name} distribution by Template — INGV (0.2–0.5){title_suffix}", metric_name)

        dataset_compare_metric_bars(df1,  METRIC_COL, "sim",      f"Mean {metric_name} — experiment vs INGV by Sim (≥ 0.5){title_suffix}", metric_name)
        dataset_compare_metric_bars(df02, METRIC_COL, "sim",      f"Mean {metric_name} — experiment vs INGV by Sim (0.2–0.5){title_suffix}", metric_name)
        dataset_compare_metric_bars(df1,  METRIC_COL, "template", f"Mean {metric_name} — experiment vs INGV by Template (≥ 0.5){title_suffix}", metric_name)
        dataset_compare_metric_bars(df02, METRIC_COL, "template", f"Mean {metric_name} — experiment vs INGV by Template (0.2–0.5){title_suffix}", metric_name)

# ==========================
# Load SWCC Clean Peak CSVs
# ==========================
def load_all_swcc_peak_csvs(peaks_root: Path) -> pd.DataFrame:
    """
    Load all clean peak CSVs from SWCC pipeline.

    Expected file structure:
    SWCC_p_wave_cleaned/
      ├── ingv/
      │   └── {station}/
      │       └── {sim}/
      │           └── {station}_{sim}_{template}_peaks.csv
      └── experiment/
          └── {station}/
              └── {sim}/
                  └── {station}_{sim}_{template}_peaks.csv

    Returns combined DataFrame with metadata columns: dataset, station, sim, template
    """
    all_peaks = []

    # Find all peak CSV files recursively
    peak_files = list(peaks_root.rglob("*_peaks.csv"))

    if not peak_files:
        raise FileNotFoundError(f"No peak CSV files found in {peaks_root}")

    print(f"[info] Found {len(peak_files)} peak CSV files")

    for csv_path in peak_files:
        # Extract metadata from path
        # Path structure: SWCC_p_wave_cleaned/{dataset}/{station}/{sim}/{station}_{sim}_{template}_peaks.csv
        parts = csv_path.parts

        # Find the index of the root directory
        try:
            root_idx = parts.index(peaks_root.name)
        except ValueError:
            print(f"[warning] Skipping {csv_path} - unexpected path structure")
            continue

        # Extract dataset, station, sim from path
        if len(parts) > root_idx + 3:
            dataset = parts[root_idx + 1]  # ingv or experiment
            station = parts[root_idx + 2]  # e.g., ECPN
            sim = parts[root_idx + 3]      # e.g., sim1

            # Extract template from filename
            # Format: {station}_{sim}_{template}_peaks.csv
            filename = csv_path.stem  # Remove .csv
            match = re.search(r'template(\d+)_peaks$', filename)
            if match:
                template = f"template{match.group(1)}"
            else:
                print(f"[warning] Could not extract template from {csv_path.name}")
                continue
        else:
            print(f"[warning] Skipping {csv_path} - path too short")
            continue

        # Load CSV
        try:
            df = pd.read_csv(csv_path)

            # Add metadata columns
            df['dataset'] = dataset
            df['station'] = station
            df['sim'] = sim
            df['template'] = template

            all_peaks.append(df)
        except Exception as e:
            print(f"[warning] Error loading {csv_path}: {e}")
            continue

    if not all_peaks:
        raise ValueError("No valid peak CSV files could be loaded")

    # Combine all dataframes
    combined = pd.concat(all_peaks, ignore_index=True)

    print(f"[info] Loaded {len(combined)} total peaks from {len(all_peaks)} files")
    print(f"[info] Datasets: {combined['dataset'].unique()}")
    print(f"[info] Stations: {sorted(combined['station'].unique())}")

    return combined

# ==========================
# Main
# ==========================
def main():
    # Load all clean peak CSVs from SWCC pipeline
    df_all = load_all_swcc_peak_csvs(PEAKS_ROOT)
    
    # Split by peak_corr: df02 has 0.2 <= peak_corr < 0.5, df05 has peak_corr >= 0.5
    df_all["peak_corr"] = pd.to_numeric(df_all["peak_corr"], errors="coerce")
    df02 = df_all[(df_all["peak_corr"] >= 0.2) & (df_all["peak_corr"] < 0.5)].copy()
    df05 = df_all[df_all["peak_corr"] >= 0.5].copy()

    # Use seg_duration_s if available, otherwise compute duration
    if "seg_duration_s" in df02.columns:
        df02["duration_s"] = df02["seg_duration_s"]
    else:
        df02 = compute_duration_s(df02, START_COL, END_COL, "duration", "duration_s")
    
    if "seg_duration_s" in df05.columns:
        df05["duration_s"] = df05["seg_duration_s"]
    else:
        df05 = compute_duration_s(df05, START_COL, END_COL, "duration", "duration_s")
    
    df1 = build_df1_from_df05(df05)

    # Normalize stations
    if "station" in df02.columns: df02["station"] = clean_station(df02["station"])
    if "station" in df1.columns:  df1["station"]  = clean_station(df1["station"])

    # Load earthquakes
    quakes = load_quake_arrivals_from_file(QUAKE_FILE)

    # Match/split
    df02_kept, df02_matched = match_split_against_quakes(df02, quakes, TOLERANCE_SEC)
    df1_kept,  df1_matched  = match_split_against_quakes(df1,  quakes, TOLERANCE_SEC)

    print(f"[info] df02 total={len(df02)} kept={len(df02_kept)} matched={len(df02_matched)}")
    print(f"[info] df1  total={len(df1)}  kept={len(df1_kept)}  matched={len(df1_matched)}")

    # Save matched details for inspection
    out_dir = Path(".")
    df02_matched.to_csv(out_dir / "df02_earthquake_matched.csv", index=False)
    df1_matched.to_csv(out_dir / "df1_earthquake_matched.csv", index=False)

    # Plot filtered (kept) as before (legendless + legend PNGs)
    run_all_plots(df1_kept, df02_kept, title_suffix=" — filtered")

    # Plot earthquake-matched events as their own cohort
    run_all_plots(df1_matched, df02_matched, title_suffix=" — earthquake-matched")

if __name__ == "__main__":
    main()


# In[8]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
swcc_conditioned_plots.py

Builds a *conditioned* dataframe using this logic:
  - peak_corr >= 0.5
  - snr_linear > 2
  - NOT matched with earthquake events (±TOLERANCE_SEC window)

Also extracts rows where total duration exceeds the template length.
Both the filtered dataframe and the "duration exceeds template" subset
are saved to CSV and plotted (legendless; separate legend PNGs are exported).

Inputs:
  EXP02 = /home/owen/Etna_signals/DATA/SWCC_0.2/EXP_PEAKS.csv
  INGV02 = /home/owen/Etna_signals/DATA/SWCC_0.2/INGV_PEAKS.csv
  CSV_05 = /home/owen/Etna_signals/ALL_0.5_PEAKS1.csv
  QUAKE_FILE = /home/owen/Etna_signals/earthquake_data/earthquakes.csv

Outputs:
  - conditioned_noquakes_ge05_snr2.csv
  - duration_exceeds_template.csv
  - figures_conditioned/*.png (figures + legend assets)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# Config
# ==========================
PEAKS_ROOT = Path("/home/owen/tilt_validation/SWCC_utc_fixed")
QUAKE_FILE = Path("/home/owen/tilt_validation/earthquakes_merged_utc.csv")

START_COL = "seg_start_time"
END_COL   = "seg_end_time"

TOLERANCE_SEC = 600
LABEL_ROT_DEG = 15
FIG_HEIGHT = 6

OUT_DIR = Path("figures_conditioned")
OUT_DIR.mkdir(exist_ok=True)

# ==========================
# Small utils
# ==========================
def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    s = re.sub(r"_+", "_", s)
    return s

def save_legend_as_png(handles, labels, filename: Path, title: str | None = None, ncol: int | None = None):
    if not handles or not labels:
        return
    fig = plt.figure(figsize=(6, 1.0))
    leg = fig.legend(handles, labels, loc="center", ncol=ncol if ncol else len(labels), frameon=False, title=title)
    if leg is not None and leg.get_title():
        leg.get_title().set_fontsize(10)
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.axis("off")
    fig.savefig(filename, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

def clean_dataset(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def clean_station(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def parse_timedelta_safe(s: pd.Series) -> pd.Series:
    return pd.to_timedelta(s, errors="coerce")

def compute_duration_s(df: pd.DataFrame, start_col: str = START_COL, end_col: str = END_COL,
                       out_td: str = "duration", out_sec: str = "duration_s") -> pd.DataFrame:
    if start_col not in df.columns or end_col not in df.columns:
        df[out_td] = pd.NaT
        df[out_sec] = np.nan
        return df
    start = parse_timedelta_safe(df[start_col])
    end   = parse_timedelta_safe(df[end_col])
    dur = end - start
    dur = dur.mask(dur < pd.Timedelta(0), dur + pd.Timedelta(days=1))
    df[out_td] = dur
    df[out_sec] = df[out_td].dt.total_seconds()
    return df

def norm_sim_to_int(sim_series: pd.Series) -> pd.Series:
    return sim_series.astype(str).str.extract(r"(\d+)")[0].astype("Int64")

def _norm_template_to_int(template_series: pd.Series) -> pd.Series:
    return template_series.astype(str).str.extract(r"(\d+)")[0].astype("Int64")

def _to_numeric_series(s):
    return pd.to_numeric(s, errors="coerce")

# ==========================
# Build unified event datetime per row
# ==========================
def coalesce_event_dt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "seg_start_dt" in df.columns:
        dt1 = pd.to_datetime(df["seg_start_dt"], errors="coerce")
    else:
        dt1 = pd.Series(pd.NaT, index=df.index)
    if "seg_end_dt" in df.columns:
        dt2 = pd.to_datetime(df["seg_end_dt"], errors="coerce")
    else:
        dt2 = pd.Series(pd.NaT, index=df.index)
    event_dt = dt1.fillna(dt2)
    if event_dt.isna().any() and {"peak_date","peak_time"}.issubset(df.columns):
        combo = (df["peak_date"].astype(str).str.strip() + " " +
                 df["peak_time"].astype(str).str.strip())
        event_dt = event_dt.fillna(pd.to_datetime(combo, errors="coerce"))
    if event_dt.isna().any():
        cand_cols = [c for c in df.columns if re.search(r"(dt|date|time)$", c, flags=re.I)]
        for c in cand_cols:
            try:
                tmp = pd.to_datetime(df[c], errors="coerce")
                event_dt = event_dt.fillna(tmp)
            except Exception:
                pass
    df["event_dt"] = event_dt
    return df

# ==========================
# Earthquake arrival loader
# ==========================
def load_quake_arrivals_from_file(quake_file: Path) -> pd.DataFrame:
    q = pd.read_csv(quake_file)
    cols = {c.lower(): c for c in q.columns}
    arrival = None
    # Prioritize p_wave_eta if available (P-wave arrival time)
    if "p_wave_eta" in cols:
        arrival = pd.to_datetime(q[cols["p_wave_eta"]], errors="coerce")
    elif "event_time" in cols:
        arrival = pd.to_datetime(q[cols["event_time"]], errors="coerce")
    elif "date" in cols and "time" in cols:
        combo = q[cols["date"]].astype(str).str.strip() + " " + q[cols["time"]].astype(str).str.strip()
        arrival = pd.to_datetime(combo, errors="coerce")
    else:
        for key in ["origin_time","timestamp","datetime","time_utc","time_local","otime"]:
            if key in cols:
                arrival = pd.to_datetime(q[cols[key]], errors="coerce")
                break
    if arrival is None:
        raise ValueError("Could not locate time columns in earthquake file.")
    out = pd.DataFrame({"arrival_dt": arrival})
    for st_key in ["station","sta","station_code","Station","STA"]:
        if st_key in q.columns:
            out["station"] = clean_station(q[st_key])
            break
    out = out.dropna(subset=["arrival_dt"]).sort_values("arrival_dt").reset_index(drop=True)
    return out

# ==========================
# Matching / splitting
# ==========================
def match_split_against_quakes(df: pd.DataFrame, quakes: pd.DataFrame, tolerance_sec: int):
    """
    Returns (kept_df, matched_df).
    matched_df has extra columns: matched_arrival_dt, match_delta_s.
    """
    if df.empty or quakes.empty:
        return df.copy(), df.head(0).copy()

    d = df.copy()
    if "station" in d.columns:
        d["station"] = clean_station(d["station"])
    d = coalesce_event_dt(d)
    d = d.dropna(subset=["event_dt"]).sort_values("event_dt").reset_index(drop=True)

    q = quakes.copy()
    tol = pd.Timedelta(seconds=int(tolerance_sec))

    if "station" in q.columns and "station" in d.columns:
        kept_chunks, match_chunks = [], []
        for st, d_g in d.groupby("station", sort=False):
            q_g = q[q["station"] == st][["arrival_dt"]].sort_values("arrival_dt")
            if q_g.empty:
                kept_chunks.append(d_g)
                continue
            merged = pd.merge_asof(d_g, q_g, left_on="event_dt", right_on="arrival_dt",
                                   direction="nearest", tolerance=tol)
            matched = merged[merged["arrival_dt"].notna()].copy()
            if not matched.empty:
                delta = (matched["event_dt"] - matched["arrival_dt"]).dt.total_seconds()
                matched["matched_arrival_dt"] = matched["arrival_dt"]
                matched["match_delta_s"] = delta
                matched = matched.drop(columns=["arrival_dt"])
                match_chunks.append(matched)
            kept = merged[merged["arrival_dt"].isna()].drop(columns=["arrival_dt"])
            kept_chunks.append(kept)
        kept_df = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else d.head(0)
        matched_df = pd.concat(match_chunks, ignore_index=True) if match_chunks else d.head(0)
    else:
        # Global time-only matching
        q_g = q[["arrival_dt"]].sort_values("arrival_dt")
        merged = pd.merge_asof(d, q_g, left_on="event_dt", right_on="arrival_dt",
                               direction="nearest", tolerance=tol)
        matched_df = merged[merged["arrival_dt"].notna()].copy()
        kept_df    = merged[merged["arrival_dt"].isna()].drop(columns=["arrival_dt"])
        if not matched_df.empty:
            matched_df["matched_arrival_dt"] = matched_df["arrival_dt"]
            matched_df["match_delta_s"] = (matched_df["event_dt"] - matched_df["arrival_dt"]).dt.total_seconds()
            matched_df = matched_df.drop(columns=["arrival_dt"])

    return kept_df, matched_df

# ==========================
# Template length handling
# ==========================
def _parse_seconds_maybe(series: pd.Series) -> pd.Series:
    """Try numeric first; else parse as timedelta-like; return seconds (float)."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.astype(float)
    td = pd.to_timedelta(series, errors="coerce")
    return td.dt.total_seconds()

def find_template_length_seconds(df: pd.DataFrame) -> pd.Series:
    """
    Look for a template length column under common names. Return seconds as float.
    """
    candidates = [
        "template_length_s","template_len_s","template_duration_s",
        "template_length","template_len","template_duration","templength",
        "tmpl_length","tmpl_len","tmpl_duration"
    ]
    for c in candidates:
        if c in df.columns:
            sec = _parse_seconds_maybe(df[c])
            if sec.notna().any():
                return sec
    # sometimes template is indicated by id 1..4 with fixed mapping
    if "template" in df.columns:
        # Fallback: user may edit these if known
        mapping = {1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan}
        tmpl = _norm_template_to_int(df["template"])
        return tmpl.map(mapping).astype(float)
    return pd.Series([np.nan]*len(df), index=df.index)

# ==========================
# Plotting helpers (legendless + legend PNG export)
# ==========================
def _bar_labels(ax, bars, values):
    labels = [f"{int(v)}" if float(v) > 0 else "" for v in values]
    ax.bar_label(bars, labels=labels, padding=2, fontsize=8)

def station_sim_pivot(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if df.empty: 
        return pd.DataFrame()
    mask = clean_dataset(df["dataset"]).eq(dataset_name.lower())
    if "sim" not in df.columns or "station" not in df.columns:
        return pd.DataFrame()
    sim_i = norm_sim_to_int(df.loc[mask, "sim"])
    tmp = (df.loc[mask, ["station"]].assign(sim=sim_i.values).dropna(subset=["sim"]))
    if tmp.empty: 
        return pd.DataFrame()
    counts = (tmp.groupby(["station","sim"], dropna=False).size().rename("n").reset_index())
    piv = counts.pivot(index="station", columns="sim", values="n").fillna(0).astype(int)
    ordered = [c for c in [1,2,3,4] if c in piv.columns]
    return piv.reindex(columns=ordered)

def plot_station_sim_grouped(pivot: pd.DataFrame, title: str):
    if pivot is None or pivot.empty:
        print(f"[info] No data to plot for: {title}")
        return
    totals = pivot.sum(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[totals.index]
    stations = pivot.index.astype(str).tolist()
    sim_cols = list(pivot.columns)
    values   = pivot.values
    n_st, n_sims = values.shape
    x = np.arange(n_st)
    width = 0.8 / max(1, n_sims)
    fig_w = max(10, min(28, 0.25 * n_st))
    fig, ax = plt.subplots(figsize=(fig_w, FIG_HEIGHT))
    cluster_tops = np.zeros(n_st, dtype=float)
    handle_cache = []
    for j, sim in enumerate(sim_cols):
        offs = (j - (n_sims - 1)/2) * width
        bars = ax.bar(x + offs, values[:, j], width=width, label=f"sim{int(sim)}", zorder=2)
        _bar_labels(ax, bars, values[:, j])
        cluster_tops = np.maximum(cluster_tops, values[:, j])
        handle_cache.append(bars[0])
    ax.set_xticks(x)
    ax.set_xticklabels(stations, rotation=LABEL_ROT_DEG, ha="right")
    ax.set_xlabel("Station")
    ax.set_ylabel("No. of correlated peaks")
    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35, zorder=1)

    labels = [f"sim{int(s)}" for s in sim_cols]
    legend_name = OUT_DIR / f"{slugify(title)}_legend.png"
    save_legend_as_png(handle_cache, labels, legend_name, title="Simulation", ncol=min(4, len(labels)))

    ymax = int(values.max()) if values.size else 1
    step = max(1, int(np.ceil(max(1, ymax)/5)))
    ax.set_yticks(np.arange(0, max(1, ymax)+step, step))
    ax.margins(y=0.1)
    # totals_arr = pivot.sum(axis=1).to_numpy()
    # y_off = max(1, int(np.ceil(ymax/20))) if ymax > 0 else 1
    # for xi, ytop, tot in zip(x, cluster_tops, totals_arr):
    # if tot > 0:
            # ax.text(xi, ytop + y_off, str(int(tot)), ha="center", va="bottom", fontsize=9)
    plt.subplots_adjust(bottom=0.22)
    plt.tight_layout()
    plt.show()

def station_template_pivot(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    mask = clean_dataset(df["dataset"]).eq(dataset_name.lower())
    if "template" not in df.columns or "station" not in df.columns:
        return pd.DataFrame()
    tmpl_i = _norm_template_to_int(df.loc[mask, "template"])
    tmp = (df.loc[mask, ["station"]].assign(template=tmpl_i.values).dropna(subset=["template"]))
    if tmp.empty:
        return pd.DataFrame()
    counts = (tmp.groupby(["station","template"], dropna=False).size().rename("n").reset_index())
    piv = counts.pivot(index="station", columns="template", values="n").fillna(0).astype(int)
    ordered = [c for c in [1,2,3,4] if c in piv.columns]
    return piv.reindex(columns=ordered)

def plot_station_template_grouped(pivot: pd.DataFrame, title: str):
    if pivot is None or pivot.empty:
        print(f"[info] No data to plot for: {title}")
        return
    totals = pivot.sum(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[totals.index]
    stations = pivot.index.astype(str).tolist()
    tmpl_cols = list(pivot.columns)
    values    = pivot.values
    n_st, n_tm = values.shape
    x = np.arange(n_st)
    width = 0.8 / max(1, n_tm)
    fig_w = max(10, min(28, 0.25 * n_st))
    fig, ax = plt.subplots(figsize=(fig_w, FIG_HEIGHT))
    cluster_tops = np.zeros(n_st, dtype=float)
    handle_cache = []
    for j, t in enumerate(tmpl_cols):
        offs = (j - (n_tm - 1)/2) * width
        bars = ax.bar(x + offs, values[:, j], width=width, label=f"T{int(t)}", zorder=2)
        _bar_labels(ax, bars, values[:, j])
        cluster_tops = np.maximum(cluster_tops, values[:, j])
        handle_cache.append(bars[0])
    ax.set_xticks(x)
    ax.set_xticklabels(stations, rotation=LABEL_ROT_DEG, ha="right")
    ax.set_xlabel("Station")
    ax.set_ylabel("No. of matched peaks")
    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35, zorder=1)

    labels = [f"T{int(t)}" for t in tmpl_cols]
    legend_name = OUT_DIR / f"{slugify(title)}_legend.png"
    save_legend_as_png(handle_cache, labels, legend_name, title="Template", ncol=min(4, len(labels)))

    ymax = int(values.max()) if values.size else 1
    step = max(1, int(np.ceil(max(1, ymax)/5)))
    ax.set_yticks(np.arange(0, max(1, ymax)+step, step))
    ax.margins(y=0.1)
    # totals_arr = pivot.sum(axis=1).to_numpy()
    # y_off = max(1, int(np.ceil(ymax/20))) if ymax > 0 else 1
    # for xi, ytop, tot in zip(x, cluster_tops, totals_arr):
    # if tot > 0:
            # ax.text(xi, ytop + y_off, str(int(tot)), ha="center", va="bottom", fontsize=9)
    plt.subplots_adjust(bottom=0.22)
    plt.tight_layout()
    plt.show()

# ==========================
# Pipeline
# ==========================
def main():
    # Load all clean peak CSVs from SWCC pipeline
    df_all = load_all_swcc_peak_csvs(PEAKS_ROOT)

    # Split by peak_corr: df02 has 0.2 <= peak_corr < 0.5, df05 has peak_corr >= 0.5
    df_all["peak_corr"] = pd.to_numeric(df_all["peak_corr"], errors="coerce")
    df02 = df_all[(df_all["peak_corr"] >= 0.2) & (df_all["peak_corr"] < 0.5)].copy()
    df05 = df_all[df_all["peak_corr"] >= 0.5].copy()

    # Use seg_duration_s if available, otherwise compute duration
    if "seg_duration_s" in df02.columns:
        df02["duration_s"] = df02["seg_duration_s"]
    else:
        df02 = compute_duration_s(df02, START_COL, END_COL, "duration", "duration_s")
    
    if "seg_duration_s" in df05.columns:
        df05["duration_s"] = df05["seg_duration_s"]
    else:
        df05 = compute_duration_s(df05, START_COL, END_COL, "duration", "duration_s")

    # We'll condition on ≥0.5 + snr filter, so start from df05-like schema if needed
    # Coalesce datetimes
    df02 = coalesce_event_dt(df02)
    df05 = coalesce_event_dt(df05)

    # Clean station and dataset
    if "station" in df02.columns: df02["station"] = clean_station(df02["station"])
    if "station" in df05.columns: df05["station"] = clean_station(df05["station"])
    if "dataset" in df02.columns: df02["dataset"] = clean_dataset(df02["dataset"])
    if "dataset" in df05.columns: df05["dataset"] = clean_dataset(df05["dataset"])

    # Load earthquakes and split matches (keep only NOT matched rows for conditioning)
    quakes = load_quake_arrivals_from_file(QUAKE_FILE)
    df02_kept, _ = match_split_against_quakes(df02, quakes, TOLERANCE_SEC)
    df05_kept, _ = match_split_against_quakes(df05, quakes, TOLERANCE_SEC)

    # Apply conditional filter: peak_corr >= 0.5 AND snr_linear > 2
    for d in (df02_kept, df05_kept):
        d["peak_corr"] = _to_numeric_series(d.get("peak_corr"))
        d["snr_linear"] = _to_numeric_series(d.get("snr_linear"))

    conditioned = pd.concat([
        df05_kept[(df05_kept["peak_corr"] >= 0.5) & (df05_kept["snr_linear"] > 4)],
        # include any >=0.2 rows that also cross thresholds (if df05 was already the ≥0.5 corpus, this adds nothing)
        df02_kept[(df02_kept["peak_corr"] >= 0.5) & (df02_kept["snr_linear"] > 4)]
    ], ignore_index=True).drop_duplicates()

    # Duration vs template length subset
    template_len_s = find_template_length_seconds(conditioned)
    duration_s = _to_numeric_series(conditioned.get("duration_s"))
    exceeds_mask = (duration_s.notna() & template_len_s.notna() & (duration_s > template_len_s))
    exceeds_df = conditioned.loc[exceeds_mask].copy()

    # Save dataframes
    conditioned.to_csv("conditioned_noquakes_ge05_snr2.csv", index=False)
    exceeds_df.to_csv("duration_exceeds_template.csv", index=False)

    print(f"[info] Conditioned rows (no-quake, corr>=0.5, snr>2): {len(conditioned)}")
    print(f"[info] Duration exceeds template rows: {len(exceeds_df)}")

    # Plotting (legendless) — Counts by Station×Sim and Station×Template for the conditioned df
    if not conditioned.empty:
        # Ensure sim/template integers
        if "sim" in conditioned.columns:
            conditioned["sim"] = norm_sim_to_int(conditioned["sim"])
        if "template" in conditioned.columns:
            conditioned["template"] = _norm_template_to_int(conditioned["template"])

        for ds in ["experiment", "ingv"]:
            piv_sim = station_sim_pivot(conditioned, ds)
            plot_station_sim_grouped(piv_sim, f"Conditioned counts by Station × Simulation — {ds}")

            piv_tmpl = station_template_pivot(conditioned, ds)
            plot_station_template_grouped(piv_tmpl, f"Conditioned counts by Station × Template — {ds}")

    # Optional: you could add heatmaps/boxplots similar to other scripts if desired.

if __name__ == "__main__":
    main()


# In[10]:


# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import datetime
from scipy import signal
import numpy as np
import pandas as pd 
import os
import scipy
import scipy.signal as signal
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from scipy.signal import detrend
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab
import scipy.fftpack
import pylab
from numpy import pi
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from dtaidistance import dtw
from scipy.stats import pearsonr
from scipy.signal import detrend
from scipy.fft import fft, fftfreq
from matplotlib import pyplot 
from geopy.distance import geodesic
import glob
from pathlib import Path
from scipy import signal
from scipy.signal import find_peaks

# ------------------------------
# 1. Load Tilt Data
# ------------------------------
def load_all_simulations(sim_dirs: dict, stations: list[str], verbose: bool = True) -> dict:
    def _read(fp: Path) -> np.ndarray | None:
        try:
            data = np.loadtxt(fp)
            if data.ndim < 2 or data.shape[1] < 2:
                raise ValueError("needs ≥2 columns")
            return data[:, 1][~np.isnan(data[:, 1])]
        except Exception as e:
            if verbose:
                print(f"⚠️  {fp.name}: {e}")
            return None

    out: dict[str, dict[str, np.ndarray]] = {}
    for sim_label, root in sim_dirs.items():
        root = Path(root)
        sim_dict: dict[str, np.ndarray] = {}
        for st in stations:
            fp = root / f"{st}.txt"
            if not fp.exists():
                if verbose:
                    print(f"❌  File not found: {fp}")
                continue
            arr = _read(fp)
            if arr is not None:
                sim_dict[st] = arr
                if verbose:
                    print(f"✅  {sim_label}/{st}: {arr.size} samples")
        out[sim_label] = sim_dict
    return out
# ------------------------------
# 2. Filter Tilt Data
# ------------------------------
def filter_all_sim_data(tilt_data: dict, fs: float = 1.0) -> dict:
    fc_highpass = 1e-5
    fc_low = 0.001
    fc_high = 0.01
    w_low = fc_low / (fs / 2)
    w_high = fc_high / (fs / 2)
    sos_bandpass = signal.butter(4, [w_low, w_high], btype="bandpass", output="sos")

    filtered_data = {}
    for sim_label, station_dict in tilt_data.items():
        filtered_data[sim_label] = {}
        for station, raw_signal in station_dict.items():
            try:
                filtered_signal = signal.sosfiltfilt(sos_bandpass, raw_signal)
                filtered_data[sim_label][station] = filtered_signal
            except Exception as e:
                print(f"❌ Error filtering {sim_label}/{station}: {e}")
    return filtered_data
# ------------------------------
# 3. Plot Tilt and Spectrograms
# ------------------------------
def plot_station_simulation_comparisons(tilt_data_raw, tilt_data_filt, stations, sim_labels, fs=1):
    NFFT = 512
    for station in stations:
        print(f"\n🔍 Plotting tilt magnitude for station: {station}")

        plt.figure(figsize=(14, 4))
        for sim_label in sim_labels:
            data = tilt_data_raw[sim_label].get(station)
            if data is not None:
                plt.plot(data, label=f"{sim_label} - raw", alpha=0.5)
        plt.title(f"Tilt Magnitude (Raw) - {station}")
        plt.xlabel("Time (s)")
        plt.ylabel("Tilt")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 4))
        for sim_label in sim_labels:
            data = tilt_data_filt[sim_label].get(station)
            if data is not None:
                plt.plot(data, label=f"{sim_label} - filtered", alpha=0.7)
        plt.title(f"Tilt Magnitude (Filtered) - {station}")
        plt.xlabel("Time (s)")
        plt.ylabel("Tilt (Filtered)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        for sim_label in sim_labels:
            raw_data = tilt_data_raw[sim_label].get(station)
            filt_data = tilt_data_filt[sim_label].get(station)

            if raw_data is not None:
                plt.figure(figsize=(12, 4))
                plt.specgram(raw_data, NFFT=NFFT, Fs=fs)
                plt.title(f"Spectrogram (Raw) - {station} - {sim_label}")
                plt.xlabel("Time (s)")
                plt.ylabel("Frequency (Hz)")
                plt.tight_layout()
                plt.show()

            if filt_data is not None:
                plt.figure(figsize=(12, 4))
                plt.specgram(filt_data, NFFT=NFFT, Fs=fs)
                plt.title(f"Spectrogram (Filtered) - {station} - {sim_label}")
                plt.xlabel("Time (s)")
                plt.ylabel("Frequency (Hz)")
                plt.tight_layout()
                plt.show()
                
                # ------------------------------
# 4. Plot FFTs
# ------------------------------
def plot_frequency_spectrum_comparisons(tilt_data_filt, stations, sim_labels, fs=1):
    for station in stations:
        print(f"\n📊 Frequency-Amplitude Spectrum for Station: {station}")

        plt.figure(figsize=(14, 4))
        for sim_label in sim_labels:
            data = tilt_data_filt[sim_label].get(station)
            if data is not None:
                freqs = fftfreq(len(data), 1/fs)
                fft_vals = np.abs(fft(data))
                plt.plot(freqs[:len(freqs)//2], fft_vals[:len(freqs)//2], label=f"{sim_label} - Tilt")
        plt.title(f"Frequency Spectrum - Tilt (Filtered) - {station}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
# ------------------------------
# 5. Execute Full Workflow
# ------------------------------
#if __name__ == "__main__":
sim_dirs = {
        "sim1": "/home/owen/Signal_Validation/solid_dofs/tilt/sim1/tilt/",
        "sim2": "/home/owen/Signal_Validation/solid_dofs/tilt/sim2/tilt/",
        "sim3": "/home/owen/Signal_Validation/solid_dofs/tilt/sim3/tilt/",
        "sim4": "/home/owen/Signal_Validation/solid_dofs/tilt/sim4/tilt/"
    }

stations = ["ECPN", "EC1", "ECIT", "EC10", "ECOR", "EMAS"]
#stations = ["ECPN","EC1"]
sim_labels = list(sim_dirs.keys())

tilt_data_raw = load_all_simulations(sim_dirs, stations)
tilt_data_filt = filter_all_sim_data(tilt_data_raw, fs=1.0)

#plot_station_simulation_comparisons(tilt_data_raw, tilt_data_filt, stations, sim_labels)
#plot_frequency_spectrum_comparisons(tilt_data_filt, stations, sim_labels)


# In[2]:


# ---------- Loader (rebuild nested dict from disk) ----------
def load_tilt_templates(outdir: str, csv_dir: str = None):
    """
    Load templates from .npy/.npz files OR from CSV files.
    
    For .npy/.npz format:
      templates[tpl]['sim']['station'] -> {'x', 't', 'i0','i1','t0','t1','t0_nom','t1_nom'}
    
    For CSV format (filename: {STATION}_{SIM}_{TEMPLATE}_{FILTER}_tpl.csv):
      Expects columns: time_seconds, x
      Example: EC1_sim1_template1_0p001-0p01Hz_tpl.csv
    
    Parameters:
    -----------
    outdir : str
        Directory containing template folders (T1/, T2/, T3/, T4/) with .npy files
    csv_dir : str, optional
        Directory containing CSV template files (flat structure)
        If provided, CSV files will be loaded in addition to .npy files
    
    Returns:
    --------
    (templates, all_stations, all_sims)
    """
    templates = {}
    all_sims, all_stations = set(), set()

    # ========== Load .npy/.npz templates (original format) ==========
    for tpl in ["T1","T2","T3","T4"]:
        tpl_dir = os.path.join(outdir, tpl)
        if not os.path.isdir(tpl_dir):
            continue
        templates[tpl] = {}
        # Find base names (pairs of .npy and _meta.npz)
        for npy in glob.glob(os.path.join(tpl_dir, "*.npy")):
            base = os.path.splitext(os.path.basename(npy))[0]
            meta_path = os.path.join(tpl_dir, base + "_meta.npz")
            if not os.path.isfile(meta_path):
                continue
            parts = base.split("_")
            if len(parts) < 2:
                sim, station = base, "UNKNOWN"
            else:
                station = parts[-1]
                sim = "_".join(parts[:-1])
            all_sims.add(sim); all_stations.add(station)

            x = np.load(npy)
            meta = dict(np.load(meta_path))
            if "t" in meta:
                t = meta["t"]
            else:
                t0 = float(meta.get("t0", 0.0))
                t1 = float(meta.get("t1", t0 + len(x)))
                if len(x) > 1:
                    t = np.linspace(t0, t1, num=len(x), endpoint=False)
                else:
                    t = np.array([t0], float)

            payload = {"x": x, "t": t}
            for k in ("i0","i1","t0","t1","t0_nom","t1_nom"):
                if k in meta:
                    payload[k] = meta[k].item() if hasattr(meta[k], "item") else meta[k]

            templates[tpl].setdefault(sim, {})[station] = payload

    # ========== Load CSV templates (new format) ==========
    if csv_dir and os.path.isdir(csv_dir):
        csv_files = glob.glob(os.path.join(csv_dir, "*_tpl.csv"))
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            base = filename.replace("_tpl.csv", "")
            parts = base.split("_")
            
            if len(parts) < 3:
                continue
            
            station = parts[0].upper()
            sim = parts[1]
            template_part = parts[2]
            
            # Station name aliases (CSV has EC1, but events may use EEC1)
            station_aliases = {
                "EC1": ["EC1", "EEC1"],  # Store under both names
                "EEC1": ["EC1", "EEC1"],
                "ECPN": ["ECPN"],
                "EMAS": ["EMAS"],
                "ECOR": ["ECOR"],
                "ECIT": ["ECIT"],
                "EC10": ["EC10"],
            }
            
            template_map_csv = {
                "template1": "T1",
                "template2": "T2", 
                "template3": "T3",
                "template4": "T4"
            }
            tpl_key = template_map_csv.get(template_part.lower())
            if not tpl_key:
                continue
            
            try:
                df = pd.read_csv(csv_file)
                if "time_seconds" not in df.columns or "x" not in df.columns:
                    continue
                
                t = df["time_seconds"].values
                x = df["x"].values
                
                payload = {
                    "x": x,
                    "t": t,
                    "t0": t[0] if len(t) > 0 else 0.0,
                    "t1": t[-1] if len(t) > 0 else 0.0,
                }
                
                if tpl_key not in templates:
                    templates[tpl_key] = {}
                
                # Store under all alias names
                alias_names = station_aliases.get(station, [station])
                for alias in alias_names:
                    templates[tpl_key].setdefault(sim, {})[alias] = payload
                    all_stations.add(alias)
                
                all_sims.add(sim)
                
            except Exception as e:
                print(f"[WARN] Failed to load CSV template {filename}: {e}")
                continue

    return templates, sorted(all_stations), sorted(all_sims)

def plot_saved_templates(
    outdir: str,
    stations: list | None = None,
    sim_labels: list | None = None,
    thirds=(0.0, 3333.0, 6666.0, 10000.0),
    time_unit: str = "seconds"
):
    """
    For each station×sim present on disk:
      - Plot Template 1, 2, 3 individually
      - Plot Template 4 (full) with the three thirds over-coloured
    Only plots items that exist (robust to missing templates).
    """
    unit_div = {"seconds":1, "minutes":60, "hours":3600, "days":86400}[time_unit]
    (loaded, all_stations, all_sims) = load_tilt_templates(outdir)
    t0, t1, t2, t3 = thirds
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # Filter scope
    stations = stations or all_stations
    sim_labels = sim_labels or all_sims

    for st in stations:
        for sim in sim_labels:
            # ------- Individual templates (1–3) -------
            for k, (label, tpl) in enumerate(zip(
                [f"Template 1 ({t0:.0f}–{t1:.0f}s)", f"Template 2 ({t1:.0f}–{t2:.0f}s)", f"Template 3 ({t2:.0f}–{t3:.0f}s)"],
                ["T1","T2","T3"]
            )):
                seg = loaded.get(tpl, {}).get(sim, {}).get(st)
                if not seg: 
                    continue
                t = np.asarray(seg["t"]) / unit_div
                x = np.asarray(seg["x"])
                if x.size <= 1:
                    continue
                plt.figure(figsize=(12,3.2))
                plt.plot(t, x, lw=1.3)
                plt.xlabel(f"Time ({time_unit})"); plt.ylabel("Tilt")
                plt.title(f"{label} – {st} – {sim}")
                plt.grid(True, ls=":")
                plt.tight_layout(); plt.show()

            # ------- Template 4 (full) with colored thirds -------
            full = loaded.get("T4", {}).get(sim, {}).get(st)
            if not full:
                continue
            tF = np.asarray(full["t"]) / unit_div
            xF = np.asarray(full["x"])
            if xF.size <= 1:
                continue
            plt.figure(figsize=(12,3.6))
            # base full trace
            plt.plot(tF, xF, lw=0.9, color="0.6", alpha=0.6, label="Full signal")

            # overlay thirds if available (use T1..T3 from disk so time aligns perfectly)
            for idx, tpl in enumerate(["T1","T2","T3"]):
                seg = loaded.get(tpl, {}).get(sim, {}).get(st)
                if not seg:
                    continue
                t = np.asarray(seg["t"]) / unit_div
                x = np.asarray(seg["x"])
                if x.size > 1:
                    lbl = ["Template 1","Template 2","Template 3"][idx]
                    plt.plot(t, x, lw=1.6, color=colors[idx], label=lbl)

            plt.xlabel(f"Time ({time_unit})"); plt.ylabel("Tilt")
            plt.title(f"Template 4 (0–{int(t3)} s) with thirds highlighted – {st} – {sim}")
            plt.grid(True, ls=":"); plt.legend(); plt.tight_layout(); plt.show()
# If you saved earlier to, e.g., "tilt_templates"
# plot_saved_templates(
#     outdir="tilt_templates",
#     stations=None,          # or a subset like ['ECPN','ECIT']
#     sim_labels=None,        # or a subset like ['sim1','sim2']
#     thirds=(0,3333,6666,10000),
#     time_unit="minutes"       # 'seconds' | 'minutes' | 'hours' | 'days'
# )


# In[3]:


import numpy as np
from scipy.signal import detrend as scipy_detrend

def build_templates_from_detrended_raw(
    tilt_data_raw: dict,          # shape: tilt_data_raw[sim][station] -> 1D array
    stations: list,
    sim_labels: list,
    fs: float = 1.0,
    thirds=(0.0, 3333.0, 6666.0, 10000.0),  # (t0, t1, t2, t3) in seconds
    detrend_type: str = "linear",  # 'linear' | 'constant'
    normalize: str | None = None   # None | 'zscore' | 'maxabs'
) -> dict:
    """
    Returns nested templates from *detrended raw* signals in your preferred shape:
      templates[sim][station]['template1'..'template4'] -> 1D np.array
    template1..3 are thirds; template4 is 'full' (t0..max(t3, end))
    """
    t0, t1, t2, t3 = thirds
    out: dict = {}

    def _prep(x):
        x = np.asarray(x, float).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return x
        if detrend_type in ("linear", "constant"):
            x = scipy_detrend(x, type=detrend_type)
        if normalize == "zscore":
            mu, sd = np.mean(x), np.std(x)
            x = (x - mu) / (sd + 1e-12)
        elif normalize == "maxabs":
            ma = np.max(np.abs(x))
            if ma > 0: x = x / ma
        return x

    def _clip_idx(N, a, b):
        i0 = max(0, int(round(a * fs)))
        i1 = min(N, int(round(b * fs)))
        return (min(i0, N), max(min(i1, N), min(i0, N)))

    for sim in sim_labels:
        out.setdefault(sim, {})
        for st in stations:
            x_full = tilt_data_raw.get(sim, {}).get(st, None)
            if x_full is None or len(x_full) < 2:
                continue
            x = _prep(x_full)
            N = len(x)

            i01 = _clip_idx(N, t0, t1)
            i12 = _clip_idx(N, t1, t2)
            i23 = _clip_idx(N, t2, t3)
            i0F = _clip_idx(N, t0, max(t3, N/fs if N>0 else t3))

            tpl = {}
            if i01[1] - i01[0] > 1: tpl["template1"] = x[i01[0]:i01[1]]
            if i12[1] - i12[0] > 1: tpl["template2"] = x[i12[0]:i12[1]]
            if i23[1] - i23[0] > 1: tpl["template3"] = x[i23[0]:i23[1]]
            if i0F[1] - i0F[0] > 1: tpl["template4"] = x[i0F[0]:i0F[1]]

            if tpl:
                out[sim][st] = tpl
    return out
templates_from_raw = build_templates_from_detrended_raw(
    tilt_data_raw=tilt_data_raw,         # your existing raw dict
    stations=stations,                   # e.g. ["ECPN","EC1","ECIT","EC10","ECOR","EMAS"]
    sim_labels=sim_labels,               # e.g. ["sim1","sim2","sim3","sim4"]
    fs=1.0,
    thirds=(0, 3333, 6666, 10000),
    detrend_type="linear",
    normalize=None
)

import numpy as np
import matplotlib.pyplot as plt

def plot_detrended_raw_templates(
    templates_nested: dict,
    station: str,
    fs: float = 1.0,
    time_unit: str = "minutes",         # "seconds" | "minutes" | "hours" | "days"
    sim_order=("sim1","sim2","sim3","sim4"),
    template_keys=("template1","template2","template3","template4"),
    figsize_per=(3.8, 2.6),
    sharey=True,
    colors=("tab:blue","tab:orange","tab:green","tab:red"),
    title_prefix=None
):
    """
    Plot detrended-raw templates for one station.
    Grid: ROWS = sims, COLS = templates 1..4. Each subplot shows the template vs time.
    """
    unit_div = {"seconds":1, "minutes":60, "hours":3600, "days":86400}[time_unit]
    # collect available sims that have this station
    sims = [s for s in sim_order if s in templates_nested and station in templates_nested[s]]
    if not sims:
        print(f"No templates found for station '{station}'.")
        return

    nrows = len(sims)
    ncols = len(template_keys)

    fig_w = figsize_per[0] * ncols
    fig_h = figsize_per[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=False, sharey=sharey)
    axes = np.atleast_2d(axes)

    # optional figure title
    if title_prefix is None:
        title_prefix = "Templates (detrended raw)"
    fig.suptitle(f"{title_prefix} — {station}", y=0.98, fontsize=13)

    # precompute y-range (if sharing y)
    if sharey:
        all_vals = []
        for sim in sims:
            for k in template_keys:
                x = templates_nested.get(sim, {}).get(station, {}).get(k)
                if isinstance(x, dict): x = x.get("x")
                if x is not None:
                    all_vals.append(np.asarray(x, float).ravel())
        if all_vals:
            z = np.concatenate([v[np.isfinite(v)] for v in all_vals if v.size])
            if z.size:
                lo, hi = np.quantile(z, [0.02, 0.98])
                span = max(hi - lo, 1e-6)
                ylims = (lo - 0.1*span, hi + 0.1*span)
            else:
                ylims = None
        else:
            ylims = None
    else:
        ylims = None

    for r, sim in enumerate(sims):
        for c, k in enumerate(template_keys):
            ax = axes[r, c]
            tpl = templates_nested.get(sim, {}).get(station, {}).get(k)
            if isinstance(tpl, dict): tpl = tpl.get("x")
            if tpl is None or len(tpl) < 2:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", alpha=0.5)
                ax.axis("on"); ax.grid(True, ls=":", alpha=0.5)
            else:
                y = np.asarray(tpl, float).ravel()
                t = np.arange(len(y)) / fs / unit_div
                ax.plot(t, y, lw=1.5, color=colors[c % len(colors)], label=k)
                ax.grid(True, ls=":", alpha=0.6)
                if ylims is not None:
                    ax.set_ylim(*ylims)

            # column headers = template names
            if r == 0:
                ax.set_title(k, fontsize=11, pad=6)
            # row labels = sim
            if c == 0:
                ax.set_ylabel(sim, rotation=0, labelpad=24, va="center", fontsize=10)
            # x label bottom row
            if r == nrows - 1:
                ax.set_xlabel(f"Time ({time_unit})")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
# Example: plot EEC1 templates (from your detrended-raw templates dict)
# plot_detrended_raw_templates(
#     templates_nested=templates_from_raw,   # the dict you built earlier
#     station="EC1",
#     fs=1.0,
#     time_unit="minutes",
#     sim_order=("sim1","sim2","sim3","sim4"),
#     template_keys=("template1","template2","template3","template4"),
#     sharey=True
# )

# Plot another station (e.g., ECOR)
#plot_detrended_raw_templates(templates_from_raw, station="ECOR", fs=1.0, time_unit="minutes")



# In[4]:


# INGV DATA Nov22 - March 23

# ECPN INGV

df_ecpn = pd.read_feather("/home/owen/Signals/experiment/INGV/ECPN.feather")
df_ecpn.datetime = pd.to_datetime(df_ecpn.datetime)
df_ecpn = df_ecpn.drop_duplicates()
df_ecpn = df_ecpn.sort_values(by=['seconds']).reset_index(drop=True)

# EEC1 INGV

df_eec1_ingv = pd.read_feather("/home/owen/Signals/experiment/INGV/EEC1.feather")
df_eec1_ingv.datetime = pd.to_datetime(df_eec1_ingv.datetime)
df_eec1_ingv = df_eec1_ingv.drop_duplicates()
df_eec1_ingv = df_eec1_ingv.sort_values(by=['seconds']).reset_index(drop=True)

import datetime as dt
import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt

# --- helper: slice a DataFrame to a datetime window ---
def _slice_df_time(df: pd.DataFrame, start_dt: dt.datetime, end_dt: dt.datetime,
                   datetime_col: str = "datetime", epoch_dt: dt.datetime | None = None) -> pd.DataFrame:
    """
    Return rows between [start_dt, end_dt].
    Priority:
      1) If `datetime_col` exists, use it (parsed with pd.to_datetime).
      2) Else if 'seconds' exists and epoch_dt is given, convert seconds -> datetime via epoch_dt + seconds.
      3) Otherwise, return df unchanged.
    """
    if datetime_col in df.columns:
        ts = pd.to_datetime(df[datetime_col])
        m = (ts >= start_dt) & (ts <= end_dt)
        return df.loc[m].reset_index(drop=True)
    elif "seconds" in df.columns and epoch_dt is not None:
        ts = epoch_dt + pd.to_timedelta(df["seconds"].to_numpy(float), unit="s")
        m = (ts >= start_dt) & (ts <= end_dt)
        return df.loc[m].reset_index(drop=True)
    else:
        # No usable time column; skip slicing
        return df

# --- (unchanged helpers you already have) ---
def _time_seconds(df: pd.DataFrame):
    if "seconds" in df.columns and np.issubdtype(df["seconds"].dtype, np.number):
        t = df["seconds"].to_numpy(float); return t - np.nanmin(t)
    if "datetime" in df.columns:
        dtcol = pd.to_datetime(df["datetime"])
        return (dtcol - dtcol.iloc[0]).dt.total_seconds().to_numpy(float)
    return np.arange(len(df), dtype=float)

def _pick_value_col(df: pd.DataFrame, prefer=None):
    prefer = prefer or ["tilt_mag","tilt","mag","value","z","y","x"]
    for c in prefer:
        if c in df.columns and np.issubdtype(df[c].dtype, np.number):
            return c
    for c in df.columns:
        if c not in {"datetime","seconds","time","timestamp","t"} and np.issubdtype(df[c].dtype, np.number):
            return c
    raise ValueError("No numeric tilt column found.")

def _estimate_fs(t_sec: np.ndarray, default=1.0):
    if t_sec.size < 2: return default
    dtm = np.median(np.diff(t_sec))
    return default if (dtm<=0 or not np.isfinite(dtm)) else 1.0/dtm

def _bandpass_001_01(x, fs, order=4, f1=0.001, f2=0.01):
    nyq = 0.5*fs
    lo = max(f1/nyq, 1e-6); hi = min(f2/nyq, 0.999999)
    sos = butter(order, [lo, hi], btype="bandpass", output="sos")
    return sosfiltfilt(sos, np.asarray(x, float))

# --- UPDATED: builder that accepts a time_window for slicing before filtering ---
def build_filtered_dataset(df_map: dict,
                           prefer_cols=None,
                           fs_override=None,
                           band=(0.003, 0.01),
                           time_window: tuple[dt.datetime, dt.datetime] | None = None,
                           datetime_col: str = "datetime",
                           epoch_dt: dt.datetime | None = None):
    """
    Build filtered series per station with optional datetime slicing.
      time_window: (start_dt, end_dt) in absolute datetimes.
      datetime_col: column name to use if present.
      epoch_dt: reference datetime to convert 'seconds' -> datetime when datetime_col absent.
    Returns:
      filt: {station -> filtered np.array}
      tmap: {station -> time seconds}
    """
    filt, tmap = {}, {}
    for st, df in df_map.items():
        if df is None or len(df) == 0:
            continue

        df_use = df
        if time_window is not None:
            start_dt, end_dt = time_window
            df_use = _slice_df_time(df, start_dt, end_dt, datetime_col=datetime_col, epoch_dt=epoch_dt)

        if df_use is None or df_use.empty:
            continue

        t = _time_seconds(df_use)
        fs = fs_override if fs_override is not None else _estimate_fs(t, default=1.0)
        col = _pick_value_col(df_use, prefer=prefer_cols)
        y = df_use[col].to_numpy(float)
        L = min(len(y), len(t)); y = y[:L]; t = t[:L]

        try:
            y_f = _bandpass_001_01(y, fs=fs, f1=band[0], f2=band[1], order=4)
        except Exception:
            y_f = np.zeros_like(y)

        filt[st] = y_f
        tmap[st] = t

    return filt, tmap
import datetime as dt

ingv_start_time = dt.datetime.strptime("2022-11-14 22:00:00", "%Y-%m-%d %H:%M:%S")
ingv_end_time   = dt.datetime.strptime("2022-11-15 02:19:59", "%Y-%m-%d %H:%M:%S")

# INGV dataframes you mentioned
station_dfs_ingv = {
    "ECPN": df_ecpn,            # has a 'datetime' column ideally
    "EEC1": df_eec1_ingv,
}


# EXPERIMENT DATA

# ECOR 
df_ecor = pd.read_feather("/home/owen/Signals/experiment/school-data/INGV_feather/ECOR.feather")
df_ecor.datetime = pd.to_datetime(df_ecor.datetime)
df_ecor = df_ecor.drop_duplicates()
df_ecor = df_ecor.sort_values(by=['seconds']).reset_index(drop=True)

# EEC1
df_eec1 = pd.read_feather("/home/owen/Signals/experiment/school-data/INGV_feather/EEC1.feather")
df_eec1.datetime = pd.to_datetime(df_eec1.datetime)
df_eec1 = df_eec1.drop_duplicates()
df_eec1 = df_eec1.sort_values(by=['seconds']).reset_index(drop=True)

#EMAS
df_emas = pd.read_feather("/home/owen/Signals/experiment/school-data/INGV_feather/EMAS.feather")
df_emas.datetime = pd.to_datetime(df_emas.datetime)
df_emas = df_emas.drop_duplicates()
df_emas = df_emas.sort_values(by=['seconds']).reset_index(drop=True)

# ECIT
df_ecit = pd.read_feather("/home/owen/Signals/experiment/school-data/INGV_feather/ECIT.feather")
df_ecit.datetime = pd.to_datetime(df_ecit.datetime)
df_ecit = df_ecit.drop_duplicates()
df_ecit = df_ecit.sort_values(by=['seconds']).reset_index(drop=True)

# EC10

df_ec10 = pd.read_feather("/home/owen/Signals/experiment/school-data/INGV_feather/EC10.feather")
df_ec10.datetime = pd.to_datetime(df_ec10.datetime)
df_ec10 = df_ec10.drop_duplicates()
df_ec10 = df_ec10.sort_values(by=['seconds']).reset_index(drop=True)


# In[11]:


# --- Experiment window (absolute datetimes) ---
import datetime as dt
import matplotlib.dates as mdates

EXP_START_DT = dt.datetime(2023, 7, 24, 0, 0, 0)
EXP_END_DT   = dt.datetime(2023, 8, 2, 23, 59, 59)

# Collect experiment dataframes
station_dfs_exp = {
    "ECOR": df_ecor,
    "ECIT": df_ecit,
    "EC10": df_ec10,
    "EMAS": df_emas,
    "EC1":  df_eec1,   # (EEC1 experiment)
}
# prune missing/empty
station_dfs_exp = {k: v for k, v in station_dfs_exp.items() if v is not None and not v.empty}

# --- Build filtered series LIMITED to the experiment window ---
exp_filt_map, exp_time_map = build_filtered_dataset(
    df_map=station_dfs_exp,
    prefer_cols=None,                 # auto-detect tilt column (tilt_mag/tilt/etc.)
    fs_override=1.0,                  # or None to infer
    band=(0.003, 0.01),
    time_window=(EXP_START_DT, EXP_END_DT),
    datetime_col="datetime",
    epoch_dt=None
)

# --- Plot filtered time series vs ABSOLUTE datetime per station ---
for st in sorted(station_dfs_exp.keys()):
    # slice again to rebuild the datetime axis aligned to the filtered y
    df_slice = _slice_df_time(
        station_dfs_exp[st],
        start_dt=EXP_START_DT,
        end_dt=EXP_END_DT,
        datetime_col="datetime",
        epoch_dt=None
    )
    if df_slice is None or df_slice.empty or st not in exp_filt_map:
        continue

    y = exp_filt_map[st]
    L = min(len(df_slice), len(y))
    if L < 2:
        continue

    ts = pd.to_datetime(df_slice["datetime"].iloc[:L])
    y = y[:L]

    plt.figure(figsize=(12, 3.2))
    plt.plot(ts, y, lw=1.2)
    plt.title(f"Experiment filtered tilt (0.003–0.01 Hz) — {st}\n{EXP_START_DT} → {EXP_END_DT}")
    plt.xlabel("Datetime"); plt.ylabel("Tilt (filtered)")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    plt.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


# Define experiment DataFrames
station_map = {
    "ECOR": df_ecor,
    "EMAS": df_emas,
    "EC1": df_eec1,
    "EC10": df_ec10,
    "ECIT": df_ecit,
    "ECPN": df_ecpn,
    "EEC1": df_eec1_ingv
}

def extract_tilt_templates(
    tilt_data: dict,
    stations: list,
    sim_labels: list,
    fs: float = 1.0,
    thirds=(0.0, 3333.0, 6666.0, 10000.0),   # (t0, t1, t2, t3)
    include_full: bool = True,               # Template 4
    detrend: str | None = None,              # None | 'linear' | 'constant'
    normalize: str | None = None,            # None | 'zscore' | 'maxabs'
    include_time: bool = True,               # also store time arrays
):
    """
    Build four signal templates per station×sim:
      T1: [t0, t1), T2: [t1, t2), T3: [t2, t3), T4 (optional full): [t0, max(t3, actual_end))

    Returns
    -------
    templates : dict
        templates['T1'|'T2'|'T3'|'T4'][sim][station] -> dict with:
            'x': segment samples (1D np.array)
            't': time in seconds (if include_time=True)
            'i0','i1': sample indices in the original series
            't0','t1': segment time bounds (seconds, clipped to available data)
    meta : dict
        Summary info: fs, thirds, options, and lengths per segment.
    """
    t0, t1, t2, t3 = thirds
    keys = ['T1', 'T2', 'T3'] + (['T4'] if include_full else [])

    # init nested dict
    templates = {k: {sim: {} for sim in sim_labels} for k in keys}

    def _prep_signal(x):
        x = np.asarray(x, float).ravel()
        x = x[np.isfinite(x)]
        if detrend in ('linear', 'constant'):
            x = scipy_detrend(x, type=detrend)
        if normalize == 'zscore':
            mu, sd = np.mean(x), np.std(x)
            if sd > 0: x = (x - mu) / sd
            else:      x = x - mu
        elif normalize == 'maxabs':
            ma = np.max(np.abs(x)) if x.size else 0.0
            if ma > 0: x = x / ma
        return x

    def _clip_idx(N, t_start, t_end):
        i0 = max(0, int(np.round(t_start * fs)))
        i1 = min(N, int(np.round(t_end   * fs)))
        i0 = min(i0, N)
        i1 = max(i1, i0)
        return i0, i1

    for st in stations:
        for sim in sim_labels:
            x_full = tilt_data.get(sim, {}).get(st)
            if x_full is None or len(x_full) < 1:
                continue
            x_full = _prep_signal(x_full)
            N = len(x_full)
            t_full = np.arange(N) / fs

            # Segment indices (clipped to available data)
            i01 = _clip_idx(N, t0, t1)
            i12 = _clip_idx(N, t1, t2)
            i23 = _clip_idx(N, t2, t3)
            i0F = _clip_idx(N, t0, max(t3, t_full[-1] if N>0 else t3))

            # Helper to package a segment
            def pack(name, rng, nominal_bounds):
                i0, i1 = rng
                seg = x_full[i0:i1]
                if seg.size == 0:
                    return None
                out = {
                    'x': seg,
                    'i0': i0, 'i1': i1,
                    't0': i0/fs, 't1': i1/fs,
                    't0_nom': nominal_bounds[0], 't1_nom': nominal_bounds[1],
                }
                if include_time:
                    out['t'] = t_full[i0:i1]
                return out

            # Store T1–T3
            out = pack('T1', i01, (t0, t1))
            if out: templates['T1'][sim][st] = out
            out = pack('T2', i12, (t1, t2))
            if out: templates['T2'][sim][st] = out
            out = pack('T3', i23, (t2, t3))
            if out: templates['T3'][sim][st] = out

            # Store T4 (full)
            if include_full:
                out = pack('T4', i0F, (t0, max(t3, t_full[-1] if N>0 else t3)))
                if out: templates['T4'][sim][st] = out

    # simple meta summary
    meta = {
        'fs': fs,
        'thirds': thirds,
        'include_full': include_full,
        'detrend': detrend,
        'normalize': normalize,
        'include_time': include_time,
    }
    return templates, meta


def save_tilt_templates(templates: dict, outdir: str, prefix: str = ""):
    """
    Save templates to disk as .npy arrays (x) and small .npz metadata per segment.
    Directory layout:
        outdir/
          T1/ <prefix>_<sim>_<station>.npy
          T1/ <prefix>_<sim>_<station>_meta.npz
          ...
    """
    os.makedirs(outdir, exist_ok=True)
    for tpl_name, sims in templates.items():
        tpl_dir = os.path.join(outdir, tpl_name)
        os.makedirs(tpl_dir, exist_ok=True)
        for sim, stations_dict in sims.items():
            for st, payload in stations_dict.items():
                base = f"{prefix+'_' if prefix else ''}{sim}_{st}"
                x = payload.get('x', np.array([]))
                np.save(os.path.join(tpl_dir, f"{base}.npy"), x)
                # Save lightweight meta (excluding 'x' to keep small)
                meta = {k: v for k, v in payload.items() if k != 'x'}
                np.savez(os.path.join(tpl_dir, f"{base}_meta.npz"), **meta)
# 1) Extract templates from your chosen dataset (raw or filtered)
templates, meta = extract_tilt_templates(
    tilt_data=tilt_data_filt, 
    #tilt_data=tilt_data_raw, # or tilt_data_raw
    stations=stations,
    sim_labels=sim_labels,
    fs=1.0,
    thirds=(0.0, 3333.0, 6666.0, 10000.0),  # Template 1–3 bounds; full is 0–max
    include_full=True,
    detrend='linear',                 # optional: None | 'linear' | 'constant'
    normalize=None,                   # optional: None | 'zscore' | 'maxabs'
    include_time=True
)

# 2) Access a template (e.g., Template 2 for one station/sim)
tpl2 = templates['T2'][sim_labels[0]][stations[0]]
x_seg = tpl2['x']           # the samples
t_seg = tpl2['t']           # matching time in seconds
i0, i1 = tpl2['i0'], tpl2['i1']

# 3) Save them for future reuse
save_tilt_templates(templates, outdir="tilt_templates", prefix="filt")


# ---------- Loader (rebuild nested dict from disk) ----------
def load_tilt_templates(outdir: str, csv_dir: str = None):
    """
    Load templates from .npy/.npz files OR from CSV files.
    
    For .npy/.npz format:
      templates[tpl]['sim']['station'] -> {'x', 't', 'i0','i1','t0','t1','t0_nom','t1_nom'}
    
    For CSV format (filename: {STATION}_{SIM}_{TEMPLATE}_{FILTER}_tpl.csv):
      Expects columns: time_seconds, x
      Example: EC1_sim1_template1_0p001-0p01Hz_tpl.csv
    
    Parameters:
    -----------
    outdir : str
        Directory containing template folders (T1/, T2/, T3/, T4/) with .npy files
    csv_dir : str, optional
        Directory containing CSV template files (flat structure)
        If provided, CSV files will be loaded in addition to .npy files
    
    Returns:
    --------
    (templates, all_stations, all_sims)
    """
    templates = {}
    all_sims, all_stations = set(), set()

    # ========== Load .npy/.npz templates (original format) ==========
    for tpl in ["T1","T2","T3","T4"]:
        tpl_dir = os.path.join(outdir, tpl)
        if not os.path.isdir(tpl_dir):
            continue
        templates[tpl] = {}
        # Find base names (pairs of .npy and _meta.npz)
        for npy in glob.glob(os.path.join(tpl_dir, "*.npy")):
            base = os.path.splitext(os.path.basename(npy))[0]
            meta_path = os.path.join(tpl_dir, base + "_meta.npz")
            if not os.path.isfile(meta_path):
                continue
            parts = base.split("_")
            if len(parts) < 2:
                sim, station = base, "UNKNOWN"
            else:
                station = parts[-1]
                sim = "_".join(parts[:-1])
            all_sims.add(sim); all_stations.add(station)

            x = np.load(npy)
            meta = dict(np.load(meta_path))
            if "t" in meta:
                t = meta["t"]
            else:
                t0 = float(meta.get("t0", 0.0))
                t1 = float(meta.get("t1", t0 + len(x)))
                if len(x) > 1:
                    t = np.linspace(t0, t1, num=len(x), endpoint=False)
                else:
                    t = np.array([t0], float)

            payload = {"x": x, "t": t}
            for k in ("i0","i1","t0","t1","t0_nom","t1_nom"):
                if k in meta:
                    payload[k] = meta[k].item() if hasattr(meta[k], "item") else meta[k]

            templates[tpl].setdefault(sim, {})[station] = payload

    # ========== Load CSV templates (new format) ==========
    if csv_dir and os.path.isdir(csv_dir):
        csv_files = glob.glob(os.path.join(csv_dir, "*_tpl.csv"))
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            base = filename.replace("_tpl.csv", "")
            parts = base.split("_")
            
            if len(parts) < 3:
                continue
            
            station = parts[0].upper()
            sim = parts[1]
            template_part = parts[2]
            
            # Station name aliases (CSV has EC1, but events may use EEC1)
            station_aliases = {
                "EC1": ["EC1", "EEC1"],  # Store under both names
                "EEC1": ["EC1", "EEC1"],
                "ECPN": ["ECPN"],
                "EMAS": ["EMAS"],
                "ECOR": ["ECOR"],
                "ECIT": ["ECIT"],
                "EC10": ["EC10"],
            }
            
            template_map_csv = {
                "template1": "T1",
                "template2": "T2", 
                "template3": "T3",
                "template4": "T4"
            }
            tpl_key = template_map_csv.get(template_part.lower())
            if not tpl_key:
                continue
            
            try:
                df = pd.read_csv(csv_file)
                if "time_seconds" not in df.columns or "x" not in df.columns:
                    continue
                
                t = df["time_seconds"].values
                x = df["x"].values
                
                payload = {
                    "x": x,
                    "t": t,
                    "t0": t[0] if len(t) > 0 else 0.0,
                    "t1": t[-1] if len(t) > 0 else 0.0,
                }
                
                if tpl_key not in templates:
                    templates[tpl_key] = {}
                
                # Store under all alias names
                alias_names = station_aliases.get(station, [station])
                for alias in alias_names:
                    templates[tpl_key].setdefault(sim, {})[alias] = payload
                    all_stations.add(alias)
                
                all_sims.add(sim)
                
            except Exception as e:
                print(f"[WARN] Failed to load CSV template {filename}: {e}")
                continue

    return templates, sorted(all_stations), sorted(all_sims)

def plot_saved_templates(
    outdir: str,
    stations: list | None = None,
    sim_labels: list | None = None,
    thirds=(0.0, 3333.0, 6666.0, 10000.0),
    time_unit: str = "seconds"
):
    """
    For each station×sim present on disk:
      - Plot Template 1, 2, 3 individually
      - Plot Template 4 (full) with the three thirds over-coloured
    Only plots items that exist (robust to missing templates).
    """
    unit_div = {"seconds":1, "minutes":60, "hours":3600, "days":86400}[time_unit]
    (loaded, all_stations, all_sims) = load_tilt_templates(outdir)
    t0, t1, t2, t3 = thirds
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # Filter scope
    stations = stations or all_stations
    sim_labels = sim_labels or all_sims

    for st in stations:
        for sim in sim_labels:
            # ------- Individual templates (1–3) -------
            for k, (label, tpl) in enumerate(zip(
                [f"Template 1 ({t0:.0f}–{t1:.0f}s)", f"Template 2 ({t1:.0f}–{t2:.0f}s)", f"Template 3 ({t2:.0f}–{t3:.0f}s)"],
                ["T1","T2","T3"]
            )):
                seg = loaded.get(tpl, {}).get(sim, {}).get(st)
                if not seg: 
                    continue
                t = np.asarray(seg["t"]) / unit_div
                x = np.asarray(seg["x"])
                if x.size <= 1:
                    continue
                plt.figure(figsize=(12,3.2))
                plt.plot(t, x, lw=1.3)
                plt.xlabel(f"Time ({time_unit})"); plt.ylabel("Tilt")
                plt.title(f"{label} – {st} – {sim}")
                plt.grid(True, ls=":")
                plt.tight_layout(); plt.show()

            # ------- Template 4 (full) with colored thirds -------
            full = loaded.get("T4", {}).get(sim, {}).get(st)
            if not full:
                continue
            tF = np.asarray(full["t"]) / unit_div
            xF = np.asarray(full["x"])
            if xF.size <= 1:
                continue
            plt.figure(figsize=(12,3.6))
            # base full trace
            plt.plot(tF, xF, lw=0.9, color="0.6", alpha=0.6, label="Full signal")

            # overlay thirds if available (use T1..T3 from disk so time aligns perfectly)
            for idx, tpl in enumerate(["T1","T2","T3"]):
                seg = loaded.get(tpl, {}).get(sim, {}).get(st)
                if not seg:
                    continue
                t = np.asarray(seg["t"]) / unit_div
                x = np.asarray(seg["x"])
                if x.size > 1:
                    lbl = ["Template 1","Template 2","Template 3"][idx]
                    plt.plot(t, x, lw=1.6, color=colors[idx], label=lbl)

            plt.xlabel(f"Time ({time_unit})"); plt.ylabel("Tilt")
            plt.title(f"Template 4 (0–{int(t3)} s) with thirds highlighted – {st} – {sim}")
            plt.grid(True, ls=":"); plt.legend(); plt.tight_layout(); plt.show()
# If you saved earlier to, e.g., "tilt_templates"
# plot_saved_templates(
#     outdir="tilt_templates",
#     stations=None,          # or a subset like ['ECPN','ECIT']
#     sim_labels=None,        # or a subset like ['sim1','sim2']
#     thirds=(0,3333,6666,10000),
#     time_unit="minutes"       # 'seconds' | 'minutes' | 'hours' | 'days'
# )

def plot_eec1_ecpn_denoised_with_templates(
    csv_path: str,
    templates_dir: str = "tilt_templates",
    denoised_dir: str = "/home/owen/tilt_validation/bandpassed_exports_001_01_csv_only_utc/ingv/",
    eec1_filename: str = "EEC1_0p001-0p01.csv",
    ecpn_filename: str = "ECPN_additional_0p001-0p01.csv",
    station_dfs: dict = None,           # must include 'EEC1' (or 'EC1') and 'ECPN' with 'datetime' if denoised lacks it
    segment_duration_s: int = 1000,
    time_unit: str = "minutes",
    detrend_type: str = "linear",       # "linear" | "constant" | None
    normalize: str = None,              # None | "zscore" | "maxabs"
    y_limit: float = 0.02,
    save_dir: str = None,
    debug: bool = True,
    eec1_template_station: str = "EC1",
    ecpn_template_station: str = "ECPN",
    # --- NEW: filtering controls ---
    apply_filter_obs: bool = True,
    band_hz: tuple = (0.001, 0.01),
    filter_templates: bool = False
):
    """
    Plot EEC1 and ECPN denoised traces with their templates (one merged plot per event).

    NEW:
      • apply_filter_obs=True applies a Butterworth band-pass (default 0.001–0.01 Hz) to *observational* denoised traces.
      • filter_templates=False by default (set True to also band-pass templates, if desired).
    """

    # ---------------- helpers ----------------
    def _prep_signal(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return x
        if detrend_type in ('linear', 'constant'):
            x = scipy_detrend(x, type=detrend_type)
        if normalize == 'zscore':
            mu, sd = np.mean(x), np.std(x)
            if sd > 0: x = (x - mu) / sd
        elif normalize == 'maxabs':
            ma = np.max(np.abs(x))
            if ma > 0: x = x / ma
        return x

    def _bandpass(x, fs, f1, f2, order=4):
        from scipy.signal import butter, sosfiltfilt
        x = np.asarray(x, float)
        nyq = 0.5 * fs
        # Clamp normalized band safely into (0, 1)
        lo = max(f1 / nyq, 1e-6)
        hi = min(f2 / nyq, 0.999999)
        if not (0 < lo < hi < 1):
            return x  # invalid band for this fs → return unfiltered
        sos = butter(order, [lo, hi], btype="bandpass", output="sos")
        return sosfiltfilt(sos, x)

    def _resample_to_grid(x, t, t_grid, seg_dur, x_fill=np.nan):
        x = np.asarray(x, float).ravel()
        t = np.asarray(t, float).ravel()
        if x.size < 1:
            return np.full_like(t_grid, x_fill, dtype=float)
        if x.size == 1 or (t[-1] - t[0]) == 0:
            return np.full_like(t_grid, x[0], dtype=float)
        t_norm = (t - t[0]) / (t[-1] - t[0]) * seg_dur
        f = interp1d(t_norm, x, kind='linear', bounds_error=False, fill_value='extrapolate')
        return f(t_grid)

    def _load_and_attach_datetime(den_path: Path, raw_df: pd.DataFrame, raw_fallback_df: pd.DataFrame = None, who: str = "EEC1"):
        """Load denoised CSV; ensure 'denoised' & 'datetime'. Borrow datetime from raw if missing."""
        if not den_path.exists():
            if debug: print(f"[DENOISED] Missing file for {who}: {den_path}")
            return None
        df = pd.read_csv(den_path)
        
        # Accept either 'denoised' or 'bandpassed' column names
        signal_col = None
        if "denoised" in df.columns:
            signal_col = "denoised"
        elif "bandpassed" in df.columns:
            signal_col = "bandpassed"
            df["denoised"] = df["bandpassed"]  # Rename for consistency
        
        if signal_col is None:
            if debug: print(f"[WARN] {den_path.name} lacks 'denoised' or 'bandpassed' column for {who}")
            return None
        dt_col = next((c for c in ["datetime","time","timestamp","DateTime","Datetime","time_seconds"] if c in df.columns), None)
        if dt_col is None:
            base = raw_df if raw_df is not None else raw_fallback_df
            if base is None or "datetime" not in base.columns:
                if debug: print(f"[ERROR] No datetime to borrow for {who}")
                return None
            base_dt = pd.to_datetime(base["datetime"], errors="coerce").dropna().reset_index(drop=True)
            n = min(len(df), len(base_dt))
            if len(df) != len(base_dt) and debug:
                print(f"[WARN] {who}: length mismatch (denoised={len(df)}, raw_dt={len(base_dt)}) → trimming to {n}")
            df = df.iloc[:n].copy()
            df["datetime"] = base_dt.iloc[:n].to_numpy()
        else:
            # Special handling for time_seconds column (elapsed seconds, not datetime)
            if dt_col == "time_seconds":
                # For experiment stations, borrow datetime from raw data if available
                base = raw_df if raw_df is not None else raw_fallback_df
                if base is not None and "datetime" in base.columns:
                    base_dt = pd.to_datetime(base["datetime"], errors="coerce").dropna().reset_index(drop=True)
                    n = min(len(df), len(base_dt))
                    if len(df) != len(base_dt) and debug:
                        print(f"[INFO] {who}: Using raw datetime for time_seconds data (denoised={len(df)}, raw_dt={len(base_dt)}) → using {n}")
                    df = df.iloc[:n].copy()
                    df["datetime"] = base_dt.iloc[:n].to_numpy()
                else:
                    # No raw datetime available - construct from time_seconds
                    # Experiment data starts at 2023-07-23 23:00:00 (UTC+1 clock → UTC) and ends at 2023-08-02 23:59:59
                    if debug: print(f"[INFO] {who}: Constructing datetime from time_seconds (reference: 2023-07-23 23:00:00)")
                    reference_time = pd.Timestamp("2023-07-23 23:00:00")
                    df["datetime"] = reference_time + pd.to_timedelta(df["time_seconds"], unit='s')
            else:
                df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=["datetime"]).reset_index(drop=True)
        if debug:
            print(f"[DENOISED] {who}: {len(df)} rows, span {df['datetime'].min()} → {df['datetime'].max()}")
        return df

    # ---------------- load events ----------------
    csv_path = Path(csv_path)
    if csv_path.is_dir():
        files = list(csv_path.glob("*.csv"))
        if not files:
            raise ValueError(f"No CSV files found in directory: {csv_path}")
        csv_path = files[0]
        if debug: print(f"[INFO] Using CSV file: {csv_path}")

    events_df = pd.read_csv(csv_path)
    required = {"station","peak_time_dt","template","sim","event_id"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"CSV must contain {sorted(required)}; missing {sorted(missing)}")
    events_df["peak_time_dt"] = pd.to_datetime(events_df["peak_time_dt"], errors="coerce")
    events_df = events_df.dropna(subset=["peak_time_dt"])
    keep = events_df["station"].astype(str).str.upper().isin(["EC1","EEC1","ECPN"])
    events_df = events_df.loc[keep].copy()
    if events_df.empty:
        raise ValueError("No EC1/EEC1/ECPN rows in event CSV.")

    template_map = {"template1":"T1","template2":"T2","template3":"T3","template4":"T4"}

    # ---------------- load templates ----------------
    templates, _, _ = load_tilt_templates(templates_dir, csv_dir=csv_templates_dir)
    if debug:
        print(f"[INFO] Templates loaded: {list(templates.keys())}")

    # ---------------- load denoised EEC1 & ECPN ----------------
    den_dir = Path(denoised_dir)
    eec1_den = _load_and_attach_datetime(
        den_dir / eec1_filename,
        raw_df=(station_dfs.get("EEC1") if station_dfs else None),
        raw_fallback_df=(station_dfs.get("EC1") if station_dfs else None),
        who="EEC1"
    )
    ecpn_den = _load_and_attach_datetime(
        den_dir / ecpn_filename,
        raw_df=(station_dfs.get("ECPN") if station_dfs else None),
        who="ECPN"
    )

    # ---------------- plotting config ----------------
    unit_div = {"seconds":1, "minutes":60, "hours":3600, "days":86400}[time_unit]
    t_common = np.linspace(0, segment_duration_s, 2000)
    obs_colors = {"EEC1":"#1f77b4", "ECPN":"#2ca02c"}  # blue, green
    template_colors = ["#E74C3C","#E67E22","#F39C12","#D35400","#C0392B",
                       "#FF6B6B","#FF8C42","#FFA600","#FF4D6D","#FF6F00"]

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ---------------- per-event loop ----------------
    for event_id, group in events_df.groupby("event_id"):
        earliest_peak = group["peak_time_dt"].min()
        half = segment_duration_s / 2
        start_time = earliest_peak - pd.Timedelta(seconds=half)
        end_time   = earliest_peak + pd.Timedelta(seconds=half)
        if debug:
            print(f"\n{'='*60}\n[EVENT {event_id}] window: {start_time} → {end_time}")

        # Slice denoised segments
        series_payloads = []  # (name, t_sec, x, color)
        for who, df_den in (("EEC1", eec1_den), ("ECPN", ecpn_den)):
            if df_den is None:
                continue
            seg = df_den[(df_den["datetime"] >= start_time) & (df_den["datetime"] <= end_time)].copy()
            if seg.empty:
                if debug: print(f"[{who}] No samples in window.")
                continue
            x_raw = seg["denoised"].to_numpy(float)
            t_s  = (seg["datetime"] - seg["datetime"].iloc[0]).dt.total_seconds().to_numpy()

            # detrend/normalize first
            x = _prep_signal(x_raw)

            # band-pass (observations) if requested and fs allows
            if apply_filter_obs and t_s.size > 1:
                fs = 1.0 / np.median(np.diff(t_s))
                x_f = _bandpass(x, fs, band_hz[0], band_hz[1])
                if debug:
                    print(f"[FILTER] {who}: fs≈{fs:.3f} Hz, band={band_hz[0]}–{band_hz[1]} Hz")
                x = x_f

            series_payloads.append((f"{who} denoised", t_s, x, obs_colors[who]))

        if not series_payloads:
            if debug: print(f"[SKIP] Event {event_id}: no EEC1/ECPN data in window.")
            continue

        # Collect templates
        tpl_curves = []  # (label, tpl_t, tpl_x)
        for _, row in group.iterrows():
            st_csv = str(row.get("station","")).upper()
            sim = str(row.get("sim",""))
            csv_tpl = str(row.get("template","")).lower()
            tpl_key = template_map.get(csv_tpl)
            if tpl_key is None or tpl_key not in templates:
                if debug: print(f"[TPL SKIP] Unknown template '{csv_tpl}'"); continue
            # resolve sim
            sim_key = sim if sim in templates[tpl_key] else None
            if sim_key is None:
                for pfx in ["filt_","raw_",""]:
                    cand = f"{pfx}{sim}"
                    if cand in templates[tpl_key]:
                        sim_key = cand; break
            if sim_key is None:
                if debug: print(f"[TPL SKIP] Sim '{sim}' not in {tpl_key}"); continue
            # station key
            target_station = eec1_template_station if st_csv in ("EC1","EEC1") else \
                             (ecpn_template_station if st_csv == "ECPN" else None)
            if target_station is None: 
                continue
            st_key = target_station if target_station in templates[tpl_key][sim_key] else None
            if st_key is None:
                for k in templates[tpl_key][sim_key].keys():
                    if k.upper() == target_station.upper():
                        st_key = k; break
            if st_key is None:
                if debug: print(f"[TPL SKIP] Station '{target_station}' not in {tpl_key}/{sim_key}")
                continue
            data = templates[tpl_key][sim_key][st_key]
            tpl_x = np.asarray(data["x"], float)
            tpl_t = np.asarray(data.get("t", np.arange(len(tpl_x))), dtype=float)
            # Optional filter templates
            if filter_templates and tpl_t.size > 1:
                fs_tpl = (len(tpl_t)-1) / (tpl_t[-1] - tpl_t[0]) if (tpl_t[-1] - tpl_t[0]) > 0 else None
                if fs_tpl and fs_tpl > 0:
                    tpl_x = _bandpass(tpl_x, fs_tpl, band_hz[0], band_hz[1])
            tpl_x = _prep_signal(tpl_x)
            tpl_curves.append((f"{st_csv} — {sim} {csv_tpl.upper()}", tpl_t, tpl_x))

        # ---------------- draw ----------------
        fig, ax = plt.subplots(figsize=(13.5, 6))

        # Observed
        for name, t_sec, x, color in series_payloads:
            y = _resample_to_grid(x, t_sec, t_common, segment_duration_s)
            ax.plot(t_common / unit_div, y, color=color, lw=2.0, ls='-', alpha=0.95, label=name, zorder=8)

        # Templates
        for j, (lab, tt, xx) in enumerate(tpl_curves):
            col = template_colors[j % len(template_colors)]
            y_tpl = _resample_to_grid(xx, tt, t_common, segment_duration_s)
            ax.plot(t_common / unit_div, y_tpl, color=col, lw=1.8, ls='--', alpha=0.9, label=lab, zorder=6)

        # Peaks
        peak_colors = {"EEC1":"#8b5cf6", "EC1":"#8b5cf6", "ECPN":"#22c55e"}
        for st_name, group_st in group.groupby(group["station"].astype(str).str.upper()):
            c = peak_colors.get(st_name, "#999999")
            for pk in sorted(group_st["peak_time_dt"].unique()):
                xpk = (pd.to_datetime(pk) - start_time).total_seconds() / unit_div
                ax.axvline(xpk, color=c, ls="-", lw=2.5, alpha=0.85)
                y_top = ax.get_ylim()[1] if y_limit is None else y_limit
                ax.text(xpk, y_top*0.96, st_name, rotation=90, ha="right", va="top", fontsize=8, color=c, alpha=0.95)
        
        # P wave arrivals (if p_wave_eta column exists)
        if "p_wave_eta" in group.columns:
            if debug:
                print(f"[P-WAVE] Checking for P wave data in event {event_id}")
                print(f"[P-WAVE] p_wave_eta values: {group['p_wave_eta'].tolist()}")
            
            # Get all P wave arrival times
            p_wave_arrivals = []
            for _, row in group.iterrows():
                if pd.notna(row.get("p_wave_eta")):
                    try:
                        p_datetime = pd.to_datetime(row["p_wave_eta"], errors="coerce")
                        if pd.notna(p_datetime):
                            p_wave_arrivals.append(p_datetime)
                            if debug:
                                print(f"[P-WAVE] Found P wave arrival: {p_datetime}")
                    except Exception as e:
                        if debug: print(f"[P-WAVE] Error parsing p_wave_eta: {e}")
            
            if debug:
                print(f"[P-WAVE] Total P wave arrivals: {len(p_wave_arrivals)}")
            
            # Plot P wave arrivals as vertical lines
            for p_arrival in sorted(set(p_wave_arrivals)):
                xp = (p_arrival - start_time).total_seconds() / unit_div
                if debug:
                    print(f"[P-WAVE] P wave at {p_arrival}, xp={xp:.2f} {time_unit}, window=[0, {segment_duration_s / unit_div:.2f}]")
                
                # Only plot if within the visible window
                if 0 <= xp <= segment_duration_s / unit_div:
                    ax.axvline(xp, color="#FF1744", ls="-", lw=2.5, alpha=0.85, zorder=9)
                    y_top = ax.get_ylim()[1] if y_limit is None else y_limit
                    ax.text(xp, y_top*0.92, "P", rotation=90, ha="right", va="top",
                            fontsize=8, color="#FF1744", alpha=0.95, fontweight="bold")
                    if debug:
                        print(f"[P-WAVE] ✓ Plotted P wave at x={xp:.2f}")
                else:
                    if debug:
                        print(f"[P-WAVE] ✗ P wave outside window at x={xp:.2f}")
        else:
            if debug:
                print(f"[P-WAVE] No p_wave_eta column in event {event_id}")

        if y_limit is not None:
            ax.set_ylim(-abs(y_limit), abs(y_limit))

        event_time_str = pd.to_datetime(earliest_peak).strftime("%Y-%m-%d %H:%M:%S")
        ax.set_title(
            f"Event {event_id} @ {event_time_str}\nEEC1 & ECPN denoised (BP {band_hz[0]}–{band_hz[1]} Hz) vs templates",
            fontsize=12, fontweight="bold", pad=8
        )
        ax.set_xlabel(f"Time ({time_unit})", fontsize=11)
        ax.set_ylabel("Tilt (normalized/detrended)" if normalize else "Tilt", fontsize=11)
        ax.grid(True, ls=":", alpha=0.35)

        leg = ax.legend(loc="upper right", fontsize=9, framealpha=1.0, facecolor="white")
        leg.get_frame().set_edgecolor("black"); leg.get_frame().set_linewidth(0.9)

        plt.tight_layout()
        if save_dir:
            name = f"eec1_ecpn_event_{int(event_id):04d}_denoised_BP_{band_hz[0]}-{band_hz[1]}Hz.png" if str(event_id).isdigit() \
                   else f"eec1_ecpn_event_{event_id}_denoised_BP_{band_hz[0]}-{band_hz[1]}Hz.png"
            out = Path(save_dir) / name
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[SAVED] {name}")
        plt.show(); plt.close()


# =============================================================================
# WAVELET ANALYSIS FUNCTIONS
# =============================================================================

def compute_cwt(signal_data, fs=1.0, wavelet='morl', freq_min=1e-4, freq_max=1e-2, n_freqs=100):
    """
    Compute Continuous Wavelet Transform using Morlet wavelet.
    Following Torrence & Compo (1998).
    
    Parameters:
    -----------
    signal_data : array-like
        Input signal
    fs : float
        Sampling frequency in Hz
    wavelet : str
        Wavelet type (default 'morl' for Morlet)
    freq_min, freq_max : float
        Frequency range for analysis (Hz)
    n_freqs : int
        Number of frequency bins
        
    Returns:
    --------
    frequencies : ndarray
        Frequency vector
    power : ndarray
        Power spectrum (shape: n_freqs × n_samples)
    """
    # Define scales for target frequency range
    # For Morlet wavelet: frequency = (center_freq * fs) / scale
    center_freq = pywt.central_frequency(wavelet)
    
    # Target frequencies (log-spaced)
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    scales = center_freq * fs / freqs
    
    # Compute CWT
    coefficients, frequencies = pywt.cwt(signal_data, scales, wavelet, 
                                         sampling_period=1.0/fs)
    
    # Power spectrum
    power = np.abs(coefficients) ** 2
    
    return frequencies, power


def compute_wavelet_coherence(signal1, signal2, fs=1.0, wavelet='morl', 
                               freq_min=1e-4, freq_max=1e-2, n_freqs=100, smooth_sigma=2):
    """
    Compute wavelet coherence between two signals.
    Following Grinsted et al. (2004).
    
    Coherence indicates similarity in time-frequency space.
    
    Parameters:
    -----------
    signal1, signal2 : array-like
        Input signals (must be same length)
    fs : float
        Sampling frequency in Hz
    wavelet : str
        Wavelet type
    freq_min, freq_max : float
        Frequency range for analysis
    n_freqs : int
        Number of frequency bins
    smooth_sigma : float
        Gaussian smoothing sigma for coherence calculation
        
    Returns:
    --------
    freqs : ndarray
        Frequency vector
    coherence : ndarray
        Wavelet coherence (shape: n_freqs × n_samples)
    """
    center_freq = pywt.central_frequency(wavelet)
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    scales = center_freq * fs / freqs
    
    # Compute CWT for both signals
    coef1, _ = pywt.cwt(signal1, scales, wavelet, sampling_period=1.0/fs)
    coef2, _ = pywt.cwt(signal2, scales, wavelet, sampling_period=1.0/fs)
    
    # Cross-wavelet spectrum
    cross_spectrum = coef1 * np.conj(coef2)
    
    # Auto-spectra
    power1 = np.abs(coef1) ** 2
    power2 = np.abs(coef2) ** 2
    
    # Wavelet coherence (smoothed)
    smooth_cross = gaussian_filter(np.abs(cross_spectrum), sigma=smooth_sigma)
    smooth_power1 = gaussian_filter(power1, sigma=smooth_sigma)
    smooth_power2 = gaussian_filter(power2, sigma=smooth_sigma)
    
    coherence = smooth_cross ** 2 / (smooth_power1 * smooth_power2 + 1e-10)
    
    return freqs, coherence


        
def plot_sync_events_with_templates(
    csv_path: str,
    templates_dir: str = "tilt_templates",
    csv_templates_dir: str = None,         # Directory with CSV templates (optional)
    denoised_dir_ingv: str = "/home/owen/tilt_validation/bandpassed_exports_001_01_csv_only_utc/ingv/",
    denoised_dir_experiment: str = "/home/owen/tilt_validation/bandpassed_exports_001_01_csv_only_utc/experiment/",
    station_dfs: dict = None,           # Optional raw station dataframes with 'datetime' column
    segment_duration_s: int = 3000,     # >= 3000 s recommended for 0.001 Hz
    time_unit: str = "minutes",
    detrend_type: str = "linear",       # "linear" | "constant" | None
    normalize: str = None,              # None | "zscore" | "maxabs"
    y_limit: float = 0.02,
    save_dir: str = None,
    debug: bool = True,
    band_hz: tuple = (0.001, 0.01),     # SAME passband both sides
    filter_order: int = 4,
    max_plot_templates_per_station: int = 8,
    # -------- EVENT SELECTION CRITERIA --------
    min_correlation: float = None,      # Minimum peak correlation (e.g., 0.3)
    min_snr_db: float = None,           # Minimum SNR as linear ratio (e.g., 2.0)
    max_snr_db: float = None,           # Maximum SNR as linear ratio (e.g., 10.0)
    stations_filter: list = None,       # Filter by stations (e.g., ["EEC1", "ECPN"])
    templates_filter: list = None,      # Filter by templates (e.g., ["template1", "template2"])
    sims_filter: list = None,           # Filter by sims (e.g., ["sim1", "sim2"])
    dedup_window_fraction: float = 0.5, # Deduplication window as fraction of segment_duration_s
    p_wave_tolerance_hours: float = 0.1,# P wave matching tolerance in hours
    # -------- WAVELET ANALYSIS --------
    enable_wavelet_analysis: bool = False,  # Enable CWT and coherence plots
    wavelet_save_dir: str = None,       # Separate directory for wavelet plots
):
 

    # ---------------- helpers ----------------
    def _butter_bandpass_zero_phase(x, fs, f1, f2, order=4):
        from scipy.signal import butter, sosfiltfilt
        x = np.asarray(x, float)
        if fs is None or fs <= 0 or f2 >= 0.5*fs:
            return x  # invalid fs/band → skip
        nyq = 0.5 * fs
        lo = max(f1 / nyq, 1e-6)
        hi = min(f2 / nyq, 0.999999)
        if not (0 < lo < hi < 1):
            return x
        sos = butter(order, [lo, hi], btype="bandpass", output="sos")
        return sosfiltfilt(sos, x)

    def _prep_signal(x: np.ndarray, fs: float) -> np.ndarray:
        """Detrend + (optional normalize) + SAME band-pass."""
        x = np.asarray(x, float).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return x
        if detrend_type in ('linear', 'constant'):
            x = scipy_detrend(x, type=detrend_type)
        # SAME band-pass (even if already denoised upstream; ensures parity)
        x = _butter_bandpass_zero_phase(x, fs, band_hz[0], band_hz[1], order=filter_order)
        if normalize == 'zscore':
            mu, sd = np.mean(x), np.std(x)
            if sd > 0: x = (x - mu) / sd
        elif normalize == 'maxabs':
            ma = np.max(np.abs(x))
            if ma > 0: x = x / ma
        return x

    def _resample_to_grid(x, t, t_grid, seg_dur):
        """Resample x(t) to uniform grid t_grid spanning [0, seg_dur]."""
        x = np.asarray(x, float).ravel()
        t = np.asarray(t, float).ravel()
        if x.size < 1:
            return np.zeros_like(t_grid)
        if x.size == 1 or (t[-1] - t[0]) == 0:
            return np.full_like(t_grid, x[0], dtype=float)
        t_norm = (t - t[0]) / (t[-1] - t[0]) * seg_dur
        f = interp1d(t_norm, x, kind="linear", bounds_error=False, fill_value="extrapolate")
        return f(t_grid)

    def _estimate_fs_from_seconds(t_sec: np.ndarray):
        if t_sec.size < 2:
            return None
        dt = np.median(np.diff(t_sec))
        return (1.0 / dt) if dt > 0 else None

    def _max_corr_and_lag(x, y, dt):
        """
        Pearson correlation for all circular lags (via FFT-like full xcorr):
        returns (r_max, lag_sec). x and y must be same length, already preprocessed.
        """
        x = np.asarray(x, float); y = np.asarray(y, float)
        n = min(len(x), len(y))
        x = x[:n]; y = y[:n]
        # demean + std-norm (global for the segment)
        x0 = x - x.mean(); y0 = y - y.mean()
        sx = x0.std(); sy = y0.std()
        if sx == 0 or sy == 0:
            return np.nan, 0.0
        x0 /= sx; y0 /= sy
        # full cross-correlation (valid lags from -(n-1) ... +(n-1))
        corr = np.correlate(x0, y0, mode='full') / n
        # best positive correlation
        k = np.argmax(corr)
        lag_samples = k - (n - 1)
        lag_sec = lag_samples * dt
        r_max = corr[k]
        return float(r_max), float(lag_sec)

    def _shift_series(y, t_grid, lag_sec):
        """Shift y by lag_sec using linear interpolation over t_grid."""
        shifted_t = t_grid - lag_sec
        f = interp1d(t_grid, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        return f(shifted_t)

    def _load_and_attach_datetime(den_path: Path, raw_df: pd.DataFrame, raw_fallback_df: pd.DataFrame = None, who: str = "EEC1"):
        """Load denoised CSV; ensure 'denoised' & 'datetime'. Borrow datetime from raw if missing."""
        if not den_path.exists():
            if debug: print(f"[DENOISED] Missing file for {who}: {den_path}")
            return None
        df = pd.read_csv(den_path)
        
        # Accept either 'denoised' or 'bandpassed' column names
        signal_col = None
        if "denoised" in df.columns:
            signal_col = "denoised"
        elif "bandpassed" in df.columns:
            signal_col = "bandpassed"
            df["denoised"] = df["bandpassed"]  # Rename for consistency
        
        if signal_col is None:
            if debug: print(f"[WARN] {den_path.name} lacks 'denoised' or 'bandpassed' column for {who}")
            return None
        dt_col = next((c for c in ["datetime","time","timestamp","DateTime","Datetime","time_seconds"] if c in df.columns), None)
        if dt_col is None:
            base = raw_df if raw_df is not None else raw_fallback_df
            if base is None or "datetime" not in base.columns:
                if debug: print(f"[ERROR] No datetime to borrow for {who}")
                return None
            base_dt = pd.to_datetime(base["datetime"], errors="coerce").dropna().reset_index(drop=True)
            n = min(len(df), len(base_dt))
            if len(df) != len(base_dt) and debug:
                print(f"[WARN] {who}: length mismatch (denoised={len(df)}, raw_dt={len(base_dt)}) → trimming to {n}")
            df = df.iloc[:n].copy()
            df["datetime"] = base_dt.iloc[:n].to_numpy()
        else:
            # Special handling for time_seconds column (elapsed seconds, not datetime)
            if dt_col == "time_seconds":
                # For experiment stations, borrow datetime from raw data if available
                base = raw_df if raw_df is not None else raw_fallback_df
                if base is not None and "datetime" in base.columns:
                    base_dt = pd.to_datetime(base["datetime"], errors="coerce").dropna().reset_index(drop=True)
                    n = min(len(df), len(base_dt))
                    if len(df) != len(base_dt) and debug:
                        print(f"[INFO] {who}: Using raw datetime for time_seconds data (denoised={len(df)}, raw_dt={len(base_dt)}) → using {n}")
                    df = df.iloc[:n].copy()
                    df["datetime"] = base_dt.iloc[:n].to_numpy()
                else:
                    # No raw datetime available - construct from time_seconds
                    # Experiment data starts at 2023-07-23 23:00:00 (UTC+1 clock → UTC) and ends at 2023-08-02 23:59:59
                    if debug: print(f"[INFO] {who}: Constructing datetime from time_seconds (reference: 2023-07-23 23:00:00)")
                    reference_time = pd.Timestamp("2023-07-23 23:00:00")
                    df["datetime"] = reference_time + pd.to_timedelta(df["time_seconds"], unit='s')
            else:
                df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=["datetime"]).reset_index(drop=True)
        if debug:
            print(f"[DENOISED] {who}: {len(df)} rows, span {df['datetime'].min()} → {df['datetime'].max()}")
        return df

    # ---------------- load events ----------------
    csv_path = Path(csv_path)
    if csv_path.is_dir():
        files = list(csv_path.glob("*.csv"))
        if not files:
            raise ValueError(f"No CSV files found in directory: {csv_path}")
        csv_path = files[0]
        if debug: print(f"[INFO] Using CSV file: {csv_path}")

    events_df = pd.read_csv(csv_path)
    required = {"station","peak_time_dt","template","sim","event_id"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"CSV must contain {sorted(required)}; missing {sorted(missing)}")
    events_df["peak_time_dt"] = pd.to_datetime(events_df["peak_time_dt"], errors="coerce")
    events_df = events_df.dropna(subset=["peak_time_dt"])
    
    # Filter by stations if stations_filter is provided, otherwise keep all
    if stations_filter is not None:
        keep = events_df["station"].astype(str).str.upper().isin([s.upper() for s in stations_filter])
        events_df = events_df.loc[keep].copy()
        if debug:
            print(f"[INFO] Filtered to stations: {stations_filter}")
    
    if events_df.empty:
        raise ValueError("No events remaining after station filtering.")
    
    # ---------------- load SNR data from peaks_merged.csv if needed ----------------
    if "snr_linear" not in events_df.columns or events_df["snr_linear"].isna().all():
        # Try to load peaks_merged.csv to get SNR values
        peaks_csv_candidates = [
            csv_path.parent / "peaks_merged.csv",
            Path("/home/owen/tilt_validation/merged_csv/peaks_merged.csv"),
            Path("peaks_merged.csv")
        ]
        
        peaks_df = None
        for peaks_path in peaks_csv_candidates:
            if peaks_path.exists():
                try:
                    peaks_df = pd.read_csv(peaks_path)
                    if "snr_linear" in peaks_df.columns:
                        peaks_df["peak_time_dt"] = pd.to_datetime(peaks_df["peak_time_dt"], errors="coerce")
                        if debug:
                            print(f"[INFO] Loaded SNR data from: {peaks_path}")
                            print(f"[INFO] Found {len(peaks_df)} peaks with SNR data")
                        break
                except Exception as e:
                    if debug: print(f"[WARN] Failed to load {peaks_path}: {e}")
        
        # Merge SNR data into events_df
        if peaks_df is not None and "snr_linear" in peaks_df.columns:
            # Merge on common columns: station, peak_time_dt, template, sim
            merge_cols = ["station", "peak_time_dt", "template", "sim"]
            # Only merge SNR columns we need
            snr_cols = ["snr_db", "snr_linear", "peak_corr"]
            merge_cols_available = [c for c in merge_cols if c in events_df.columns and c in peaks_df.columns]
            snr_cols_available = [c for c in snr_cols if c in peaks_df.columns and c not in events_df.columns]
            
            if merge_cols_available:
                events_df = events_df.merge(
                    peaks_df[merge_cols_available + snr_cols_available].drop_duplicates(),
                    on=merge_cols_available,
                    how="left"
                )
                if debug:
                    print(f"[INFO] Merged SNR data into events_df")
                    print(f"[INFO] SNR linear values available: {events_df['snr_linear'].notna().sum()}/{len(events_df)}")
                    if "snr_db" in events_df.columns:
                        print(f"[INFO] SNR dB values available: {events_df['snr_db'].notna().sum()}/{len(events_df)}")
    
    # ---------------- apply event selection criteria ----------------
    if debug:
        print(f"\n[INFO] Applying event selection criteria...")
        print(f"[INFO] Events before filtering: {len(events_df)}")
    
    # Filter by correlation
    if min_correlation is not None:
        if "peak_corr" in events_df.columns:
            before = len(events_df)
            events_df = events_df[events_df["peak_corr"] >= min_correlation]
            if debug:
                print(f"[FILTER] Correlation >= {min_correlation}: removed {before - len(events_df)} events")
        else:
            if debug: print(f"[WARN] peak_corr column not found, skipping correlation filter")
    
    # Filter by SNR (using linear ratio)
    if min_snr_db is not None:
        if "snr_linear" in events_df.columns:
            before = len(events_df)
            events_df = events_df[events_df["snr_linear"] >= min_snr_db]
            if debug:
                print(f"[FILTER] SNR (linear) >= {min_snr_db}: removed {before - len(events_df)} events")
        else:
            if debug: print(f"[WARN] snr_linear column not found, skipping min SNR filter")
    
    if max_snr_db is not None:
        if "snr_linear" in events_df.columns:
            before = len(events_df)
            events_df = events_df[events_df["snr_linear"] <= max_snr_db]
            if debug:
                print(f"[FILTER] SNR (linear) <= {max_snr_db}: removed {before - len(events_df)} events")
        else:
            if debug: print(f"[WARN] snr_linear column not found, skipping max SNR filter")
    
    # Filter by stations
    if stations_filter is not None:
        before = len(events_df)
        stations_upper = [s.upper() for s in stations_filter]
        events_df = events_df[events_df["station"].astype(str).str.upper().isin(stations_upper)]
        if debug:
            print(f"[FILTER] Stations in {stations_filter}: removed {before - len(events_df)} events")
    
    # Filter by templates
    if templates_filter is not None:
        before = len(events_df)
        templates_lower = [t.lower() for t in templates_filter]
        events_df = events_df[events_df["template"].astype(str).str.lower().isin(templates_lower)]
        if debug:
            print(f"[FILTER] Templates in {templates_filter}: removed {before - len(events_df)} events")
    
    # Filter by sims
    if sims_filter is not None:
        before = len(events_df)
        sims_lower = [s.lower() for s in sims_filter]
        events_df = events_df[events_df["sim"].astype(str).str.lower().isin(sims_lower)]
        if debug:
            print(f"[FILTER] Sims in {sims_filter}: removed {before - len(events_df)} events")
    
    if events_df.empty:
        if debug: print("[ERROR] No events remaining after filtering!")
        return
    
    if debug:
        print(f"[INFO] Events after filtering: {len(events_df)}")
    
    # ---------------- load and merge earthquake P wave data ----------------
    # Try to find earthquakes CSV with P wave data in the same directory or common locations
    eq_csv_candidates = [
        Path("/home/owen/tilt_validation/earthquakes_merged_utc.csv")
        #csv_path.parent / "earthquakes_merged_utc.csv",
        #Path("/home/owen/tilt_validation/earthquakes_merged_utc.csv"),
        #Path("earthquakes_merged_utc.csv")
    ]
    
    earthquakes_df = None
    for eq_path in eq_csv_candidates:
        if eq_path.exists():
            try:
                earthquakes_df = pd.read_csv(eq_path)
                if "p_wave_eta" in earthquakes_df.columns:
                    earthquakes_df["datetime"] = pd.to_datetime(earthquakes_df["datetime"], errors="coerce")
                    earthquakes_df["p_wave_eta"] = pd.to_datetime(earthquakes_df["p_wave_eta"], errors="coerce")
                    if debug:
                        print(f"[INFO] Loaded earthquake P wave data from: {eq_path}")
                        print(f"[INFO] Found {len(earthquakes_df)} earthquakes with P wave data")
                    break
            except Exception as e:
                if debug: print(f"[WARN] Failed to load {eq_path}: {e}")
    
    # Merge P wave data with events based on temporal proximity
    if earthquakes_df is not None and "p_wave_eta" in earthquakes_df.columns:
        # For each event, find matching earthquakes based on origin time proximity to detected peak
        # and drop any event whose 3333 s plotting window overlaps a catalog P-wave arrival.
        tolerance = pd.Timedelta(hours=p_wave_tolerance_hours)
        half_window = pd.Timedelta(seconds=segment_duration_s / 2)
        guard_pad = tolerance
        contaminated_ids = set()
        eq_p_times = earthquakes_df["p_wave_eta"].dropna().sort_values()
        
        if debug:
            print(f"[INFO] Attempting to match {len(events_df)} events with {len(earthquakes_df)} earthquakes")
            print(f"[INFO] Event time range: {events_df['peak_time_dt'].min()} to {events_df['peak_time_dt'].max()}")
            print(f"[INFO] Earthquake time range: {earthquakes_df['datetime'].min()} to {earthquakes_df['datetime'].max()}")
        
        if not eq_p_times.empty:
            for event_id, group in events_df.groupby("event_id"):
                earliest_peak = group["peak_time_dt"].min()
                if pd.isna(earliest_peak):
                    continue
                window_start = earliest_peak - half_window - guard_pad
                window_end = earliest_peak + half_window + guard_pad
                if ((eq_p_times >= window_start) & (eq_p_times <= window_end)).any():
                    contaminated_ids.add(event_id)
            if contaminated_ids:
                if debug:
                    print(f"[P-WAVE] Removing {len(contaminated_ids)} event_ids that overlap catalog P arrivals "
                          f"within ±{half_window.total_seconds()/60:.1f} min (+ tolerance).")
                events_df = events_df[~events_df["event_id"].isin(contaminated_ids)].copy()
                if events_df.empty:
                    if debug:
                        print("[WARN] All events removed after P-wave contamination guard.")
                    return
        
        p_wave_matches = []
        for idx, event_row in events_df.iterrows():
            peak_time = event_row["peak_time_dt"]
            
            # Strategy 1: Match by earthquake origin time (detected peak might be near origin or surface wave)
            time_diffs_origin = (earthquakes_df["datetime"] - peak_time).abs()
            nearby_origin = earthquakes_df[time_diffs_origin <= tolerance]
            
            # Strategy 2: Match by P wave arrival time (detected peak might be the P wave)
            time_diffs_p = (earthquakes_df["p_wave_eta"] - peak_time).abs()
            nearby_p = earthquakes_df[time_diffs_p <= tolerance]
            
            # Use whichever strategy finds closer matches
            if not nearby_origin.empty or not nearby_p.empty:
                if not nearby_origin.empty and not nearby_p.empty:
                    # Use the strategy with the closest match
                    min_diff_origin = time_diffs_origin[time_diffs_origin <= tolerance].min()
                    min_diff_p = time_diffs_p[time_diffs_p <= tolerance].min()
                    use_origin = min_diff_origin < min_diff_p
                elif not nearby_origin.empty:
                    use_origin = True
                else:
                    use_origin = False
                
                if use_origin:
                    closest_idx = time_diffs_origin[time_diffs_origin <= tolerance].idxmin()
                    match_type = "origin"
                else:
                    closest_idx = time_diffs_p[time_diffs_p <= tolerance].idxmin()
                    match_type = "P-wave"
                
                eq_datetime = earthquakes_df.loc[closest_idx, "datetime"]
                p_eta = earthquakes_df.loc[closest_idx, "p_wave_eta"]
                
                p_wave_matches.append({
                    "event_idx": idx,
                    "p_wave_eta": p_eta,
                    "eq_datetime": eq_datetime,
                    "eq_magnitude": earthquakes_df.loc[closest_idx, "magnitude"] if "magnitude" in earthquakes_df.columns else None,
                    "match_type": match_type
                })
                
                if debug:
                    time_diff = (p_eta - peak_time).total_seconds() / 60
                    print(f"[MATCH] Event {idx} peak={peak_time} matched to EQ origin={eq_datetime}, P-wave={p_eta} (Δ={time_diff:.1f} min, via {match_type})")
        
        # Add P wave data to events_df
        if p_wave_matches:
            for match in p_wave_matches:
                events_df.loc[match["event_idx"], "p_wave_eta"] = match["p_wave_eta"]
                events_df.loc[match["event_idx"], "eq_datetime"] = match["eq_datetime"]
                if match["eq_magnitude"] is not None:
                    events_df.loc[match["event_idx"], "eq_magnitude"] = match["eq_magnitude"]
            if debug:
                print(f"[INFO] ✓ Matched {len(p_wave_matches)} events with earthquake P waves")
                print(f"[INFO] events_df now has columns: {list(events_df.columns)}")
        else:
            if debug: print("[WARN] No P wave arrivals matched to events within tolerance window")
    else:
        if debug: print("[WARN] No earthquake P wave data found - P waves will not be plotted")

    # ---------------- deduplicate events within template duration ----------------
    # Events within ~template duration are likely the same physical event
    # Strategy: For each event_id group, keep only the earliest peak time
    dedup_window = pd.Timedelta(seconds=segment_duration_s * dedup_window_fraction)
    
    if debug:
        print(f"\n[INFO] Deduplicating events within {dedup_window.total_seconds():.0f}s ({dedup_window.total_seconds()/60:.1f} min) window...")
        print(f"[INFO] Events before deduplication: {len(events_df)}")
        print(f"[INFO] Unique event_ids: {events_df['event_id'].nunique()}")
    
    # Sort by time
    events_df = events_df.sort_values('peak_time_dt').reset_index(drop=True)
    
    # Group-based deduplication: within each event_id, keep all rows (they're part of same sync event)
    # But between different event_ids, remove those that are too close in time
    keep_indices = []
    event_groups = events_df.groupby('event_id')
    
    for event_id, group in event_groups:
        # Get earliest peak time for this event_id
        earliest_peak = group['peak_time_dt'].min()
        
        # Check if this event_id is a duplicate of an already-kept event
        is_duplicate = False
        for kept_idx in keep_indices:
            kept_time = events_df.loc[kept_idx, 'peak_time_dt']
            time_diff = abs((earliest_peak - kept_time).total_seconds())
            if time_diff < dedup_window.total_seconds():
                is_duplicate = True
                if debug:
                    print(f"  [DEDUP] Event_id {event_id} at {earliest_peak} is duplicate (Δ={time_diff:.0f}s from existing event)")
                break
        
        if not is_duplicate:
            # Keep all rows from this event_id
            keep_indices.extend(group.index.tolist())
    
    events_df_original = events_df.copy()  # Keep original for reference
    events_df = events_df.loc[keep_indices].sort_values('peak_time_dt').reset_index(drop=True)
    
    if debug:
        print(f"[INFO] Events after deduplication: {len(events_df)}")
        print(f"[INFO] Unique event_ids after dedup: {events_df['event_id'].nunique()}")
        print(f"[INFO] Removed {len(events_df_original) - len(events_df)} duplicate event detections")


    template_map = {"template1":"T1","template2":"T2","template3":"T3","template4":"T4"}

    # ---------------- load templates ----------------
    templates, _, _ = load_tilt_templates(templates_dir, csv_dir=csv_templates_dir)
    if debug:
        print(f"[INFO] Templates loaded: {list(templates.keys())}")

    # ---------------- dynamically load denoised data for all stations in events ----------------
    # Get unique stations from the filtered events
    unique_stations = events_df["station"].astype(str).str.upper().unique()
    if debug:
        print(f"\n[INFO] Loading denoised data for stations: {list(unique_stations)}")
    
    # Dictionary to store loaded denoised data for each station
    station_denoised_data = {}
    
    # Build paths for all known stations from both directories
    denoised_paths = {
        # INGV stations - look in ingv directory with flexible naming
        "EEC1": Path(denoised_dir_ingv) / "EEC1_0p001-0p01.csv",
        "EC1": Path(denoised_dir_ingv) / "EEC1_0p001-0p01.csv",  # Same as EEC1
        "ECPN": Path(denoised_dir_ingv) / "ECPN_0p001-0p01.csv",
        # Experiment stations - look in experiment directory
        "EMAS": Path(denoised_dir_experiment) / "EMAS_0p001-0p01Hz_bp.csv",
        "ECOR": Path(denoised_dir_experiment) / "ECOR_0p001-0p01Hz_bp.csv",
        "ECIT": Path(denoised_dir_experiment) / "ECIT_0p001-0p01Hz_bp.csv",
        "EC10": Path(denoised_dir_experiment) / "EC10_0p001-0p01Hz_bp.csv",
    }
    
    # Also try alternate naming patterns if primary not found
    alternate_paths = {
        "EEC1": [
            Path(denoised_dir_ingv) / "EEC1_0p001-0p01Hz_bp.csv",  # Alternate naming
            Path(denoised_dir_experiment) / "EEC1_0p001-0p01Hz_bp.csv",  # May be in experiment dir
        ],
        "EC1": [
            Path(denoised_dir_ingv) / "EC1_0p001-0p01Hz_bp.csv",
            Path(denoised_dir_experiment) / "EC1_0p001-0p01Hz_bp.csv",
        ],
        "ECPN": [
            Path(denoised_dir_ingv) / "ECPN_0p001-0p01Hz_bp.csv",  # Alternate naming
            Path(denoised_dir_ingv) / "ECPN_additional_0p001-0p01.csv",
            Path(denoised_dir_experiment) / "ECPN_0p001-0p01Hz_bp.csv",  # May be in experiment dir
        ]
    }
    
    # Load data for each station present in events
    for station in unique_stations:
        station_upper = station.upper()
        
        # Try primary path first
        denoised_path = denoised_paths.get(station_upper)
        paths_to_try = [denoised_path] if denoised_path else []
        
        # Add alternate paths if available
        if station_upper in alternate_paths:
            paths_to_try.extend(alternate_paths[station_upper])
        
        loaded = False
        for path in paths_to_try:
            if path and path.exists():
                try:
                    # Load denoised data
                    raw_df = station_dfs.get(station_upper) if station_dfs else None
                    raw_fallback_df = None
                    
                    # Special handling for EEC1/EC1 fallback
                    if station_upper == "EEC1" and station_dfs:
                        raw_fallback_df = station_dfs.get("EC1")
                    elif station_upper == "EC1" and station_dfs:
                        raw_fallback_df = station_dfs.get("EEC1")
                    
                    denoised_data = _load_and_attach_datetime(
                        path,
                        raw_df=raw_df,
                        raw_fallback_df=raw_fallback_df,
                        who=station_upper
                    )
                    
                    station_denoised_data[station_upper] = denoised_data
                    loaded = True
                    
                    if debug:
                        print(f"  ✓ Loaded {station_upper} from {path.name}")
                        if denoised_data is not None:
                            print(f"    Time range: {denoised_data['datetime'].min()} to {denoised_data['datetime'].max()}")
                            print(f"    Samples: {len(denoised_data)}")
                    break  # Successfully loaded, stop trying paths
                    
                except Exception as e:
                    if debug:
                        print(f"  ⚠ Failed to load {station_upper} from {path}: {e}")
                    continue
        
        if not loaded:
            if debug:
                print(f"  ✗ No denoised file found for {station_upper}")
                print(f"    Tried paths: {[str(p) for p in paths_to_try if p]}")
    
    if not station_denoised_data:
        raise ValueError(f"No denoised data loaded for any station in events: {list(unique_stations)}")
    
    if debug:
        print(f"[INFO] Successfully loaded denoised data for: {list(station_denoised_data.keys())}")

    # ---------------- plotting config ----------------
    unit_div = {"seconds":1, "minutes":60, "hours":3600, "days":86400}[time_unit]
    t_common = np.linspace(0, segment_duration_s, 3000)  # 1 Hz plotting grid over the event window
    
    # Dynamic color assignment for stations
    station_colors = {
        "EEC1": "#1f77b4",  # blue
        "EC1": "#1f77b4",   # blue (same as EEC1)
        "ECPN": "#2ca02c",  # green
        "EMAS": "#d62728",  # red
        "ECOR": "#9467bd",  # purple
        "ECIT": "#8c564b",  # brown
        "EC10": "#e377c2",  # pink
    }
    
    # Assign colors to any stations not predefined
    default_colors = ["#ff7f0e", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78"]
    color_idx = 0
    for station in unique_stations:
        if station.upper() not in station_colors:
            station_colors[station.upper()] = default_colors[color_idx % len(default_colors)]
            color_idx += 1
    
    template_colors = ["#E74C3C","#E67E22","#F39C12","#D35400","#C0392B",
                       "#FF6B6B","#FF8C42","#FFA600","#FF4D6D","#FF6F00"]

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ---------------- per-event loop ----------------
    for event_id, group in events_df.groupby("event_id"):
        earliest_peak = group["peak_time_dt"].min()
        half = segment_duration_s / 2
        start_time = earliest_peak - pd.Timedelta(seconds=half)
        end_time   = earliest_peak + pd.Timedelta(seconds=half)
        if debug:
            print(f"\n{'='*60}\n[EVENT {event_id}] window: {start_time} → {end_time}")

        # Slice & preprocess observed segments for all stations in this event
        obs_payloads = {}  # station -> dict(t_grid, y_pre, color)
        
        # Get unique stations for this specific event
        event_stations = group["station"].astype(str).str.upper().unique()
        
        for station_name in event_stations:
            station_upper = station_name.upper()
            
            # Get the denoised data for this station
            df_den = station_denoised_data.get(station_upper)
            if df_den is None:
                if debug:
                    print(f"[{station_upper}] No denoised data available (not loaded)")
                continue
            
            # Extract segment within time window
            seg = df_den[(df_den["datetime"] >= start_time) & (df_den["datetime"] <= end_time)].copy()
            if seg.empty:
                if debug: 
                    print(f"[{station_upper}] No samples in window.")
                continue
            
            # Preprocess the segment
            x_raw = seg["denoised"].to_numpy(float)
            t_s  = (seg["datetime"] - seg["datetime"].iloc[0]).dt.total_seconds().to_numpy()
            fs   = _estimate_fs_from_seconds(t_s)
            x_pre = _prep_signal(x_raw, fs)  # detrend + SAME band-pass (+ optional normalize)
            y_obs = _resample_to_grid(x_pre, t_s, t_common, segment_duration_s)
            
            # Store with assigned color
            obs_payloads[station_upper] = {
                "y": y_obs, 
                "color": station_colors.get(station_upper, "#999999")
            }

            if debug:
                print(f"[OBS] {station_upper}: fs≈{fs:.3f} Hz, preprocess= detrend:{detrend_type}, band:{band_hz}")

        if not obs_payloads:
            if debug: 
                print(f"[SKIP] Event {event_id}: no observed data in window for any station.")
            continue

        # Collect & preprocess templates for stations present in group
        tpl_curves = []  # (label, station, y_pre_resampled, r_max, lag_sec)
        for _, row in group.iterrows():
            st_csv = str(row.get("station","")).upper()
            sim = str(row.get("sim",""))
            csv_tpl = str(row.get("template","")).lower()
            tpl_key = template_map.get(csv_tpl)
            if tpl_key is None or tpl_key not in templates:
                if debug: print(f"[TPL SKIP] Unknown template '{csv_tpl}'"); continue
            # resolve sim
            sim_key = sim if sim in templates[tpl_key] else None
            if sim_key is None:
                for pfx in ["filt_","raw_",""]:
                    cand = f"{pfx}{sim}"
                    if cand in templates[tpl_key]:
                        sim_key = cand; break
            if sim_key is None:
                if debug: print(f"[TPL SKIP] Sim '{sim}' not in {tpl_key}"); continue
            # station key - try to match the CSV station name to template station names
            # First try exact match, then try common variations
            target_station = st_csv  # Use the station name from CSV
            
            # Map common variations
            station_variations = {
                "EEC1": ["EEC1", "EC1"],
                "EC1": ["EC1", "EEC1"],
                "ECPN": ["ECPN"],
                "EMAS": ["EMAS"],
                "ECOR": ["ECOR"],
                "ECIT": ["ECIT"],
                "EC10": ["EC10"],
            }
            
            # Try to find this station in the template
            st_key = None
            search_names = station_variations.get(st_csv, [st_csv])
            
            for search_name in search_names:
                if search_name in templates[tpl_key][sim_key]:
                    st_key = search_name
                    break
                # Try case-insensitive match
                for k in templates[tpl_key][sim_key].keys():
                    if k.upper() == search_name.upper():
                        st_key = k
                        break
                if st_key:
                    break
            
            if st_key is None:
                if debug: 
                    print(f"[TPL SKIP] Station '{st_csv}' not found in {tpl_key}/{sim_key}")
                    print(f"    Available stations: {list(templates[tpl_key][sim_key].keys())}")
                continue

            data = templates[tpl_key][sim_key][st_key]
            tpl_x_raw = np.asarray(data["x"], float)
            tpl_t = np.asarray(data.get("t", np.arange(len(tpl_x_raw))), dtype=float)

            # Estimate template fs (if t provided and spans time)
            fs_tpl = None
            if tpl_t.size > 1 and (tpl_t[-1] - tpl_t[0]) > 0:
                fs_tpl = (len(tpl_t) - 1) / (tpl_t[-1] - tpl_t[0])

            # SAME preprocessing on template
            tpl_x_pre = _prep_signal(tpl_x_raw, fs_tpl if fs_tpl else 1.0)

            # Resample template to the same plotting grid
            y_tpl = _resample_to_grid(tpl_x_pre, tpl_t if tpl_t.size>1 else np.linspace(0, segment_duration_s, len(tpl_x_pre)),
                                      t_common, segment_duration_s)

            # Compute best lag & correlation against the corresponding observed track (if present)
            target_obs = st_csv  # Use the station name directly
            r_max, lag_sec = (np.nan, 0.0)
            
            if target_obs in obs_payloads:
                dt = t_common[1] - t_common[0]
                # Calculate correlation and lag with unscaled template
                r_max, lag_sec = _max_corr_and_lag(y_tpl, obs_payloads[target_obs]["y"], dt)

            # Extract just the number from sim (e.g., "sim1" -> "1")
            sim_num = sim.replace("sim", "").replace("Sim", "").replace("SIM", "")
            tpl_num = csv_tpl.upper().replace('TEMPLATE', '').replace('T', '')
            label = f"{st_csv} — s{sim_num} T{tpl_num} (r={r_max:.2f})"
            
            # Store UNSCALED template - scaling will happen after shifting
            tpl_curves.append((label, target_obs, y_tpl, r_max, lag_sec))

            if debug:
                print(f"[TPL] {st_csv}/{sim}/{csv_tpl}: fs_tpl≈{fs_tpl if fs_tpl else 'n/a'} Hz, r_max={r_max:.2f}, lag={lag_sec:.1f}s")

        # Limit clutter if too many templates
        if len(tpl_curves) > max_plot_templates_per_station:
            tpl_curves = tpl_curves[:max_plot_templates_per_station]

        # ---------------- draw ----------------
        fig, ax = plt.subplots(figsize=(13.5, 6))

        # Observed (preprocessed)
        for who, payload in obs_payloads.items():
            ax.plot(t_common / unit_div, payload["y"], color=payload["color"], lw=2.0, ls='-',
                    alpha=0.95, label=f"{who}", zorder=8)

        # Templates (preprocessed + shifted by τ* + fitted scale/offset to match obs)
        for j, (lab, target_obs, y_tpl, rmax, lag_sec) in enumerate(tpl_curves):
            col = ["#E74C3C","#E67E22","#F39C12","#D35400","#C0392B",
                   "#FF6B6B","#FF8C42","#FFA600","#FF4D6D","#FF6F00"][j % 10]
            
            # Step 1: Shift the template by lag
            y_shift = _shift_series(y_tpl, t_common, lag_sec) if np.isfinite(rmax) else y_tpl
            
            # Step 2: Fit scale and offset using least-squares on overlapping window
            scale_factor = 1.0
            offset = 0.0
            
            if target_obs in obs_payloads:
                obs_y = obs_payloads[target_obs]["y"]
                
                # Use central 80% window where template and obs should align best
                # This avoids edge effects from shifting
                n = len(obs_y)
                start_idx = int(n * 0.1)
                end_idx = int(n * 0.9)
                
                obs_window = obs_y[start_idx:end_idx]
                tpl_window = y_shift[start_idx:end_idx]
                
                # Least-squares fit: obs ≈ a*tpl + b
                # Solve: [tpl, ones] * [a; b] = obs
                if len(tpl_window) >= 10:
                    try:
                        T = np.column_stack([tpl_window, np.ones_like(tpl_window)])
                        sol, *_ = np.linalg.lstsq(T, obs_window, rcond=None)
                        scale_factor, offset = sol
                        
                        # Sanity checks on fitted parameters
                        if not np.isfinite(scale_factor):
                            scale_factor = 1.0
                        if not np.isfinite(offset):
                            offset = 0.0
                        
                        # Limit extreme scaling
                        if abs(scale_factor) > 100:
                            if debug:
                                print(f"[SCALE WARNING] Extreme scale {scale_factor:.1f}x for {lab}, limiting to ±100x")
                            scale_factor = 100 if scale_factor > 0 else -100
                        elif abs(scale_factor) < 0.01:
                            if debug:
                                print(f"[SCALE WARNING] Tiny scale {scale_factor:.4f}x for {lab}, limiting to ±0.01x")
                            scale_factor = 0.01 if scale_factor > 0 else -0.01
                        
                        # Apply scale and offset
                        y_shift = scale_factor * y_shift + offset
                        
                        if debug:
                            print(f"[FIT] {lab}: scale={scale_factor:.2f}, offset={offset:.4f}")
                            
                    except Exception as e:
                        if debug:
                            print(f"[FIT ERROR] Failed to fit {lab}: {e}, using unscaled template")
            
            ax.plot(t_common / unit_div, y_shift, color=col, lw=1.8, ls='--', alpha=0.9, label=lab, zorder=6)

        # Peaks - use darker version of station color for vertical lines
        peak_colors = {
            "EEC1": "#8b5cf6",  # purple
            "EC1": "#8b5cf6",   # purple
            "ECPN": "#22c55e",  # green
            "EMAS": "#b91c1c",  # dark red
            "ECOR": "#6b21a8",  # dark purple
            "ECIT": "#78350f",  # dark brown
            "EC10": "#be185d",  # dark pink
        }
        
        for st_name, group_st in group.groupby(group["station"].astype(str).str.upper()):
            # Use peak color if defined, otherwise use a darker shade of station color
            c = peak_colors.get(st_name, "#666666")
            for pk in sorted(group_st["peak_time_dt"].unique()):
                xpk = (pd.to_datetime(pk) - start_time).total_seconds() / unit_div
                ax.axvline(xpk, color=c, ls="-", lw=2.5, alpha=0.85)
                y_top = ax.get_ylim()[1] if y_limit is None else y_limit
                ax.text(xpk, y_top*0.96, st_name, rotation=90, ha="right", va="top",
                        fontsize=8, color=c, alpha=0.95)
        
        # P wave arrivals (if p_wave_eta column exists) - deduplicate to avoid multiples
        if "p_wave_eta" in group.columns:
            if debug:
                print(f"[P-WAVE] Checking for P wave data in event {event_id}")
                print(f"[P-WAVE] p_wave_eta values: {group['p_wave_eta'].tolist()}")
            
            # Get all P wave arrival times and deduplicate within 1 second
            p_wave_arrivals = []
            seen_times = []
            for _, row in group.iterrows():
                if pd.notna(row.get("p_wave_eta")):
                    try:
                        p_datetime = pd.to_datetime(row["p_wave_eta"], errors="coerce")
                        if pd.notna(p_datetime):
                            # Check if we've already seen this time (within 1 second tolerance)
                            is_duplicate = False
                            for seen_time in seen_times:
                                if abs((p_datetime - seen_time).total_seconds()) < 1.0:
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                p_wave_arrivals.append(p_datetime)
                                seen_times.append(p_datetime)
                                if debug:
                                    print(f"[P-WAVE] Found unique P wave arrival: {p_datetime}")
                            elif debug:
                                print(f"[P-WAVE] Skipped duplicate P wave: {p_datetime}")
                    except Exception as e:
                        if debug: print(f"[P-WAVE] Error parsing p_wave_eta: {e}")
            
            if debug:
                print(f"[P-WAVE] Total unique P wave arrivals: {len(p_wave_arrivals)}")
            
            # Plot P wave arrivals as vertical lines with magnitude annotations
            for p_arrival in sorted(set(p_wave_arrivals)):
                xp = (p_arrival - start_time).total_seconds() / unit_div
                
                # Find magnitude for this P-wave arrival
                magnitude = None
                for _, row in group.iterrows():
                    if pd.notna(row.get("p_wave_eta")):
                        p_datetime = pd.to_datetime(row["p_wave_eta"], errors="coerce")
                        if pd.notna(p_datetime) and abs((p_datetime - p_arrival).total_seconds()) < 1.0:
                            # Found matching P-wave, get magnitude
                            if "magnitude" in row and pd.notna(row["magnitude"]):
                                magnitude = row["magnitude"]
                                break
                            elif "mag" in row and pd.notna(row["mag"]):
                                magnitude = row["mag"]
                                break
                
                if debug:
                    mag_str = f", mag={magnitude}" if magnitude is not None else ""
                    print(f"[P-WAVE] P wave at {p_arrival}, xp={xp:.2f} {time_unit}{mag_str}, window=[0, {segment_duration_s / unit_div:.2f}]")
                
                # Only plot if within the visible window
                if 0 <= xp <= segment_duration_s / unit_div:
                    ax.axvline(xp, color="#FF1744", ls="-", lw=2.5, alpha=0.85, zorder=9)
                    y_top = ax.get_ylim()[1] if y_limit is None else y_limit
                    
                    # Create label with magnitude if available
                    if magnitude is not None:
                        label_text = f"P\nM{magnitude:.1f}"
                    else:
                        label_text = "P"
                    
                    ax.text(xp, y_top*0.88, label_text, rotation=90, ha="right", va="top",
                            fontsize=8, color="#FF1744", alpha=0.95, fontweight="bold")
                    
                    if debug:
                        print(f"[P-WAVE] ✓ Plotted P wave at x={xp:.2f}")
                else:
                    if debug:
                        print(f"[P-WAVE] ✗ P wave outside window at x={xp:.2f}")
        else:
            if debug:
                print(f"[P-WAVE] No p_wave_eta column in event {event_id}")

        if y_limit is not None:
            ax.set_ylim(-abs(y_limit), abs(y_limit))

        event_time_str = pd.to_datetime(earliest_peak).strftime("%Y-%m-%d %H:%M:%S")
        
        # Create station list for title
        event_station_names = sorted(event_stations)
        station_str = " & ".join(event_station_names)
        
        ax.set_title(
            f"Event {event_id} @ {event_time_str}\n"
            f"{station_str}",
            fontsize=12, fontweight="bold", pad=8
        )
        ax.set_xlabel(f"Time ({time_unit})", fontsize=11)
        ax.set_ylabel("Tilt" if normalize is None else "Tilt (normalized)", fontsize=11)
        ax.grid(True, ls=":", alpha=0.35)

        # Compute average correlation and SNR for this event
        corr_values = [r for _, _, _, r, _ in tpl_curves if np.isfinite(r)]
        snr_values = group["snr_linear"].dropna().values if "snr_linear" in group.columns else []
        
        avg_corr = np.mean(corr_values) if len(corr_values) > 0 else np.nan
        avg_snr = np.mean(snr_values) if len(snr_values) > 0 else np.nan
        
        # Create statistics text box (top left)
        stats_text = f"Avg. Correlation: {avg_corr:.3f}\nAvg. SNR (linear): {avg_snr:.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=0.9, alpha=1.0))

        # Main legend (upper right)
        leg = ax.legend(loc="upper right", fontsize=9, framealpha=1.0, facecolor="white")
        leg.get_frame().set_edgecolor("black"); leg.get_frame().set_linewidth(0.9)

        plt.tight_layout()
        if save_dir:
            # Create filename with station names
            station_prefix = "_".join(sorted(event_station_names)).lower()
            name = f"{station_prefix}_event_{int(event_id):04d}_proc_shifted.png" if str(event_id).isdigit() \
                   else f"{station_prefix}_event_{event_id}_proc_shifted.png"
            out = Path(save_dir) / name
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[SAVED] {name}")
        plt.show(); plt.close()
        
        # ============================================================================
        # WAVELET ANALYSIS (if enabled and multiple stations)
        # ============================================================================
        if enable_wavelet_analysis and len(event_station_names) >= 1:
            if debug:
                print(f"\n[WAVELET] Computing CWT and coherence for event {event_id}")
            
            # Setup wavelet save directory
            wav_dir = Path(wavelet_save_dir) if wavelet_save_dir else Path(save_dir) / "wavelet_analysis"
            wav_dir.mkdir(parents=True, exist_ok=True)
            
            # Estimate sampling frequency from first station
            fs = 1.0 / (t_common[1] - t_common[0])  # Hz
            
            if debug:
                print(f"[WAVELET] Sampling frequency: {fs:.4f} Hz")
                print(f"[WAVELET] Analyzing stations: {list(obs_payloads.keys())}")
            
            # ========================================================================
            # PART 1: Station vs Template Coherence (like your wavelet_coherence.py)
            # ========================================================================
            # For each station, compare its signal to its templates
            for station_name in obs_payloads.keys():
                # Get observed signal for this station
                obs_signal = obs_payloads[station_name]["y"]
                
                # Find templates for this station
                station_templates = [(lab, y_tpl, rmax) for lab, target_obs, y_tpl, rmax, lag_sec 
                                    in tpl_curves if target_obs == station_name]
                
                if not station_templates:
                    if debug:
                        print(f"[WAVELET] No templates for {station_name}, skipping")
                    continue
                
                # Use the best template (highest correlation)
                best_template = max(station_templates, key=lambda x: x[2] if np.isfinite(x[2]) else -1)
                tpl_label, tpl_signal, tpl_corr = best_template
                
                if debug:
                    print(f"[WAVELET] {station_name}: Using template {tpl_label} (r={tpl_corr:.3f})")
                
                try:
                    # Compute CWT for observed signal
                    freqs_obs, power_obs = compute_cwt(obs_signal, fs=fs, 
                                                      freq_min=band_hz[1], freq_max=band_hz[0])
                    
                    # Compute CWT for template (use shifted/scaled version from tpl_curves)
                    # Find the shifted template for this station
                    tpl_shifted = tpl_signal  # This is the unshifted template
                    for lab, target_obs, y_tpl, rmax, lag_sec in tpl_curves:
                        if target_obs == station_name and lab == tpl_label:
                            # Shift and scale like in the plot (using the _shift_series function defined above)
                            tpl_shifted = _shift_series(y_tpl, t_common, lag_sec) if np.isfinite(rmax) else y_tpl
                            
                            # Apply same scaling as in plot
                            n = len(obs_signal)
                            start_idx = int(n * 0.1)
                            end_idx = int(n * 0.9)
                            obs_window = obs_signal[start_idx:end_idx]
                            tpl_window = tpl_shifted[start_idx:end_idx]
                            
                            if len(tpl_window) >= 10:
                                T = np.column_stack([tpl_window, np.ones_like(tpl_window)])
                                sol, *_ = np.linalg.lstsq(T, obs_window, rcond=None)
                                scale_factor, offset = sol
                                if np.isfinite(scale_factor) and np.isfinite(offset):
                                    if abs(scale_factor) > 100:
                                        scale_factor = 100 if scale_factor > 0 else -100
                                    elif abs(scale_factor) < 0.01:
                                        scale_factor = 0.01 if scale_factor > 0 else -0.01
                                    tpl_shifted = scale_factor * tpl_shifted + offset
                            break
                    
                    freqs_tpl, power_tpl = compute_cwt(tpl_shifted, fs=fs,
                                                      freq_min=band_hz[1], freq_max=band_hz[0])
                    
                    # Compute wavelet coherence between observed and template
                    freqs_coh, coherence = compute_wavelet_coherence(
                        obs_signal, tpl_shifted, fs=fs,
                        freq_min=band_hz[1], freq_max=band_hz[0]
                    )
                    
                    # Create 3-panel plot: Obs CWT, Template CWT, Coherence
                    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
                    
                    time_num = t_common / unit_div
                    
                    # Panel 1: Observed Signal CWT
                    power_obs_db = 10 * np.log10(power_obs + 1e-10)
                    im1 = axes[0].pcolormesh(time_num, freqs_obs, power_obs_db,
                                            shading='auto', cmap='viridis',
                                            vmin=np.percentile(power_obs_db, 5),
                                            vmax=np.percentile(power_obs_db, 95))
                    axes[0].set_ylabel('Frequency (Hz)', fontsize=11)
                    axes[0].set_title(f'{station_name} - Observed Tilt CWT Power', 
                                     fontsize=12, fontweight='bold')
                    axes[0].set_yscale('log')
                    axes[0].grid(True, alpha=0.3)
                    cbar1 = plt.colorbar(im1, ax=axes[0])
                    cbar1.set_label('Power (dB)', fontsize=10)
                    
                    # Panel 2: Template CWT
                    power_tpl_db = 10 * np.log10(power_tpl + 1e-10)
                    im2 = axes[1].pcolormesh(time_num, freqs_tpl, power_tpl_db,
                                            shading='auto', cmap='viridis',
                                            vmin=np.percentile(power_tpl_db, 5),
                                            vmax=np.percentile(power_tpl_db, 95))
                    axes[1].set_ylabel('Frequency (Hz)', fontsize=11)
                    axes[1].set_title(f'Template CWT Power ({tpl_label})',
                                     fontsize=12, fontweight='bold')
                    axes[1].set_yscale('log')
                    axes[1].grid(True, alpha=0.3)
                    cbar2 = plt.colorbar(im2, ax=axes[1])
                    cbar2.set_label('Power (dB)', fontsize=10)
                    
                    # Panel 3: Wavelet Coherence
                    im3 = axes[2].pcolormesh(time_num, freqs_coh, coherence,
                                            shading='auto', cmap='hot',
                                            vmin=0, vmax=1)
                    axes[2].set_ylabel('Frequency (Hz)', fontsize=11)
                    axes[2].set_xlabel(f'Time ({time_unit})', fontsize=11)
                    axes[2].set_title(f'Wavelet Coherence: Observed vs Template',
                                     fontsize=12, fontweight='bold')
                    axes[2].set_yscale('log')
                    axes[2].grid(True, alpha=0.3)
                    cbar3 = plt.colorbar(im3, ax=axes[2])
                    cbar3.set_label('Coherence', fontsize=10)
                    
                    # Add event info
                    fig.suptitle(f'Event {event_id} @ {pd.to_datetime(earliest_peak).strftime("%Y-%m-%d %H:%M:%S")} - {station_name}',
                               fontsize=14, fontweight='bold')
                    
                    plt.tight_layout(rect=[0, 0, 1, 0.97])
                    
                    # Save
                    wav_name = f"wavelet_{station_name}_vs_template_event_{int(event_id):04d}.png" if str(event_id).isdigit() \
                              else f"wavelet_{station_name}_vs_template_event_{event_id}.png"
                    wav_out = wav_dir / wav_name
                    plt.savefig(wav_out, dpi=300, bbox_inches='tight')
                    print(f"[WAVELET SAVED] {wav_name}")
                    plt.close()
                    
                except Exception as e:
                    if debug:
                        print(f"[WAVELET ERROR] Failed for {station_name}: {e}")
                    plt.close()
            
            # ========================================================================
            # PART 2: Station-Station Coherence (if multiple stations)
            # ========================================================================
            if len(obs_payloads) >= 2:
                if debug:
                    print(f"[WAVELET] Computing station-station coherence")
                
                station_list = list(obs_payloads.keys())
                for i in range(len(station_list)):
                    for j in range(i+1, len(station_list)):
                        st1, st2 = station_list[i], station_list[j]
                        sig1 = obs_payloads[st1]["y"]
                        sig2 = obs_payloads[st2]["y"]
                        
                        if debug:
                            print(f"[WAVELET] Computing coherence: {st1} <-> {st2}")
                        
                        try:
                            # Compute CWT for each station
                            freqs1, power1 = compute_cwt(sig1, fs=fs, freq_min=band_hz[1], freq_max=band_hz[0])
                            freqs2, power2 = compute_cwt(sig2, fs=fs, freq_min=band_hz[1], freq_max=band_hz[0])
                            
                            # Compute wavelet coherence
                            freqs_coh, coherence = compute_wavelet_coherence(
                                sig1, sig2, fs=fs,
                                freq_min=band_hz[1], freq_max=band_hz[0]
                            )
                            
                            # Create 3-panel plot
                            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
                            
                            time_num = t_common / unit_div
                            
                            # Plot 1: CWT of station 1
                            power1_db = 10 * np.log10(power1 + 1e-10)
                            im1 = axes[0].pcolormesh(time_num, freqs1, power1_db,
                                                    shading='auto', cmap='viridis',
                                                    vmin=np.percentile(power1_db, 5),
                                                    vmax=np.percentile(power1_db, 95))
                            axes[0].set_ylabel('Frequency (Hz)', fontsize=11)
                            axes[0].set_title(f'{st1} - CWT Power', fontsize=12, fontweight='bold')
                            axes[0].set_yscale('log')
                            axes[0].grid(True, alpha=0.3)
                            cbar1 = plt.colorbar(im1, ax=axes[0])
                            cbar1.set_label('Power (dB)', fontsize=10)
                            
                            # Plot 2: CWT of station 2
                            power2_db = 10 * np.log10(power2 + 1e-10)
                            im2 = axes[1].pcolormesh(time_num, freqs2, power2_db,
                                                    shading='auto', cmap='viridis',
                                                    vmin=np.percentile(power2_db, 5),
                                                    vmax=np.percentile(power2_db, 95))
                            axes[1].set_ylabel('Frequency (Hz)', fontsize=11)
                            axes[1].set_title(f'{st2} - CWT Power', fontsize=12, fontweight='bold')
                            axes[1].set_yscale('log')
                            axes[1].grid(True, alpha=0.3)
                            cbar2 = plt.colorbar(im2, ax=axes[1])
                            cbar2.set_label('Power (dB)', fontsize=10)
                            
                            # Plot 3: Wavelet Coherence
                            im3 = axes[2].pcolormesh(time_num, freqs_coh, coherence,
                                                    shading='auto', cmap='RdYlBu_r',
                                                    vmin=0, vmax=1)
                            axes[2].set_ylabel('Frequency (Hz)', fontsize=11)
                            axes[2].set_xlabel(f'Time ({time_unit})', fontsize=11)
                            axes[2].set_title(f'Wavelet Coherence: {st1} <-> {st2}',
                                            fontsize=12, fontweight='bold')
                            axes[2].set_yscale('log')
                            axes[2].grid(True, alpha=0.3)
                            cbar3 = plt.colorbar(im3, ax=axes[2])
                            cbar3.set_label('Coherence', fontsize=10)
                            
                            # Add event info
                            fig.suptitle(f'Event {event_id} @ {pd.to_datetime(earliest_peak).strftime("%Y-%m-%d %H:%M:%S")}',
                                       fontsize=14, fontweight='bold')
                            
                            plt.tight_layout(rect=[0, 0, 1, 0.97])
                            
                            # Save
                            wav_name = f"wavelet_{st1}_{st2}_event_{int(event_id):04d}.png" if str(event_id).isdigit() \
                                      else f"wavelet_{st1}_{st2}_event_{event_id}.png"
                            wav_out = wav_dir / wav_name
                            plt.savefig(wav_out, dpi=300, bbox_inches='tight')
                            print(f"[WAVELET SAVED] {wav_name}")
                            plt.close()
                            
                        except Exception as e:
                            if debug:
                                print(f"[WAVELET ERROR] Failed for {st1}-{st2}: {e}")
                            plt.close()
    
    # ---------------- export screened events to CSV ----------------
    if save_dir:
        csv_out_path = Path(save_dir) / "screened_events_used_in_plots.csv"
        events_df.to_csv(csv_out_path, index=False)
        if debug:
            print(f"\n[CSV EXPORT] Saved {len(events_df)} screened events to: {csv_out_path}")
            print(f"[CSV EXPORT] Columns: {list(events_df.columns)}")

# Make sure load_tilt_templates(...) is importable in this scope.

# Define station_dfs_dict for the plot function (includes ALL stations: INGV + experiment)
station_dfs_dict = station_map  # Use station_map to include experiment stations with datetime

# ============================================================================
# EVENT SELECTION CRITERIA - CUSTOMIZE THESE PARAMETERS
# ============================================================================
# These parameters control which events are included in the plots:
#
# min_correlation:  Minimum peak correlation coefficient (e.g., 0.3, 0.5)
# min_snr_db:       Minimum SNR as LINEAR RATIO (e.g., 1.5, 2.0, 3.0) - uses snr_linear column
# max_snr_db:       Maximum SNR as LINEAR RATIO (e.g., 10.0, 20.0) - uses snr_linear column
# stations_filter:  List of stations to include - None = ALL stations from peaks_merged.csv
#                   e.g., ["EEC1", "ECPN"] or ["ECPN", "EMAS", "ECIT"] etc.
# templates_filter: List of templates to include (e.g., ["template1", "template2"])
# sims_filter:      List of simulations to include (e.g., ["sim1", "sim2"])
# dedup_window_fraction: Fraction of segment_duration_s for deduplication (0.5 = 50%)
# p_wave_tolerance_hours: Hours tolerance for matching P waves to events (e.g., 1.0)
#
# EVENT-CENTRIC APPROACH: Each plot shows ALL stations that detected the same event_id
# ============================================================================



# In[4]:


print(QUAKE_FILE)
print(PEAKS_ROOT)


# In[2]:



# In[2]:


#!/usr/bin/env python3
"""
Function to process individual event CSV files from synchronous_events directory.

These CSV files contain all detections for a single event across multiple stations.
Each file represents one synchronized event.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def process_single_event_file(
    event_csv_path: str,
    templates_dir: str = "tilt_templates",
    csv_templates_dir: str = "/home/owen/tilt_validation/tilt_templates_csv/experiment/",
    denoised_dir_ingv: str = "/home/owen/tilt_validation/bandpassed_exports_001_01_csv_only_utc/ingv/",
    denoised_dir_experiment: str = "/home/owen/tilt_validation/bandpassed_exports_001_01_csv_only_utc/experiment/",
    station_dfs: dict = None,
    segment_duration_s: int = 3000,
    time_unit: str = "minutes",
    detrend_type: str = "linear",
    normalize: str = None,
    y_limit: float = 0.1,
    save_dir: str = None,
    debug: bool = True,
    band_hz: tuple = (0.01, 0.001),
    filter_order: int = 4,
    enable_wavelet_analysis: bool = True,
    wavelet_save_dir: str = None,
):
    """
    Process a single event CSV file from the synchronous_events directory.
    
    Parameters
    ----------
    event_csv_path : str
        Path to the event CSV file (e.g., '/home/owen/tilt_validation/synchronous_events/event_026.csv')
    templates_dir : str
        Directory containing .npy/.npz templates
    csv_templates_dir : str
        Directory containing CSV templates for experiment/INGV stations
    denoised_dir_ingv : str
        Directory containing denoised INGV station data
    denoised_dir_experiment : str
        Directory containing bandpassed experiment station data
    station_dfs : dict
        Dictionary of raw station dataframes with datetime columns
    segment_duration_s : int
        Duration of segment window in seconds
    time_unit : str
        Time unit for plots ('seconds', 'minutes', 'hours', 'days')
    detrend_type : str
        Detrending method ('linear', 'constant', or None)
    normalize : str
        Normalization method (None, 'zscore', 'maxabs')
    y_limit : float
        Y-axis limit for plots
    save_dir : str
        Directory to save output plots and CSVs
    debug : bool
        Enable debug output
    band_hz : tuple
        Bandpass filter range (high, low) in Hz
    filter_order : int
        Butterworth filter order
    enable_wavelet_analysis : bool
        Enable wavelet coherence analysis
    wavelet_save_dir : str
        Directory for wavelet plots (None = save_dir/wavelet_analysis)
    
    Returns
    -------
    dict
        Results dictionary containing:
        - 'event_id': Event ID
        - 'event_csv': Path to input CSV
        - 'num_detections': Number of detections
        - 'stations': List of stations that detected the event
        - 'time_range': (earliest, latest) detection times
        - 'plots_saved': List of saved plot paths
        - 'success': True if processed successfully
    
    Examples
    --------
    >>> results = process_single_event_file(
    ...     event_csv_path='/home/owen/tilt_validation/synchronous_events/event_026.csv',
    ...     station_dfs=station_map,
    ...     save_dir='./event_026_output',
    ...     debug=True
    ... )
    >>> print(f"Processed event {results['event_id']} with {results['num_detections']} detections")
    """
    
    # Import the main plotting function
    # This assumes plot_sync_events_with_templates is defined in the same notebook/script
    # or can be imported from a module
    #from plot_sync_events_with_templates import plot_sync_events_with_templates
    
    event_csv_path = Path(event_csv_path)
    
    if not event_csv_path.exists():
        raise FileNotFoundError(f"Event CSV not found: {event_csv_path}")
    
    # Extract event ID from filename (e.g., event_026.csv -> 026)
    event_name = event_csv_path.stem  # 'event_026'
    event_id = event_name.split('_')[-1]  # '026'
    
    if debug:
        print("=" * 70)
        print(f"PROCESSING EVENT FILE: {event_csv_path.name}")
        print(f"Event ID: {event_id}")
        print("=" * 70)
    
    # Load event data
    event_df = pd.read_csv(event_csv_path)
    
    if debug:
        print(f"\n[INFO] Loaded {len(event_df)} detections from {event_csv_path.name}")
        print(f"[INFO] Columns: {list(event_df.columns)}")
    
    # Verify required columns
    required_cols = {'station', 'peak_time_dt', 'template', 'sim'}
    missing = required_cols - set(event_df.columns)
    if missing:
        raise ValueError(f"Event CSV missing required columns: {missing}")
    
    # Parse datetime
    event_df['peak_time_dt'] = pd.to_datetime(event_df['peak_time_dt'], errors='coerce')
    event_df = event_df.dropna(subset=['peak_time_dt'])
    
    if event_df.empty:
        raise ValueError(f"No valid detections in {event_csv_path.name}")
    
    # Get event info
    stations = event_df['station'].astype(str).str.upper().unique().tolist()
    earliest_time = event_df['peak_time_dt'].min()
    latest_time = event_df['peak_time_dt'].max()
    
    if debug:
        print(f"\n[INFO] Event statistics:")
        print(f"  Stations: {', '.join(stations)}")
        print(f"  Time range: {earliest_time} to {latest_time}")
        print(f"  Duration: {(latest_time - earliest_time).total_seconds():.1f} seconds")
        if 'avg_correlation' in event_df.columns:
            print(f"  Avg correlation: {event_df['avg_correlation'].iloc[0]:.3f}")
        if 'avg_snr_linear' in event_df.columns:
            print(f"  Avg SNR (linear): {event_df['avg_snr_linear'].iloc[0]:.3f}")
    
    # Set up save directory
    if save_dir is None:
        save_dir = f"./event_{event_id}_output"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if debug:
        print(f"\n[INFO] Output directory: {save_dir}")
    
    # Call the main processing function
    try:
        plot_sync_events_with_templates(
            csv_path=str(event_csv_path),
            templates_dir=templates_dir,
            csv_templates_dir=csv_templates_dir,
            denoised_dir_ingv=denoised_dir_ingv,
            denoised_dir_experiment=denoised_dir_experiment,
            station_dfs=station_dfs,
            segment_duration_s=segment_duration_s,
            time_unit=time_unit,
            detrend_type=detrend_type,
            normalize=normalize,
            y_limit=y_limit,
            save_dir=str(save_dir),
            debug=debug,
            band_hz=band_hz,
            filter_order=filter_order,
            max_plot_templates_per_station=8,
            # No filtering - process the entire event as-is
            min_correlation=None,
            min_snr_db=None,
            max_snr_db=None,
            stations_filter=None,
            templates_filter=None,
            sims_filter=None,
            dedup_window_fraction=0.0,  # No deduplication needed (already grouped)
            p_wave_tolerance_hours=0.0,  # Skip P wave matching
            enable_wavelet_analysis=enable_wavelet_analysis,
            wavelet_save_dir=wavelet_save_dir,
        )
        
        # Collect output files
        plot_files = list(save_dir.glob("*.png"))
        # Also check wavelet subdirectory
        wavelet_dir = save_dir / "wavelet_analysis"
        if wavelet_dir.exists():
            plot_files.extend(list(wavelet_dir.glob("*.png")))
        
        csv_files = list(save_dir.glob("*.csv"))
        
        if debug:
            print(f"\n[SUCCESS] Event {event_id} processed!")
            print(f"[INFO] Generated {len(plot_files)} plots")
            print(f"[INFO] Generated {len(csv_files)} CSV files")
        
        results = {
            'event_id': event_id,
            'event_csv': str(event_csv_path),
            'num_detections': len(event_df),
            'stations': stations,
            'time_range': (earliest_time, latest_time),
            'plots_saved': [str(p) for p in plot_files],
            'csv_saved': [str(c) for c in csv_files],
            'output_dir': str(save_dir),
            'success': True
        }
        
        return results
        
    except Exception as e:
        if debug:
            print(f"\n[ERROR] Failed to process event {event_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            'event_id': event_id,
            'event_csv': str(event_csv_path),
            'success': False,
            'error': str(e)
        }


def process_all_events_in_directory(
    events_dir: str = "/home/owen/tilt_validation/synchronous_events",
    output_base_dir: str = "./all_events_output",
    station_dfs: dict = None,
    templates_dir: str = "tilt_templates",
    csv_templates_dir: str = "/home/owen/tilt_validation/tilt_templates_csv/experiment/",
    **kwargs
):
    """
    Process all event CSV files in a directory.
    
    Parameters
    ----------
    events_dir : str
        Directory containing event CSV files (event_001.csv, event_002.csv, etc.)
    output_base_dir : str
        Base directory for all event outputs
    station_dfs : dict
        Dictionary of raw station dataframes
    templates_dir : str
        Directory with .npy templates
    csv_templates_dir : str
        Directory with CSV templates
    **kwargs : dict
        Additional arguments passed to process_single_event_file
    
    Returns
    -------
    list
        List of result dictionaries, one per event
    
    Examples
    --------
    >>> results = process_all_events_in_directory(
    ...     events_dir='/home/owen/tilt_validation/synchronous_events',
    ...     station_dfs=station_map,
    ...     enable_wavelet_analysis=True
    ... )
    >>> successful = [r for r in results if r['success']]
    >>> print(f"Processed {len(successful)}/{len(results)} events successfully")
    """
    
    events_dir = Path(events_dir)
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    if not events_dir.exists():
        raise FileNotFoundError(f"Events directory not found: {events_dir}")
    
    # Find all event CSV files
    event_files = sorted(events_dir.glob("event_*.csv"))
    
    if not event_files:
        print(f"[WARN] No event CSV files found in {events_dir}")
        return []
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {len(event_files)} events")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, event_file in enumerate(event_files, 1):
        print(f"\n[{i}/{len(event_files)}] Processing {event_file.name}...")
        
        # Extract event ID for subdirectory
        event_id = event_file.stem.split('_')[-1]
        event_output_dir = output_base_dir / f"event_{event_id}"
        
        result = process_single_event_file(
            event_csv_path=str(event_file),
            save_dir=str(event_output_dir),
            station_dfs=station_dfs,
            templates_dir=templates_dir,
            csv_templates_dir=csv_templates_dir,
            **kwargs
        )
        
        results.append(result)
        
        if result['success']:
            print(f"  ✓ Success: {result['num_detections']} detections, {len(result['stations'])} stations")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total events: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed events:")
        for r in failed:
            print(f"  - {r['event_id']}: {r.get('error', 'Unknown')}")
    
    print(f"\nOutput directory: {output_base_dir}")
    print(f"{'='*70}\n")
    
    return results




# In[ ]:





# In[3]:


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Process a single EXPERIMENT event (event_026)
    print("\nExample 1: Process single EXPERIMENT event")
    print("-" * 70)
    
    # You'll need to provide your station_dfs_dict
    # from your_script import station_dfs_dict
    
    result = process_single_event_file(
        event_csv_path="/home/owen/tilt_validation/synchronous_events/event_6745.csv",
        station_dfs=None,  # Replace with station_dfs_dict
        save_dir="./event_6745_output",
        templates_dir="tilt_templates",
        csv_templates_dir="/home/owen/tilt_validation/tilt_templates_csv/experiment/",  # Experiment templates
        enable_wavelet_analysis=True,
        debug=True
    )
    
    if result['success']:
        print(f"\n✓ Event {result['event_id']} processed successfully!")
        print(f"  Output: {result['output_dir']}")
        print(f"  Plots: {len(result['plots_saved'])}")
    
    # Example 2: Process a single INGV event (event_003)
    print("\n\nExample 2: Process single INGV event")
    print("-" * 70)
    
    result = process_single_event_file(
        event_csv_path="/home/owen/tilt_validation/synchronous_events/event_003.csv",
        station_dfs=station_dfs_dict,  # Replace with station_dfs_dict
        save_dir="./event_003_1_output",
        templates_dir="tilt_templates",
        csv_templates_dir="/home/owen/tilt_validation/tilt_templates_csv/ingv/",  # INGV templates
        enable_wavelet_analysis=True,
        debug=True
    )
    
    result = process_single_event_file(
        event_csv_path="/home/owen/tilt_validation/synchronous_events/event_103.csv",
        station_dfs=station_dfs_dict,  # Replace with station_dfs_dict
        save_dir="./event_103_output",
        templates_dir="tilt_templates",
        csv_templates_dir="/home/owen/tilt_validation/tilt_templates_csv/ingv/",  # INGV templates
        enable_wavelet_analysis=True,
        debug=True
    )
    
    if result['success']:
        print(f"\n✓ Event {result['event_id']} processed successfully!")
        print(f"  Output: {result['output_dir']}")
        print(f"  Plots: {len(result['plots_saved'])}")
    
    # Example 3: Process all events in directory (commented out)
    print("\n\nExample 3: Process all events (commented out)")
    print("-" * 70)
    print("# Uncomment to process all events:")
    print("#")
    print("# results = process_all_events_in_directory(")
    print("#     events_dir='/home/owen/tilt_validation/synchronous_events',")
    print("#     output_base_dir='./all_events_output',")
    print("#     station_dfs=station_dfs_dict,")
    print("#     enable_wavelet_analysis=True,")
    print("#     debug=False  # Less verbose for batch")
    print("# )")


# In[4]:


# ============================================================================
# COPY-PASTE THIS ENTIRE CELL TO END OF YOUR NOTEBOOK
# ============================================================================

def plot_events_from_csvs(ecpn_csv_path, eec1_csv_path, event_csv_path, templates,
                          buffer_seconds=1000, sampling_rate=1.0):
    """
    Plot event windows from CSV files with templates overlaid.
    
    File format expected:
    - Tilt CSVs: columns = time_seconds, denoised, denoised_0p001_0p01Hz, raw_detrended
    - Event CSV: includes peak_index, peak_time_dt, peak_corr, snr_db, etc.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load data
    print("Loading data...")
    events = pd.read_csv(event_csv_path)
    df_ecpn = pd.read_csv(ecpn_csv_path)
    df_eec1 = pd.read_csv(eec1_csv_path)
    
    print(f"✓ Events: {len(events)}")
    print(f"✓ ECPN: {len(df_ecpn)} samples")
    print(f"✓ EEC1: {len(df_eec1)} samples")
    
    station_data = {'ECPN': df_ecpn, 'EEC1': df_eec1}
    buffer_samples = int(buffer_seconds * sampling_rate)
    
    # Template colors and styles
    colors = {'template1': '#E74C3C', 'template2': '#27AE60', 
              'template3': '#F39C12', 'template4': '#3498DB'}
    styles = {'sim1': '-', 'sim2': '--', 'sim3': '-.', 'sim4': ':'}
    
    # Plot each station
    for station in events['station'].unique():
        station_events = events[events['station'] == station]
        df = station_data.get(station)
        
        if df is None:
            print(f"⚠ No data for {station}")
            continue
        
        print(f"\n{'='*60}")
        print(f"{station}: {len(station_events)} events")
        print(f"{'='*60}")
        
        # Find columns
        denoised_col = 'denoised_0p001_0p01Hz' if 'denoised_0p001_0p01Hz' in df.columns else 'denoised'
        raw_col = 'raw_detrended' if 'raw_detrended' in df.columns else 'raw'
        
        # Get first event
        first_event = station_events.iloc[0]
        peak_idx = int(first_event['peak_index'])
        peak_time = first_event['peak_time_dt']
        
        print(f"Peak: index={peak_idx}, time={peak_time}")
        
        # Extract window
        start_idx = max(0, peak_idx - buffer_samples)
        end_idx = min(len(df), peak_idx + buffer_samples)
        df_window = df.iloc[start_idx:end_idx].copy()
        
        print(f"Window: [{start_idx}:{end_idx}] = {len(df_window)} samples")
        
        if len(df_window) == 0:
            print("⚠ Empty window")
            continue
        
        # Time array centered at peak
        peak_pos = peak_idx - start_idx
        time_arr = (np.arange(len(df_window)) - peak_pos) / sampling_rate
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot signals
        if raw_col in df_window.columns:
            ax.plot(time_arr, df_window[raw_col], 'gray', linewidth=0.8,
                   label='Raw (detrended)', alpha=0.4, zorder=0)
        
        ax.plot(time_arr, df_window[denoised_col], 'b-', linewidth=1.5,
               label='Filtered (0.001-0.01 Hz)', alpha=0.8, zorder=1)
        
        # Event window shading
        ax.axvspan(-buffer_seconds, buffer_seconds, alpha=0.1,
                  color='lightblue', label='Event window', zorder=0)
        ax.axvline(0, color='red', linestyle=':', linewidth=1.5,
                  alpha=0.6, label='Peak', zorder=2)
        
        # Plot templates
        plotted = set()
        
        for _, evt in station_events.iterrows():
            sim, tpl = evt['sim'], evt['template']
            corr, snr = evt['peak_corr'], evt['snr_db']
            
            # Get template (handle EEC1/EC1 aliasing)
            try:
                if sim in templates and station in templates[sim]:
                    tpl_sig = templates[sim][station][tpl]
                elif station == 'EEC1' and sim in templates and 'EC1' in templates[sim]:
                    tpl_sig = templates[sim]['EC1'][tpl]
                elif station == 'EC1' and sim in templates and 'EEC1' in templates[sim]:
                    tpl_sig = templates[sim]['EEC1'][tpl]
                else:
                    print(f"  ⚠ Not found: {sim}/{station}/{tpl}")
                    continue
            except:
                continue
            
            # Normalize and scale
            tpl_norm = (tpl_sig - np.mean(tpl_sig)) / np.std(tpl_sig)
            sig_std = np.std(df_window[denoised_col])
            sig_mean = np.mean(df_window[denoised_col])
            tpl_scaled = tpl_norm * sig_std + sig_mean
            
            # Time array for template
            n = len(tpl_scaled)
            tpl_time = (np.arange(n) - n//2) / sampling_rate
            
            # Plot
            color = colors.get(tpl, '#95A5A6')
            style = styles.get(sim, '-')
            
            label = f"{tpl.upper()} ({sim.upper()}, ρ={corr:.2f}, SNR={snr:.1f}dB)"
            combo = (sim, tpl)
            if combo in plotted:
                label = None
            else:
                plotted.add(combo)
            
            ax.plot(tpl_time, tpl_scaled, color=color, linestyle=style,
                   linewidth=2.2, label=label, alpha=0.9, zorder=3)
        
        # Format
        ax.set_xlabel('Time relative to peak (s)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Tilt (μrad)', fontsize=13, fontweight='bold')
        ax.set_title(f'{station}\n{peak_time}', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize=9, framealpha=0.95, edgecolor='gray')
        ax.set_xlim(-buffer_seconds*1.05, buffer_seconds*1.05)
        ax.tick_params(axis='both', labelsize=11)
        
        plt.tight_layout()
        plt.show()
        
        print(f"✓ Plotted {station}")
    
    print(f"\n{'='*60}\n✓ COMPLETE\n{'='*60}")


# ============================================================================
# RUN IT - Update these paths to your actual file locations
# ============================================================================

plot_events_from_csvs(
    ecpn_csv_path='/home/owen/Etna_signals/denoised_exports_001_csv_only/ingv/ECPN_0p001-0p01.csv',      # UPDATE THIS PATH
    eec1_csv_path='/home/owen/Etna_signals/denoised_exports_001_csv_only/ingv/EEC1_0p001-0p01.csv',      # UPDATE THIS PATH
    event_csv_path='/home/owen/tilt_validation/synchronous_events/event_1311.csv',
    templates=templates,  # Already loaded in your notebook
    buffer_seconds=1000,
    sampling_rate=1.0
)


# In[10]:


# ============================================================================
# CORRECTED USAGE - Load denoised CSVs from directories
# ============================================================================

# DO NOT pass the raw station dataframes (df_ecpn, df_eec1_ingv)
# Those contain raw data: ['datetime', 'seconds', 'x', 'y', 'na', 'east', 'north', 'mag']
# 
# Instead, let the function load the DENOISED CSVs from the directories

results = plot_event_windows_with_templates(
    event_csv_path='/home/owen/tilt_validation/synchronous_events/event_1311.csv',
    denoised_dir_ingv='/home/owen/tilt_validation/bandpassed_exports_001_01_csv_only_utc/ingv/',
    denoised_dir_experiment='/home/owen/tilt_validation/bandpassed_exports_001_01_csv_only_utc/experiment/',
    station_dfs=None,  # ← Set to None so it loads from directories
    buffer_seconds=1000,
    save_dir='./event_plots',
    debug=True,
    show_plots=True
)

# Check results
print(f"\n{'='*70}")
if results['success']:
    print(f"SUCCESS!")
    print(f"  Plotted: {results['stations']}")
    print(f"  Saved {len(results['plots_saved'])} plots")
else:
    print(f"FAILED!")
    print(f"  Error: {results.get('error', 'Unknown error')}")
print(f"{'='*70}")


# ============================================================================
# ALTERNATIVE: Load the denoised dataframes yourself
# ============================================================================

# If you want to use pre-loaded dataframes, load the DENOISED ones:

import pandas as pd

# Load denoised CSVs
df_ecpn_denoised = pd.read_csv('/home/owen/tilt_validation/bandpassed_exports_001_01_csv_only_utc/ingv/ECPN_0p001-0p01Hz_bp.csv')
df_eec1_denoised = pd.read_csv('/home/owen/tilt_validation/bandpassed_exports_001_01_csv_only_utc/ingv/EEC1_0p001-0p01Hz_bp.csv')
# or
# df_eec1_denoised = pd.read_csv('/home/owen/tilt_validation/bandpassed_exports_001_01_csv_only_utc/experiment/EEC1_0p001-0p01.csv')

print(f"ECPN denoised columns: {df_ecpn_denoised.columns.tolist()}")
print(f"EEC1 denoised columns: {df_eec1_denoised.columns.tolist()}")

# Then pass these denoised dataframes:
station_data_denoised = {
    'ECPN': df_ecpn_denoised,
    'EEC1': df_eec1_denoised,
}

results = plot_event_windows_with_templates(
    event_csv_path='/home/owen/tilt_validation/synchronous_events/event_1311.csv',
    station_dfs=station_data_denoised,  # ← Pass denoised dataframes
    buffer_seconds=3300,
    save_dir='./event_plots',
    debug=True,
    show_plots=True
)






