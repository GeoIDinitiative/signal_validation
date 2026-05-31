"""
network_matched_filter.py  —  cross-station coherent stack (network matched filter)
==================================================================================
Stacks the per-station correlation traces across stations for the same simulated
event (sim, template). A real coherent source peaks at the same time at every
station, so the stack adds the signal ~linearly while incoherent noise adds ~sqrt(N)
=> sensitivity improves by ~sqrt(N_stations). This is the gold-standard detector and
is more sensitive than independent detection + post-hoc synchrony.

Moveout: the simulation templates are co-registered (same sim/template = same source
event at each station), so the differential timing is already baked into the templates
— stacking the per-station r(t) needs no extra shift.

Detector at time t:  NetMax(t) = max over (sim,template) of mean_over_stations |r_st(t)|.

This script quantifies the gain with injection-recovery: inject the SAME event
coherently into all stations at a given SNR and compare network vs single-station
detection probability, giving the network SNR50/SNR90 (the sensitivity floor for
very small tilt amplitudes).

Output: SWCC_comprehensive/network/  (recovery.csv, recovery.png, summary.txt)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from swcc_gapaware import swcc_gapaware
from swcc_comprehensive import load_template, SIMS

warnings.filterwarnings("ignore")
BASE = Path("/home/owen/tilt_validation")
CONT = BASE / "continuous_bandpassed"
OUT  = BASE / "SWCC_comprehensive" / "network"
STATIONS = {"ingv": ["ECPN", "EEC1"], "experiment": ["EC1", "EC10", "ECIT", "ECOR", "EMAS"]}
TEMPLATES = ["template1", "template2", "template3"]
COMP = "dir"
SNRS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
N_TRIALS, N_FLOOR = 60, 200
WIN, TOL = 10002, 400


def load_dataset(ds):
    sigs, starts, tpls = {}, {}, {}
    for st in STATIONS[ds]:
        f = CONT / ds / f"{st}_{COMP}_0p001-0p01Hz_cont_bp.feather"
        if not f.exists():
            continue
        d = pd.read_feather(f)
        ok = (~d["veto"].to_numpy(bool)) & np.isfinite(d["bandpassed"].to_numpy())
        cs = np.concatenate(([0], np.cumsum(ok.astype(int))))
        vs = np.where((cs[WIN:] - cs[:-WIN]) == WIN)[0]
        if len(vs) == 0:
            continue
        sigs[st] = d["bandpassed"].to_numpy()
        starts[st] = vs
        for sim in SIMS:
            for tn in TEMPLATES:
                tpls[(st, sim, tn)] = load_template(ds, st, sim, tn)
    return sigs, starts, tpls


def score(wins, stations, tpls):
    """NetMax (cross-station stacked) and per-station single max, near the centre."""
    netbest, singbest = 0.0, {st: 0.0 for st in stations}
    M0 = None
    for sim in SIMS:
        for tn in TEMPLATES:
            rs = {}
            for st in stations:
                T = tpls.get((st, sim, tn))
                if T is None:
                    continue
                rs[st] = np.abs(swcc_gapaware(T, wins[st], min_valid_frac=0.8))
                M0 = len(T)
            if len(rs) < 2:
                continue
            L = min(len(r) for r in rs.values())
            net = np.nanmean(np.vstack([rs[st][:L] for st in rs]), axis=0)
            p = (WIN - M0) // 2
            lo, hi = max(0, p - TOL), min(L, p + TOL)
            if hi > lo:
                netbest = max(netbest, np.nanmax(net[lo:hi]))
                for st in rs:
                    singbest[st] = max(singbest[st], np.nanmax(rs[st][lo:hi]))
    return netbest, singbest


def draw_windows(stations, sigs, starts, rng, inject=None):
    wins = {}
    for st in stations:
        s0 = starts[st][rng.integers(len(starts[st]))]
        w = sigs[st][s0:s0+WIN].copy()
        if inject is not None:
            sim, tn = inject
            T = tpls_global[(st, sim, tn)]
            M = len(T); p = (WIN - M) // 2
            nrms = np.sqrt(np.mean(w**2))
            if T.std() > 0 and nrms > 0:
                w[p:p+M] += inject_snr * nrms * (T - T.mean()) / T.std()
        wins[st] = w
    return wins


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    global tpls_global, inject_snr
    rows = []
    L = ["NETWORK MATCHED FILTER — coherent cross-station stack", "=" * 54, ""]
    for ds in STATIONS:
        sigs, starts, tpls = load_dataset(ds)
        stations = [s for s in STATIONS[ds] if s in sigs]
        if len(stations) < 2:
            continue
        tpls_global = tpls
        rng = np.random.default_rng(21)

        # noise-only floors (no injection): 99th-pct of NetMax and single max
        inject_snr = 0
        net_n, sing_n = [], []
        for _ in range(N_FLOOR):
            wins = draw_windows(stations, sigs, starts, rng, inject=None)
            nb, sb = score(wins, stations, tpls)
            net_n.append(nb); sing_n.append(max(sb.values()))
        net_floor = float(np.percentile(net_n, 99))
        sing_floor = float(np.percentile(sing_n, 99))
        L.append(f"{ds}: {len(stations)} stations  net_floor={net_floor:.3f}  single_floor={sing_floor:.3f}")

        for snr in SNRS:
            inject_snr = snr
            rec_net = rec_sing = 0
            for _ in range(N_TRIALS):
                inj = (SIMS[rng.integers(len(SIMS))], TEMPLATES[rng.integers(len(TEMPLATES))])
                wins = draw_windows(stations, sigs, starts, rng, inject=inj)
                nb, sb = score(wins, stations, tpls)
                rec_net += nb > net_floor
                rec_sing += max(sb.values()) > sing_floor
            rows.append({"dataset": ds, "n_stations": len(stations), "snr": snr,
                         "p_network": rec_net/N_TRIALS, "p_single": rec_sing/N_TRIALS})
            print(f"  {ds} SNR={snr:>4}: P_network={rec_net/N_TRIALS:.2f}  P_single={rec_sing/N_TRIALS:.2f}")
        L.append("")
    df = pd.DataFrame(rows); df.to_csv(OUT / "recovery.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    for ds, c in [("ingv", "#1f2937"), ("experiment", "#dc2626")]:
        g = df[df.dataset == ds].sort_values("snr")
        if g.empty:
            continue
        n = int(g.n_stations.iloc[0])
        ax.plot(g.snr, g.p_network, "-", color=c, marker="o", label=f"{ds} · NETWORK (N={n})")
        ax.plot(g.snr, g.p_single, "--", color=c, marker="s", alpha=0.7, label=f"{ds} · single station")
        for col, lab in [("p_network", "NETWORK"), ("p_single", "single")]:
            y, x = g[col].to_numpy(), g.snr.to_numpy()
            s90 = float(np.interp(0.9, y, x)) if y.max() >= 0.9 else np.nan
            s50 = float(np.interp(0.5, y, x)) if y.max() >= 0.5 else np.nan
            L.append(f"{ds:11s} {lab:8s}: SNR50={s50:.2f}  SNR90={s90:.2f}")
    ax.axhline(0.9, color="gray", ls=":", alpha=0.6); ax.axhline(0.5, color="gray", ls=":", alpha=0.6)
    ax.set(xlabel="injection SNR (template band-RMS / noise band-RMS)", ylabel="detection probability",
           title="Network matched filter vs single-station sensitivity")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "recovery.png", dpi=300); plt.close(fig)
    L += ["", "A lower NETWORK SNR90 than single-station = the coherent stack detects smaller",
          "tilt amplitudes (~sqrt(N) gain). This sets the deepest sensitivity floor of the search."]
    (OUT / "summary.txt").write_text("\n".join(L))
    print("\n" + "\n".join(L)); print(f"\nOutputs → {OUT}")


if __name__ == "__main__":
    main()
