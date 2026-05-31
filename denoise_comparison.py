"""
denoise_comparison.py — bandpass-only vs scalar-β regression vs coherent admittance H(f).
Shows the in-band noise removed by each thermal-denoising method per station. The coherent
admittance (transfer function, gain+phase) is the best-practice method now used in stage 1.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from build_clean_bandpassed_continuous import estimate_admittance, apply_admittance

warnings.filterwarnings("ignore")
OUT = Path("/home/owen/tilt_validation/outputs/denoise_comparison"); OUT.mkdir(parents=True, exist_ok=True)
SIG = Path("/home/owen/Signals/experiment")
SOS = butter(4, [0.001/0.5, 0.01/0.5], btype="bandpass", output="sos")
bp = lambda x: sosfiltfilt(SOS, x - np.mean(x))
SRC = {
    "ECPN": (SIG/"INGV/ECPN.feather", "f", "east"), "EEC1": (SIG/"INGV/EEC1.feather", "f", "east"),
    "EC1": (SIG/"EC1.csv", "c", "x"), "EC10": (SIG/"school-data/INGV_feather/EC10.feather", "f", "x"),
    "ECIT": (SIG/"school-data/INGV_feather/ECIT.feather", "f", "x"),
    "ECOR": (SIG/"school-data/INGV_feather/ECOR.feather", "f", "x"),
    "EMAS": (SIG/"school-data/INGV_feather/EMAS.feather", "f", "x"),
}


def rms_drop(before, after):
    return 100*(1 - np.std(after)/np.std(before))


def main():
    rows = []
    for st, (path, fmt, col) in SRC.items():
        d = pd.read_csv(path) if fmt == "c" else pd.read_feather(path)
        d = d.rename(columns={" x": "x"})
        if "na" not in d.columns:
            continue
        x = d[col].to_numpy(float)[:400000]; na = d["na"].to_numpy(float)[:400000]
        m = np.isfinite(x) & np.isfinite(na); x, na = x[m], na[m]
        xb, nb = bp(x), bp(na)
        # scalar β
        beta = np.cov(xb, nb)[0, 1] / (np.var(nb) + 1e-30)
        scalar = xb - beta*nb
        # coherent admittance H(f)
        adm = estimate_admittance(xb, nb)
        adm_res = xb - apply_admittance(nb, adm) if adm is not None else xb
        rows.append({"station": st,
                     "scalar_beta_drop": round(rms_drop(xb, scalar), 0),
                     "admittance_drop": round(rms_drop(xb, adm_res), 0)})
    df = pd.DataFrame(rows); df.to_csv(OUT/"comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    ax.bar(x-0.2, df.scalar_beta_drop, 0.4, color="#9ca3af", label="scalar β (zero-lag)")
    ax.bar(x+0.2, df.admittance_drop, 0.4, color="#2563eb", label="coherent admittance H(f) [stage 1]")
    ax.set_xticks(x); ax.set_xticklabels(df.station)
    ax.set(ylabel="in-band RMS reduction (%)", title="Thermal denoising: scalar β vs coherent admittance H(f)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT/"comparison.png", dpi=140); plt.close(fig)

    print("THERMAL DENOISING — in-band RMS reduction (%)")
    print(df.to_string(index=False))
    print(f"\nmean: scalar β {df.scalar_beta_drop.mean():.0f}%  vs  admittance H(f) {df.admittance_drop.mean():.0f}%")
    print(f"Outputs → {OUT}")


if __name__ == "__main__":
    main()
