"""Single-column version of the PVC-extraction example figure for the paper (patient 822228, LVOT).

Same content as the review figures of scripts/18_pvc_extraction_check.py, laid out for one journal column
(3.45 in wide): (a) lead II with the flagged PVCs, (b) sorted QRS widths, (c) overlay of lead II, (d) overlay of V1.
Needs data/processed (run on the PC).

Usage:
    python scripts/19_paper_fig_extraction.py
Output: results/pvc_check/paper_fig_extraction.png (600 dpi) and .pdf
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check18", HERE / "18_pvc_extraction_check.py")
check18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check18)
config, FS, REF, V1 = check18.config, check18.FS, check18.REF, check18.V1

PATIENT = 822228
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def main():
    index = check18.load_patient_index()
    row = index[index.hospital_id == PATIENT].iloc[0]
    ecg, peaks, widths, wide, kept = check18.analyse(row)
    sig, t = ecg[:, REF] / 1000, np.arange(len(ecg)) / FS  # amplitude in thousands of ADC units

    fig = plt.figure(figsize=(3.45, 3.9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, sig, lw=0.5, color="black")
    for i, p in enumerate(peaks):
        c = "red" if wide[i] else "tab:blue"
        ax.plot(t[p], sig[p], "v" if wide[i] else "o", color=c, ms=3)
        ax.annotate(f"{widths[i]:.0f}", (t[p], sig[p]), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=5, color=c)
        if wide[i] and kept[i]:
            ax.axvspan(t[p] - config.BEAT_PRE_MS / 1000, t[p] + config.BEAT_POST_MS / 1000, color="red", alpha=0.07, lw=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Lead {config.RPEAK_DETECTION_LEAD}")
    ax.margins(x=0.01, y=0.25)
    ax.text(-0.11, 1.0, "(a)", transform=ax.transAxes, fontweight="bold", va="top")

    ax = fig.add_subplot(gs[1, :])
    order = np.argsort(widths)
    ax.bar(range(len(widths)), widths[order], color=["red" if wide[j] else "tab:blue" for j in order])
    ax.set_xlabel("Beats sorted by QRS width")
    ax.set_ylabel("QRS width (ms)")
    ax.set_xticks([])
    ax.text(-0.11, 1.0, "(b)", transform=ax.transAxes, fontweight="bold", va="top")

    tt = np.arange(-config.BEAT_PRE_SAMPLES, config.BEAT_POST_SAMPLES) / FS * 1000
    for k, (lead, name) in enumerate([(REF, config.RPEAK_DETECTION_LEAD), (V1, "V1")]):
        ax = fig.add_subplot(gs[2, k])
        for i, p in enumerate(peaks):
            if kept[i]:
                ax.plot(tt, ecg[p - config.BEAT_PRE_SAMPLES:p + config.BEAT_POST_SAMPLES, lead] / 1000,
                        color="red" if wide[i] else "tab:blue", lw=0.5, alpha=0.6)
        ax.set_xlabel("ms from R peak")
        ax.set_ylabel(f"Lead {name}")
        ax.text(-0.3, 1.0, f"({'cd'[k]})", transform=ax.transAxes, fontweight="bold", va="top")

    out = check18.OUT / "paper_fig_extraction"
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".pdf"))
    print("written", out.with_suffix(".png"), out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
