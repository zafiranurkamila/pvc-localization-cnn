"""Figures of the J Med Syst manuscript from the saved results (no training).

  fig_coverage.pdf     macro F1 and balanced accuracy of the classified patients against the realised coverage
                       (scripts/26, mean +- SD over the splits) for fusion, ECGFounder and the CNN
  fig_calibration.pdf  reliability diagram of the Platt-scaled probabilities (scripts/27), patients of all splits
                       pooled, ten quantile bins per model

Usage (after scripts 26 and 27):
    python scripts/28_paper_figures.py
Output: results/nested_cv/figures/fig_coverage.pdf, fig_calibration.pdf (copy them to jurnal-jmeds/figures)
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location("nested21", Path(__file__).resolve().parent / "21_nested_cv_dual_rhythm.py")
n21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n21)

OUT = n21.OUT
FIG = OUT / "figures"
MODELS = {"fusion": ("Fusion (proposed)", "#0F6E78", "o"), "founder": ("ECGFounder", "#D17A22", "s"),
          "cnn": ("PSD--CWT CNN".replace("--", "–"), "#7A7A7A", "^")}
PRIMARY = 0.90
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def coverage_figure():
    res = pd.read_csv(OUT / "selective.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharex=True)
    for ax, metric, title in zip(axes, ("macro_f1", "balanced_accuracy"), ("Macro F1-score", "Balanced accuracy")):
        for m, (label, color, marker) in MODELS.items():
            g = res[res.model == m].groupby("target_coverage").agg(cov=("coverage", "mean"), mean=(metric, "mean"),
                                                                    sd=(metric, "std")).sort_index()
            ax.errorbar(100 * g["cov"], g["mean"], yerr=g["sd"], color=color, marker=marker, ms=3.5, lw=1.2,
                        capsize=2, label=label)
            p = g.loc[np.isclose(g.index, PRIMARY)]
            ax.scatter(100 * p["cov"], p["mean"], s=70, facecolor="none", edgecolor=color, lw=1.2, zorder=3)
        ax.axhline(0.9, color="#BBBBBB", lw=0.8, ls="--", zorder=0)
        ax.set_xlabel("Patients classified (%)")
        ax.set_title(title, fontsize=8.5)
        ax.invert_xaxis()
        ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig_coverage.pdf")
    plt.close(fig)


def calibration_figure():
    cal = pd.read_csv(OUT / "paper_calibration_patients.csv")
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls="--")
    for m, (label, color, marker) in MODELS.items():
        bins = pd.qcut(cal[f"p_{m}"], 10, duplicates="drop")
        g = cal.groupby(bins, observed=True).agg(p=(f"p_{m}", "mean"), y=("label", "mean"))
        ax.plot(g.p, g.y, color=color, marker=marker, ms=3.5, lw=1.2, label=label)
    ax.set_xlabel("Predicted LVOT probability")
    ax.set_ylabel("Observed LVOT proportion")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_calibration.pdf")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    coverage_figure()
    calibration_figure()
    print("written", FIG)


if __name__ == "__main__":
    main()
