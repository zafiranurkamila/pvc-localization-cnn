"""Visual review of the PVC beats extracted from the UPF mapping-point strips (reviewer point A2; no training).

For a random sample of the UPF outflow-tract patients with PVC beats (10 RVOT and 10 LVOT, seed 42, drawn from
results/external_upf/patients.csv), four mapping-point strips per patient are processed exactly as in scripts/30 and
scripts/31 (2000 Hz, band-pass and notch filter, R peaks on lead II, QRS width index, largest-gap rule). One figure
per patient shows leads II and V1 of the four strips with every detected beat marked (red triangle = flagged as PVC,
blue circle = other beat) and its width index in ms.

The reviewer fills review_form.csv: for each patient, the number of flagged beats that are PVCs, the number of
flagged beats that are not (artefact, sinus or other), the number of PVCs that were missed, and a note. Precision
and sensitivity are then computed with scripts/34 --summarize.

Usage (on the PC, after scripts/29):
    python scripts/34_upf_visual_review.py data/external/otva_upf/all_points_may_2024.pkl
    python scripts/34_upf_visual_review.py --summarize        # after filling the form
Output: data/external/otva_upf/visual_review/<patient>.png, review_form.csv (inside the ignored data folder)
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location("check30", Path(__file__).resolve().parent / "30_check_external_pvc.py")
n30 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n30)

from pvc_localization import config  # noqa: E402
from pvc_localization.preprocessing.beat_detection import flag_wide_beats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "otva_upf" / "visual_review"
PER_SIDE = 10
STRIPS = 4


def sample_patients():
    df = pd.read_csv(ROOT / "results" / "external_upf" / "patients.csv")
    df = df[df.pvc_beats > 0]
    rng = np.random.default_rng(config.RANDOM_SEED)
    pick = []
    for lab in (0, 1):
        ids = df[df.label == lab].patient.to_numpy()
        pick += list(rng.choice(ids, min(PER_SIDE, len(ids)), replace=False))
    return pick


def figure(h, p, rng):
    side = n30.first(p["SOO_chamber"])
    strips = n30.strips_of(p, STRIPS, rng)
    fig, axes = plt.subplots(len(strips), 2, figsize=(12, 2.2 * len(strips)), squeeze=False)
    n_flag = 0
    for r, (m, pt, x) in enumerate(strips):
        xf, peaks, widths = n30.process(x)
        wide = flag_wide_beats(widths) if len(widths) >= 2 else np.zeros(len(widths), bool)
        n_flag += int(wide.sum())
        t = np.arange(len(xf)) / n30.FS
        for c, lead in enumerate((1, 6)):
            ax = axes[r, c]
            ax.plot(t, xf[:, lead], lw=0.7, color="#333333")
            for pk, w, wd in zip(peaks, widths, wide):
                ax.plot(pk / n30.FS, xf[pk, lead], "v" if wd else "o", color="#C0392B" if wd else "#2E86C1", ms=5)
                if c == 0:
                    ax.annotate(f"{w:.0f}", (pk / n30.FS, xf[pk, lead]), textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=7)
            ax.set_title(f"{h} ({side}) map {m} point {pt}  lead {'II' if lead == 1 else 'V1'}", fontsize=8)
            ax.set_yticks([])
    axes[-1, 0].set_xlabel("s"); axes[-1, 1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(OUT / f"{h}.png", dpi=110)
    plt.close(fig)
    return side, len(strips), n_flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", nargs="?", help="the extracted all_points_may_2024.pkl")
    ap.add_argument("--summarize", action="store_true", help="compute precision and sensitivity from the form")
    args = ap.parse_args()
    form = OUT / "review_form.csv"
    if args.summarize:
        f = pd.read_csv(form)
        tp, fp, fn = f.flagged_pvc.sum(), f.flagged_not_pvc.sum(), f.missed_pvc.sum()
        print(f"{len(f)} patients reviewed; flagged beats {tp + fp}: {tp} PVC, {fp} not PVC; missed PVCs {fn}")
        print(f"precision {tp / max(tp + fp, 1):.1%}, sensitivity {tp / max(tp + fn, 1):.1%}")
        for side, g in f.groupby("side"):
            print(f"  {side}: precision {g.flagged_pvc.sum() / max(g.flagged_pvc.sum() + g.flagged_not_pvc.sum(), 1):.1%}"
                  f", sensitivity {g.flagged_pvc.sum() / max(g.flagged_pvc.sum() + g.missed_pvc.sum(), 1):.1%}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    data = n30.n29.load_pickle(args.pkl)
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for h in sample_patients():
        side, n_strips, n_flag = figure(h, data[h], rng)
        rows.append({"patient": h, "side": side, "strips": n_strips, "flagged_beats": n_flag,
                     "flagged_pvc": "", "flagged_not_pvc": "", "missed_pvc": "", "note": ""})
        print(f"{h} ({side}): {n_strips} strips, {n_flag} beats flagged")
    pd.DataFrame(rows).to_csv(form, index=False)
    print("figures and form written to", OUT)


if __name__ == "__main__":
    main()
