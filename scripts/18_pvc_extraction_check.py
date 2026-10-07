"""Check of the automatic PVC-beat extraction (reviewer major comment 13).

Two parts:
1. Automatic consistency check on ALL patients: QRS width of the narrow (sinus) and the wide (PVC) group, the
   gap between them, and a template check (each flagged PVC is correlated, on lead II, with the mean of the
   other flagged PVCs and with the mean of the narrow beats; a true PVC should resemble the PVC template more).
2. Manual review of a random, class-stratified sample of patients: one figure per patient with every R peak
   numbered and marked as PVC (red) or not (blue). Fill in results/pvc_check/manual_review.csv by looking at
   the figures, then run with --summarize to get precision and sensitivity of the extraction.

No training. Needs data/processed and data/interim/patient_index.csv (run on the PC).

Usage:
    python scripts/18_pvc_extraction_check.py                 # part 1 + figures + empty review sheet
    python scripts/18_pvc_extraction_check.py --n-per-class 15
    python scripts/18_pvc_extraction_check.py --summarize     # after filling in manual_review.csv
Output: results/pvc_check/
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config  # noqa: E402
from pvc_localization.data.dataset import create_train_val_test_split  # noqa: E402
from pvc_localization.data.loader import load_ecg, load_patient_index  # noqa: E402
from pvc_localization.preprocessing.beat_detection import (  # noqa: E402
    detect_r_peaks, flag_wide_beats, qrs_width_ms)
from pvc_localization.preprocessing.filtering import clean_signal  # noqa: E402

OUT = config.RESULTS_DIR / "pvc_check"
FS = config.SAMPLING_RATE_HZ
REF = config.LEADS.index(config.RPEAK_DETECTION_LEAD)
V1 = config.LEADS.index("V1")
HALF = int(0.1 * FS)  # +-100 ms around the R peak for the template check
SEED = 42


def analyse(row):
    ecg = clean_signal(load_ecg(row["processed_path"]).to_numpy())
    sig = ecg[:, REF]
    peaks = detect_r_peaks(sig)
    widths = np.array([qrs_width_ms(sig, p) for p in peaks]) if len(peaks) else np.array([])
    wide = flag_wide_beats(widths) if len(peaks) >= 2 else np.zeros(len(peaks), dtype=bool)
    kept = (peaks - config.BEAT_PRE_SAMPLES >= 0) & (peaks + config.BEAT_POST_SAMPLES <= len(sig))
    return ecg, peaks, widths, wide, kept


def template_check(sig, peaks, wide):
    """Fraction of flagged PVCs whose lead-II QRS correlates more with the (leave-one-out) PVC template than with
    the narrow-beat template. NaN when a patient has fewer than 2 PVCs or no narrow beat."""
    ok = (peaks - HALF >= 0) & (peaks + HALF <= len(sig))
    seg = np.array([sig[p - HALF:p + HALF] for p in peaks[ok]]) if ok.any() else np.empty((0, 2 * HALF))
    w = wide[ok]
    if w.sum() < 2 or (~w).sum() < 1:
        return np.nan, np.nan, np.nan
    pvc, nor = seg[w], seg[~w]
    t_nor = nor.mean(0)
    r_pvc, r_nor = [], []
    for i in range(len(pvc)):
        t_pvc = np.delete(pvc, i, 0).mean(0)
        r_pvc.append(np.corrcoef(pvc[i], t_pvc)[0, 1])
        r_nor.append(np.corrcoef(pvc[i], t_nor)[0, 1])
    r_pvc, r_nor = np.array(r_pvc), np.array(r_nor)
    return float(np.mean(r_pvc > r_nor)), float(np.median(r_pvc)), float(np.median(r_nor))


def plot_patient(row, ecg, peaks, widths, wide, kept, path):
    sig, t = ecg[:, REF], np.arange(len(ecg)) / FS
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, sig, lw=0.6, color="black")
    for i, p in enumerate(peaks):
        c = "red" if wide[i] else "tab:blue"
        ax.plot(t[p], sig[p], "v" if wide[i] else "o", color=c, ms=7)
        ax.annotate(f"{i + 1}\n{widths[i]:.0f}ms", (t[p], sig[p]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color=c)
        if wide[i] and kept[i]:
            ax.axvspan(t[p] - config.BEAT_PRE_MS / 1000, t[p] + config.BEAT_POST_MS / 1000, color="red", alpha=0.07)
    ax.set_title(f"Patient {row['hospital_id']} ({row['label']}) - lead {config.RPEAK_DETECTION_LEAD}: red = flagged "
                 f"PVC (shaded = extracted window), blue = not PVC; number = beat id, ms = estimated QRS width", fontsize=9)
    ax.set_xlabel("time (s)")
    ax.margins(y=0.2)
    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(widths)
    ax.bar(range(len(widths)), widths[order], color=["red" if wide[j] else "tab:blue" for j in order])
    ax.set_title("QRS widths, sorted (split at the largest gap)", fontsize=9)
    ax.set_ylabel("ms")
    for k, (lead, name) in enumerate([(REF, config.RPEAK_DETECTION_LEAD), (V1, "V1")]):
        ax = fig.add_subplot(gs[1, k + 1])
        tt = (np.arange(-config.BEAT_PRE_SAMPLES, config.BEAT_POST_SAMPLES)) / FS * 1000
        for i, p in enumerate(peaks):
            if not kept[i]:
                continue
            ax.plot(tt, ecg[p - config.BEAT_PRE_SAMPLES:p + config.BEAT_POST_SAMPLES, lead],
                    color="red" if wide[i] else "tab:blue", lw=0.7, alpha=0.6)
        ax.set_title(f"Overlay lead {name}: red = PVC, blue = other beats", fontsize=9)
        ax.set_xlabel("ms from R peak")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def summarize():
    df = pd.read_csv(OUT / "manual_review.csv")
    need = ["correct_pvc", "false_pvc", "missed_pvc"]
    done = df.dropna(subset=need)
    if done.empty:
        sys.exit("manual_review.csv has no filled rows (columns correct_pvc, false_pvc, missed_pvc)")
    tp, fp, fn = (int(done[c].sum()) for c in need)
    out = {"patients_reviewed": int(len(done)), "by_class": done.label.value_counts().to_dict(),
           "true_pvc_flagged": tp, "false_pvc_flagged": fp, "missed_pvc": fn,
           "precision": tp / max(tp + fp, 1), "precision_ci95": wilson(tp, tp + fp),
           "sensitivity": tp / max(tp + fn, 1), "sensitivity_ci95": wilson(tp, tp + fn),
           "patients_without_any_error": int(((done.false_pvc == 0) & (done.missed_pvc == 0)).sum()),
           "patients_with_at_least_one_correct_pvc": int((done.correct_pvc > 0).sum())}
    (OUT / "manual_review_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=15)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        return summarize()

    index = load_patient_index()
    train_ids, test_ids = create_train_val_test_split()
    split = {pid: "train" for pid in train_ids} | {pid: "test" for pid in test_ids}
    sample = (index.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(args.n_per_class, len(g)), random_state=SEED)))
    sample_ids = set(sample.hospital_id)
    (OUT / "figures").mkdir(exist_ok=True)

    rows, review = [], []
    for _, row in tqdm(index.iterrows(), total=len(index), desc="patients"):
        ecg, peaks, widths, wide, kept = analyse(row)
        frac, r_pvc, r_nor = template_check(ecg[:, REF], peaks, wide)
        rows.append({"hospital_id": row["hospital_id"], "label": row["label"], "split": split.get(row["hospital_id"]),
                     "r_peaks": int(len(peaks)), "flagged_pvc": int(wide.sum()),
                     "pvc_kept": int((wide & kept).sum()),
                     "narrow_width_ms": float(np.median(widths[~wide])) if (~wide).any() else np.nan,
                     "wide_width_ms": float(np.median(widths[wide])) if wide.any() else np.nan,
                     "gap_ms": float(widths[wide].min() - widths[~wide].max()) if wide.any() and (~wide).any() else np.nan,
                     "frac_pvc_closer_to_pvc_template": frac, "median_r_pvc_template": r_pvc,
                     "median_r_normal_template": r_nor})
        if row["hospital_id"] in sample_ids:
            plot_patient(row, ecg, peaks, widths, wide, kept, OUT / "figures" / f"{row['label']}_{row['hospital_id']}.png")
            review.append({"hospital_id": row["hospital_id"], "label": row["label"],
                           "split": split.get(row["hospital_id"]), "r_peaks": int(len(peaks)),
                           "flagged_pvc": int(wide.sum()), "flagged_ids": " ".join(str(i + 1) for i in np.where(wide)[0]),
                           "correct_pvc": "", "false_pvc": "", "missed_pvc": "", "notes": ""})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "auto_check_per_patient.csv", index=False)
    sheet = OUT / "manual_review.csv"
    if sheet.exists():
        print(f"{sheet} already exists, not overwritten (delete it to start the review again)")
    else:
        pd.DataFrame(review).to_csv(sheet, index=False)

    both = df.dropna(subset=["gap_ms"])
    tc = df.dropna(subset=["frac_pvc_closer_to_pvc_template"])
    out = {"n_patients": int(len(df)), "n_with_narrow_and_wide_beats": int(len(both)),
           "narrow_width_ms_median": float(both.narrow_width_ms.median()),
           "wide_width_ms_median": float(both.wide_width_ms.median()),
           "gap_ms_median": float(both.gap_ms.median()), "gap_ms_q1": float(both.gap_ms.quantile(.25)),
           "patients_gap_below_10ms": int((both.gap_ms < 10).sum()),
           "patients_without_flagged_pvc": int((df.flagged_pvc == 0).sum()),
           "template_check_patients": int(len(tc)),
           "pvc_closer_to_pvc_template_mean_fraction": float(tc.frac_pvc_closer_to_pvc_template.mean()),
           "patients_all_pvc_closer_to_pvc_template": int((tc.frac_pvc_closer_to_pvc_template == 1).sum()),
           "median_r_pvc_template": float(tc.median_r_pvc_template.median()),
           "median_r_normal_template": float(tc.median_r_normal_template.median()),
           "manual_sample": {"n_patients": len(review), "n_per_class": args.n_per_class, "seed": SEED}}
    (OUT / "auto_check_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"\nFigures for manual review: {OUT / 'figures'} ({len(review)} patients)")
    print(f"Fill in {sheet} (correct_pvc / false_pvc / missed_pvc per patient), then run with --summarize")


if __name__ == "__main__":
    main()
