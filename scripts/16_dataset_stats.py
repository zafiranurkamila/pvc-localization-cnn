"""Dataset and PVC-extraction statistics (reviewer comment 13 and minor 12).

For every PVC patient: recording length, number of detected R peaks, number of beats flagged as PVC
(wide QRS) and number of PVC beats kept after segmentation. Summaries per class and per train/test
split: beats per patient (median, IQR, range) and the number of patients without any extracted beat.

No training. Needs data/processed and data/interim/patient_index.csv (run on the PC).

Usage:
    python scripts/16_dataset_stats.py
Output: results/dataset_stats.json and results/dataset_stats_per_patient.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config  # noqa: E402
from pvc_localization.data.dataset import create_train_val_test_split  # noqa: E402
from pvc_localization.data.loader import load_ecg, load_patient_index  # noqa: E402
from pvc_localization.preprocessing.beat_detection import (  # noqa: E402
    detect_r_peaks, extract_pvc_beats, flag_wide_beats, qrs_width_ms)
from pvc_localization.preprocessing.filtering import clean_signal  # noqa: E402


def summary(v):
    v = np.asarray(v)
    return {"n_patients": int(v.size), "total_beats": int(v.sum()), "median": float(np.median(v)),
            "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
            "min": int(v.min()), "max": int(v.max()), "zero_beat_patients": int((v == 0).sum())}


def main():
    index = load_patient_index()
    train_ids, test_ids = create_train_val_test_split()
    split = {pid: "train" for pid in train_ids} | {pid: "test" for pid in test_ids}
    ref = config.LEADS.index(config.RPEAK_DETECTION_LEAD)
    rows = []
    for _, row in tqdm(index.iterrows(), total=len(index), desc="patients"):
        ecg = clean_signal(load_ecg(row["processed_path"]).to_numpy())
        r_peaks = detect_r_peaks(ecg[:, ref])
        wide = 0
        if len(r_peaks) >= 2:
            widths = np.array([qrs_width_ms(ecg[:, ref], p) for p in r_peaks])
            wide = int(flag_wide_beats(widths).sum())
        beats, _ = extract_pvc_beats(ecg)
        rows.append({"hospital_id": row["hospital_id"], "label": row["label"], "split": split.get(row["hospital_id"]),
                     "duration_s": round(len(ecg) / config.SAMPLING_RATE_HZ, 2), "r_peaks": int(len(r_peaks)),
                     "wide_beats": wide, "pvc_beats_kept": int(len(beats))})
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "dataset_stats_per_patient.csv", index=False)

    out = {"all": summary(df.pvc_beats_kept),
           "by_class": {lab: summary(g.pvc_beats_kept) for lab, g in df.groupby("label")},
           "by_split": {s: summary(g.pvc_beats_kept) for s, g in df.groupby("split")},
           "by_split_class": {f"{s}_{lab}": summary(g.pvc_beats_kept) for (s, lab), g in df.groupby(["split", "label"])},
           "recording_duration_s": {"median": float(df.duration_s.median()), "min": float(df.duration_s.min()),
                                    "max": float(df.duration_s.max())},
           "fraction_of_r_peaks_flagged_wide": float(df.wide_beats.sum() / max(df.r_peaks.sum(), 1)),
           "wide_beats_dropped_at_recording_edges": int(df.wide_beats.sum() - df.pvc_beats_kept.sum())}
    (config.RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
