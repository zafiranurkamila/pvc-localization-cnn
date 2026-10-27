"""Sinus (non-PVC) reference beats per patient for the dual-rhythm model of the journal paper.

For every patient the beats that the PVC extraction does NOT flag as wide (the narrow group of the QRS-width split)
are cut with the same 800-ms window as the PVC beats, z-scored per lead and averaged by the median into one
12 x 1600 sinus template. The template is the patient's own reference rhythm, so the model can compare each PVC
with the conducted beat of the same patient (cf. Nakasone et al. 2024; Miyamae et al. 2026).

Patients without a usable narrow beat (recordings that consist almost only of PVCs, or where no narrow group exists)
get no template; they are listed so that the fallback (an all-zero template plus a flag) can be reported.

No training. Needs data/processed and data/interim/patient_index.csv (run on the PC).

Usage:
    python scripts/20_sinus_beats.py
Output:
    data/interim/sinus/<hospital_id>.npy      12 x 1600 float32 median sinus template
    results/sinus/sinus_beats_per_patient.csv  counts per patient
    results/sinus/sinus_summary.json           availability by class and split
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
from pvc_localization.preprocessing.beat_detection import detect_r_peaks, flag_wide_beats, qrs_width_ms  # noqa: E402
from pvc_localization.preprocessing.filtering import clean_signal  # noqa: E402
from pvc_localization.preprocessing.normalization import zscore_normalize  # noqa: E402

OUT_BEATS = config.INTERIM_DIR / "sinus"
OUT_RES = config.RESULTS_DIR / "sinus"
REF = config.LEADS.index(config.RPEAK_DETECTION_LEAD)
PRE, POST = config.BEAT_PRE_SAMPLES, config.BEAT_POST_SAMPLES


def main():
    OUT_BEATS.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)
    index = load_patient_index()
    train_ids, test_ids = create_train_val_test_split()
    split = {pid: "train" for pid in train_ids} | {pid: "test" for pid in test_ids}

    rows = []
    for _, row in tqdm(index.iterrows(), total=len(index), desc="patients"):
        ecg = clean_signal(load_ecg(row["processed_path"]).to_numpy())
        sig = ecg[:, REF]
        peaks = detect_r_peaks(sig)
        widths = np.array([qrs_width_ms(sig, p) for p in peaks]) if len(peaks) else np.array([])
        wide = flag_wide_beats(widths) if len(peaks) >= 2 else np.zeros(len(peaks), dtype=bool)
        kept = (peaks - PRE >= 0) & (peaks + POST <= len(sig))
        narrow_idx = np.where(~wide & kept)[0]
        pvc_idx = np.where(wide & kept)[0]
        has_sinus = len(narrow_idx) > 0 and len(pvc_idx) > 0
        if has_sinus:
            beats = np.stack([zscore_normalize(ecg[peaks[i] - PRE:peaks[i] + POST].T) for i in narrow_idx])
            np.save(OUT_BEATS / f"{row['hospital_id']}.npy", np.median(beats, axis=0).astype(np.float32))
        rows.append({"hospital_id": row["hospital_id"], "label": row["label"], "split": split.get(row["hospital_id"]),
                     "r_peaks": int(len(peaks)), "pvc_beats": int(len(pvc_idx)), "sinus_beats": int(len(narrow_idx)),
                     "sinus_qrs_ms_median": float(np.median(widths[narrow_idx])) if len(narrow_idx) else np.nan,
                     "has_sinus_template": bool(has_sinus)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_RES / "sinus_beats_per_patient.csv", index=False)
    with_pvc = df[df.pvc_beats > 0]
    summary = {
        "patients": int(len(df)), "patients_with_pvc_beats": int(len(with_pvc)),
        "with_sinus_template": int(with_pvc.has_sinus_template.sum()),
        "without_sinus_template": int((~with_pvc.has_sinus_template).sum()),
        "by_label": {k: {"with_pvc": int(len(g)), "with_sinus": int(g.has_sinus_template.sum())}
                     for k, g in with_pvc.groupby("label")},
        "by_split": {k: {"with_pvc": int(len(g)), "with_sinus": int(g.has_sinus_template.sum())}
                     for k, g in with_pvc.groupby("split")},
        "sinus_beats_per_patient_median": float(with_pvc.sinus_beats.median()),
        "sinus_beats_per_patient_q1_q3": [float(with_pvc.sinus_beats.quantile(.25)),
                                          float(with_pvc.sinus_beats.quantile(.75))],
        "patients_without_sinus": with_pvc.loc[~with_pvc.has_sinus_template, ["hospital_id", "label", "pvc_beats"]]
                                          .to_dict("records"),
    }
    (OUT_RES / "sinus_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "patients_without_sinus"}, indent=1))
    print(f"patients without a sinus template: {summary['without_sinus_template']} (listed in sinus_summary.json)")


if __name__ == "__main__":
    main()
