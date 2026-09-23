"""Fase 2 full run: segment PVC beats for ALL patients, save to data/interim/beats/

Output:
  - data/interim/beats/{hospital_id}.npy  (n_beats, 12, 1600) array
  - data/interim/beats_metadata.csv       (beat-level index: hospital_id, beat_idx, label)

Usage:
    python scripts/03_full_segmentation.py [--sample-only | --sample-size N]
        --sample-only   : run only on 10 patients (5 RVOT + 5 LVOT) for testing
        --sample-size N : run on N patients (default: all)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config
from pvc_localization.data.loader import load_ecg, load_patient_index
from pvc_localization.preprocessing.beat_detection import extract_pvc_beats
from pvc_localization.preprocessing.filtering import clean_signal
from pvc_localization.preprocessing.normalization import zscore_normalize


def segment_patient(row: pd.Series) -> dict:
    """Load, segment, and save beats for one patient. Returns metadata row."""
    hospital_id = row["hospital_id"]
    label = row["label"]

    try:
        ecg = load_ecg(row["processed_path"]).to_numpy()
        ecg_filtered = clean_signal(ecg)

        pvc_beats, pvc_peaks = extract_pvc_beats(ecg_filtered)

        if len(pvc_beats) == 0:
            return {
                "hospital_id": hospital_id,
                "n_beats": 0,
                "label": label,
                "status": "no_beats_found"
            }

        # Normalize each beat
        pvc_beats_norm = np.array([zscore_normalize(beat) for beat in pvc_beats])

        # Save beats array
        beats_dir = config.INTERIM_DIR / "beats"
        beats_dir.mkdir(parents=True, exist_ok=True)
        beats_file = beats_dir / f"{hospital_id}.npy"
        np.save(beats_file, pvc_beats_norm)

        return {
            "hospital_id": hospital_id,
            "n_beats": len(pvc_beats),
            "label": label,
            "status": "success"
        }

    except Exception as e:
        return {
            "hospital_id": hospital_id,
            "n_beats": 0,
            "label": label,
            "status": f"error: {str(e)}"
        }


def build_beat_metadata(index: pd.DataFrame) -> pd.DataFrame:
    """Build beat-level metadata from beat arrays in beats/ dir."""
    beats_dir = config.INTERIM_DIR / "beats"
    rows = []

    for hospital_id in index["hospital_id"]:
        beats_file = beats_dir / f"{hospital_id}.npy"
        if beats_file.exists():
            beats = np.load(beats_file)
            label = index[index["hospital_id"] == hospital_id]["label"].values[0]
            for beat_idx in range(len(beats)):
                rows.append({
                    "hospital_id": hospital_id,
                    "beat_idx": beat_idx,
                    "label": label,
                    "label_int": config.LABEL_TO_INT[label]
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-only", action="store_true",
                        help="Run on 10 patients (5 RVOT + 5 LVOT) only")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Run on N patients (if --sample-only not set)")
    args = parser.parse_args()

    index = load_patient_index()

    if args.sample_only:
        rvot = index[index["label"] == "RVOT"].head(5)
        lvot = index[index["label"] == "LVOT"].head(5)
        index = pd.concat([rvot, lvot]).reset_index(drop=True)
        print(f"Running on {len(index)} sample patients (5 RVOT + 5 LVOT)")
    elif args.sample_size:
        per_class = args.sample_size // 2
        rvot = index[index["label"] == "RVOT"].head(per_class)
        lvot = index[index["label"] == "LVOT"].head(args.sample_size - per_class)
        index = pd.concat([rvot, lvot]).reset_index(drop=True)
        print(f"Running on {len(index)} sample patients")
    else:
        print(f"Running on all {len(index)} patients")

    # Segment all patients
    results = []
    for _, row in tqdm(index.iterrows(), total=len(index), desc="Segmenting beats"):
        result = segment_patient(row)
        results.append(result)

    results_df = pd.DataFrame(results)
    success_count = (results_df["status"] == "success").sum()
    print(f"\nSegmentation complete: {success_count}/{len(results_df)} patients successful")
    print(f"Total beats segmented: {results_df['n_beats'].sum()}")

    # Build beat-level metadata
    beat_metadata = build_beat_metadata(index)
    beat_metadata_path = config.INTERIM_DIR / "beats_metadata.csv"
    beat_metadata.to_csv(beat_metadata_path, index=False)
    print(f"Beat metadata saved: {beat_metadata_path}")
    print(f"  - {len(beat_metadata)} beats across all patients")
    print(f"  - RVOT: {(beat_metadata['label'] == 'RVOT').sum()}")
    print(f"  - LVOT: {(beat_metadata['label'] == 'LVOT').sum()}")

    # Save segmentation results summary
    results_path = config.INTERIM_DIR / "segmentation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSegmentation summary: {results_path}")


if __name__ == "__main__":
    main()
