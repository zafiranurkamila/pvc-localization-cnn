"""Build the patient-level label index used by every later stage.

Reads data/external/Diagnosis.xlsx, keeps only PVC patients, maps
LeftRight -> {RVOT, LVOT}, and cross-checks that a matching raw and
noise-reduced ECG CSV exists for each patient. Writes the result to
data/interim/patient_index.csv.

Run from the project root:
    python scripts/01_build_label_index.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config


def main():
    diagnosis = pd.read_excel(config.DIAGNOSIS_XLSX)

    pvc = diagnosis[diagnosis["Type"] == "PVC"].copy()
    dropped_vt = len(diagnosis) - len(pvc)

    pvc["label"] = pvc["LeftRight"].map(config.LEFTRIGHT_TO_LABEL)
    unmapped = pvc[pvc["label"].isna()]
    if len(unmapped):
        print(f"WARNING: {len(unmapped)} rows have unmapped LeftRight values, dropping them")
        pvc = pvc.dropna(subset=["label"])

    records = []
    missing_raw, missing_processed = 0, 0
    for _, row in pvc.iterrows():
        hospital_id = int(row["HospitalID"])
        raw_path = config.RAW_ECG_DIR / f"{hospital_id}.csv"
        processed_path = config.PROCESSED_ECG_DIR / f"{hospital_id}.csv"

        if not raw_path.exists():
            missing_raw += 1
            continue
        if not processed_path.exists():
            missing_processed += 1
            continue

        records.append({
            "hospital_id": hospital_id,
            "label": row["label"],
            "sublocation": row["Sublocation"],
            "gender": row["Gender"],
            "raw_path": str(raw_path.relative_to(config.PROJECT_ROOT)),
            "processed_path": str(processed_path.relative_to(config.PROJECT_ROOT)),
        })

    index = pd.DataFrame.from_records(records)
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    index.to_csv(config.PATIENT_INDEX_CSV, index=False)

    print(f"Total diagnosis rows: {len(diagnosis)}")
    print(f"Dropped (non-PVC, e.g. VT): {dropped_vt}")
    print(f"Missing raw ECG file: {missing_raw}")
    print(f"Missing processed ECG file: {missing_processed}")
    print(f"Final indexed patients: {len(index)}")
    print(index["label"].value_counts())
    print(f"\nSaved to {config.PATIENT_INDEX_CSV}")


if __name__ == "__main__":
    main()
