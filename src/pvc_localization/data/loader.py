"""Load per-patient 12-lead ECG CSVs and the label index, with a guaranteed
lead column order (the raw CSV files store leads alphabetically, not in the
clinical I/II/III/aVR/... order used throughout this project)."""
import pandas as pd

from pvc_localization import config


def load_patient_index() -> pd.DataFrame:
    if not config.PATIENT_INDEX_CSV.exists():
        raise FileNotFoundError(
            f"{config.PATIENT_INDEX_CSV} not found — run scripts/01_build_label_index.py first"
        )
    return pd.read_csv(config.PATIENT_INDEX_CSV)


def load_ecg(path) -> pd.DataFrame:
    """Load one patient's ECG CSV, columns reordered to config.LEADS."""
    df = pd.read_csv(config.PROJECT_ROOT / path)
    return df[config.LEADS]
