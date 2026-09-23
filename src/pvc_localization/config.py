"""Central paths and constants for the PVC localization pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_ECG_DIR = DATA_DIR / "raw"
PROCESSED_ECG_DIR = DATA_DIR / "processed"
DIAGNOSIS_XLSX = DATA_DIR / "external" / "Diagnosis.xlsx"
INTERIM_DIR = DATA_DIR / "interim"
PATIENT_INDEX_CSV = INTERIM_DIR / "patient_index.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Confirmed from Zheng et al., "A 12-Lead ECG database to identify origins of
# idiopathic ventricular arrhythmia containing 334 patients", Scientific Data
# (2020): signals were collected with the EP WorkMate System (Abbott) at
# 2000 Hz. https://doi.org/10.1038/s41597-020-0440-8
SAMPLING_RATE_HZ = 2000

# Label mapping used throughout the project: dataset's LeftRight -> research label
LEFTRIGHT_TO_LABEL = {"Right": "RVOT", "Left": "LVOT"}
LABEL_TO_INT = {"RVOT": 0, "LVOT": 1}

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5

# Beat segmentation window around each detected R-peak. PVC QRS complexes are
# wide (often >150-200ms) with no preceding P-wave, so the window is asymmetric
# and generous enough to capture the full QRS + ST-T segment of a PVC beat.
BEAT_PRE_MS = 300
BEAT_POST_MS = 500
BEAT_PRE_SAMPLES = int(BEAT_PRE_MS / 1000 * SAMPLING_RATE_HZ)
BEAT_POST_SAMPLES = int(BEAT_POST_MS / 1000 * SAMPLING_RATE_HZ)
BEAT_LENGTH_SAMPLES = BEAT_PRE_SAMPLES + BEAT_POST_SAMPLES

# Lead used for R-peak detection (best QRS visibility in most 12-lead ECGs).
RPEAK_DETECTION_LEAD = "II"

# CWT settings for the Wavelet scalogram branch (2D-Conv input).
CWT_WAVELET = "cmor1.5-1.0"
CWT_NUM_SCALES = 64

FEATURE_SCENARIOS = {
    "baseline": [],
    "psd": ["psd"],
    "wavelet": ["wavelet"],
    "hos": ["hos"],
    "psd_wavelet": ["psd", "wavelet"],
    "psd_hos": ["psd", "hos"],
    "wavelet_hos": ["wavelet", "hos"],
    "psd_wavelet_hos": ["psd", "wavelet", "hos"],
}
