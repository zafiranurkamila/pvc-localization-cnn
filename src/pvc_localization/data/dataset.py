"""PyTorch Dataset for PVC beats with feature caching and lazy extraction."""
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pvc_localization import config
from pvc_localization.data.loader import load_ecg, load_patient_index
from pvc_localization.preprocessing.beat_detection import extract_pvc_beats
from pvc_localization.preprocessing.filtering import clean_signal
from pvc_localization.preprocessing.normalization import zscore_normalize
from pvc_localization.features.psd import flatten_psd_features
from pvc_localization.features.wavelet import flatten_cwt_features, extract_cwt_scalogram
from pvc_localization.features.hos import flatten_hos_features


class PVCBeatsDataset(Dataset):
    """Load patient ECG → segment PVC beats → extract features on-the-fly.

    Args:
        patient_ids: list of HospitalID values to include
        feature_scenario: which features to extract ("psd", "wavelet", "hos", or combinations)
        cache_dir: directory to save/load feature cache (optional, for faster loading on GPU machine)
    """

    def __init__(
        self,
        patient_ids: list[int],
        feature_scenario: list[str],
        cache_dir: Path = None,
    ):
        self.index = load_patient_index()
        self.index = self.index[self.index["hospital_id"].isin(patient_ids)].reset_index(drop=True)
        self.feature_scenario = feature_scenario
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        self.beats_list = []
        self.load_all_beats()

    def load_all_beats(self):
        """Pre-load all PVC beats from all patients (one-time, at init)."""
        for _, row in self.index.iterrows():
            beats = self.extract_patient_beats(row)
            for beat_idx, beat in enumerate(beats):
                self.beats_list.append((row["hospital_id"], row["label"], beat_idx, beat))

    def extract_patient_beats(self, row) -> np.ndarray:
        """Return array of PVC beats from one patient. Shape: (n_beats, n_leads, window_len)."""
        ecg = load_ecg(row["processed_path"]).to_numpy()
        ecg_filtered = clean_signal(ecg)
        pvc_beats, _ = extract_pvc_beats(ecg_filtered)
        if len(pvc_beats) == 0:
            return np.empty((0, 12, config.BEAT_LENGTH_SAMPLES))
        return pvc_beats

    def __len__(self):
        return len(self.beats_list)

    def labels(self) -> list[int]:
        return [config.LABEL_TO_INT[label] for _, label, _, _ in self.beats_list]

    def patient_groups(self) -> list[int]:
        return [hospital_id for hospital_id, _, _, _ in self.beats_list]

    def __getitem__(self, idx: int):
        hospital_id, label, beat_idx, beat_raw = self.beats_list[idx]
        beat = zscore_normalize(beat_raw)
        label_int = config.LABEL_TO_INT[label]

        cache_key = f"{hospital_id}_beat{beat_idx}_" + "_".join(sorted(self.feature_scenario) or ["baseline"])

        features = {}
        if not self.feature_scenario:
            features["raw"] = beat.astype(np.float32)
        if "psd" in self.feature_scenario:
            features["psd"] = self._get_or_compute(cache_key + "_psd", lambda: flatten_psd_features(beat))
        if "wavelet" in self.feature_scenario:
            features["wavelet"] = self._get_or_compute(
                cache_key + "_wavelet",
                lambda: extract_cwt_scalogram(beat)  # return 3D, not flattened
            )
        if "hos" in self.feature_scenario:
            features["hos"] = self._get_or_compute(cache_key + "_hos", lambda: flatten_hos_features(beat))

        # Convert to PyTorch tensors
        for key in features:
            if isinstance(features[key], np.ndarray):
                features[key] = torch.from_numpy(features[key]).float()

        return {
            "hospital_id": hospital_id,
            "beat_idx": beat_idx,
            "label": label_int,
            **features,
        }

    def _get_or_compute(self, cache_key: str, compute_fn):
        """Load from cache or compute and cache."""
        if not self.cache_dir:
            return compute_fn()

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        result = compute_fn()
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        return result


def create_train_val_test_split(
    random_state: int = config.RANDOM_SEED,
    test_size: float = config.TEST_SIZE,
) -> tuple[list[int], list[int]]:
    """Stratified split at patient level (not beat level) to avoid data leakage.

    Returns:
        (train_patient_ids, test_patient_ids)
    """
    from sklearn.model_selection import train_test_split

    index = load_patient_index()
    train_ids, test_ids = train_test_split(
        index["hospital_id"].values,
        test_size=test_size,
        stratify=index["label"].values,
        random_state=random_state,
    )
    return list(train_ids), list(test_ids)
