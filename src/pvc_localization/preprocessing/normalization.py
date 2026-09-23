"""Per-beat normalization so amplitude scale differences (e.g. raw vs
noise-reduced ECG use different units) don't confound the model."""
import numpy as np


def zscore_normalize(beat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize each lead independently. `beat` shape: (n_leads, window_len)."""
    mean = beat.mean(axis=1, keepdims=True)
    std = beat.std(axis=1, keepdims=True)
    return (beat - mean) / (std + eps)
