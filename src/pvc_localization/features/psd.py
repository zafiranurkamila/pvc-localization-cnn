"""Power Spectral Density (PSD) via Welch's method.

For each beat (n_leads, window_len), computes PSD and extracts a fixed-size
feature vector per lead via frequency binning or peak detection.

Persamaan 2.1/3.1 dari proposal: PSD menggunakan Welch dengan jumlah FFT points
yang sesuai untuk window panjang 1600 samples.
"""
import numpy as np
from scipy.signal import welch

from pvc_localization import config


def extract_psd_features(beat: np.ndarray, fs: int = config.SAMPLING_RATE_HZ,
                         nperseg: int = None, n_freqs: int = 32,
                         fmin: float = config.FEATURE_FMIN_HZ,
                         fmax: float = config.FEATURE_FMAX_HZ) -> np.ndarray:
    """Extract log-PSD features in the ECG band from one beat across all 12 leads.

    Args:
        beat: (n_leads=12, window_len) array, already normalized
        fs: sampling rate in Hz
        nperseg: FFT window length for Welch (default: beat window length)
        n_freqs: number of frequency bins per lead, evenly spaced over [fmin, fmax]

    Returns:
        features: (12, n_freqs) array of log10 mean power per bin
    """
    if nperseg is None:
        nperseg = beat.shape[1]

    freqs, psd = welch(beat, fs=fs, nperseg=nperseg, axis=-1)
    bin_idx = np.digitize(freqs, np.linspace(fmin, fmax, n_freqs + 1)) - 1

    features = np.zeros((beat.shape[0], n_freqs))
    for b in range(n_freqs):
        features[:, b] = psd[:, bin_idx == b].mean(axis=1)

    return np.log10(features + 1e-12)


def flatten_psd_features(beat: np.ndarray, fs: int = config.SAMPLING_RATE_HZ,
                        n_freqs: int = 32) -> np.ndarray:
    """Extract PSD and return as 1D feature vector (12 leads × n_freqs).

    Output shape: (12 * n_freqs,)
    """
    features = extract_psd_features(beat, fs=fs, n_freqs=n_freqs)
    return features.flatten()
