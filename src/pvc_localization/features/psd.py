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
                         nperseg: int = None, n_freqs: int = 32) -> np.ndarray:
    """Extract PSD features from one beat across all 12 leads.

    Args:
        beat: (n_leads=12, window_len) array, already normalized
        fs: sampling rate in Hz
        nperseg: FFT window length for Welch (default: beat window length)
        n_freqs: number of frequency bins to return per lead

    Returns:
        features: (12, n_freqs) array with PSD values for each lead
    """
    if nperseg is None:
        nperseg = beat.shape[1]

    n_leads = beat.shape[0]
    features = np.zeros((n_leads, n_freqs))

    for lead_idx in range(n_leads):
        signal = beat[lead_idx, :]
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

        # Resample PSD to fixed number of frequency bins
        if len(psd) > n_freqs:
            # Downsample by taking every nth element
            step = len(psd) // n_freqs
            features[lead_idx, :] = psd[::step][:n_freqs]
        else:
            # Upsample by interpolation
            features[lead_idx, :] = np.interp(
                np.linspace(0, len(psd) - 1, n_freqs),
                np.arange(len(psd)),
                psd
            )

    return features


def flatten_psd_features(beat: np.ndarray, fs: int = config.SAMPLING_RATE_HZ,
                        n_freqs: int = 32) -> np.ndarray:
    """Extract PSD and return as 1D feature vector (12 leads × n_freqs).

    Output shape: (12 * n_freqs,)
    """
    features = extract_psd_features(beat, fs=fs, n_freqs=n_freqs)
    return features.flatten()
