"""Higher-Order Statistics (HOS) features: Bispectrum and higher-order moments.

Proposal Bab 2.4.2 & Persamaan 2.2/3.4: bispectrum sebagai fitur non-linear
untuk menangkap coupling frekuensi, plus momen orde-3 (skewness) sebagai
pelengkap yang murah dihitung.

Bispectrum: B(f1, f2) = E[X(f1) * X(f2) * X*(f1+f2)] (magnitude & phase)
Skewness: γ = E[(X - μ)³] / σ³ (per lead)
"""
from functools import lru_cache

import numpy as np
from scipy import signal as scipy_signal

from pvc_localization import config


@lru_cache(maxsize=8)
def _bispectrum_grid(n_freqs: int):
    f1_idx, f2_idx = np.meshgrid(np.arange(n_freqs), np.arange(n_freqs), indexing="ij")
    mask = (f1_idx + f2_idx) < n_freqs
    f3_idx = np.where(mask, f1_idx + f2_idx, 0)
    return f1_idx, f2_idx, f3_idx, mask


def bispectrum_direct(x: np.ndarray, nperseg: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bispectrum via direct FFT-based triple product."""
    if nperseg is None:
        nperseg = len(x)

    n_segments = max(1, len(x) // nperseg)
    bispectrum_sum = None

    for seg in range(n_segments):
        start = seg * nperseg
        end = min(start + nperseg, len(x))
        x_seg = x[start:end]

        nfft = 2 ** int(np.ceil(np.log2(len(x_seg))))
        X = np.fft.fft(x_seg, n=nfft)

        # B(f1, f2) = X(f1) * X(f2) * conj(X(f1 + f2)), region f1 + f2 < n_freqs
        n_freqs = nfft // 2
        f1_idx, f2_idx, f3_idx, mask = _bispectrum_grid(n_freqs)
        bispectrum_seg = X[f1_idx] * X[f2_idx] * np.conj(X[f3_idx])
        bispectrum_seg[~mask] = 0

        if bispectrum_sum is None:
            bispectrum_sum = bispectrum_seg
        else:
            bispectrum_sum += bispectrum_seg

    bispectrum = np.abs(bispectrum_sum) / n_segments
    freqs = np.fft.fftfreq(nfft, d=1.0)[:n_freqs]

    return bispectrum, freqs, freqs


def bispectrum_features(bispectrum: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Reduce 2D bispectrum to fixed-size (n_bins * n_bins,) feature vector."""
    if bispectrum.shape[0] != n_bins or bispectrum.shape[1] != n_bins:
        src_i = np.arange(n_bins) * bispectrum.shape[0] // n_bins
        src_j = np.arange(n_bins) * bispectrum.shape[1] // n_bins
        bispectrum_resized = bispectrum[np.ix_(src_i, src_j)]
    else:
        bispectrum_resized = bispectrum

    return bispectrum_resized.flatten()


def higher_order_moments(x: np.ndarray) -> dict:
    """Compute skewness and excess kurtosis."""
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return {'skewness': 0.0, 'kurtosis': 0.0}

    m3 = np.mean((x - mean) ** 3) / (std ** 3)
    m4 = np.mean((x - mean) ** 4) / (std ** 4)

    return {
        'skewness': m3,
        'kurtosis': m4 - 3.0
    }


def extract_hos_features(beat: np.ndarray, fs: int = config.SAMPLING_RATE_HZ,
                         bispectrum_n_bins: int = 8) -> np.ndarray:
    """Extract HOS features (bispectrum + moments) from one beat across all 12 leads.

    Total size: 12 * (n_bins² + 2)
    """
    n_leads = beat.shape[0]
    bispectrum_size = bispectrum_n_bins ** 2
    total_size = 12 * (bispectrum_size + 2)

    features = np.zeros(total_size)
    idx = 0

    for lead_idx in range(n_leads):
        signal = beat[lead_idx, :]

        bisp, _, _ = bispectrum_direct(signal, nperseg=len(signal) // 2)
        bisp_features = bispectrum_features(bisp, n_bins=bispectrum_n_bins)
        features[idx : idx + bispectrum_size] = bisp_features
        idx += bispectrum_size

        moments = higher_order_moments(signal)
        features[idx] = moments['skewness']
        features[idx + 1] = moments['kurtosis']
        idx += 2

    return features


def flatten_hos_features(beat: np.ndarray, fs: int = config.SAMPLING_RATE_HZ,
                        bispectrum_n_bins: int = 8) -> np.ndarray:
    """Wrapper for consistent interface."""
    return extract_hos_features(beat, fs=fs, bispectrum_n_bins=bispectrum_n_bins)