"""Continuous Wavelet Transform (CWT) for scalogram features.

Computes CWT for each lead independently, producing a scalogram (time × scale)
that serves as 2D input for the Wavelet branch of the multi-branch CNN.

Proposal Bab 2.6.2 (b): menggunakan CWT scalogram untuk multi-branch CNN dengan
2D convolution. Mother wavelet: complex Morlet (cmor1.5-1.0).
"""
import numpy as np
import pywt

from pvc_localization import config


def extract_cwt_scalogram(beat: np.ndarray,
                          wavelet: str = config.CWT_WAVELET,
                          n_scales: int = config.CWT_NUM_SCALES,
                          fs: int = config.SAMPLING_RATE_HZ) -> np.ndarray:
    """Compute CWT scalogram for all 12 leads.

    Args:
        beat: (n_leads=12, window_len) array, already normalized
        wavelet: mother wavelet name (default: "cmor1.5-1.0")
        n_scales: number of scale levels to compute
        fs: sampling rate

    Returns:
        scalograms: (12, n_scales, window_len) array where each [lead, :, :]
                   is a time-frequency representation suitable for 2D-Conv
    """
    n_leads, window_len = beat.shape

    # Linear scale spacing: maps to frequency range roughly 5 Hz to 500 Hz
    min_freq = 5.0
    max_freq = fs / 2  # Nyquist
    scales = np.linspace(1, n_scales, n_scales)
    scales = fs / (2 * np.pi * np.linspace(min_freq, max_freq, n_scales))

    scalograms = np.zeros((n_leads, n_scales, window_len))

    for lead_idx in range(n_leads):
        signal = beat[lead_idx, :]
        # Compute CWT for this lead
        # cwt returns (n_scales, n_samples)
        coeffs, _ = pywt.cwt(signal, scales, wavelet)
        # Take absolute value (magnitude)
        scalograms[lead_idx, :, :] = np.abs(coeffs)

    return scalograms


def flatten_cwt_features(beat: np.ndarray,
                        wavelet: str = config.CWT_WAVELET,
                        n_scales: int = config.CWT_NUM_SCALES,
                        fs: int = config.SAMPLING_RATE_HZ) -> np.ndarray:
    """Compute CWT scalogram and flatten to 1D feature vector.

    Output shape: (12 * n_scales * window_len,)
    Intended for comparison with other feature methods in early experiments;
    normally the multi-branch CNN uses the 3D form directly.
    """
    scalograms = extract_cwt_scalogram(beat, wavelet=wavelet, n_scales=n_scales, fs=fs)
    return scalograms.flatten()
