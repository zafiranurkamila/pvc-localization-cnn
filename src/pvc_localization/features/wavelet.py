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
        scalograms: (12, n_scales, window_len // CWT_TIME_DOWNSAMPLE) log-magnitude
                   time-frequency maps over CWT_FMIN_HZ..FEATURE_FMAX_HZ, for 2D-Conv
    """
    n_leads, window_len = beat.shape

    freqs = np.geomspace(config.CWT_FMIN_HZ, config.FEATURE_FMAX_HZ, n_scales)
    scales = pywt.central_frequency(wavelet) * fs / freqs

    coeffs, _ = pywt.cwt(beat, scales, wavelet, method="fft", axis=-1)  # (n_scales, n_leads, window_len)
    magnitude = np.abs(coeffs).transpose(1, 0, 2)

    ds = config.CWT_TIME_DOWNSAMPLE
    n_time = window_len // ds
    magnitude = magnitude[:, :, :n_time * ds].reshape(n_leads, n_scales, n_time, ds).mean(axis=-1)

    return np.log1p(magnitude)


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
