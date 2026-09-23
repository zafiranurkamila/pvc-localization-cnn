"""Bandpass + powerline notch filtering for 12-lead ECG signals.

Uses second-order-section (SOS) filter representations rather than the
transfer-function (b, a) form. At fs=2000 Hz a 0.5 Hz bandpass edge has a
normalized frequency of 0.0005, which is numerically unstable as (b, a)
coefficients (filtfilt can blow the signal up by 10-20x) but is stable as SOS.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos

from pvc_localization import config


def bandpass_filter(signal: np.ndarray, low_hz: float = 0.5, high_hz: float = 40.0,
                     fs: int = config.SAMPLING_RATE_HZ, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass. `signal` shape: (n_samples,) or (n_samples, n_leads)."""
    nyquist = fs / 2
    sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, signal, axis=0)


def notch_filter(signal: np.ndarray, freq_hz: float = 50.0,
                  fs: int = config.SAMPLING_RATE_HZ, quality: float = 30.0) -> np.ndarray:
    """Remove powerline interference (50 Hz default; use 60 Hz if the recording site needs it)."""
    b, a = iirnotch(freq_hz / (fs / 2), quality)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, signal, axis=0)


def clean_signal(signal: np.ndarray, notch_hz: float = 50.0) -> np.ndarray:
    """Bandpass + notch, in the order recommended for ECG baseline wander / powerline removal."""
    signal = bandpass_filter(signal)
    signal = notch_filter(signal, freq_hz=notch_hz)
    return signal
