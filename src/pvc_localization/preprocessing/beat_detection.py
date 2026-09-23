"""R-peak detection, PVC beat flagging, and beat segmentation.

We only care about the PVC beats for feature extraction (that's what carries
the RVOT/LVOT morphology signature), not every normal sinus beat in the strip.

Design note (found by inspecting real recordings, not from the proposal):
recordings in this dataset are short (5-30s) and often show stable bigeminy/
trigeminy (a PVC every 2nd or 3rd beat), not an isolated one-off ectopic beat
in an otherwise normal rhythm. neurokit2's Kubios-style RR-interval artifact
detector (`signal_fixpeaks`) is tuned for the latter case and misses PVCs in
these periodic patterns entirely (verified: it flagged 0 ectopic beats on
several patients with an obvious visual PVC pattern).

What does work reliably is QRS width: a PVC's QRS is wide (no His-Purkinje
conduction) while a normal beat's is narrow, and on every patient inspected
so far this produces a clean bimodal split (~30-40ms normal vs ~80-135ms PVC)
even though neurokit2's own delineation (`ecg_delineate`) fails to find
onset/offset on the wide beats (it's tuned to normal QRS morphology too).
So beat classification here uses a custom amplitude-threshold width estimate
plus a max-gap 1D split, not a fixed ms cutoff or RR statistics.
"""
import numpy as np
import neurokit2 as nk

from pvc_localization import config


def detect_r_peaks(lead_signal: np.ndarray, fs: int = config.SAMPLING_RATE_HZ) -> np.ndarray:
    """Return R-peak sample indices detected on a single lead."""
    _, info = nk.ecg_peaks(lead_signal, sampling_rate=fs)
    return np.asarray(info["ECG_R_Peaks"])


def qrs_width_ms(signal: np.ndarray, peak: int, fs: int = config.SAMPLING_RATE_HZ,
                  half_window_ms: float = 150, amplitude_fraction: float = 0.25) -> float:
    """Estimate QRS width at one peak: how far the signal stays beyond
    `amplitude_fraction` of the peak's height above local baseline, contiguous
    around the peak. Morphology-agnostic (works even on wide/aberrant beats),
    unlike wavelet-based delineation which assumes normal QRS shape.
    """
    half_window = int(half_window_ms / 1000 * fs)
    lo, hi = max(0, peak - half_window), min(len(signal), peak + half_window)
    window = signal[lo:hi]
    local_peak = peak - lo

    baseline = np.median(np.concatenate([window[:10], window[-10:]]))
    amplitude = signal[peak] - baseline
    if amplitude == 0:
        return 0.0
    threshold = baseline + amplitude_fraction * amplitude
    above = window > threshold if amplitude > 0 else window < threshold

    left = local_peak
    while left > 0 and above[left - 1]:
        left -= 1
    right = local_peak
    while right < len(above) - 1 and above[right + 1]:
        right += 1
    return (right - left) / fs * 1000


def flag_wide_beats(widths_ms: np.ndarray) -> np.ndarray:
    """Split widths into a narrow (normal) and wide (PVC) group using the
    largest gap in the sorted values, rather than a fixed ms threshold or a
    multiple of the median — robust regardless of the normal:PVC ratio in a
    given recording (some patients here are majority-PVC).

    Returns a boolean array (same order as input) marking the wide group.
    Needs at least 2 beats with distinguishable widths; otherwise nothing is
    flagged (can't tell PVC from normal off a single beat).
    """
    n = len(widths_ms)
    if n < 2:
        return np.zeros(n, dtype=bool)

    order = np.argsort(widths_ms)
    sorted_widths = widths_ms[order]
    gaps = np.diff(sorted_widths)
    split_at = np.argmax(gaps)

    if gaps[split_at] <= 0:  # all widths identical -> nothing stands out
        return np.zeros(n, dtype=bool)

    is_wide_sorted = np.arange(n) > split_at
    is_wide = np.zeros(n, dtype=bool)
    is_wide[order] = is_wide_sorted
    return is_wide


def segment_beats(multilead_signal: np.ndarray, peak_indices: np.ndarray,
                   pre_samples: int = config.BEAT_PRE_SAMPLES,
                   post_samples: int = config.BEAT_POST_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """Cut fixed-length windows around each peak from a (n_samples, n_leads) array.

    Returns (beats, kept_peak_indices): beats has shape (n_kept, n_leads, window_len).
    Peaks too close to either edge of the recording are dropped.
    """
    n_samples = multilead_signal.shape[0]
    beats, kept = [], []
    for peak in peak_indices:
        start, end = peak - pre_samples, peak + post_samples
        if start < 0 or end > n_samples:
            continue
        window = multilead_signal[start:end, :]
        beats.append(window.T)  # (n_leads, window_len)
        kept.append(peak)
    if not beats:
        return np.empty((0, multilead_signal.shape[1], pre_samples + post_samples)), np.array([])
    return np.stack(beats), np.array(kept)


def extract_pvc_beats(multilead_signal: np.ndarray, fs: int = config.SAMPLING_RATE_HZ,
                       reference_lead: str = config.RPEAK_DETECTION_LEAD) -> tuple[np.ndarray, np.ndarray]:
    """End-to-end: detect R-peaks on the reference lead, flag the wide (PVC)
    ones by QRS width, segment all 12 leads around the flagged peaks only.

    Returns (pvc_beats, pvc_peak_indices).
    """
    reference_signal = multilead_signal[:, config.LEADS.index(reference_lead)]
    r_peaks = detect_r_peaks(reference_signal, fs=fs)
    if len(r_peaks) < 2:
        return segment_beats(multilead_signal, np.array([]))

    widths = np.array([qrs_width_ms(reference_signal, p, fs=fs) for p in r_peaks])
    is_wide = flag_wide_beats(widths)
    pvc_peaks = r_peaks[is_wide]
    return segment_beats(multilead_signal, pvc_peaks)
