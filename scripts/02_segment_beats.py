"""Fase 2 sanity check: detect + segment PVC beats for a handful of patients
and save plots so the segmentation can be verified visually before running it
on the full dataset.

Usage:
    python scripts/02_segment_beats.py --n-patients 5
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config
from pvc_localization.data.loader import load_ecg, load_patient_index
from pvc_localization.preprocessing.beat_detection import (
    detect_r_peaks,
    flag_wide_beats,
    qrs_width_ms,
    segment_beats,
)
from pvc_localization.preprocessing.filtering import clean_signal
from pvc_localization.preprocessing.normalization import zscore_normalize


def process_patient(row, out_dir: Path):
    ecg = load_ecg(row["processed_path"]).to_numpy()
    ecg_filtered = clean_signal(ecg)

    ref_lead_idx = config.LEADS.index(config.RPEAK_DETECTION_LEAD)
    ref_signal = ecg_filtered[:, ref_lead_idx]

    r_peaks = detect_r_peaks(ref_signal)
    widths = np.array([qrs_width_ms(ref_signal, p) for p in r_peaks])
    is_wide = flag_wide_beats(widths)
    pvc_peaks = r_peaks[is_wide]

    beats, kept_peaks = segment_beats(ecg_filtered, pvc_peaks)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    t = np.arange(len(ref_signal)) / config.SAMPLING_RATE_HZ
    axes[0].plot(t, ref_signal, linewidth=0.6)
    axes[0].scatter(r_peaks / config.SAMPLING_RATE_HZ, ref_signal[r_peaks],
                     color="green", s=15, label=f"all R-peaks ({len(r_peaks)})")
    axes[0].scatter(pvc_peaks / config.SAMPLING_RATE_HZ, ref_signal[pvc_peaks],
                     color="red", s=40, marker="x", label=f"flagged PVC (wide QRS) ({len(pvc_peaks)})")
    axes[0].set_title(f"Patient {row['hospital_id']} ({row['label']}) — lead {config.RPEAK_DETECTION_LEAD}")
    axes[0].legend(loc="upper right")
    axes[0].set_xlabel("time (s)")

    if len(beats) > 0:
        beat_norm = zscore_normalize(beats[0])
        for lead_i, lead_name in enumerate(config.LEADS):
            axes[1].plot(beat_norm[lead_i], label=lead_name, linewidth=0.8)
        axes[1].set_title(f"First segmented PVC beat, all 12 leads (z-scored), window={config.BEAT_LENGTH_SAMPLES} samples")
        axes[1].legend(loc="upper right", ncol=4, fontsize=7)
    else:
        axes[1].set_title("No PVC beats segmented (edge-of-recording or none flagged)")

    fig.tight_layout()
    out_path = out_dir / f"segmentation_check_{row['hospital_id']}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {
        "hospital_id": row["hospital_id"],
        "label": row["label"],
        "n_total_peaks": len(r_peaks),
        "n_flagged_pvc": len(pvc_peaks),
        "n_segmented_beats": len(beats),
        "plot": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=5)
    args = parser.parse_args()

    index = load_patient_index()
    sample = pd_concat_balanced(index, args.n_patients)

    out_dir = config.FIGURES_DIR / "segmentation_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in sample.iterrows():
        result = process_patient(row, out_dir)
        print(result)


def pd_concat_balanced(index, n_patients):
    import pandas as pd
    per_class = max(1, n_patients // 2)
    rvot = index[index["label"] == "RVOT"].head(per_class)
    lvot = index[index["label"] == "LVOT"].head(n_patients - per_class)
    return pd.concat([rvot, lvot]).reset_index(drop=True)


if __name__ == "__main__":
    main()
