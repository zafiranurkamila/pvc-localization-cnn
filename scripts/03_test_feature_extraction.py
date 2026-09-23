"""Fase 3 sanity check: extract PSD, Wavelet, HOS features dari beberapa sample beats.

Usage:
    python scripts/03_test_feature_extraction.py --n-patients 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config
from pvc_localization.data.loader import load_ecg, load_patient_index
from pvc_localization.preprocessing.beat_detection import extract_pvc_beats
from pvc_localization.preprocessing.filtering import clean_signal
from pvc_localization.preprocessing.normalization import zscore_normalize
from pvc_localization.features.psd import flatten_psd_features
from pvc_localization.features.wavelet import flatten_cwt_features
from pvc_localization.features.hos import flatten_hos_features


def extract_features_from_patient(row):
    ecg = load_ecg(row["processed_path"]).to_numpy()
    ecg_filtered = clean_signal(ecg)

    pvc_beats, pvc_peaks = extract_pvc_beats(ecg_filtered)
    if len(pvc_beats) == 0:
        return []

    results = []
    for beat_idx, beat_raw in enumerate(pvc_beats):
        beat = zscore_normalize(beat_raw)

        try:
            psd_feat = flatten_psd_features(beat)
            wavelet_feat = flatten_cwt_features(beat)
            hos_feat = flatten_hos_features(beat)

            results.append({
                "hospital_id": row["hospital_id"],
                "label": row["label"],
                "beat_idx": beat_idx,
                "psd_shape": psd_feat.shape,
                "psd_nan": np.isnan(psd_feat).sum(),
                "psd_inf": np.isinf(psd_feat).sum(),
                "wavelet_shape": wavelet_feat.shape,
                "wavelet_nan": np.isnan(wavelet_feat).sum(),
                "wavelet_inf": np.isinf(wavelet_feat).sum(),
                "hos_shape": hos_feat.shape,
                "hos_nan": np.isnan(hos_feat).sum(),
                "hos_inf": np.isinf(hos_feat).sum(),
            })
        except Exception as e:
            print(f"ERROR extracting features from {row['hospital_id']}.beat{beat_idx}: {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=3)
    args = parser.parse_args()

    index = load_patient_index()
    sample = pd_concat_balanced(index, args.n_patients)

    all_results = []
    for _, row in sample.iterrows():
        results = extract_features_from_patient(row)
        all_results.extend(results)

    print(f"\nExtracted {len(all_results)} beats across {len(sample)} patients")
    if all_results:
        r = all_results[0]
        print(f"\nExample shapes per beat:")
        print(f"  PSD: {r['psd_shape']} (any NaN: {r['psd_nan']}, any Inf: {r['psd_inf']})")
        print(f"  Wavelet: {r['wavelet_shape']} (any NaN: {r['wavelet_nan']}, any Inf: {r['wavelet_inf']})")
        print(f"  HOS: {r['hos_shape']} (any NaN: {r['hos_nan']}, any Inf: {r['hos_inf']})")


def pd_concat_balanced(index, n_patients):
    import pandas as pd
    per_class = max(1, n_patients // 2)
    rvot = index[index["label"] == "RVOT"].head(per_class)
    lvot = index[index["label"] == "LVOT"].head(n_patients - per_class)
    return pd.concat([rvot, lvot]).reset_index(drop=True)


if __name__ == "__main__":
    main()
