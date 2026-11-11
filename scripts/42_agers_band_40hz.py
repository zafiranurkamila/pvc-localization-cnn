"""AGERS 2026 camera-ready, reviewer 3: PSD and CWT limited to 0.5-40 Hz instead of 0.5-60 Hz.

The deployed AGERS model is the fixed-setting PSD + CWT model (results/models/psd_wavelet.pt; split 263/66, seed 42,
50 epochs, Adam 1e-3, class-weighted loss). This script trains the same model twice with the same setting and seed:
  fmax60   PSD bins and CWT frequencies up to 60 Hz (as in the paper), rerun to measure run-to-run variation
  fmax40   PSD bins (32 over 0.5-40 Hz) and CWT frequencies (64 over 2-40 Hz) up to 40 Hz
Each run = 5-fold patient CV on the 263 training patients + final fit on all of them, evaluated on the 66 held-out
patients at patient level (mean logit margin, LVOT if > 0), as in the AGERS paper; the stored predictions of the
paper's model are evaluated the same way for reference. Features of the 40 Hz run are cached separately
(data/cache_fmax40); the original cache and results are not touched. The pywt CWT time per beat is measured for both
bands on the same 200 held-out beats.

Run on the PC (GPU; data/processed). Usage:  python scripts/42_agers_band_40hz.py
Output: results/agers_revision/band_40hz.json and band_<arm>_test_predictions.csv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config  # noqa: E402
import pvc_localization.data.dataset as dataset_mod  # noqa: E402
import pvc_localization.training.trainer as trainer_mod  # noqa: E402
from pvc_localization.data.dataset import PVCBeatsDataset, create_train_val_test_split  # noqa: E402
from pvc_localization.features.psd import extract_psd_features  # noqa: E402
from pvc_localization.features.wavelet import extract_cwt_scalogram  # noqa: E402
from pvc_localization.preprocessing.normalization import zscore_normalize  # noqa: E402
from pvc_localization.training.trainer import CVTrainer, set_seed  # noqa: E402

OUT = config.RESULTS_DIR / "agers_revision"
ROOT = Path(__file__).resolve().parents[1]
ARMS = {"fmax60": (60.0, ROOT / "data" / "cache"), "fmax40": (40.0, ROOT / "data" / "cache_fmax40")}


def set_band(fmax):
    config.FEATURE_FMAX_HZ = fmax  # read at call time by the CWT
    dataset_mod.flatten_psd_features = lambda beat: extract_psd_features(beat, fmax=fmax).flatten()


def patient_metrics(df):
    p = df.prob_lvot.clip(1e-7, 1 - 1e-7)
    df = df.assign(m=np.log(p / (1 - p)))
    g = df.groupby("hospital_id").agg(y=("label", "first"), s=("m", "mean"))
    y, s = g.y.to_numpy(), g.s.to_numpy()
    pred = (s > 0).astype(int)
    sens, spec = (pred[y == 1] == 1).mean(), (pred[y == 0] == 0).mean()
    return {"macro_f1": float(f1_score(y, pred, average="macro")), "sensitivity_lvot": float(sens),
            "specificity": float(spec), "balanced_accuracy": float((sens + spec) / 2), "auc": float(roc_auc_score(y, s)),
            "false_lvot": int(((pred == 1) & (y == 0)).sum()), "missed_lvot": int(((pred == 0) & (y == 1)).sum()),
            "patients": int(len(y))}


def main():
    ckpt = torch.load(config.RESULTS_DIR / "models" / "psd_wavelet.pt", map_location="cpu", weights_only=False)
    hp = ckpt["hyperparameters"]
    features = ckpt["features"]
    print("deployed model hyperparameters:", hp, features)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ids, test_ids = create_train_val_test_split()
    OUT.mkdir(parents=True, exist_ok=True)

    res = {"hyperparameters": hp, "features": features, "device": device,
           "paper_model_stored_predictions": patient_metrics(
               pd.read_csv(config.RESULTS_DIR / "predictions" / "psd_wavelet_test_predictions.csv")),
           "arms": {}}

    # CWT time per beat (pywt, CPU) for both bands on the same held-out beats
    probe = PVCBeatsDataset(test_ids, [], cache_dir=None)
    beats = [zscore_normalize(b[3]) for b in probe.beats_list[:200]]
    for arm, (fmax, _) in ARMS.items():
        set_band(fmax)
        extract_cwt_scalogram(beats[0])
        t0 = time.perf_counter()
        for b in beats:
            extract_cwt_scalogram(b)
        res.setdefault("cwt_ms_per_beat_pywt", {})[arm] = (time.perf_counter() - t0) * 1000 / len(beats)

    for arm, (fmax, cache) in ARMS.items():
        set_band(fmax)
        trainer_mod.CACHE_DIR = cache
        set_seed(config.RANDOM_SEED)
        trainer = CVTrainer(features, num_folds=config.N_FOLDS, device=device, class_weight=True,
                            optimizer=hp["optimizer"], model_params={"n_filters": hp["n_filters"],
                                                                     "kernel_size": hp["kernel_size"],
                                                                     "dropout": hp["dropout"]})
        t0 = time.time()
        folds, _ = trainer.run(train_ids, test_ids, epochs=hp["epochs"], batch_size=hp["batch_size"],
                               learning_rate=hp["learning_rate"])
        p = pd.DataFrame(trainer.test_predictions)
        p.to_csv(OUT / f"band_{arm}_test_predictions.csv", index=False)
        res["arms"][arm] = {"fmax_hz": fmax, "train_minutes": round((time.time() - t0) / 60, 1),
                            "cv_macro_f1_beat_mean": float(np.mean([f["macro_f1"] for f in folds])),
                            "cv_auc_beat_mean": float(np.mean([f["auc"] for f in folds])),
                            "test_patient": patient_metrics(p)}
        print(arm, json.dumps(res["arms"][arm], indent=1))
        (OUT / "band_40hz.json").write_text(json.dumps(res, indent=1))

    print(json.dumps(res, indent=1))
    print("written", OUT / "band_40hz.json")


if __name__ == "__main__":
    main()
