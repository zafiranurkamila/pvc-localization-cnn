"""Re-export test-set predictions from the saved final models at full precision.

Loads results/models/<name>.pt, predicts the test patients again (no training) and
writes results/predictions/<name>_test_predictions.csv with the full-precision
probability and the logit margin (logit_LVOT - logit_RVOT), which keeps the ranking
even when probabilities saturate at 0 or 1. Checks that macro F1 matches the
results JSON.

Usage:
    python scripts/10_export_predictions.py
"""
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config
from pvc_localization.data.dataset import PVCBeatsDataset, create_train_val_test_split
from pvc_localization.evaluation.metrics import compute_metrics
from pvc_localization.models.fusion import BaselineCNN, FusionCNN
from pvc_localization.training.trainer import CACHE_DIR

NAMES = ["baseline_nocw", "psd_tuned", "wavelet_tuned", "hos_tuned", "psd_wavelet_tuned",
         "psd_hos_tuned", "wavelet_hos_tuned", "psd_wavelet_hos_tuned"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_ids = create_train_val_test_split()
    model_dir = config.RESULTS_DIR / "models"
    pred_dir = config.RESULTS_DIR / "predictions"
    res_dir = config.RESULTS_DIR / "scenarios"

    for name in NAMES:
        ckpt = torch.load(model_dir / f"{name}.pt", map_location=device)
        features, hp = ckpt["features"], ckpt["hyperparameters"]
        if features:
            model = FusionCNN(features, n_filters=hp["n_filters"], kernel_size=hp["kernel_size"],
                              dropout=hp["dropout"])
        else:
            model = BaselineCNN()
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()

        test_set = PVCBeatsDataset(test_ids, features, cache_dir=CACHE_DIR)
        loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
        rows = []
        with torch.no_grad():
            for batch in loader:
                b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                logits = model(b["raw"]) if not features else model(b)
                probs = torch.softmax(logits.double(), dim=1)
                margin = (logits[:, 1] - logits[:, 0]).double()
                for i in range(len(b["label"])):
                    rows.append((int(batch["hospital_id"][i]), int(batch["beat_idx"][i]), int(batch["label"][i]),
                                 repr(float(probs[i, 1])), repr(float(margin[i])), int(probs[i].argmax())))

        with open(pred_dir / f"{name}_test_predictions.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hospital_id", "beat_idx", "label", "prob_lvot", "logit_margin", "pred"])
            writer.writerows(rows)

        m = compute_metrics([r[2] for r in rows], [r[5] for r in rows], [float(r[4]) for r in rows])
        saved = json.loads((res_dir / f"{name}_results.json").read_text())["test"]
        ok = abs(m["macro_f1"] - saved["macro_f1"]) < 1e-9
        print(f"{name:<24} macro_f1 {m['macro_f1']:.4f} (json {saved['macro_f1']:.4f}) {'OK' if ok else 'BEDA!'} | "
              f"AUC {m['auc']:.4f} (json {saved['auc']:.4f})")


if __name__ == "__main__":
    main()
