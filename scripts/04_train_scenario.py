"""Train a single scenario: patient-level 5-fold CV, then final fit evaluated on the test set.

Usage:
    python scripts/04_train_scenario.py --scenario psd --epochs 50 --batch-size 64
    python scripts/04_train_scenario.py --scenario baseline --epochs 50 --batch-size 64 --no-class-weight
    python scripts/04_train_scenario.py --scenario psd_wavelet --tuned

Output: results/scenarios/<scenario>[_nocw|_tuned]_results.json
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config
from pvc_localization.data.dataset import create_train_val_test_split
from pvc_localization.training.trainer import CVTrainer, OPTIMIZERS, set_seed

METRIC_KEYS = ["accuracy", "precision_rvot", "precision_lvot", "recall_rvot", "recall_lvot",
               "f1_rvot", "f1_lvot", "macro_f1", "balanced_accuracy", "auc"]


def summarize(fold_results: list[dict]) -> tuple[dict, dict]:
    mean, std = {}, {}
    for key in METRIC_KEYS:
        values = [r[key] for r in fold_results if not np.isnan(r[key])]
        mean[key] = float(np.mean(values)) if values else float("nan")
        std[key] = float(np.std(values)) if values else float("nan")
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, choices=list(config.FEATURE_SCENARIOS.keys()), required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=list(OPTIMIZERS), default="adam")
    parser.add_argument("--n-filters", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=None, help="default: 5 for 1D branches, 3 for 2D")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-class-weight", action="store_true",
                        help="Train without class weights (baseline as defined in the proposal)")
    parser.add_argument("--tuned", action="store_true",
                        help="Use results/tuning/<scenario>_best.json from scripts/08_tune_scenarios.py")
    args = parser.parse_args()

    hp = {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
          "optimizer": args.optimizer, "n_filters": args.n_filters, "kernel_size": args.kernel_size,
          "dropout": args.dropout}
    if args.tuned:
        best_file = config.RESULTS_DIR / "tuning" / f"{args.scenario}_best.json"
        hp.update(json.loads(best_file.read_text())["config"])
    use_class_weight = not args.no_class_weight

    set_seed(config.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    feature_types = config.FEATURE_SCENARIOS[args.scenario]
    print(f"\nScenario: {args.scenario}")
    print(f"Features: {feature_types if feature_types else 'baseline (raw 12-lead beat)'}")
    print(f"Class weight: {'balanced' if use_class_weight else 'none'}")
    print(f"Hyperparameters: {hp}")

    train_ids, test_ids = create_train_val_test_split()
    print(f"\nTrain patients: {len(train_ids)}")
    print(f"Test patients: {len(test_ids)}")

    trainer = CVTrainer(feature_types, num_folds=config.N_FOLDS, device=device,
                        class_weight=use_class_weight, optimizer=hp["optimizer"],
                        model_params={"n_filters": hp["n_filters"], "kernel_size": hp["kernel_size"],
                                      "dropout": hp["dropout"]})
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    fold_results, test_metrics = trainer.run(
        train_ids, test_ids,
        epochs=hp["epochs"],
        batch_size=hp["batch_size"],
        learning_rate=hp["learning_rate"],
    )
    efficiency = {
        "train_time_min": round((time.time() - start) / 60, 2),
        "peak_gpu_memory_mb": trainer.train_peak_memory_mb,
        "inference": trainer.inference,
    }
    cv_mean, cv_std = summarize(fold_results)

    print(f"\nCross-validation ({config.N_FOLDS}-fold per pasien, mean ± std):")
    for key in METRIC_KEYS:
        print(f"  {key:<18} {cv_mean[key]:.4f} ± {cv_std[key]:.4f}")

    print(f"\nTest set ({len(test_ids)} pasien):")
    for key in METRIC_KEYS:
        print(f"  {key:<18} {test_metrics[key]:.4f}")
    print(f"  confusion_matrix   {test_metrics['confusion_matrix']}  (baris: RVOT, LVOT asli)")
    inf = efficiency["inference"]
    print(f"\nWaktu training: {efficiency['train_time_min']} menit | "
          f"Memori GPU puncak: {efficiency['peak_gpu_memory_mb']} MB | "
          f"Waktu inferensi: {inf['per_beat_ms']} ms/beat ({inf['test_beats']} beat test)")

    result_dir = config.RESULTS_DIR / "scenarios"
    result_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("" if use_class_weight else "_nocw") + ("_tuned" if args.tuned else "")
    name = f"{args.scenario}{suffix}"
    result_file = result_dir / f"{name}_results.json"

    model_dir = config.RESULTS_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{name}.pt"
    torch.save({"state_dict": trainer.final_model.state_dict(), "scenario": args.scenario,
                "features": feature_types, "hyperparameters": hp,
                "class_weight": use_class_weight}, model_file)

    pred_dir = config.RESULTS_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = pred_dir / f"{name}_test_predictions.csv"
    p = trainer.test_predictions
    with open(pred_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hospital_id", "beat_idx", "label", "prob_lvot", "pred"])
        writer.writerows(zip(p["hospital_id"], p["beat_idx"], p["label"],
                             [round(x, 6) for x in p["prob_lvot"]], p["pred"]))
    print(f"Model saved to {model_file}")
    print(f"Test predictions saved to {pred_file}")
    with open(result_file, "w") as f:
        json.dump({
            "scenario": args.scenario,
            "features": feature_types,
            "config": {**hp, "n_folds": config.N_FOLDS, "cv": "StratifiedGroupKFold per pasien",
                       "class_weight": "balanced" if use_class_weight else "none",
                       "seed": config.RANDOM_SEED, "tuned": args.tuned},
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_folds": fold_results,
            "test": test_metrics,
            "efficiency": efficiency,
        }, f, indent=2)
    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    main()
