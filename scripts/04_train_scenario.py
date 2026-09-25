"""Train a single scenario: patient-level 5-fold CV, then final fit evaluated on the test set.

Usage:
    python scripts/04_train_scenario.py --scenario psd --epochs 50 --batch-size 64

Output: results/scenarios/<scenario>_results.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config
from pvc_localization.data.dataset import create_train_val_test_split
from pvc_localization.training.trainer import CVTrainer

METRIC_KEYS = ["accuracy", "precision_rvot", "precision_lvot", "recall_rvot", "recall_lvot",
               "f1_rvot", "f1_lvot", "macro_f1", "balanced_accuracy", "auc"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--no-class-weight", action="store_true",
                        help="Train without class weights (baseline as defined in the proposal)")
    args = parser.parse_args()
    use_class_weight = not args.no_class_weight

    set_seed(config.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    feature_types = config.FEATURE_SCENARIOS[args.scenario]
    print(f"\nScenario: {args.scenario}")
    print(f"Features: {feature_types if feature_types else 'baseline (raw 12-lead beat)'}")
    print(f"Class weight: {'balanced' if use_class_weight else 'none'}")

    train_ids, test_ids = create_train_val_test_split()
    print(f"\nTrain patients: {len(train_ids)}")
    print(f"Test patients: {len(test_ids)}")

    trainer = CVTrainer(feature_types, num_folds=config.N_FOLDS, device=device, class_weight=use_class_weight)
    fold_results, test_metrics = trainer.run(
        train_ids, test_ids,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    cv_mean, cv_std = summarize(fold_results)

    print(f"\nCross-validation ({config.N_FOLDS}-fold per pasien, mean ± std):")
    for key in METRIC_KEYS:
        print(f"  {key:<18} {cv_mean[key]:.4f} ± {cv_std[key]:.4f}")

    print(f"\nTest set ({len(test_ids)} pasien):")
    for key in METRIC_KEYS:
        print(f"  {key:<18} {test_metrics[key]:.4f}")
    print(f"  confusion_matrix   {test_metrics['confusion_matrix']}  (baris: RVOT, LVOT asli)")

    result_dir = config.RESULTS_DIR / "scenarios"
    result_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if use_class_weight else "_nocw"
    result_file = result_dir / f"{args.scenario}{suffix}_results.json"
    with open(result_file, "w") as f:
        json.dump({
            "scenario": args.scenario,
            "features": feature_types,
            "config": {"epochs": args.epochs, "batch_size": args.batch_size,
                       "learning_rate": args.learning_rate, "n_folds": config.N_FOLDS,
                       "cv": "StratifiedGroupKFold per pasien", "class_weight": "balanced" if use_class_weight else "none",
                       "seed": config.RANDOM_SEED},
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_folds": fold_results,
            "test": test_metrics,
        }, f, indent=2)
    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    main()
