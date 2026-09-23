"""Train a single scenario with 5-fold CV and evaluate on test set.

Usage:
    python scripts/04_train_scenario.py --scenario psd --epochs 10 --batch-size 32
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config
from pvc_localization.data.dataset import create_train_val_test_split
from pvc_localization.training.trainer import CVTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, choices=list(config.FEATURE_SCENARIOS.keys()), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    feature_types = config.FEATURE_SCENARIOS[args.scenario]
    print(f"\nScenario: {args.scenario}")
    print(f"Features: {feature_types if feature_types else 'baseline (none)'}")

    train_ids, test_ids = create_train_val_test_split()
    print(f"\nTrain patients: {len(train_ids)}")
    print(f"Test patients: {len(test_ids)}")

    trainer = CVTrainer(feature_types, num_folds=config.N_FOLDS, device=device)
    fold_results = trainer.train(
        train_ids,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    avg_metrics = {}
    for key in fold_results[0].keys():
        if key != "confusion_matrix":
            values = [r[key] for r in fold_results if not isinstance(r[key], float) or not np.isnan(r[key])]
            if values:
                avg_metrics[key] = float(np.mean(values))

    print(f"\nCross-validation results ({config.N_FOLDS}-fold average):")
    for key, val in avg_metrics.items():
        print(f"  {key}: {val:.4f}")

    result_dir = config.RESULTS_DIR / "scenarios"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.scenario}_cv_results.json"
    with open(result_file, "w") as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    import numpy as np
    main()
