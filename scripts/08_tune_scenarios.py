"""Stage-wise hyperparameter tuning (Proposal Bab 3.1.11, Tabel 3.5).

One hyperparameter is tuned per stage while the others stay at their current best
value. Each candidate is scored by patient-level 5-fold CV on the training patients
only (the test set is never loaded). Selection: highest mean macro F1, ties by AUC.

Resumable: every evaluated config is stored in results/tuning/<scenario>_tuning.json
and is not re-run. Best config is written to results/tuning/<scenario>_best.json,
then train the final model with:  python scripts/04_train_scenario.py --scenario X --tuned

Usage:
    python scripts/08_tune_scenarios.py                       # all 7 feature scenarios
    python scripts/08_tune_scenarios.py --scenarios psd_wavelet psd
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config
from pvc_localization.data.dataset import create_train_val_test_split
from pvc_localization.training.trainer import CVTrainer, set_seed

START_CONFIG = {"learning_rate": 0.001, "optimizer": "adam", "batch_size": 64, "n_filters": 64,
                "kernel_size": 3, "dropout": 0.3, "epochs": 50}

# Tabel 3.5; epochs last because 100/150 epochs are the most expensive candidates.
STAGES = [
    ("learning_rate", [0.001, 0.0005, 0.0001]),
    ("optimizer", ["adam", "rmsprop"]),
    ("batch_size", [16, 32, 64]),
    ("n_filters", [32, 64, 128]),
    ("kernel_size", [3, 5]),
    ("dropout", [0.2, 0.3, 0.5]),
    ("epochs", [50, 100, 150]),
]

FEATURE_SCENARIOS = [s for s, feats in config.FEATURE_SCENARIOS.items() if feats]


def key_of(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


def evaluate(scenario: str, cfg: dict, train_ids: list[int]) -> dict:
    set_seed(config.RANDOM_SEED)
    trainer = CVTrainer(config.FEATURE_SCENARIOS[scenario], num_folds=config.N_FOLDS,
                        class_weight=True, optimizer=cfg["optimizer"],
                        model_params={"n_filters": cfg["n_filters"], "kernel_size": cfg["kernel_size"],
                                      "dropout": cfg["dropout"]})
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    folds, _ = trainer.run(train_ids, None, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                           learning_rate=cfg["learning_rate"])
    score = lambda k: [f[k] for f in folds if not np.isnan(f[k])]
    return {
        "config": cfg,
        "macro_f1": float(np.mean(score("macro_f1"))),
        "macro_f1_std": float(np.std(score("macro_f1"))),
        "auc": float(np.mean(score("auc"))),
        "balanced_accuracy": float(np.mean(score("balanced_accuracy"))),
        "recall_lvot": float(np.mean(score("recall_lvot"))),
        "time_min": round((time.time() - start) / 60, 2),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None,
    }


def tune(scenario: str, train_ids: list[int], out_dir: Path):
    log_file = out_dir / f"{scenario}_tuning.json"
    log = json.loads(log_file.read_text()) if log_file.exists() else {"runs": {}, "stages": []}
    best = dict(START_CONFIG)
    log["stages"] = []

    for stage_idx, (param, values) in enumerate(STAGES, 1):
        candidates = []
        for value in values:
            cfg = {**best, param: value}
            k = key_of(cfg)
            if k not in log["runs"]:
                print(f"\n[{scenario}] Tahap {stage_idx}/{len(STAGES)} {param}={value}  config={cfg}")
                log["runs"][k] = evaluate(scenario, cfg, train_ids)
                log_file.write_text(json.dumps(log, indent=2))
            r = log["runs"][k]
            candidates.append((value, r))
            print(f"[{scenario}] {param}={value}: macro_f1={r['macro_f1']:.4f} ± {r['macro_f1_std']:.4f}, "
                  f"auc={r['auc']:.4f}")

        chosen_value, chosen = max(candidates, key=lambda c: (c[1]["macro_f1"], c[1]["auc"]))
        best[param] = chosen_value
        log["stages"].append({"param": param, "chosen": chosen_value,
                              "candidates": {str(v): r["macro_f1"] for v, r in candidates}})
        print(f"[{scenario}] >> Tahap {stage_idx} dipilih {param}={chosen_value} (macro_f1={chosen['macro_f1']:.4f})")
        log_file.write_text(json.dumps(log, indent=2))

    best_run = log["runs"][key_of(best)]
    (out_dir / f"{scenario}_best.json").write_text(json.dumps(best_run, indent=2))
    print(f"\n[{scenario}] KONFIGURASI TERBAIK: {best}  macro_f1 CV={best_run['macro_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=FEATURE_SCENARIOS, default=FEATURE_SCENARIOS)
    args = parser.parse_args()

    out_dir = config.RESULTS_DIR / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids, _ = create_train_val_test_split()

    for scenario in args.scenarios:
        tune(scenario, train_ids, out_dir)


if __name__ == "__main__":
    main()
