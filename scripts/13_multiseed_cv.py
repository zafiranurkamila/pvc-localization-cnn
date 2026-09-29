"""Multi-seed patient-level 5-fold CV with the fixed setting (training patients only).

Each seed changes both the fold split (StratifiedGroupKFold random_state) and the model
initialisation / batch order. The train/test split stays fixed (seed 42) and the test set is
never loaded, so the feature-combination decision is made from training data only.

Fixed setting (same as scripts/04_train_scenario.py defaults): Adam, lr 0.001, batch 64,
50 epochs, 64 filters, default kernel, dropout 0.3, class weight (baseline included).

Resumable: each (scenario, seed) is saved to results/multiseed/<scenario>_seed<seed>.json.

Usage:
    python scripts/13_multiseed_cv.py                      # 8 scenarios x seeds 1-5
    python scripts/13_multiseed_cv.py --seeds 1 2 3
    python scripts/13_multiseed_cv.py --summary-only       # only rebuild summary.json
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

HP = {"epochs": 50, "batch_size": 64, "learning_rate": 0.001, "optimizer": "adam",
      "n_filters": 64, "kernel_size": None, "dropout": 0.3}
SCENARIOS = list(config.FEATURE_SCENARIOS.keys())
OUT_DIR = config.RESULTS_DIR / "multiseed"


def run_one(scenario: str, seed: int, train_ids: list[int]) -> dict:
    set_seed(seed)
    trainer = CVTrainer(config.FEATURE_SCENARIOS[scenario], num_folds=config.N_FOLDS,
                        class_weight=True, optimizer=HP["optimizer"], cv_seed=seed,
                        model_params={"n_filters": HP["n_filters"], "kernel_size": HP["kernel_size"],
                                      "dropout": HP["dropout"]})
    start = time.time()
    folds, _ = trainer.run(train_ids, None, epochs=HP["epochs"], batch_size=HP["batch_size"],
                           learning_rate=HP["learning_rate"])
    f1 = [f["macro_f1"] for f in folds]
    auc = [f["auc"] for f in folds if not np.isnan(f["auc"])]
    return {"scenario": scenario, "seed": seed, "config": {**HP, "class_weight": "balanced"},
            "macro_f1_mean": float(np.mean(f1)), "macro_f1_std": float(np.std(f1)),
            "auc_mean": float(np.mean(auc)), "folds": folds,
            "time_min": round((time.time() - start) / 60, 2)}


def summarize(seeds: list[int]):
    runs = {s: {} for s in SCENARIOS}
    for s in SCENARIOS:
        for seed in seeds:
            f = OUT_DIR / f"{s}_seed{seed}.json"
            if f.exists():
                runs[s][seed] = json.loads(f.read_text())
    done_seeds = [seed for seed in seeds if all(seed in runs[s] for s in SCENARIOS)]
    if not done_seeds:
        print("Belum ada seed yang lengkap untuk semua skenario.")
        return

    # Rank per seed among the 7 feature scenarios (1 = highest mean CV macro F1).
    features = [s for s in SCENARIOS if config.FEATURE_SCENARIOS[s]]
    ranks = {s: [] for s in features}
    for seed in done_seeds:
        order = sorted(features, key=lambda s: -runs[s][seed]["macro_f1_mean"])
        for r, s in enumerate(order, 1):
            ranks[s].append(r)

    summary = {"seeds": done_seeds, "config": {**HP, "class_weight": "balanced"}, "scenarios": {}}
    for s in SCENARIOS:
        seed_means = [runs[s][seed]["macro_f1_mean"] for seed in done_seeds]
        all_folds = [f["macro_f1"] for seed in done_seeds for f in runs[s][seed]["folds"]]
        summary["scenarios"][s] = {
            "macro_f1_mean": float(np.mean(all_folds)),
            "macro_f1_std_folds": float(np.std(all_folds)),
            "macro_f1_std_seeds": float(np.std(seed_means)),
            "auc_mean": float(np.mean([runs[s][seed]["auc_mean"] for seed in done_seeds])),
            "per_seed": dict(zip(map(str, done_seeds), seed_means)),
            "rank_per_seed": ranks.get(s),
            "mean_rank": float(np.mean(ranks[s])) if s in ranks else None,
            "times_rank1": ranks[s].count(1) if s in ranks else None,
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nRINGKASAN MULTI-SEED CV (seed {done_seeds}, {len(done_seeds) * config.N_FOLDS} fold per skenario)")
    print(f"{'skenario':<17}{'macro F1':>9}{'sd fold':>9}{'sd seed':>9}{'AUC':>7}  {'rank per seed':<20}{'rata2 rank':>10}")
    for s in sorted(SCENARIOS, key=lambda s: -summary["scenarios"][s]["macro_f1_mean"]):
        r = summary["scenarios"][s]
        rank = " ".join(map(str, r["rank_per_seed"])) if r["rank_per_seed"] else "(acuan)"
        mr = f"{r['mean_rank']:.1f}" if r["mean_rank"] else "-"
        print(f"{s:<17}{r['macro_f1_mean']:>9.3f}{r['macro_f1_std_folds']:>9.3f}{r['macro_f1_std_seeds']:>9.3f}"
              f"{r['auc_mean']:>7.3f}  {rank:<20}{mr:>10}")
    print(f"\nSaved: {OUT_DIR / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        train_ids, _ = create_train_val_test_split()
        total_start = time.time()
        for seed in args.seeds:
            for scenario in args.scenarios:
                f = OUT_DIR / f"{scenario}_seed{seed}.json"
                if f.exists():
                    print(f"[seed {seed}] {scenario}: sudah ada, skip")
                    continue
                print(f"\n{'=' * 70}\n[seed {seed}] {scenario}\n{'=' * 70}")
                r = run_one(scenario, seed, train_ids)
                f.write_text(json.dumps(r, indent=2))
                print(f"[seed {seed}] {scenario}: macro_f1={r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} "
                      f"({r['time_min']} menit, total {(time.time() - total_start) / 60:.1f} menit)")

    summarize(args.seeds)


if __name__ == "__main__":
    main()
