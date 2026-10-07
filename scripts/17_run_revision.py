"""Run every experiment requested by the journal reviewers, in order (resumable).

  1. Tune the raw-signal baseline with the same stage-wise budget       (reviewer comment 6)
  2. Train and test the tuned baseline                                  (reviewer comment 6)
  3. End-to-end latency and software versions                           (reviewer comment 8, minor 8)
  4. Beats per patient and PVC-extraction statistics                    (reviewer comment 13, minor 12)
  5. Class-specific SHAP and stability across seeds                     (reviewer comment 10, minor 7)

Steps whose output already exists are skipped, so the script can simply be run again after an
interruption. Run on the PC (needs the data, the feature cache and results/models).

Usage:
    python scripts/17_run_revision.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

STEPS = [
    ("tune raw-signal baseline", ["08_tune_scenarios.py", "--scenarios", "baseline"], RES / "tuning" / "baseline_best.json"),
    ("train + test tuned baseline", ["04_train_scenario.py", "--scenario", "baseline", "--tuned"],
     RES / "scenarios" / "baseline_tuned_results.json"),
    ("end-to-end latency", ["14_latency.py"], RES / "latency.json"),
    ("dataset statistics", ["16_dataset_stats.py"], RES / "dataset_stats.json"),
    ("class-specific SHAP + seed stability", ["15_xai_class_seeds.py"], RES / "xai" / "stability_psd_wavelet_hos.json"),
]


def main():
    t_all = time.time()
    for i, (name, cmd, done_file) in enumerate(STEPS, 1):
        if done_file.exists():
            print(f"[{i}/{len(STEPS)}] {name}: sudah ada ({done_file.relative_to(ROOT)}), skip")
            continue
        print(f"\n{'=' * 70}\n[{i}/{len(STEPS)}] {name}\n{'=' * 70}")
        t0 = time.time()
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / cmd[0]), *cmd[1:]], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"\nGAGAL di langkah {i} ({name}). Perbaiki errornya lalu jalankan lagi script ini.")
        print(f"[{i}/{len(STEPS)}] {name}: selesai ({(time.time() - t0) / 60:.1f} menit)")
    print(f"\nSemua langkah selesai ({(time.time() - t_all) / 60:.1f} menit). Hasil yang perlu di-push:")
    for p in ["results/tuning/baseline_tuning.json", "results/tuning/baseline_best.json",
              "results/scenarios/baseline_tuned_results.json", "results/predictions/baseline_tuned_test_predictions.csv",
              "results/latency.json", "results/dataset_stats.json", "results/dataset_stats_per_patient.csv",
              "results/xai/ (file *_class_* dan stability_*)"]:
        print("  -", p)


if __name__ == "__main__":
    main()
