"""Master script: Run all 8 scenarios end-to-end (resumable).

Flow:
  0. Check which scenarios already have results
  1. Build patient index (script 01)
  2. Segment beats (script 03) unless --skip-beats
  3. Train only unfinished scenarios (script 04)

Usage:
    python scripts/06_run_all_scenarios.py --epochs 50 --batch-size 64 --skip-beats
    python scripts/06_run_all_scenarios.py --epochs 50 --batch-size 64 --force   # ulang semua
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config


def run_script(script_path: str, args: list) -> bool:
    """Run a script and return success status."""
    cmd = [sys.executable, script_path] + args
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Master orchestration for all training scenarios")
    parser.add_argument("--sample-only", action="store_true",
                        help="Run on 10 sample patients (5 RVOT + 5 LVOT) for testing")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per scenario")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--skip-beats", action="store_true",
                        help="Skip beat segmentation (assume data/interim/beats/ exists)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all scenarios even if results exist")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results" / "scenarios"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0: Check status of all scenarios
    scenarios = list(config.FEATURE_SCENARIOS.keys())
    print("\n" + "="*70)
    print("CEK STATUS SEMUA SKENARIO")
    print("="*70)
    todo = []
    for scenario in scenarios:
        done = (results_dir / f"{scenario}_cv_results.json").exists()
        print(f"{scenario:<20} {'SUDAH' if done else 'BELUM'}")
        if not done or args.force:
            todo.append(scenario)

    if not todo:
        print("\nSemua skenario sudah selesai. Tidak ada yang dijalankan.")
        return 0
    print(f"\nAkan dijalankan ({len(todo)}): {', '.join(todo)}")

    # Phase 1: Build patient index
    print("\n" + "="*70)
    print("PHASE 1: Building patient index...")
    print("="*70)
    if not run_script(str(project_root / "scripts" / "01_build_label_index.py"), []):
        print("ERROR: Failed to build patient index")
        return 1

    # Phase 2: Segment beats
    if not args.skip_beats:
        print("\n" + "="*70)
        print("PHASE 2: Segmenting PVC beats...")
        print("="*70)
        seg_args = ["--sample-only"] if args.sample_only else []
        if not run_script(str(project_root / "scripts" / "03_full_segmentation.py"), seg_args):
            print("ERROR: Failed to segment beats")
            return 1

    # Phase 8: Train unfinished scenarios
    print("\n" + "="*70)
    print(f"PHASE 8: Training {len(todo)} scenarios...")
    print("="*70)

    all_results = {s: "skipped (sudah ada)" for s in scenarios if s not in todo}
    durations = {}

    total_start = time.time()
    for i, scenario in enumerate(todo, 1):
        start = time.time()
        print(f"\n{'─'*70}")
        print(f"Scenario {i}/{len(todo)}: {scenario.upper()}")
        print(f"{'─'*70}")

        train_args = [
            "--scenario", scenario,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate)
        ]

        success = run_script(str(project_root / "scripts" / "04_train_scenario.py"), train_args)
        all_results[scenario] = "completed" if success else "failed"

        menit = (time.time() - start) / 60
        total = (time.time() - total_start) / 60
        durations[scenario] = round(menit, 1)
        print(f"{'OK' if success else 'GAGAL'}: {scenario} ({menit:.1f} menit, total {total:.1f} menit)")

    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for scenario in scenarios:
        waktu = f"  ({durations[scenario]} menit)" if scenario in durations else ""
        print(f"{scenario:<20} {all_results[scenario]}{waktu}")

    failed = [s for s, st in all_results.items() if st == "failed"]
    total_menit = (time.time() - total_start) / 60
    print(f"\nGagal: {len(failed)} | Selesai/skip: {len(scenarios) - len(failed)}/{len(scenarios)}")
    print(f"Total waktu training: {total_menit:.1f} menit ({total_menit / 60:.1f} jam)")

    summary_path = results_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"status": all_results, "durasi_menit": durations}, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())