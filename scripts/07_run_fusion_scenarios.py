"""Check all 8 scenarios, then run only the ones without results.

Usage:
    python scripts/07_run_fusion_scenarios.py --epochs 50 --batch-size 64
    python scripts/07_run_fusion_scenarios.py --scenarios psd_hos wavelet_hos
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_SCENARIOS = ["baseline", "psd", "wavelet", "hos",
                     "psd_wavelet", "psd_hos", "wavelet_hos", "psd_wavelet_hos"]


def run_script(script_path: Path, args: list, cwd: Path) -> bool:
    cmd = [sys.executable, str(script_path)] + args
    print(f"\n{'='*70}\nRunning: {' '.join(cmd)}\n{'='*70}\n")
    return subprocess.run(cmd, cwd=str(cwd)).returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run remaining scenarios")
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results" / "scenarios"
    results_dir.mkdir(parents=True, exist_ok=True)
    train_script = project_root / "scripts" / "04_train_scenario.py"

    print(f"\n{'='*70}\nCEK STATUS SEMUA SKENARIO\n{'='*70}")
    todo = []
    for scenario in args.scenarios:
        done = (results_dir / f"{scenario}_cv_results.json").exists()
        print(f"{scenario:<20} {'SUDAH' if done else 'BELUM'}")
        if not done or args.force:
            todo.append(scenario)

    if not todo:
        print("\nSemua skenario sudah selesai.")
        return 0
    print(f"\nAkan dijalankan: {', '.join(todo)}")

    all_results = {}
    for scenario in todo:
        print(f"\n{'─'*70}\nScenario: {scenario.upper()}\n{'─'*70}")
        ok = run_script(train_script, [
            "--scenario", scenario,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
        ], project_root)
        all_results[scenario] = "completed" if ok else "failed"
        print(f"{'OK' if ok else 'GAGAL'}: {scenario}")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for scenario, status in all_results.items():
        print(f"{scenario:<20} {status}")

    with open(results_dir / "remaining_training_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return 0 if "failed" not in all_results.values() else 1


if __name__ == "__main__":
    sys.exit(main())