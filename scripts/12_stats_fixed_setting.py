"""Bootstrap CI and paired significance tests for the FIXED-SETTING run (same ML config
for all 8 scenarios, only features varied — the experiment requested by the advisor).

Uses results/predictions/<scenario>_test_predictions.csv (plain scenario name, no _tuned
or _nocw suffix). Never touches the tuned run's files.

Usage:
    python scripts/12_stats_fixed_setting.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvc_localization import config

SCEN = [("baseline", "Baseline"), ("psd", "PSD"), ("wavelet", "Wavelet"), ("hos", "HOS"),
        ("psd_wavelet", "PSD+Wavelet"), ("psd_hos", "PSD+HOS"), ("wavelet_hos", "Wavelet+HOS"),
        ("psd_wavelet_hos", "PSD+Wavelet+HOS")]
PAIRS = [(s, "baseline") for s, _ in SCEN[1:]] + [
    ("psd_wavelet", "psd"), ("psd_wavelet", "wavelet"), ("psd_hos", "psd"), ("psd_hos", "hos"),
    ("wavelet_hos", "wavelet"), ("wavelet_hos", "hos"), ("psd_wavelet_hos", "psd_wavelet"),
    ("psd_wavelet_hos", "psd_hos"), ("psd_wavelet_hos", "wavelet_hos")]
B, SEED = 2000, 42


def main():
    pred_dir = config.RESULTS_DIR / "predictions"
    D = {}
    for s, _ in SCEN:
        f = pred_dir / f"{s}_test_predictions.csv"
        if not f.exists():
            sys.exit(f"Missing {f} -- run scripts/04_train_scenario.py --scenario {s} first (no --tuned).")
        D[s] = pd.read_csv(f).sort_values(["hospital_id", "beat_idx"]).reset_index(drop=True)

    ref = D["baseline"][["hospital_id", "beat_idx", "label"]]
    for s, _ in SCEN:
        assert (D[s][["hospital_id", "beat_idx", "label"]].values == ref.values).all(), s
    y = ref.label.values
    pids = ref.hospital_id.unique()
    idx = {p: np.where(ref.hospital_id.values == p)[0] for p in pids}
    rng = np.random.default_rng(SEED)
    samples = [np.concatenate([idx[p] for p in rng.choice(pids, len(pids))]) for _ in range(B)]

    f1 = lambda s, i: f1_score(y[i], D[s].pred.values[i], average="macro")
    auc = lambda s, i: roc_auc_score(y[i], D[s].prob_lvot.values[i])
    full = np.arange(len(y))
    boot = {s: (np.array([f1(s, i) for i in samples]), np.array([auc(s, i) for i in samples])) for s, _ in SCEN}

    print(f"Test set: {len(y)} beats, {len(pids)} patients\n")
    print(f"{'Scenario':<18}{'Test F1 [95% CI]':<28}{'Test AUC [95% CI]'}")
    for s, name in SCEN:
        fb, ab = boot[s]
        print(f"{name:<18}{f1(s, full):.3f} [{np.percentile(fb, 2.5):.3f}, {np.percentile(fb, 97.5):.3f}]"
              f"{'':<6}{auc(s, full):.3f} [{np.percentile(ab, 2.5):.3f}, {np.percentile(ab, 97.5):.3f}]")

    print("\nPaired differences (two-sided bootstrap p-value):")
    results = {"scenarios": {}, "pairs": []}
    for s, _ in SCEN:
        fb, ab = boot[s]
        results["scenarios"][s] = {"test_f1": f1(s, full), "test_f1_ci": [float(np.percentile(fb, 2.5)), float(np.percentile(fb, 97.5))],
                                   "test_auc": auc(s, full), "test_auc_ci": [float(np.percentile(ab, 2.5)), float(np.percentile(ab, 97.5))]}
    for a, b in PAIRS:
        row = {"a": a, "b": b}
        for k, j in [("f1", 0), ("auc", 1)]:
            d = boot[a][j] - boot[b][j]
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            row[f"d_{k}"], row[f"p_{k}"] = float(d.mean()), float(p)
        results["pairs"].append(row)
        sig = "*" if min(row["p_f1"], row["p_auc"]) < 0.05 else " "
        print(f"  {sig} {a:<16} vs {b:<10} F1 {row['d_f1']:+.3f} (p={row['p_f1']:.3f})  "
              f"AUC {row['d_auc']:+.3f} (p={row['p_auc']:.3f})")

    out = config.RESULTS_DIR / "stats_fixed_setting.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
