"""Secondary analysis: selective classification (the model may abstain and refer the patient to EP mapping).

No retraining: uses the outer-test scores and the inner out-of-fold scores saved by scripts/24_nested_cv_fusion.py.
Per outer fold, a patient is abstained on when |score - threshold| < delta, the score close to the decision threshold.
delta is chosen on the inner out-of-fold scores of the outer training patients only, as the quantile of
|score - threshold| that leaves the target coverage (fraction of patients classified); the outer test fold is not used.

Specified before looking at the results (committed with this script):
  primary target coverage   90 % (about one patient in ten referred)
  also reported             target coverage 100, 95, 90, 85, 80, 75, 70 % (risk-coverage curve)
Reported per coverage: realised test coverage, metrics on the classified patients (bootstrap 95 % CI per split),
mean +- SD over the splits, and the LVOT share among the abstained patients.

Usage (after the fusion runs):
    python scripts/26_selective_classification.py
Output: results/nested_cv/selective.csv, selective.json
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("nested21", Path(__file__).resolve().parent / "21_nested_cv_dual_rhythm.py")
n21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n21)

OUT = n21.OUT
RUNS = ["fusion_psd12_aug_wd0.0001_ens5", "fusion_psd12_aug_wd0.0001_ens5_split7",
        "fusion_psd12_aug_wd0.0001_ens5_split123"]
PRIMARY_COVERAGE = 0.90
COVERAGES = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
MODELS = ("fusion", "founder", "cnn", "fusion_equal")
KEYS = ("macro_f1", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "auc")


def select(test, inner, model, coverage):
    """Classified test patients of every outer fold; delta from the inner out-of-fold scores of that fold."""
    keep = []
    for fo, te in test.groupby("outer_fold"):
        thr = float(te[f"{model}_threshold"].iloc[0])
        dist_in = np.abs(inner.loc[inner.outer_fold == fo, f"{model}_score"].to_numpy() - thr)
        delta = 0.0 if coverage >= 1 else float(np.quantile(dist_in, 1 - coverage))
        keep.append(te[np.abs(te[f"{model}_score"].to_numpy() - thr) >= delta])
    return pd.concat(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS, help="fusion result names (one per split)")
    args = ap.parse_args()
    rows = []
    for r in args.runs:
        test = pd.read_csv(OUT / f"{r}_patients.csv"); inner = pd.read_csv(OUT / f"{r}_inner_oof.csv")
        for m in MODELS:
            for cov in COVERAGES:
                k = select(test, inner, m, cov)
                y, sc, th = k.label.to_numpy(), k[f"{m}_score"].to_numpy(), k[f"{m}_threshold"].to_numpy()
                met = n21.metrics(y, sc, th)
                ci = n21.bootstrap_ci(y, sc, th)
                ab = test[~test.hospital_id.isin(k.hospital_id)]
                rows.append({"run": r, "model": m, "target_coverage": cov, "coverage": len(k) / len(test),
                             "classified": len(k), "abstained": len(ab), "abstained_lvot": int(ab.label.sum()),
                             **{k_: met[k_] for k_ in KEYS},
                             **{f"{k_}_ci": ci[k_] for k_ in ("macro_f1", "balanced_accuracy")}})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "selective.csv", index=False)
    agg = res.groupby(["model", "target_coverage"]).agg(
        coverage_mean=("coverage", "mean"), abstained_mean=("abstained", "mean"),
        abstained_lvot_mean=("abstained_lvot", "mean"),
        **{f"{k}_mean": (k, "mean") for k in KEYS}, **{f"{k}_sd": (k, "std") for k in KEYS}).reset_index()
    (OUT / "selective.json").write_text(json.dumps(
        {"runs": args.runs, "primary_coverage": PRIMARY_COVERAGE, "mean_sd_over_runs": agg.to_dict(orient="records")},
        indent=1))
    print(f"mean +- SD over {len(args.runs)} splits (primary target coverage {PRIMARY_COVERAGE:.0%}):")
    for m in MODELS:
        print(f"  {m}")
        for _, a in agg[agg.model == m].sort_values("target_coverage", ascending=False).iterrows():
            mark = "  <- primary" if np.isclose(a.target_coverage, PRIMARY_COVERAGE) else ""
            print(f"    target {a.target_coverage:.0%}: test coverage {a.coverage_mean:.1%} (abstained "
                  f"{a.abstained_mean:.1f}, of which LVOT {a.abstained_lvot_mean:.1f})  macro F1 {a.macro_f1_mean:.3f} +- "
                  f"{a.macro_f1_sd:.3f}  bal. acc {a.balanced_accuracy_mean:.3f} +- {a.balanced_accuracy_sd:.3f}  "
                  f"sens {a.sensitivity_mean:.3f}  spec {a.specificity_mean:.3f}{mark}")
    print("written", OUT / "selective.csv")


if __name__ == "__main__":
    main()
