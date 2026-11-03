"""Fusion vs ECGFounder at equal REALIZED coverage (reviewer round 2, point 2; no retraining).

scripts/33 compares the two models at equal TARGET coverage: each model's referral band comes from its own inner
out-of-fold scores, so the share of test patients actually classified (realized coverage) can differ between them.
Here both models classify exactly the same number of test patients: per split and coverage c, each model keeps the
round(c * n) test patients with the largest |score - threshold| (its own outer-fold threshold) and refers the rest.
This uses the test scores but no test label, so it is a comparison device, not a deployable rule.

Per split and coverage (95, 90, 85, 80 %): macro F1, balanced accuracy, sensitivity, specificity and errors of each
model; paired patient bootstrap (2000 resamples; within every resample both models keep the same number of patients)
of the difference fusion - ECGFounder with 95 % CI (unadjusted).

Usage:  python scripts/37_fusion_vs_founder_realized.py
Output: results/nested_cv/fusion_vs_founder_realized.csv, fusion_vs_founder_realized_bootstrap.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

OUT = Path(__file__).resolve().parents[1] / "results" / "nested_cv"
RUNS = {"42": "fusion_psd12_aug_wd0.0001_ens5", "7": "fusion_psd12_aug_wd0.0001_ens5_split7",
        "123": "fusion_psd12_aug_wd0.0001_ens5_split123"}
COVERAGES = [0.95, 0.90, 0.85, 0.80]
N_BOOT = 2000
SEED = 42


def metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max((y == 1).sum(), 1); spec = tn / max((y == 0).sum(), 1)
    return {"macro_f1": f1_score(y, pred, average="macro", zero_division=0), "balanced_accuracy": (sens + spec) / 2,
            "sensitivity": sens, "specificity": spec, "errors": int((pred != y).sum())}


def keep_top(dist, k):
    """Mask of the k patients farthest from the threshold (ties broken by position)."""
    m = np.zeros(len(dist), bool)
    m[np.argsort(-dist, kind="stable")[:k]] = True
    return m


def main():
    rows, boot = [], []
    rng = np.random.default_rng(SEED)
    for split, run in RUNS.items():
        t = pd.read_csv(OUT / f"{run}_patients.csv")
        y = t.label.to_numpy(); n = len(t)
        pred = {m: (t[f"{m}_score"] > t[f"{m}_threshold"]).to_numpy().astype(int) for m in ("fusion", "founder")}
        dist = {m: np.abs(t[f"{m}_score"] - t[f"{m}_threshold"]).to_numpy() for m in ("fusion", "founder")}
        for c in COVERAGES:
            k = int(round(c * n))
            keep = {m: keep_top(dist[m], k) for m in pred}
            res = {m: metrics(y[keep[m]], pred[m][keep[m]]) for m in pred}
            for m in pred:
                rows.append({"split": split, "coverage": c, "classified": k, "model": m, **res[m]})
            d = {"macro_f1": [], "balanced_accuracy": [], "errors_per_100": []}
            for _ in range(N_BOOT):
                i = rng.integers(0, n, n)
                a = {}
                for m in pred:
                    km = keep_top(dist[m][i], k)
                    yy = y[i][km]
                    if len(set(yy)) < 2:
                        break
                    a[m] = metrics(yy, pred[m][i][km])
                if len(a) < 2:
                    continue
                d["macro_f1"].append(a["fusion"]["macro_f1"] - a["founder"]["macro_f1"])
                d["balanced_accuracy"].append(a["fusion"]["balanced_accuracy"] - a["founder"]["balanced_accuracy"])
                d["errors_per_100"].append(100 * (a["fusion"]["errors"] - a["founder"]["errors"]) / n)
            point = {"macro_f1": res["fusion"]["macro_f1"] - res["founder"]["macro_f1"],
                     "balanced_accuracy": res["fusion"]["balanced_accuracy"] - res["founder"]["balanced_accuracy"],
                     "errors_per_100": 100 * (res["fusion"]["errors"] - res["founder"]["errors"]) / n}
            for key, v in d.items():
                boot.append({"split": split, "coverage": c, "metric": key, "difference": point[key],
                             "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5))})
    res = pd.DataFrame(rows); b = pd.DataFrame(boot)
    res.to_csv(OUT / "fusion_vs_founder_realized.csv", index=False)
    b.to_csv(OUT / "fusion_vs_founder_realized_bootstrap.csv", index=False)
    mean = res.groupby(["coverage", "model"])[["macro_f1", "balanced_accuracy", "sensitivity", "specificity",
                                                "errors"]].mean().reset_index()
    pd.set_option("display.width", 200)
    print("mean over the three splits (equal realized coverage)"); print(mean.round(3).to_string(index=False))
    print(b.round(3).to_string(index=False))
    excl = b[(b.ci_low > 0) | (b.ci_high < 0)]
    print(f"\nintervals excluding zero: {len(excl)} of {len(b)}"); print(excl.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
