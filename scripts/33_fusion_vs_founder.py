"""Does the fusion add anything over ECGFounder alone? (reviewer point A3; no retraining)

Uses the outer-test scores and inner out-of-fold scores of the three nested-CV splits (scripts/24) and the referral
rule of scripts/26 applied to each model on its own: per outer fold the band delta is the (1 - c) quantile of
|score - threshold| over that model's inner out-of-fold scores. Both models therefore abstain by the same rule at the
same target coverage c.

Reported per split and target coverage (100, 95, 90, 85, 80 %):
  coverage, classified patients, errors, macro F1, balanced accuracy, sensitivity, specificity of each model;
  paired patient bootstrap (2000 resamples) of the difference fusion - ECGFounder in macro F1, balanced accuracy and
  number of errors per 100 patients, with 95 % CI (unadjusted).
Discordance at full coverage (per split): patients correct by one model and wrong by the other (CNN vs ECGFounder,
fusion vs ECGFounder), exact McNemar test, split by class (LVOT/RVOT) and by ECGFounder certainty (the third of
patients with the smallest |ECGFounder score - threshold| = uncertain).

Usage:  python scripts/33_fusion_vs_founder.py
Output: results/nested_cv/fusion_vs_founder.csv, fusion_vs_founder_discordance.csv, fusion_vs_founder.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

OUT = Path(__file__).resolve().parents[1] / "results" / "nested_cv"
RUNS = {"42": "fusion_psd12_aug_wd0.0001_ens5", "7": "fusion_psd12_aug_wd0.0001_ens5_split7",
        "123": "fusion_psd12_aug_wd0.0001_ens5_split123"}
COVERAGES = [1.0, 0.95, 0.90, 0.85, 0.80]
N_BOOT = 2000
SEED = 42


def classified(test, inner, model, coverage):
    """Boolean mask of the test patients the model classifies at the target coverage (scripts/26 rule)."""
    keep = np.zeros(len(test), bool)
    for fo in test.outer_fold.unique():
        idx = np.where(test.outer_fold.to_numpy() == fo)[0]
        thr = float(test[f"{model}_threshold"].iloc[idx[0]])
        dist_in = np.abs(inner.loc[inner.outer_fold == fo, f"{model}_score"].to_numpy() - thr)
        delta = 0.0 if coverage >= 1 else float(np.quantile(dist_in, 1 - coverage))
        keep[idx] = np.abs(test[f"{model}_score"].to_numpy()[idx] - thr) >= delta
    return keep


def metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max((y == 1).sum(), 1); spec = tn / max((y == 0).sum(), 1)
    return {"macro_f1": f1_score(y, pred, average="macro", zero_division=0), "balanced_accuracy": (sens + spec) / 2,
            "sensitivity": sens, "specificity": spec, "errors": int((pred != y).sum())}


def main():
    rows, disc_rows, boot_rows = [], [], []
    rng = np.random.default_rng(SEED)
    for split, run in RUNS.items():
        test = pd.read_csv(OUT / f"{run}_patients.csv")
        inner = pd.read_csv(OUT / f"{run}_inner_oof.csv")
        y = test.label.to_numpy()
        pred = {m: (test[f"{m}_score"] > test[f"{m}_threshold"]).to_numpy().astype(int) for m in ("fusion", "founder", "cnn")}
        n = len(test)
        for cov in COVERAGES:
            keep = {m: classified(test, inner, m, cov) for m in ("fusion", "founder")}
            for m in ("fusion", "founder"):
                k = keep[m]
                rows.append({"split": split, "target_coverage": cov, "model": m, "coverage": k.mean(),
                             "classified": int(k.sum()), **metrics(y[k], pred[m][k])})
            # paired bootstrap over patients; each resample keeps each model's own classified subset
            d = {"macro_f1": [], "balanced_accuracy": [], "errors_per_100": []}
            for _ in range(N_BOOT):
                i = rng.integers(0, n, n)
                mf, mo = keep["fusion"][i], keep["founder"][i]
                yf, yo = y[i][mf], y[i][mo]
                if len(set(yf)) < 2 or len(set(yo)) < 2:
                    continue
                a, b = metrics(yf, pred["fusion"][i][mf]), metrics(yo, pred["founder"][i][mo])
                d["macro_f1"].append(a["macro_f1"] - b["macro_f1"])
                d["balanced_accuracy"].append(a["balanced_accuracy"] - b["balanced_accuracy"])
                d["errors_per_100"].append(100 * (a["errors"] - b["errors"]) / n)
            a = metrics(y[keep["fusion"]], pred["fusion"][keep["fusion"]])
            b = metrics(y[keep["founder"]], pred["founder"][keep["founder"]])
            point = {"macro_f1": a["macro_f1"] - b["macro_f1"],
                     "balanced_accuracy": a["balanced_accuracy"] - b["balanced_accuracy"],
                     "errors_per_100": 100 * (a["errors"] - b["errors"]) / n}
            for k_, v in d.items():
                boot_rows.append({"split": split, "target_coverage": cov, "metric": k_, "difference": point[k_],
                                  "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5))})
        # discordance at full coverage
        thr_f = test.founder_threshold.to_numpy()
        certainty = np.abs(test.founder_score.to_numpy() - thr_f)
        uncertain = certainty <= np.quantile(certainty, 1 / 3)
        groups = {"all": np.ones(n, bool), "LVOT": y == 1, "RVOT": y == 0,
                  "ECGFounder uncertain (lowest third)": uncertain, "ECGFounder certain": ~uncertain}
        for other in ("cnn", "fusion"):
            for g, mask in groups.items():
                ok_o = pred[other][mask] == y[mask]; ok_f = pred["founder"][mask] == y[mask]
                only_o = int((ok_o & ~ok_f).sum()); only_f = int((~ok_o & ok_f).sum())
                p = binomtest(only_o, only_o + only_f, 0.5).pvalue if only_o + only_f else 1.0
                disc_rows.append({"split": split, "comparison": f"{other} vs founder", "group": g,
                                  "patients": int(mask.sum()), f"correct_only_{other}": only_o,
                                  "correct_only_founder": only_f, "both_wrong": int((~ok_o & ~ok_f).sum()),
                                  "mcnemar_p": float(p)})
    res = pd.DataFrame(rows); boot = pd.DataFrame(boot_rows); disc = pd.DataFrame(disc_rows)
    res.to_csv(OUT / "fusion_vs_founder.csv", index=False)
    disc.to_csv(OUT / "fusion_vs_founder_discordance.csv", index=False)
    boot.to_csv(OUT / "fusion_vs_founder_bootstrap.csv", index=False)
    summary = res.groupby(["target_coverage", "model"]).agg(
        coverage=("coverage", "mean"), errors=("errors", "mean"), macro_f1=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"), balanced_accuracy=("balanced_accuracy", "mean"),
        sensitivity=("sensitivity", "mean"), specificity=("specificity", "mean")).reset_index()
    (OUT / "fusion_vs_founder.json").write_text(json.dumps(
        {"mean_over_splits": summary.to_dict(orient="records"), "bootstrap": boot.to_dict(orient="records")}, indent=1))

    print("mean over 3 splits")
    for cov in COVERAGES:
        print(f" target {cov:.0%}")
        for _, r in summary[summary.target_coverage == cov].iterrows():
            print(f"   {r.model:8s} coverage {r.coverage:.1%} errors {r.errors:5.1f}  macro F1 {r.macro_f1:.3f} "
                  f"(SD {r.macro_f1_sd:.3f})  bal.acc {r.balanced_accuracy:.3f}  sens {r.sensitivity:.3f} "
                  f"spec {r.specificity:.3f}")
        for _, b in boot[(boot.target_coverage == cov)].iterrows():
            print(f"     split {b.split:>3s} {b.metric:17s} {b.difference:+.3f} [{b.ci_low:+.3f}, {b.ci_high:+.3f}]")
    print("\ndiscordance at full coverage (summed over splits)")
    agg = disc.groupby(["comparison", "group"]).sum(numeric_only=True).reset_index()
    for _, r in agg.iterrows():
        o = r.get("correct_only_cnn", 0) if r.comparison.startswith("cnn") else r.get("correct_only_fusion", 0)
        print(f"  {r.comparison:18s} {r.group:36s} only other {int(o):3d}  only founder {int(r.correct_only_founder):3d}"
              f"  both wrong {int(r.both_wrong):3d}")
    print("written", OUT)


if __name__ == "__main__":
    main()
