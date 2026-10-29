"""Statistics for the J Med Syst manuscript, from the saved predictions (no network training).

1. Clinical ECG criteria on the same three splits: L2 logistic regression on the clinical features of
   scripts/22_nested_cv_clinical_stack.py (results/nested_cv/clinical_features.csv), C and threshold chosen on the
   inner folds of every outer fold exactly as in script 22, with the outer and inner folds of scripts/24.
2. Paired patient-level bootstrap (2000 resamples) of the differences in macro F1, balanced accuracy and AUC between
   fusion, ECGFounder, the PSD + CWT CNN and the clinical model, per split; Holm-adjusted within each split.
3. Selective classification (scripts/26, primary target coverage 90 %) per split, with bootstrap 95 % CIs.
4. Profile of the referred (abstained) patients at the primary coverage: class, sub-location, and how many of them
   the model would have misclassified.

Usage (after scripts 22 --features-only, 24 for the three splits, and 26):
    python scripts/27_paper_statistics.py
Output: results/nested_cv/paper_stats.json, paper_pairwise.csv, paper_clinical_patients.csv, paper_referred.csv
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n21 = _load("nested21", "21_nested_cv_dual_rhythm.py")  # also puts src/ on sys.path
n22 = _load("nested22", "22_nested_cv_clinical_stack.py")
n26 = _load("selective26", "26_selective_classification.py")

from pvc_localization import config  # noqa: E402
from pvc_localization.data.loader import load_patient_index  # noqa: E402

OUT = n21.OUT
SEED = n21.SEED
RUNS = {42: "fusion_psd12_aug_wd0.0001_ens5", 7: "fusion_psd12_aug_wd0.0001_ens5_split7",
        123: "fusion_psd12_aug_wd0.0001_ens5_split123"}
MODELS = ("fusion", "founder", "cnn", "clinical")
PAIRS = [("fusion", "founder"), ("fusion", "cnn"), ("fusion", "clinical"), ("founder", "cnn"),
         ("founder", "clinical"), ("cnn", "clinical")]
METRICS = ("macro_f1", "balanced_accuracy", "auc")
N_BOOT = 2000


def clinical_split(patients, clinical, split_seed, fold_of):
    """Outer-test decision values and thresholds of the clinical logistic model for one split."""
    all_p = patients.index.to_numpy(); all_y = patients.to_numpy()
    cols = list(clinical.columns)
    rows = []
    for fo, (tr_p, te_p) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
                                      .split(all_p, all_y)):
        tr_pat, te_pat = all_p[tr_p], all_p[te_p]
        assert all(fold_of[h] == fo for h in te_pat), "outer folds differ from the fusion run"
        inner = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed + fo)
                     .split(tr_pat, patients[tr_pat].to_numpy()))
        model, C, auc, thr = n22.fit_logistic(clinical.reindex(tr_pat)[cols].to_numpy(),
                                              patients[tr_pat].to_numpy(), inner)
        sc = model.decision_function(clinical.reindex(te_pat)[cols].to_numpy())
        rows += [{"hospital_id": h, "clinical_score": float(s), "clinical_threshold": thr, "clinical_C": C}
                 for h, s in zip(te_pat, sc)]
    return pd.DataFrame(rows)


def paired_bootstrap(y, s, t, a, b, rng):
    """Observed difference a - b, 95 % CI and two-sided p for every metric (patients resampled)."""
    obs = {k: n21.metrics(y, s[a], t[a])[k] - n21.metrics(y, s[b], t[b])[k] for k in METRICS}
    boots = {k: [] for k in METRICS}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        ma, mb = n21.metrics(y[i], s[a][i], t[a][i]), n21.metrics(y[i], s[b][i], t[b][i])
        for k in METRICS:
            boots[k].append(ma[k] - mb[k])
    out = {}
    for k in METRICS:
        d = np.array(boots[k])
        p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
        out[k] = {"diff": obs[k], "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p": p}
    return out


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def main():
    index = load_patient_index()
    patients = index.set_index("hospital_id").label.map(config.LABEL_TO_INT)
    subloc = index.set_index("hospital_id").sublocation.fillna("not recorded")
    clinical = pd.read_csv(OUT / "clinical_features.csv", index_col="hospital_id")
    stats = {"clinical": {}, "selective": {}, "referred": {}}
    pair_rows, clin_rows, ref_rows = [], [], []
    for split, run in RUNS.items():
        df = pd.read_csv(OUT / f"{run}_patients.csv")
        clin = clinical_split(patients, clinical, split, df.set_index("hospital_id").outer_fold.to_dict())
        df = df.merge(clin, on="hospital_id")
        clin_rows.append(clin.assign(split=split))
        y = df.label.to_numpy()
        s = {m: df[f"{m}_score"].to_numpy() for m in MODELS}
        t = {m: df[f"{m}_threshold"].to_numpy() for m in MODELS}
        stats["clinical"][split] = n21.metrics(y, s["clinical"], t["clinical"]) \
            | {"ci95": n21.bootstrap_ci(y, s["clinical"], t["clinical"])}
        # 2. paired bootstrap, Holm within the split over all pairs x metrics
        rng = np.random.default_rng(SEED)
        res = {(a, b): paired_bootstrap(y, s, t, a, b, rng) for a, b in PAIRS}
        flat = [(a, b, k) for a, b in PAIRS for k in METRICS]
        adj = holm(np.array([res[(a, b)][k]["p"] for a, b, k in flat]))
        for (a, b, k), p_h in zip(flat, adj):
            r = res[(a, b)][k]
            pair_rows.append({"split": split, "model_a": a, "model_b": b, "metric": k, "diff": r["diff"],
                              "ci_low": r["ci95"][0], "ci_high": r["ci95"][1], "p": r["p"], "p_holm": p_h})
        # 3. selective classification at the primary coverage
        inner = pd.read_csv(OUT / f"{run}_inner_oof.csv")
        kept = n26.select(df, inner, "fusion", n26.PRIMARY_COVERAGE)
        yk, sk, tk = kept.label.to_numpy(), kept.fusion_score.to_numpy(), kept.fusion_threshold.to_numpy()
        stats["selective"][split] = n21.metrics(yk, sk, tk) | {"ci95": n21.bootstrap_ci(yk, sk, tk),
                                                               "coverage": len(kept) / len(df)}
        # 4. referred patients
        ref = df[~df.hospital_id.isin(kept.hospital_id)]
        wrong = (ref.fusion_score > ref.fusion_threshold).astype(int) != ref.label
        stats["referred"][split] = {"patients": int(len(ref)), "lvot": int(ref.label.sum()),
                                    "would_be_wrong": int(wrong.sum()),
                                    "errors_among_classified": int(((kept.fusion_score > kept.fusion_threshold)
                                                                    .astype(int) != kept.label).sum())}
        ref_rows += [{"split": split, "hospital_id": h, "label": int(l), "sublocation": subloc[h],
                      "would_be_wrong": bool(w)} for h, l, w in zip(ref.hospital_id, ref.label, wrong)]
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT / "paper_pairwise.csv", index=False)
    pd.concat(clin_rows).to_csv(OUT / "paper_clinical_patients.csv", index=False)
    refs = pd.DataFrame(ref_rows)
    refs.to_csv(OUT / "paper_referred.csv", index=False)
    (OUT / "paper_stats.json").write_text(json.dumps({k: {str(s): v for s, v in d.items()} for k, d in stats.items()},
                                                     indent=1))

    print("1. clinical ECG criteria (logistic regression), per split:")
    for split, m in stats["clinical"].items():
        print(f"   split {split}: macro F1 {m['macro_f1']:.3f} [{m['ci95']['macro_f1'][0]:.3f}, "
              f"{m['ci95']['macro_f1'][1]:.3f}]  bal. acc {m['balanced_accuracy']:.3f}  acc {m['accuracy']:.3f}  "
              f"sens {m['sensitivity']:.3f}  spec {m['specificity']:.3f}  AUC {m['auc']:.3f}")
    c = pd.DataFrame(stats["clinical"]).T.drop(columns="ci95").astype(float)
    print(f"   mean +- SD: macro F1 {c.macro_f1.mean():.3f} +- {c.macro_f1.std():.3f}  bal. acc "
          f"{c.balanced_accuracy.mean():.3f} +- {c.balanced_accuracy.std():.3f}  AUC {c.auc.mean():.3f} +- {c.auc.std():.3f}")
    print("\n2. paired bootstrap differences (a - b), Holm-adjusted p within each split:")
    for (a, b, k), g in pairs.groupby(["model_a", "model_b", "metric"], sort=False):
        print(f"   {a:8s} - {b:8s} {k:17s} " + "  ".join(
            f"[{r.split}] {r.diff:+.3f} ({r.ci_low:+.3f}, {r.ci_high:+.3f}) p={r.p_holm:.3f}" for r in g.itertuples())
              + f"  | mean {g['diff'].mean():+.3f}, positive in {int((g['diff'] > 0).sum())}/3")
    print("\n3. selective classification, fusion, primary target coverage 90 %, per split:")
    for split, m in stats["selective"].items():
        print(f"   split {split}: coverage {m['coverage']:.1%}  macro F1 {m['macro_f1']:.3f} "
              f"[{m['ci95']['macro_f1'][0]:.3f}, {m['ci95']['macro_f1'][1]:.3f}]  bal. acc "
              f"{m['balanced_accuracy']:.3f} [{m['ci95']['balanced_accuracy'][0]:.3f}, "
              f"{m['ci95']['balanced_accuracy'][1]:.3f}]  sens {m['sensitivity']:.3f}  spec {m['specificity']:.3f}")
    print("\n4. referred patients at the primary coverage:")
    for split, r in stats["referred"].items():
        print(f"   split {split}: {r['patients']} referred ({r['lvot']} LVOT); {r['would_be_wrong']} of them would have "
              f"been misclassified; {r['errors_among_classified']} errors among the classified patients")
    print("   sub-locations of the referred patients (all splits; a patient can be referred in several splits):")
    for (lab, sl), n in refs.groupby(["label", "sublocation"]).size().sort_values(ascending=False).items():
        print(f"     {'LVOT' if lab else 'RVOT'} {sl}: {n}")
    print("written", OUT / "paper_stats.json")


if __name__ == "__main__":
    main()
