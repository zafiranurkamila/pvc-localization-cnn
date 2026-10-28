"""Secondary analysis: performance by the number of PVC beats in the recording (no retraining).

Uses the outer-test predictions of scripts/24_nested_cv_fusion.py (every patient predicted once per split, thresholds
chosen on inner out-of-fold scores of all training patients). The number of PVC beats is known before prediction
(it does not use the label), so "at least N PVC beats" is an inclusion criterion that a user of the model can apply.

Specified before looking at the results (committed with this script):
  primary secondary-analysis cutoff  N >= 3 PVC beats
  also reported                      every cutoff N = 1 .. 8 (the whole curve), and the excluded patients
Each cutoff: patients and LVOT kept, pooled metrics with bootstrap 95 % CI per split, mean +- SD over the splits.

Usage (after the three fusion runs):
    python scripts/25_min_beats_subgroup.py
    python scripts/25_min_beats_subgroup.py --runs fusion_psd12_aug_wd0.0001_ens5 fusion_psd12_aug_wd0.0001_ens5_split7
Output: results/nested_cv/min_beats_subgroup.csv, min_beats_subgroup.json
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("nested21", Path(__file__).resolve().parent / "21_nested_cv_dual_rhythm.py")
n21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n21)  # also puts src/ on sys.path

from pvc_localization.data.dataset import PVCBeatsDataset  # noqa: E402
from pvc_localization.data.loader import load_patient_index  # noqa: E402
from pvc_localization.training.trainer import CACHE_DIR  # noqa: E402

OUT = n21.OUT
RUNS = ["fusion_psd12_aug_wd0.0001_ens5", "fusion_psd12_aug_wd0.0001_ens5_split7",
        "fusion_psd12_aug_wd0.0001_ens5_split123"]
PRIMARY_CUTOFF = 3
CUTOFFS = list(range(1, 9))
MODELS = ("fusion", "founder", "cnn", "fusion_equal")
KEYS = ("macro_f1", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "auc")


def beat_counts():
    ds = PVCBeatsDataset(load_patient_index().hospital_id.tolist(), ["psd", "wavelet"], cache_dir=CACHE_DIR)
    return pd.Series(ds.patient_groups()).value_counts().rename("n_beats")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS, help="fusion result names (one per split)")
    args = ap.parse_args()
    counts = beat_counts()
    print("PVC beats per patient: median", int(counts.median()), "min", int(counts.min()), "max", int(counts.max()))
    print("patients with >= N beats:", {n: int((counts >= n).sum()) for n in CUTOFFS})
    runs = {r: pd.read_csv(OUT / f"{r}_patients.csv") for r in args.runs}
    rows = []
    for r, df in runs.items():
        df = df.assign(n_beats=df.hospital_id.map(counts))
        for n in CUTOFFS:
            for subset, d in (("kept", df[df.n_beats >= n]), ("excluded", df[df.n_beats < n])):
                if len(d) == 0:
                    continue
                y = d.label.to_numpy()
                for m in MODELS:
                    sc, th = d[f"{m}_score"].to_numpy(), d[f"{m}_threshold"].to_numpy()
                    met = n21.metrics(y, sc, th)
                    ci = n21.bootstrap_ci(y, sc, th) if subset == "kept" and len(set(y)) > 1 else {}
                    rows.append({"run": r, "cutoff": n, "subset": subset, "model": m, "patients": len(d),
                                 "lvot": int(y.sum()), **{k: met[k] for k in KEYS},
                                 **{f"{k}_ci": ci.get(k) for k in ("macro_f1", "balanced_accuracy")}})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "min_beats_subgroup.csv", index=False)
    kept = res[res.subset == "kept"]
    agg = kept.groupby(["model", "cutoff"]).agg(patients=("patients", "first"), lvot=("lvot", "first"),
                                               **{f"{k}_mean": (k, "mean") for k in KEYS},
                                               **{f"{k}_sd": (k, "std") for k in KEYS}).reset_index()
    (OUT / "min_beats_subgroup.json").write_text(json.dumps(
        {"runs": list(runs), "primary_cutoff": PRIMARY_CUTOFF, "mean_sd_over_runs": agg.to_dict(orient="records")},
        indent=1))
    print(f"\nmean +- SD over {len(runs)} splits, patients with >= N PVC beats (primary cutoff N = {PRIMARY_CUTOFF}):")
    for m in MODELS:
        print(f"  {m}")
        for _, a in agg[agg.model == m].iterrows():
            mark = "  <- primary" if a.cutoff == PRIMARY_CUTOFF else ""
            print(f"    N>={int(a.cutoff)}: {int(a.patients):3d} patients ({int(a.lvot)} LVOT)  macro F1 {a.macro_f1_mean:.3f} +- "
                  f"{a.macro_f1_sd:.3f}  bal. acc {a.balanced_accuracy_mean:.3f} +- {a.balanced_accuracy_sd:.3f}  "
                  f"acc {a.accuracy_mean:.3f}  AUC {a.auc_mean:.3f}{mark}")
    exc = res[(res.subset == "excluded") & (res.model == "fusion") & (res.cutoff == PRIMARY_CUTOFF)]
    if len(exc):
        print(f"\nfusion on the excluded patients (< {PRIMARY_CUTOFF} beats), per split: "
              + "; ".join(f"{e.patients} patients ({e.lvot} LVOT) macro F1 {e.macro_f1:.3f} bal. acc "
                          f"{e.balanced_accuracy:.3f}" for e in exc.itertuples()))
    print("written", OUT / "min_beats_subgroup.csv")


if __name__ == "__main__":
    main()
