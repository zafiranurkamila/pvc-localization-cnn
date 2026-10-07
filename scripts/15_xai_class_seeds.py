"""Class-specific explanations and their stability across seeds (reviewer comment 10 and minor 7).

1. Class-specific, signed SHAP for the tuned models (default psd_wavelet_tuned, psd_wavelet_hos_tuned).
   SHAP values on the logit margin (LVOT - RVOT) keep their sign: positive pushes towards LVOT.
   They are averaged per branch, lead, PSD bin and wavelet time-frequency cell, separately for the true
   RVOT and LVOT test beats, and also patient-weighted (beats -> patient mean -> class mean) so that
   patients with many beats do not dominate.
2. Stability across seeds: final fixed-setting models (all training patients, no CV) are trained with
   seeds 1-5 for psd_wavelet and psd_wavelet_hos, then branch permutation importance, |SHAP| lead shares
   and the test performance (beat and patient level) are compared across seeds.

Resumable: trained seed models are saved as results/models/<scenario>_seed<k>.pt and reused.

Usage:
    python scripts/15_xai_class_seeds.py
    python scripts/15_xai_class_seeds.py --skip-seeds          # only part 1
Output: results/xai/<model>_class_signed.json and .csv, results/xai/stability_<scenario>.json
"""
import argparse
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pvc_localization import config  # noqa: E402
from pvc_localization.data.dataset import PVCBeatsDataset, create_train_val_test_split  # noqa: E402
from pvc_localization.training.trainer import CACHE_DIR, CVTrainer, set_seed  # noqa: E402

_spec = importlib.util.spec_from_file_location("xai11", ROOT / "scripts" / "11_xai_shap.py")
xai = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xai)

LEADS = config.LEADS
FIXED = {"epochs": 50, "batch_size": 64, "learning_rate": 0.001, "optimizer": "adam",
         "n_filters": 64, "kernel_size": None, "dropout": 0.3}
OUT = config.RESULTS_DIR / "xai"


def per_lead(v: np.ndarray, branch: str) -> np.ndarray:
    """(n, ...) SHAP of one branch -> (n, 12) summed per lead (signs kept)."""
    n = v.shape[0]
    if branch == "psd":
        return v.reshape(n, len(LEADS), -1).sum(axis=2)
    if branch == "hos":
        return v.reshape(n, len(LEADS), -1).sum(axis=2)
    return v.sum(axis=(2, 3))  # wavelet: (n, 12, scales, time)


def explain(name, device, train_set, test_set, n_background=100):
    model, features = xai.load_model(name, device)
    x_test, y_test = xai.stack(test_set, features)
    pids = np.array([test_set.beats_list[i][0] for i in range(len(test_set))])
    rng = np.random.default_rng(config.RANDOM_SEED)
    bg_idx = rng.choice(len(train_set), size=min(n_background, len(train_set)), replace=False)
    x_bg, _ = xai.stack(train_set, features, bg_idx)
    sv = xai.shap_values(model, features, x_bg, x_test, device)
    return model, features, x_test, y_test, pids, sv


def patient_mean(values: np.ndarray, pids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over the beats of each patient, then over patients (rows selected by mask)."""
    df = pd.DataFrame(values[mask])
    return df.groupby(pids[mask]).mean().mean(axis=0).to_numpy()


def class_signed(name, device, train_set, test_set):
    model, features, x_test, y_test, pids, sv = explain(name, device, train_set, test_set)
    res = {"model": name, "sign": "positive = towards LVOT (logit margin LVOT - RVOT)", "classes": {}}
    for cls, lab in (("RVOT", 0), ("LVOT", 1)):
        m = y_test == lab
        branch = {f: float(sv[f][m].reshape(m.sum(), -1).sum(axis=1).mean()) for f in features}
        lead_beats = sum(per_lead(sv[f], f) for f in features)
        res["classes"][cls] = {
            "n_beats": int(m.sum()), "n_patients": int(len(set(pids[m]))),
            "branch_mean_signed": branch,
            "lead_mean_signed_beat": dict(zip(LEADS, lead_beats[m].mean(axis=0).round(6).tolist())),
            "lead_mean_signed_patient": dict(zip(LEADS, patient_mean(lead_beats, pids, m).round(6).tolist())),
        }
        if "psd" in features:
            psd = sv["psd"][m].reshape(m.sum(), len(LEADS), -1).mean(axis=(0, 1))
            res["classes"][cls]["psd_freq_mean_signed"] = psd.round(6).tolist()
        if "wavelet" in features:
            tf = sv["wavelet"][m].mean(axis=(0, 1))
            np.savetxt(OUT / f"{name}_class_{cls}_wavelet_signed.csv", tf, delimiter=",", fmt="%.6f")
    (OUT / f"{name}_class_signed.json").write_text(json.dumps(res, indent=1))
    print(f"\n=== {name}: signed SHAP per class (positive = LVOT)")
    for cls, v in res["classes"].items():
        top = sorted(v["lead_mean_signed_patient"].items(), key=lambda kv: -abs(kv[1]))[:4]
        print(f"  {cls}: branch {({k: round(b, 3) for k, b in v['branch_mean_signed'].items()})} "
              f"| strongest leads (patient-weighted) {[(l, round(s, 3)) for l, s in top]}")


def train_seed_model(scenario, seed, train_ids, device):
    path = config.RESULTS_DIR / "models" / f"{scenario}_seed{seed}.pt"
    if path.exists():
        return path.stem
    features = config.FEATURE_SCENARIOS[scenario]
    set_seed(seed)
    trainer = CVTrainer(features, num_folds=config.N_FOLDS, device=device, class_weight=True,
                        optimizer=FIXED["optimizer"],
                        model_params={k: FIXED[k] for k in ("n_filters", "kernel_size", "dropout")})
    train_set = PVCBeatsDataset(train_ids, features, cache_dir=CACHE_DIR)
    labels = np.array(train_set.labels())
    pbar = tqdm(total=FIXED["epochs"], desc=f"{scenario} seed {seed}", unit="epoch")
    model = trainer._fit(train_set, labels, FIXED["epochs"], FIXED["batch_size"], FIXED["learning_rate"], pbar, f"seed{seed}")
    pbar.close()
    torch.save({"state_dict": model.state_dict(), "scenario": scenario, "features": features,
                "hyperparameters": FIXED, "class_weight": True, "seed": seed}, path)
    return path.stem


def patient_scores(y, margin, pids):
    df = pd.DataFrame({"pid": pids, "y": y, "m": margin}).groupby("pid").agg(y=("y", "first"), m=("m", "mean"))
    pred = (df.m > 0).astype(int)
    return float(f1_score(df.y, pred, average="macro")), float(roc_auc_score(df.y, df.m))


def stability(scenario, seeds, train_ids, test_ids, device):
    features = config.FEATURE_SCENARIOS[scenario]
    train_set = PVCBeatsDataset(train_ids, features, cache_dir=CACHE_DIR)
    test_set = PVCBeatsDataset(test_ids, features, cache_dir=CACHE_DIR)
    rows, lead_shares = [], []
    for seed in seeds:
        name = train_seed_model(scenario, seed, train_ids, device)
        model, feats, x_test, y_test, pids, sv = explain(name, device, train_set, test_set)
        margin = xai.predict_margin(model, x_test, device)
        perm = xai.branch_permutation(model, x_test, y_test, feats, device)
        lead_abs = sum(per_lead(np.abs(sv[f]), f) for f in feats).mean(axis=0)
        share = lead_abs / lead_abs.sum()
        lead_shares.append(share)
        pf1, pauc = patient_scores(y_test, margin, pids)
        rows.append({"seed": seed,
                     "beat_macro_f1": float(f1_score(y_test, (margin > 0).astype(int), average="macro")),
                     "beat_auc": float(roc_auc_score(y_test, margin)),
                     "patient_macro_f1": pf1, "patient_auc": pauc,
                     "perm_auc_drop": {r["branch"]: r["auc_drop_mean"] for r in perm["branches"]},
                     "lead_share": dict(zip(LEADS, share.round(4).tolist())),
                     "precordial_share": float(share[LEADS.index("V1"):].sum())})
        print(f"  {scenario} seed {seed}: beat F1 {rows[-1]['beat_macro_f1']:.3f} patient F1 {pf1:.3f} | "
              f"perm {({k: round(v, 3) for k, v in rows[-1]['perm_auc_drop'].items()})} | V1-V6 {rows[-1]['precordial_share']:.1%}")
    rho = [spearmanr(a, b)[0] for a, b in combinations(lead_shares, 2)]
    summary = {"scenario": scenario, "seeds": seeds, "setting": FIXED, "per_seed": rows,
               "mean_pairwise_spearman_lead_share": float(np.mean(rho)),
               "perm_auc_drop_mean_sd": {f: [float(np.mean([r["perm_auc_drop"][f] for r in rows])),
                                             float(np.std([r["perm_auc_drop"][f] for r in rows]))] for f in features},
               "patient_macro_f1_mean_sd": [float(np.mean([r["patient_macro_f1"] for r in rows])),
                                            float(np.std([r["patient_macro_f1"] for r in rows]))],
               "beat_macro_f1_mean_sd": [float(np.mean([r["beat_macro_f1"] for r in rows])),
                                         float(np.std([r["beat_macro_f1"] for r in rows]))]}
    (OUT / f"stability_{scenario}.json").write_text(json.dumps(summary, indent=1))
    print(f"  => {scenario}: lead-share rank agreement (mean Spearman) {summary['mean_pairwise_spearman_lead_share']:.2f}, "
          f"patient F1 {summary['patient_macro_f1_mean_sd'][0]:.3f} ± {summary['patient_macro_f1_mean_sd'][1]:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["psd_wavelet_tuned", "psd_wavelet_hos_tuned"])
    parser.add_argument("--stability", nargs="+", default=["psd_wavelet", "psd_wavelet_hos"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--skip-seeds", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    train_ids, test_ids = create_train_val_test_split()

    for name in args.models:
        set_seed(config.RANDOM_SEED)
        features = torch.load(config.RESULTS_DIR / "models" / f"{name}.pt", map_location="cpu")["features"]
        class_signed(name, device, PVCBeatsDataset(train_ids, features, cache_dir=CACHE_DIR),
                     PVCBeatsDataset(test_ids, features, cache_dir=CACHE_DIR))

    if not args.skip_seeds:
        for scenario in args.stability:
            print(f"\n=== stability across seeds: {scenario}")
            stability(scenario, args.seeds, train_ids, test_ids, device)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
