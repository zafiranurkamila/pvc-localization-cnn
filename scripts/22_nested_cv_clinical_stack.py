"""Nested patient-level CV of the PSD + CWT CNN stacked with clinical ECG criteria of the outflow-tract origin.

Same outer and inner folds, grid, options and CNN as scripts/21_nested_cv_dual_rhythm.py (scenario pvc). Added:

Clinical ECG features per PVC beat (median over the patient's beats), measured on the filtered, unnormalised beat:
  QRS duration (multi-lead slope envelope); R/(R+S) in V1-V6; precordial transition (interpolated lead number at
  which R/(R+S) reaches 0.5); V2 and V3 transition ratio (PVC R/(R+S) over that of the patient's sinus template,
  cf. Betensky et al. 2011); R-wave duration index in V1 and V2 (cf. Ouyang et al. 2002); V2S/V3R index
  (cf. Yoshida et al. 2014); polarity (net / absolute QRS area) of all 12 leads; inferior-lead net amplitude.

Three patient-level models per outer fold, all chosen on the outer training part only:
  cnn       ensemble of K seeds; the threshold is chosen on inner out-of-fold scores of a K-seed ensemble, so the
            score scale of the selection matches that of the outer model (script 21 used one seed in the inner loop)
  clinical  L2 logistic regression on the clinical features
  stack     L2 logistic regression on the CNN ensemble score + the clinical features; the CNN input of the
            training patients is their inner out-of-fold ensemble score
  For the logistic models C in {0.01, 0.1, 1} is chosen by inner out-of-fold AUC and the threshold on the inner
  out-of-fold decision values (criterion --select), using the same inner folds as the CNN.

Run on the PC (GPU).
Usage:
    python scripts/22_nested_cv_clinical_stack.py --features-only          # clinical features + univariate AUC
    python scripts/22_nested_cv_clinical_stack.py --quick                  # smoke test (1 outer fold, few epochs)
    python scripts/22_nested_cv_clinical_stack.py --psd12 --augment --wd 0.0001 --ensemble 5
Output: results/nested_cv/clinical_features.csv, stack<suffix>_patients.csv, stack<suffix>_folds.json,
        summary_stack.json
"""
import argparse
import importlib.util
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

_spec = importlib.util.spec_from_file_location("nested21", Path(__file__).resolve().parent / "21_nested_cv_dual_rhythm.py")
n21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n21)  # also puts src/ on sys.path

from pvc_localization import config  # noqa: E402
from pvc_localization.data.dataset import PVCBeatsDataset  # noqa: E402
from pvc_localization.data.loader import load_patient_index  # noqa: E402
from pvc_localization.training.trainer import CACHE_DIR  # noqa: E402

OUT = n21.OUT
SEED = n21.SEED
FS = config.SAMPLING_RATE_HZ
R_IDX = config.BEAT_PRE_SAMPLES
L = {name: i for i, name in enumerate(config.LEADS)}
PRECORDIAL = [L[f"V{k}"] for k in range(1, 7)]
C_GRID = [0.01, 0.1, 1.0]
EPS = 1e-6


# ---------------------------------------------------------------- clinical ECG features
def qrs_bounds(beat):
    """QRS onset and offset (sample indices): the stretch around the maximum of the smoothed multi-lead slope envelope
    where it exceeds its floor (10th percentile over the whole beat) by 20 % of the range (gaps < 20 ms bridged),
    searched from 120 ms before to 160 ms after the detected R peak."""
    lo, hi = R_IDX - int(0.12 * FS), R_IDX + int(0.16 * FS)
    w = int(0.01 * FS)
    smooth = lambda a: np.convolve(a, np.ones(w) / w, mode="same")
    env_all = smooth(np.abs(np.diff(np.apply_along_axis(smooth, 1, beat), axis=1)).sum(axis=0))
    env = env_all[lo:hi]
    floor = np.percentile(env_all, 10)
    above = np.where(env > floor + 0.2 * (env.max() - floor))[0]
    if len(above) < 2:
        return R_IDX - int(0.06 * FS), R_IDX + int(0.06 * FS)
    runs = np.split(above, np.where(np.diff(above) > int(0.02 * FS))[0] + 1)
    run = next(r for r in runs if r[0] <= np.argmax(env) <= r[-1])
    return lo + run[0], lo + run[-1] + 1


def qrs_deflections(beat):
    """Baseline-corrected QRS of every lead (baseline: median of the 5-20 ms before onset) and the QRS duration (ms)."""
    on, off = qrs_bounds(beat)
    base = np.median(beat[:, max(on - int(0.02 * FS), 0):max(on - int(0.005 * FS), 1)], axis=1, keepdims=True)
    return beat[:, on:off] - base, (off - on) / FS * 1000


def rs_ratio(x):
    r, s = max(x.max(), 0.0), max(-x.min(), 0.0)
    return r / (r + s + EPS), r, s


def transition_lead(rs):
    """Interpolated precordial lead number (1-6) at which R/(R+S) first reaches 0.5; 7 if it never does."""
    k = int(np.argmax(rs >= 0.5)) if (rs >= 0.5).any() else None
    if k is None:
        return 7.0
    if k == 0:
        return 1.0
    return k + (0.5 - rs[k - 1]) / (rs[k] - rs[k - 1] + EPS)


def beat_features(beat, sinus_rs):
    x, qrs_ms = qrs_deflections(beat)
    rs, r, s = map(np.array, zip(*[rs_ratio(x[i]) for i in range(12)]))
    f = {"qrs_ms": qrs_ms}
    for k, i in enumerate(PRECORDIAL, 1):
        f[f"rs_V{k}"] = rs[i]
    f["transition"] = transition_lead(rs[PRECORDIAL])
    for lead in ("V2", "V3"):
        f[f"tz_ratio_{lead}"] = np.log1p(np.clip(rs[L[lead]] / (sinus_rs[lead] + EPS), 0, 10)) \
            if np.isfinite(sinus_rs[lead]) else np.nan
    for lead in ("V1", "V2"):
        xi = x[L[lead]]
        f[f"r_dur_idx_{lead}"] = float(np.mean(xi > 0.1 * r[L[lead]])) if r[L[lead]] > 0 else 0.0
    f["v2s_v3r"] = np.log1p(np.clip(s[L["V2"]] / (r[L["V3"]] + EPS), 0, 20))
    for name, i in L.items():
        f[f"polarity_{name}"] = x[i].sum() / (np.abs(x[i]).sum() + EPS)
    inferior = [L["II"], L["III"], L["aVF"]]
    f["inferior_net"] = (r[inferior] - s[inferior]).sum() / ((r + s).sum() + EPS)
    return f


def sinus_rs(h):
    path = n21.SINUS / f"{h}.npy"
    if not path.exists():
        return {"V2": np.nan, "V3": np.nan}
    x, _ = qrs_deflections(np.load(path).astype(np.float64))
    return {lead: rs_ratio(x[L[lead]])[0] for lead in ("V2", "V3")}


# ---------------------------------------------------------------- data
def load_data(device):
    """PSD + CWT of every PVC beat (as script 21) and the patient-level clinical features."""
    index = load_patient_index()
    ids = index.hospital_id.tolist()
    ds = PVCBeatsDataset(ids, ["psd", "wavelet"], cache_dir=CACHE_DIR)
    sin = {h: sinus_rs(h) for h in tqdm(ids, desc="sinus templates", unit="patient")}
    psd, cwt, pid, lab, clin = [], [], [], [], []
    for i in tqdm(range(len(ds)), desc="PVC beats", unit="beat"):
        item = ds[i]
        psd.append(item["psd"].numpy()); cwt.append(item["wavelet"].numpy())
        h = int(item["hospital_id"]); pid.append(h); lab.append(int(item["label"]))
        clin.append({"hospital_id": h, **beat_features(ds.beats_list[i][3].astype(np.float64), sin[h])})
    t = lambda a: torch.tensor(np.stack(a), dtype=torch.float32, device=device)
    feats = {"psd": t(psd), "cwt": t(cwt)}
    clinical = pd.DataFrame(clin).groupby("hospital_id").median()
    patients = index.set_index("hospital_id").label.map(config.LABEL_TO_INT)
    return feats, np.array(pid), np.array(lab), patients, clinical


# ---------------------------------------------------------------- logistic models
def logistic(C):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def fit_logistic(X, y, cv):
    """C by inner out-of-fold AUC, threshold on the inner out-of-fold decision values; refit on all of X."""
    best = None
    for C in C_GRID:
        oof = np.zeros(len(y))
        for tr, va in cv:
            oof[va] = logistic(C).fit(X[tr], y[tr]).decision_function(X[va])
        auc = roc_auc_score(y, oof)
        if best is None or auc > best[1]:
            best = (C, auc, oof)
    C, auc, oof = best
    return logistic(C).fit(X, y), C, auc, n21.best_threshold(oof, y)


# ---------------------------------------------------------------- nested CV
def run(feats, pid, labels, patients, clinical, device, n_outer, quick, ensemble):
    all_p = patients.index.to_numpy(); all_y = patients.to_numpy()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    ckpts = [3, 5] if quick else n21.EPOCH_CHECKPOINTS
    grid = [dict(zip(n21.GRID, v)) for v in product(*n21.GRID.values())][: (1 if quick else None)]
    clin_cols = list(clinical.columns)
    rows, folds = [], []
    for fo, (tr_p, te_p) in enumerate(outer.split(all_p, all_y)):
        if fo >= n_outer:
            break
        t0 = time.time()
        tr_pat, te_pat = all_p[tr_p], all_p[te_p]
        inner = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fo)
                     .split(tr_pat, patients[tr_pat].to_numpy()))
        # 1) CNN setting: as script 21, one seed per inner fold and setting
        oof = {}
        for itr, iva in inner:
            b_tr = np.where(np.isin(pid, tr_pat[itr]))[0]; b_va = np.where(np.isin(pid, tr_pat[iva]))[0]
            for hp in grid:
                _, snaps = n21.train(feats, b_tr, labels, "pvc", hp["learning_rate"], hp["dropout"], max(ckpts),
                                     device, eval_sets=[b_va], checkpoints=ckpts)
                for ep, (m_va,) in snaps.items():
                    oof.setdefault((hp["learning_rate"], hp["dropout"], ep), []).append(
                        n21.patient_scores(m_va, pid[b_va]))
        best, best_c = None, -1
        for key, parts in oof.items():
            s = pd.concat(parts); y = patients[s.index].to_numpy()
            c = n21.criterion_at(s.to_numpy(), y, n21.best_threshold(s.to_numpy(), y))
            if c > best_c:
                best, best_c = key, c
        lr, dropout, ep = best
        # 2) inner out-of-fold scores of a K-seed ensemble with the chosen setting; seed SEED is the grid run above
        #    (its snapshot at epoch ep equals a run of ep epochs: same seed, constant learning rate)
        ens_parts = []
        for (itr, iva), seed0 in zip(inner, oof[best]):
            b_tr = np.where(np.isin(pid, tr_pat[itr]))[0]; b_va = np.where(np.isin(pid, tr_pat[iva]))[0]
            extra = [n21.patient_scores(n21.margins(n21.train(feats, b_tr, labels, "pvc", lr, dropout, ep, device,
                                                              seed=SEED + k)[0], feats, b_va, "pvc"), pid[b_va])
                     for k in range(1, ensemble)]
            ens_parts.append(pd.concat([seed0] + extra, axis=1).mean(axis=1))
        cnn_tr = pd.concat(ens_parts)
        tr_ids = cnn_tr.index.to_numpy(); y_tr = patients[tr_ids].to_numpy()
        pos = {h: i for i, h in enumerate(tr_ids)}
        cv = [(np.array([pos[h] for h in tr_pat[itr] if h in pos]), np.array([pos[h] for h in tr_pat[iva] if h in pos]))
              for itr, iva in inner]
        thr_cnn = n21.best_threshold(cnn_tr.to_numpy(), y_tr)
        X_clin_tr = clinical.reindex(tr_ids)[clin_cols].to_numpy()
        clin_model, c_clin, auc_clin, thr_clin = fit_logistic(X_clin_tr, y_tr, cv)
        X_st_tr = np.column_stack([cnn_tr.to_numpy(), X_clin_tr])
        st_model, c_st, auc_st, thr_st = fit_logistic(X_st_tr, y_tr, cv)
        # 3) outer fit with K seeds; predict the outer test fold once
        b_tr = np.where(np.isin(pid, tr_pat))[0]; b_te = np.where(np.isin(pid, te_pat))[0]
        m_te = np.mean([n21.margins(n21.train(feats, b_tr, labels, "pvc", lr, dropout, ep, device, seed=SEED + k)[0],
                                    feats, b_te, "pvc") for k in range(ensemble)], axis=0)
        cnn_te = n21.patient_scores(m_te, pid[b_te])
        te_ids = cnn_te.index.to_numpy(); y_te = patients[te_ids].to_numpy()
        X_clin_te = clinical.reindex(te_ids)[clin_cols].to_numpy()
        sc = {"cnn": cnn_te.to_numpy(), "clinical": clin_model.decision_function(X_clin_te),
              "stack": st_model.decision_function(np.column_stack([cnn_te.to_numpy(), X_clin_te]))}
        thr = {"cnn": thr_cnn, "clinical": thr_clin, "stack": thr_st}
        for i, h in enumerate(te_ids):
            rows.append({"hospital_id": h, "label": int(y_te[i]), "outer_fold": fo,
                         **{f"{m}_score": float(sc[m][i]) for m in sc}, **{f"{m}_threshold": thr[m] for m in sc}})
        fm = {m: n21.metrics(y_te, sc[m], thr[m]) for m in sc}
        coef = st_model[-1].coef_[0]
        folds.append({"outer_fold": fo, "learning_rate": lr, "dropout": dropout, "epochs": ep,
                      "inner_criterion_cnn_1seed": best_c, "C_clinical": c_clin, "C_stack": c_st,
                      "inner_auc_clinical": auc_clin, "inner_auc_stack": auc_st,
                      "stack_coef": dict(zip(["cnn_score"] + clin_cols, map(float, coef))),
                      **{f"test_{m}_{k}": v for m in fm for k, v in fm[m].items()},
                      "minutes": round((time.time() - t0) / 60, 1)})
        tqdm.write(f"outer {fo + 1}/5: lr={lr} dropout={dropout} epochs={ep} | test macro F1 / bal. acc / AUC  "
                   + "  ".join(f"{m} {fm[m]['macro_f1']:.3f}/{fm[m]['balanced_accuracy']:.3f}/{fm[m]['auc']:.3f}"
                               for m in fm) + f"  ({folds[-1]['minutes']} min)")
    df = pd.DataFrame(rows)
    y = df.label.to_numpy()
    pooled = {m: n21.metrics(y, df[f"{m}_score"].to_numpy(), df[f"{m}_threshold"].to_numpy())
              | {"ci95": n21.bootstrap_ci(y, df[f"{m}_score"].to_numpy(), df[f"{m}_threshold"].to_numpy())}
              for m in ("cnn", "clinical", "stack")}
    return df, folds, pooled | {"patients": int(len(df)), "lvot": int(y.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", type=int, default=5, help="number of seeds of the CNN ensemble")
    ap.add_argument("--psd12", action="store_true", help="PSD branch with one channel per lead")
    ap.add_argument("--augment", action="store_true", help="training-only time-shift and noise augmentation")
    ap.add_argument("--wd", type=float, default=0.0, help="Adam weight decay")
    ap.add_argument("--select", choices=["f1", "bacc"], default="f1",
                    help="inner-loop criterion for the CNN setting and all thresholds")
    ap.add_argument("--features-only", action="store_true", help="only compute the clinical features")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 outer fold, 1 setting, 5 epochs")
    args = ap.parse_args()
    n21.OPTS.update(psd12=args.psd12, augment=args.augment, wd=args.wd, select=args.select)
    suffix = ("_psd12" if args.psd12 else "") + ("_aug" if args.augment else "") + (f"_wd{args.wd:g}" if args.wd else "") \
        + ("_bacc" if args.select == "bacc" else "") + f"_ens{args.ensemble}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    OUT.mkdir(parents=True, exist_ok=True)
    feats, pid, labels, patients, clinical = load_data(device)
    print(f"{len(pid)} PVC beats from {len(np.unique(pid))} patients, {clinical.shape[1]} clinical features")
    clinical.to_csv(OUT / "clinical_features.csv")
    # descriptive only (all patients, no selection is based on it): univariate AUC of each feature for LVOT
    y = patients[clinical.index].to_numpy()
    auc = {c: roc_auc_score(y, clinical[c].fillna(clinical[c].median())) for c in clinical.columns}
    print("univariate AUC (LVOT vs RVOT; <0.5 means lower in LVOT):")
    for c, a in sorted(auc.items(), key=lambda kv: -abs(kv[1] - 0.5)):
        print(f"  {c:18s} {a:.3f}   missing {int(clinical[c].isna().sum())}")
    if args.features_only:
        print("written", OUT / "clinical_features.csv")
        return
    df, folds, pooled = run(feats, pid, labels, patients, clinical, device, 1 if args.quick else 5, args.quick,
                            args.ensemble)
    tag = "stack" + suffix + ("_quick" if args.quick else "")
    df.to_csv(OUT / f"{tag}_patients.csv", index=False)
    (OUT / f"{tag}_folds.json").write_text(json.dumps(folds, indent=1))
    summary_path = OUT / ("summary_stack_quick.json" if args.quick else "summary_stack.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary[tag] = pooled
    summary_path.write_text(json.dumps(summary, indent=1))
    print(f"\n[{tag}] pooled over {pooled['patients']} patients ({pooled['lvot']} LVOT), inner-selected thresholds:")
    for m in ("cnn", "clinical", "stack"):
        r = pooled[m]
        print(f"  {m:8s} macro F1 {r['macro_f1']:.3f} [{r['ci95']['macro_f1'][0]:.3f}, {r['ci95']['macro_f1'][1]:.3f}]"
              f"  bal. acc {r['balanced_accuracy']:.3f}  acc {r['accuracy']:.3f}  sens {r['sensitivity']:.3f}"
              f"  spec {r['specificity']:.3f}  AUC {r['auc']:.3f}  (TP {r['tp']} FN {r['fn']} TN {r['tn']} FP {r['fp']})")
    print("written", OUT)


if __name__ == "__main__":
    main()
