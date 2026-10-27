"""Nested patient-level CV of the late fusion of the fine-tuned ECGFounder and the PSD + CWT CNN (LVOT vs RVOT).

Same outer and inner patient folds, criterion (--select) and pooled metrics as scripts/21-23. Per outer fold, using
the outer training part only:
  founder       ECGFounder as script 23: learning rate x epochs chosen on the inner folds (one seed), then inner
                out-of-fold scores of a K-seed ensemble with the chosen setting (script 23 took the threshold from the
                one-seed scores; here, as script 22, from the K-seed ones, so its scale matches the outer ensemble)
  cnn           PSD + CWT CNN as script 22 (scenario pvc; --psd12 --augment --wd): setting chosen on the inner folds,
                then inner out-of-fold scores of a K-seed ensemble
  fusion        weighted mean of the two scores, each standardised with the mean and standard deviation of its inner
                out-of-fold scores; the ECGFounder weight w in {0, 0.25, 0.5, 0.75, 1} is chosen by inner out-of-fold
                AUC (ties: the one closest to 0.5)
  fusion_equal  the same with w = 0.5 fixed in advance (no weight selection)
Every threshold is chosen on inner out-of-fold scores (criterion --select). The outer test fold is predicted once, by
K-seed ensembles trained on the whole outer training part.

Run on the PC (GPU); ECGFounder setup as in script 23.
Usage:
    python scripts/24_nested_cv_fusion.py --quick --ensemble 2      # smoke test (1 outer fold, short grids)
    python scripts/24_nested_cv_fusion.py --psd12 --augment --wd 0.0001 --ensemble 5
Output: results/nested_cv/fusion<suffix>_patients.csv, fusion<suffix>_inner_oof.csv, fusion<suffix>_folds.json,
        summary_fusion.json
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n21 = _load("nested21", "21_nested_cv_dual_rhythm.py")  # also puts src/ on sys.path
n23 = _load("nested23", "23_nested_cv_ecgfounder.py")

OUT = n21.OUT
SEED = n21.SEED
FOUNDER_LRS = [3e-4, 1e-4]
FOUNDER_EPOCHS = [20, 30, 45, 60]
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
MODELS = ("founder", "cnn", "fusion", "fusion_equal")


def select(oof, patients):
    """Setting with the best inner out-of-fold criterion (at its best threshold)."""
    best, best_c = None, -1
    for key, parts in oof.items():
        s = pd.concat(parts); y = patients[s.index].to_numpy()
        c = n21.criterion_at(s.to_numpy(), y, n21.best_threshold(s.to_numpy(), y))
        if c > best_c:
            best, best_c = key, c
    return best, best_c


def beats(pid, ids):
    return np.where(np.isin(pid, ids))[0]


# ---------------------------------------------------------------- the two models on one outer fold
def founder_fold(rec, patients, tr_pat, te_pat, inner, device, ensemble, batch, lrs, ckpts, bar):
    """Inner out-of-fold K-seed ensemble scores, outer test ensemble scores, chosen setting."""
    bar.set_postfix_str("founder 1/3 inner grid")
    oof = {}
    for itr, iva in inner:
        for lr in lrs:
            _, snaps = n23.train(rec, patients, list(tr_pat[itr]), lr, max(ckpts), device, eval_ids=list(tr_pat[iva]),
                                 checkpoints=ckpts, batch=batch, progress=bar)
            for ep, s in snaps.items():
                oof.setdefault((lr, ep), []).append(s)
    (lr, ep), c = select(oof, patients)
    bar.total += (len(inner) * (ensemble - 1) + ensemble) * ep
    bar.refresh()
    # seed SEED of the inner ensemble is the grid run above (its snapshot at epoch ep equals a run of ep epochs)
    bar.set_postfix_str("founder 2/3 inner ensemble")
    parts = []
    for (itr, iva), seed0 in zip(inner, oof[(lr, ep)]):
        extra = [n23.margins(n23.train(rec, patients, list(tr_pat[itr]), lr, ep, device, seed=SEED + k, batch=batch,
                                       progress=bar)[0], rec, list(tr_pat[iva]), device) for k in range(1, ensemble)]
        parts.append(pd.concat([seed0] + extra, axis=1).mean(axis=1))
    bar.set_postfix_str("founder 3/3 outer ensemble")
    s_te = pd.concat([n23.margins(n23.train(rec, patients, list(tr_pat), lr, ep, device, seed=SEED + k, batch=batch,
                                            progress=bar)[0], rec, list(te_pat), device) for k in range(ensemble)],
                     axis=1).mean(axis=1)
    return pd.concat(parts), s_te, {"learning_rate": lr, "epochs": ep, "inner_criterion_1seed": c}


def cnn_fold(feats, pid, labels, patients, tr_pat, te_pat, inner, device, ensemble, grid, ckpts, bar):
    """As founder_fold, for the PSD + CWT CNN (scenario pvc)."""
    bar.set_postfix_str("cnn 1/3 inner grid")
    oof = {}
    for itr, iva in inner:
        b_tr, b_va = beats(pid, tr_pat[itr]), beats(pid, tr_pat[iva])
        for hp in grid:
            _, snaps = n21.train(feats, b_tr, labels, "pvc", hp["learning_rate"], hp["dropout"], max(ckpts), device,
                                 eval_sets=[b_va], checkpoints=ckpts, progress=bar)
            for ep, (m_va,) in snaps.items():
                oof.setdefault((hp["learning_rate"], hp["dropout"], ep), []).append(
                    n21.patient_scores(m_va, pid[b_va]))
    (lr, dropout, ep), c = select(oof, patients)
    bar.total += (len(inner) * (ensemble - 1) + ensemble) * ep
    bar.refresh()
    bar.set_postfix_str("cnn 2/3 inner ensemble")
    parts = []
    for (itr, iva), seed0 in zip(inner, oof[(lr, dropout, ep)]):
        b_tr, b_va = beats(pid, tr_pat[itr]), beats(pid, tr_pat[iva])
        extra = [n21.patient_scores(n21.margins(n21.train(feats, b_tr, labels, "pvc", lr, dropout, ep, device,
                                                          seed=SEED + k, progress=bar)[0], feats, b_va, "pvc"),
                                    pid[b_va]) for k in range(1, ensemble)]
        parts.append(pd.concat([seed0] + extra, axis=1).mean(axis=1))
    bar.set_postfix_str("cnn 3/3 outer ensemble")
    b_tr, b_te = beats(pid, tr_pat), beats(pid, te_pat)
    m_te = np.mean([n21.margins(n21.train(feats, b_tr, labels, "pvc", lr, dropout, ep, device, seed=SEED + k,
                                          progress=bar)[0], feats, b_te, "pvc") for k in range(ensemble)], axis=0)
    return pd.concat(parts), n21.patient_scores(m_te, pid[b_te]), \
        {"learning_rate": lr, "dropout": dropout, "epochs": ep, "inner_criterion_1seed": c}


def standardiser(s):
    mu, sd = float(s.mean()), float(s.std())
    return lambda x: (np.asarray(x) - mu) / (sd + 1e-8)


# ---------------------------------------------------------------- nested CV
def run(rec, feats, pid, labels, patients, device, n_outer, quick, ensemble, batch):
    all_p = patients.index.to_numpy(); all_y = patients.to_numpy()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    f_lrs, f_ckpts = (FOUNDER_LRS[:1], [1, 2]) if quick else (FOUNDER_LRS, FOUNDER_EPOCHS)
    c_ckpts = [3, 5] if quick else n21.EPOCH_CHECKPOINTS
    grid = [dict(zip(n21.GRID, v)) for v in product(*n21.GRID.values())][: (1 if quick else None)]
    rows, oof_rows, folds = [], [], []
    for fo, (tr_p, te_p) in enumerate(outer.split(all_p, all_y)):
        if fo >= n_outer:
            break
        t0 = time.time()
        tr_pat, te_pat = all_p[tr_p], all_p[te_p]
        inner = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fo)
                     .split(tr_pat, patients[tr_pat].to_numpy()))
        # progress in epochs (ECGFounder and CNN epochs alike); grows once the chosen epochs are known
        bar = tqdm(total=len(inner) * (len(f_lrs) * max(f_ckpts) + len(grid) * max(c_ckpts)),
                   desc=f"outer {fo + 1}/5", unit="epoch", leave=False)
        f_tr, f_te, f_set = founder_fold(rec, patients, tr_pat, te_pat, inner, device, ensemble, batch, f_lrs, f_ckpts,
                                         bar)
        if device == "cuda":
            torch.cuda.empty_cache()
        c_tr, c_te, c_set = cnn_fold(feats, pid, labels, patients, tr_pat, te_pat, inner, device, ensemble, grid,
                                     c_ckpts, bar)
        bar.close()
        tr_ids = f_tr.index.intersection(c_tr.index); te_ids = f_te.index.intersection(c_te.index)
        if len(tr_ids) < len(f_tr) or len(te_ids) < len(f_te):
            tqdm.write(f"  note: {len(f_tr) - len(tr_ids)} training / {len(f_te) - len(te_ids)} test patients "
                       "without a CNN score are left out of this fold")
        y_tr, y_te = patients[tr_ids].to_numpy(), patients[te_ids].to_numpy()
        zf, zc = standardiser(f_tr[tr_ids]), standardiser(c_tr[tr_ids])
        fuse = lambda w, f, c: w * zf(f) + (1 - w) * zc(c)
        aucs = {w: roc_auc_score(y_tr, fuse(w, f_tr[tr_ids], c_tr[tr_ids])) for w in W_GRID}
        w = min((v for v in W_GRID if np.isclose(aucs[v], max(aucs.values()))), key=lambda v: abs(v - 0.5))
        inner_sc = {"founder": f_tr[tr_ids].to_numpy(), "cnn": c_tr[tr_ids].to_numpy(),
                    "fusion": fuse(w, f_tr[tr_ids], c_tr[tr_ids]), "fusion_equal": fuse(0.5, f_tr[tr_ids], c_tr[tr_ids])}
        test_sc = {"founder": f_te[te_ids].to_numpy(), "cnn": c_te[te_ids].to_numpy(),
                   "fusion": fuse(w, f_te[te_ids], c_te[te_ids]), "fusion_equal": fuse(0.5, f_te[te_ids], c_te[te_ids])}
        thr = {m: n21.best_threshold(inner_sc[m], y_tr) for m in MODELS}
        for i, h in enumerate(te_ids):
            rows.append({"hospital_id": h, "label": int(y_te[i]), "outer_fold": fo,
                         **{f"{m}_score": float(test_sc[m][i]) for m in MODELS},
                         **{f"{m}_threshold": thr[m] for m in MODELS}})
        for i, h in enumerate(tr_ids):
            oof_rows.append({"hospital_id": h, "label": int(y_tr[i]), "outer_fold": fo,
                             **{f"{m}_score": float(inner_sc[m][i]) for m in MODELS}})
        fm = {m: n21.metrics(y_te, test_sc[m], thr[m]) for m in MODELS}
        folds.append({"outer_fold": fo, "founder": f_set, "cnn": c_set, "weight_founder": w,
                      "inner_auc_by_weight": {str(k): v for k, v in aucs.items()},
                      "inner_auc": {m: roc_auc_score(y_tr, inner_sc[m]) for m in MODELS},
                      "thresholds": thr, **{f"test_{m}_{k}": v for m in fm for k, v in fm[m].items()},
                      "minutes": round((time.time() - t0) / 60, 1)})
        tqdm.write(f"outer {fo + 1}/5: founder lr={f_set['learning_rate']} ep={f_set['epochs']} | cnn "
                   f"lr={c_set['learning_rate']} dropout={c_set['dropout']} ep={c_set['epochs']} | w={w} | "
                   "test macro F1 / bal. acc / AUC  "
                   + "  ".join(f"{m} {fm[m]['macro_f1']:.3f}/{fm[m]['balanced_accuracy']:.3f}/{fm[m]['auc']:.3f}"
                               for m in MODELS) + f"  ({folds[-1]['minutes']} min)")
    df = pd.DataFrame(rows)
    y = df.label.to_numpy()
    pooled = {m: n21.metrics(y, df[f"{m}_score"].to_numpy(), df[f"{m}_threshold"].to_numpy())
              | {"ci95": n21.bootstrap_ci(y, df[f"{m}_score"].to_numpy(), df[f"{m}_threshold"].to_numpy())}
              for m in MODELS}
    return df, pd.DataFrame(oof_rows), folds, pooled | {"patients": int(len(df)), "lvot": int(y.sum())}


def main():
    global FOUNDER_LRS, FOUNDER_EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", type=int, default=5, help="number of seeds of every ensemble (both models)")
    ap.add_argument("--select", choices=["f1", "bacc"], default="f1",
                    help="inner-loop criterion for the settings and all thresholds")
    ap.add_argument("--psd12", action="store_true", help="CNN: PSD branch with one channel per lead")
    ap.add_argument("--augment", action="store_true", help="CNN: training-only time-shift and noise augmentation")
    ap.add_argument("--wd", type=float, default=0.0, help="CNN: Adam weight decay")
    ap.add_argument("--lrs", type=float, nargs="+", default=FOUNDER_LRS, help="ECGFounder: learning rates of the grid")
    ap.add_argument("--epochs", type=int, nargs="+", default=FOUNDER_EPOCHS, help="ECGFounder: epoch checkpoints")
    ap.add_argument("--batch", type=int, default=16, help="ECGFounder batch size")
    ap.add_argument("--trust-checkpoint", action="store_true", help="as script 23")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 outer fold, short grids")
    args = ap.parse_args()
    n21.OPTS.update(psd12=args.psd12, augment=args.augment, wd=args.wd, select=args.select)
    grid_changed = sorted(args.lrs) != sorted(FOUNDER_LRS) or sorted(args.epochs) != sorted(FOUNDER_EPOCHS)
    FOUNDER_LRS, FOUNDER_EPOCHS = list(args.lrs), sorted(args.epochs)
    n23.TRUST_CHECKPOINT = args.trust_checkpoint
    suffix = ("_psd12" if args.psd12 else "") + ("_aug" if args.augment else "") + (f"_wd{args.wd:g}" if args.wd else "") \
        + ("_bacc" if args.select == "bacc" else "") \
        + (f"_lr{'-'.join(f'{v:g}' for v in FOUNDER_LRS)}_ep{'-'.join(map(str, FOUNDER_EPOCHS))}" if grid_changed else "") \
        + f"_ens{args.ensemble}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if not n23.CKPT.exists():
        sys.exit(f"checkpoint not found: {n23.CKPT}\nsee the setup lines at the top of scripts/23_nested_cv_ecgfounder.py")
    OUT.mkdir(parents=True, exist_ok=True)
    rec, patients = n23.load_recordings()
    feats, pid, labels, _ = n21.load_features(device)
    print(f"{len(rec)} recordings; {len(pid)} PVC beats from {len(np.unique(pid))} patients")
    df, oof, folds, pooled = run(rec, feats, pid, labels, patients, device, 1 if args.quick else 5, args.quick,
                                 args.ensemble, args.batch)
    tag = "fusion" + suffix + ("_quick" if args.quick else "")
    df.to_csv(OUT / f"{tag}_patients.csv", index=False)
    oof.to_csv(OUT / f"{tag}_inner_oof.csv", index=False)
    (OUT / f"{tag}_folds.json").write_text(json.dumps(folds, indent=1))
    summary_path = OUT / ("summary_fusion_quick.json" if args.quick else "summary_fusion.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary[tag] = pooled
    summary_path.write_text(json.dumps(summary, indent=1))
    print(f"\n[{tag}] pooled over {pooled['patients']} patients ({pooled['lvot']} LVOT), inner-selected thresholds; "
          f"ECGFounder weight per fold {[f['weight_founder'] for f in folds]}:")
    for m in MODELS:
        r = pooled[m]
        print(f"  {m:12s} macro F1 {r['macro_f1']:.3f} [{r['ci95']['macro_f1'][0]:.3f}, {r['ci95']['macro_f1'][1]:.3f}]"
              f"  bal. acc {r['balanced_accuracy']:.3f}  acc {r['accuracy']:.3f}  sens {r['sensitivity']:.3f}"
              f"  spec {r['specificity']:.3f}  AUC {r['auc']:.3f}  (TP {r['tp']} FN {r['fn']} TN {r['tn']} FP {r['fp']})")
    print("written", OUT)


if __name__ == "__main__":
    main()
