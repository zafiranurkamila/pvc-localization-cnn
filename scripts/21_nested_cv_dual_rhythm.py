"""Nested patient-level cross-validation of the PVC-only and the dual-rhythm PSD + CWT models (journal paper).

Primary evaluation of the journal paper. All 329 patients are used:
  outer loop  5 stratified patient folds; every patient is predicted exactly once by a model that never saw it.
  inner loop  3 stratified patient folds inside the outer training part; selects the learning rate, the dropout,
              the number of epochs and the patient-level decision threshold. The outer test fold is never used
              for any choice.

Scenarios (same protocol, same folds):
  pvc   PSD (12 x 32) + CWT (12 x 64 x 200) of each PVC beat (the earlier fusion model)
  dual  as pvc, plus the PSD and CWT of the patient's median sinus beat (scripts/20_sinus_beats.py):
        PSD branch input 768 = PVC || sinus, CWT branch input 24 channels = PVC (12) + sinus (12)
  diff  as dual, but the second half is the difference PVC - sinus (how the PVC deviates from the patient's own
        conducted beat) instead of the sinus representation itself

Options (all applied identically in the inner and the outer loop; never chosen on the outer test fold):
  --psd12    PSD branch as a 1D CNN over the 32 frequency bins with one input channel per lead (and per sinus/diff
             half), instead of one 384-value sequence; kernels no longer cross lead boundaries
  --augment  training-only augmentation: random circular time shift of the scalogram by up to +-5 columns (+-20 ms)
             and Gaussian noise (sd 0.05) on both inputs
  --wd W     Adam weight decay
  --select   inner-loop criterion for the setting and the patient threshold: f1 (macro F1, default) or bacc
             (balanced accuracy)

Ensemble (--ensemble K): hyperparameters, epochs and threshold are selected in the inner loop with one seed; the
outer model is then trained with K seeds and the patient margins of the K models are averaged.

Patient decision: mean logit margin (logit_LVOT - logit_RVOT) over the patient's PVC beats, compared with the
threshold chosen in the inner loop (and, for reference, with 0).

Run on the PC (GPU). Needs data/processed, data/interim/patient_index.csv, data/interim/sinus/ and the feature cache.
Usage:
    python scripts/21_nested_cv_dual_rhythm.py                 # both scenarios
    python scripts/21_nested_cv_dual_rhythm.py --scenario dual --quick   # smoke test (1 outer fold, few epochs)
    python scripts/21_nested_cv_dual_rhythm.py --scenario pvc diff --ensemble 5
Output: results/nested_cv/<scenario>_patients.csv, <scenario>_folds.json, summary.json
"""
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config  # noqa: E402
from pvc_localization.data.dataset import PVCBeatsDataset  # noqa: E402
from pvc_localization.data.loader import load_patient_index  # noqa: E402
from pvc_localization.features.psd import flatten_psd_features  # noqa: E402
from pvc_localization.features.wavelet import extract_cwt_scalogram  # noqa: E402
from pvc_localization.models.branches import PSDBlock, WaveletBlock  # noqa: E402
from pvc_localization.preprocessing.normalization import zscore_normalize  # noqa: E402
from pvc_localization.training.trainer import CACHE_DIR, set_seed  # noqa: E402

OUT = config.RESULTS_DIR / "nested_cv"
SINUS = config.INTERIM_DIR / "sinus"
GRID = {"learning_rate": [1e-3, 3e-4], "dropout": [0.3, 0.5]}
EPOCH_CHECKPOINTS = [20, 35, 50]
BATCH = 64
SEED = config.RANDOM_SEED


class PSDLeadBlock(nn.Module):
    """1D CNN over frequency with one channel per lead: input (batch, channels * 32) reshaped to (batch, channels, 32)."""

    def __init__(self, channels: int, n_filters: int = 64, n_bins: int = 32):
        super().__init__()
        self.channels, self.n_bins = channels, n_bins
        self.net = nn.Sequential(nn.BatchNorm1d(channels),
                                 nn.Conv1d(channels, n_filters // 2, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(n_filters // 2, n_filters, 3, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool1d(16))
        self.output_size = n_filters * 16

    def forward(self, x):
        return self.net(x.view(x.size(0), self.channels, self.n_bins)).flatten(1)


class TwoBranchCNN(nn.Module):
    """PSD branch (1D) + CWT branch (2D) + classifier, as the earlier fusion model; input sizes set per scenario."""

    def __init__(self, psd_len: int, cwt_channels: int, dropout: float, n_filters: int = 64, hidden: int = 64,
                 psd12: bool = False):
        super().__init__()
        self.psd = PSDLeadBlock(psd_len // 32, n_filters) if psd12 else PSDBlock(input_size=psd_len, hidden=n_filters)
        self.cwt = WaveletBlock(n_leads=cwt_channels, hidden=n_filters)
        self.head = nn.Sequential(nn.Linear(self.psd.output_size + self.cwt.output_size, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 2))

    def forward(self, psd, cwt):
        return self.head(torch.cat([self.psd(psd), self.cwt(cwt)], dim=1))


def load_features(device):
    """PVC-beat features from the cache, plus the sinus-template features of every patient (on the device)."""
    index = load_patient_index()
    ids = index.hospital_id.tolist()
    ds = PVCBeatsDataset(ids, ["psd", "wavelet"], cache_dir=CACHE_DIR)
    psd, cwt, pid, lab = [], [], [], []
    for i in tqdm(range(len(ds)), desc="PVC beat features", unit="beat"):
        item = ds[i]
        psd.append(item["psd"].numpy()); cwt.append(item["wavelet"].numpy())
        pid.append(int(item["hospital_id"])); lab.append(int(item["label"]))
    sin_psd, sin_cwt = {}, {}
    for h in tqdm(ids, desc="sinus template features", unit="patient"):
        tpl = zscore_normalize(np.load(SINUS / f"{h}.npy").astype(np.float64))
        sin_psd[h] = flatten_psd_features(tpl).astype(np.float32)
        sin_cwt[h] = extract_cwt_scalogram(tpl).astype(np.float32)
    pid = np.array(pid)
    t = lambda a: torch.tensor(np.stack(a), dtype=torch.float32, device=device)
    feats = {
        "psd": t(psd), "cwt": t(cwt),
        "sin_psd": t([sin_psd[h] for h in pid]), "sin_cwt": t([sin_cwt[h] for h in pid]),
    }
    patients = index.set_index("hospital_id").label.map(config.LABEL_TO_INT)
    return feats, pid, np.array(lab), patients


def inputs(feats, idx, scenario):
    idx = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=feats["psd"].device)
    psd, cwt = feats["psd"][idx], feats["cwt"][idx]
    if scenario == "pvc":
        return psd, cwt
    if scenario == "diff":
        return (torch.cat([psd, psd - feats["sin_psd"][idx]], dim=1),
                torch.cat([cwt, cwt - feats["sin_cwt"][idx]], dim=1))
    return (torch.cat([psd, feats["sin_psd"][idx]], dim=1), torch.cat([cwt, feats["sin_cwt"][idx]], dim=1))


OPTS = {"psd12": False, "augment": False, "wd": 0.0, "select": "f1"}


def augment(psd, cwt, gen):
    """Training-only augmentation on the device: per-sample circular time shift of the scalogram and Gaussian noise."""
    b, t = cwt.shape[0], cwt.shape[-1]
    shift = torch.randint(-5, 6, (b,), generator=gen).to(cwt.device)
    idx = (torch.arange(t, device=cwt.device)[None, :] - shift[:, None]) % t
    cwt = torch.gather(cwt, -1, idx[:, None, None, :].expand_as(cwt))
    return psd + 0.05 * torch.randn_like(psd), cwt + 0.05 * torch.randn_like(cwt)


def train(feats, beat_idx, labels, scenario, lr, dropout, epochs, device, eval_sets=(), checkpoints=(), seed=SEED,
          progress=None):
    """Train one model; returns it and, for every checkpoint epoch, the beat margins of every eval set.
    `progress` (optional tqdm bar) is advanced by one per epoch."""
    set_seed(seed)
    psd0, cwt0 = inputs(feats, beat_idx[:1], scenario)
    model = TwoBranchCNN(psd0.shape[1], cwt0.shape[1], dropout, psd12=OPTS["psd12"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=OPTS["wd"])
    y = torch.tensor(labels[beat_idx], device=device)
    counts = np.bincount(labels[beat_idx], minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(beat_idx) / (2 * np.maximum(counts, 1)),
                                                      dtype=torch.float32, device=device))
    g = torch.Generator(device="cpu").manual_seed(seed)
    snaps = {}
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(beat_idx), generator=g)
        for k in range(0, len(perm) - BATCH + 1, BATCH):  # drop_last, as in the earlier trainer
            b = perm[k:k + BATCH]
            psd, cwt = inputs(feats, beat_idx[b.numpy()], scenario)
            if OPTS["augment"]:
                psd, cwt = augment(psd, cwt, g)
            opt.zero_grad()
            loss_fn(model(psd, cwt), y[b.to(device)]).backward()
            opt.step()
        if ep in checkpoints:
            snaps[ep] = [margins(model, feats, s, scenario) for s in eval_sets]
        if progress is not None:
            progress.update(1)
    return model, snaps


@torch.no_grad()
def margins(model, feats, beat_idx, scenario):
    model.eval()
    out = []
    for k in range(0, len(beat_idx), 256):
        psd, cwt = inputs(feats, beat_idx[k:k + 256], scenario)
        z = model(psd, cwt)
        out.append((z[:, 1] - z[:, 0]).double().cpu().numpy())
    return np.concatenate(out)


def patient_scores(beat_margins, beat_pid):
    s = pd.Series(beat_margins).groupby(beat_pid).mean()
    return s


def criterion_at(scores, y, thr):
    if OPTS["select"] == "bacc":
        pred = (scores > thr).astype(int)
        return (np.mean(pred[y == 1] == 1) + np.mean(pred[y == 0] == 0)) / 2
    return f1_score(y, (scores > thr).astype(int), average="macro", zero_division=0)


def best_threshold(scores, y):
    cand = np.unique(np.concatenate([[0.0], (np.sort(scores)[1:] + np.sort(scores)[:-1]) / 2]))
    f = [criterion_at(scores, y, c) for c in cand]
    best = np.max(f)
    # among ties, the threshold closest to 0 (the uncalibrated default)
    return float(cand[np.argmin(np.where(np.isclose(f, best), np.abs(cand), np.inf))])


def metrics(y, score, thr):
    pred = (score > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {"macro_f1": f1_score(y, pred, average="macro", zero_division=0), "accuracy": (tp + tn) / len(y),
            "balanced_accuracy": (sens + spec) / 2, "sensitivity": sens, "specificity": spec,
            "auc": roc_auc_score(y, score) if len(set(y)) > 1 else float("nan"),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def bootstrap_ci(y, score, thr, n=2000):
    rng = np.random.default_rng(SEED)
    vals = {k: [] for k in ("macro_f1", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "auc")}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        m = metrics(y[i], score[i], thr[i] if np.ndim(thr) else thr)
        for k in vals:
            vals[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def run_scenario(scenario, feats, pid, labels, patients, device, n_outer, quick, ensemble=1):
    all_p = patients.index.to_numpy(); all_y = patients.to_numpy()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, folds = [], []
    ckpts = [3, 5] if quick else EPOCH_CHECKPOINTS
    grid = [dict(zip(GRID, v)) for v in product(*GRID.values())][: (1 if quick else None)]
    for fo, (tr_p, te_p) in enumerate(outer.split(all_p, all_y)):
        if fo >= n_outer:
            break
        t0 = time.time()
        tr_pat, te_pat = all_p[tr_p], all_p[te_p]
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fo)
        # inner loop: pooled out-of-fold patient scores for every (hyperparameters, epochs)
        oof = {}
        for fi, (itr, iva) in enumerate(inner.split(tr_pat, patients[tr_pat].to_numpy())):
            b_tr = np.where(np.isin(pid, tr_pat[itr]))[0]; b_va = np.where(np.isin(pid, tr_pat[iva]))[0]
            for hp in grid:
                _, snaps = train(feats, b_tr, labels, scenario, hp["learning_rate"], hp["dropout"], max(ckpts),
                                 device, eval_sets=[b_va], checkpoints=ckpts)
                for ep, (m_va,) in snaps.items():
                    oof.setdefault((hp["learning_rate"], hp["dropout"], ep), []).append(patient_scores(m_va, pid[b_va]))
        best, best_f1, best_thr = None, -1, 0.0
        for key, parts in oof.items():
            s = pd.concat(parts); y = patients[s.index].to_numpy()
            thr = best_threshold(s.to_numpy(), y)
            f = criterion_at(s.to_numpy(), y, thr)
            if f > best_f1:
                best, best_f1, best_thr = key, f, thr
        lr, dropout, ep = best
        # outer fit on the whole outer training part with the chosen setting; predict the outer test fold once
        b_tr = np.where(np.isin(pid, tr_pat))[0]; b_te = np.where(np.isin(pid, te_pat))[0]
        m_te = np.mean([margins(train(feats, b_tr, labels, scenario, lr, dropout, ep, device, seed=SEED + k)[0],
                                feats, b_te, scenario) for k in range(ensemble)], axis=0)
        s_te = patient_scores(m_te, pid[b_te])
        for h, sc in s_te.items():
            rows.append({"hospital_id": h, "label": int(patients[h]), "outer_fold": fo, "score": float(sc),
                         "threshold": best_thr, "pred_thr": int(sc > best_thr), "pred_zero": int(sc > 0),
                         "n_beats": int((pid == h).sum())})
        y_te = patients[s_te.index].to_numpy()
        fm = metrics(y_te, s_te.to_numpy(), best_thr)
        folds.append({"outer_fold": fo, "learning_rate": lr, "dropout": dropout, "epochs": ep, "threshold": best_thr,
                      "inner_macro_f1": best_f1, "test_patients": int(len(te_pat)), "test_lvot": int(y_te.sum()),
                      **{f"test_{k}": v for k, v in fm.items()}, "minutes": round((time.time() - t0) / 60, 1)})
        tqdm.write(f"[{scenario}] outer {fo + 1}/5: lr={lr} dropout={dropout} epochs={ep} thr={best_thr:.3f} "
                   f"inner F1={best_f1:.3f} | test macro F1={fm['macro_f1']:.3f} acc={fm['accuracy']:.3f} "
                   f"AUC={fm['auc']:.3f} ({folds[-1]['minutes']} min)")
    df = pd.DataFrame(rows)
    y, sc, th = df.label.to_numpy(), df.score.to_numpy(), df.threshold.to_numpy()
    pooled = {"threshold_inner": metrics(y, sc, th) | {"ci95": bootstrap_ci(y, sc, th)},
              "threshold_zero": metrics(y, sc, 0.0) | {"ci95": bootstrap_ci(y, sc, 0.0)},
              "patients": int(len(df)), "lvot": int(y.sum())}
    return df, folds, pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", nargs="+", choices=["pvc", "dual", "diff"], default=["pvc", "dual"])
    ap.add_argument("--ensemble", type=int, default=1, help="number of seeds averaged in the outer model")
    ap.add_argument("--psd12", action="store_true", help="PSD branch with one channel per lead")
    ap.add_argument("--augment", action="store_true", help="training-only time-shift and noise augmentation")
    ap.add_argument("--wd", type=float, default=0.0, help="Adam weight decay")
    ap.add_argument("--select", choices=["f1", "bacc"], default="f1",
                    help="inner-loop criterion for the setting and the threshold: macro F1 or balanced accuracy")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 outer fold, 1 setting, 5 epochs")
    args = ap.parse_args()
    OPTS.update(psd12=args.psd12, augment=args.augment, wd=args.wd, select=args.select)
    suffix = ("_psd12" if args.psd12 else "") + ("_aug" if args.augment else "") + (f"_wd{args.wd:g}" if args.wd else "") \
        + ("_bacc" if args.select == "bacc" else "")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    OUT.mkdir(parents=True, exist_ok=True)
    feats, pid, labels, patients = load_features(device)
    print(f"{len(pid)} PVC beats from {len(np.unique(pid))} patients")
    summary_path = OUT / ("summary_quick.json" if args.quick else "summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for sc in args.scenario:
        df, folds, pooled = run_scenario(sc, feats, pid, labels, patients, device, 1 if args.quick else 5, args.quick,
                                         args.ensemble)
        name = sc + suffix + ("" if args.ensemble == 1 else f"_ens{args.ensemble}")
        tag = f"{name}_quick" if args.quick else name
        df.to_csv(OUT / f"{tag}_patients.csv", index=False)
        (OUT / f"{tag}_folds.json").write_text(json.dumps(folds, indent=1))
        summary[name] = pooled
        summary_path.write_text(json.dumps(summary, indent=1))
        m = pooled["threshold_inner"]
        print(f"\n[{name}] pooled over {pooled['patients']} patients ({pooled['lvot']} LVOT), inner-selected threshold: "
              f"macro F1 {m['macro_f1']:.3f} {m['ci95']['macro_f1']}, accuracy {m['accuracy']:.3f}, "
              f"balanced acc {m['balanced_accuracy']:.3f}, sens {m['sensitivity']:.3f}, spec {m['specificity']:.3f}, "
              f"AUC {m['auc']:.3f}  (TP {m['tp']} FN {m['fn']} TN {m['tn']} FP {m['fp']})")
        z = pooled["threshold_zero"]
        print(f"[{name}] threshold 0 (reference): macro F1 {z['macro_f1']:.3f}, accuracy {z['accuracy']:.3f}, "
              f"balanced acc {z['balanced_accuracy']:.3f}, sens {z['sensitivity']:.3f}, spec {z['specificity']:.3f}")
    print("written", OUT)


if __name__ == "__main__":
    main()
