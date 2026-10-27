"""Nested patient-level CV of the ECGFounder foundation model fine-tuned for PVC origin (LVOT vs RVOT).

ECGFounder (Li et al., arXiv 2410.04133; MIT licence) is a Net1D network pretrained on over 10 million 12-lead ECGs of
the Harvard-Emory ECG Database. Its 12-lead checkpoint takes 12 x 5000 samples (10 s at 500 Hz). Here its last layer
is replaced by a 2-class layer and the whole network is fine-tuned on the PVC recordings.

Same outer and inner patient folds, criterion (--select), threshold rule, ensemble and pooled metrics as
scripts/21_nested_cv_dual_rhythm.py, so the results are directly comparable with the PSD + CWT CNN.

Input per patient: the whole recording, band-pass 0.5-40 Hz and 50 Hz notch at 2000 Hz (as the rest of the project),
resampled to 500 Hz, baseline removed with a 0.4 s median filter and z-scored over the window (the ECGFounder
preprocessing). Training takes a random 10 s crop of the recording every epoch; prediction averages the logit margin
of the first, middle and last 10 s crops. Recordings shorter than 10 s are zero-padded at the end (as ECGFounder).

Inner loop (outer test fold never used): learning rate x number of epochs, and the patient threshold.

Setup on the PC (once):
    git clone https://github.com/PKUDigitalHealth/ECGFounder.git external/ECGFounder
    curl.exe -L -o external/ECGFounder/checkpoint/12_lead_ECGFounder.pth --create-dirs
        https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth
Usage:
    python scripts/23_nested_cv_ecgfounder.py --check          # load checkpoint and data, one forward pass
    python scripts/23_nested_cv_ecgfounder.py --quick          # smoke test (1 outer fold, 1 lr, 2 epochs)
    python scripts/23_nested_cv_ecgfounder.py --ensemble 3
Output: results/nested_cv/founder<suffix>_patients.csv, founder<suffix>_folds.json, summary.json (key founder...)
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import medfilt, resample_poly
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

_spec = importlib.util.spec_from_file_location("nested21", Path(__file__).resolve().parent / "21_nested_cv_dual_rhythm.py")
n21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n21)  # also puts src/ on sys.path

from pvc_localization import config  # noqa: E402
from pvc_localization.data.loader import load_ecg, load_patient_index  # noqa: E402
from pvc_localization.preprocessing.filtering import clean_signal  # noqa: E402
from pvc_localization.training.trainer import set_seed  # noqa: E402

FOUNDER_DIR = config.PROJECT_ROOT / "external" / "ECGFounder"
CKPT = FOUNDER_DIR / "checkpoint" / "12_lead_ECGFounder.pth"
CACHE = config.INTERIM_DIR / "founder500"
OUT = n21.OUT
SEED = n21.SEED
FS_OUT = 500
WIN = 10 * FS_OUT
LR_GRID = [1e-4, 3e-5]
EPOCH_CHECKPOINTS = [10, 20, 30]


# ---------------------------------------------------------------- data
def preprocess(h, path):
    """Filtered recording at 500 Hz with the baseline removed, (12, n) float32; cached per patient."""
    f = CACHE / f"{h}.npy"
    if f.exists():
        return np.load(f)
    ecg = clean_signal(load_ecg(path).to_numpy())  # (n, 12) at 2000 Hz
    x = resample_poly(ecg, 1, config.SAMPLING_RATE_HZ // FS_OUT, axis=0).T
    x = x - np.stack([medfilt(lead, int(0.4 * FS_OUT) + 1) for lead in x])
    x = x.astype(np.float32)
    np.save(f, x)
    return x


def load_recordings():
    CACHE.mkdir(parents=True, exist_ok=True)
    index = load_patient_index()
    rec = {int(r.hospital_id): preprocess(int(r.hospital_id), r.processed_path)
           for r in tqdm(index.itertuples(), total=len(index), desc="recordings (500 Hz)", unit="patient")}
    patients = index.set_index("hospital_id").label.map(config.LABEL_TO_INT)
    return rec, patients


def crop(x, start):
    w = x[:, start:start + WIN]
    if w.shape[1] < WIN:
        w = np.pad(w, ((0, 0), (0, WIN - w.shape[1])))
    return (w - w.mean()) / (w.std() + 1e-8)


def eval_crops(x):
    n = x.shape[1]
    starts = sorted({0, max((n - WIN) // 2, 0), max(n - WIN, 0)})
    return np.stack([crop(x, s) for s in starts])


# ---------------------------------------------------------------- model
TRUST_CHECKPOINT = False


def load_checkpoint():
    """torch.load with weights_only=True; the checkpoint also stores numpy scalars (training metadata such as the
    validation AUROC), so those numpy types are allow-listed. --trust-checkpoint falls back to a full unpickling."""
    if TRUST_CHECKPOINT:
        return torch.load(CKPT, map_location="cpu", weights_only=False)
    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import scalar
    safe = [(scalar, "numpy.core.multiarray.scalar"), (scalar, "numpy._core.multiarray.scalar"), np.dtype]
    safe += [getattr(np.dtypes, n) for n in ("Float64DType", "Float32DType", "Int64DType", "Int32DType", "BoolDType")
             if hasattr(np, "dtypes") and hasattr(np.dtypes, n)]
    with torch.serialization.safe_globals(safe):
        return torch.load(CKPT, map_location="cpu", weights_only=True)


def founder(device):
    sys.path.insert(0, str(FOUNDER_DIR))
    from net1d import Net1D  # noqa: E402  (ECGFounder repository)
    model = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                  m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16, verbose=False,
                  use_bn=False, use_do=False, n_classes=150)
    state = load_checkpoint()["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items() if not k.removeprefix("module.").startswith("dense.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.dense = nn.Linear(model.dense.in_features, 2)
    return model.to(device), missing, unexpected


def train(rec, patients, train_ids, lr, epochs, device, eval_ids=(), checkpoints=(), seed=SEED, batch=16,
          progress=None):
    """Fine-tune one model; returns it and, for every checkpoint epoch, the patient margins of eval_ids."""
    set_seed(seed)
    model, _, _ = founder(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    y_all = patients[train_ids].to_numpy()
    counts = np.bincount(y_all, minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(y_all) / (2 * np.maximum(counts, 1)), dtype=torch.float32,
                                                      device=device))
    rng = np.random.default_rng(seed)
    snaps = {}
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_ids))
        for k in range(0, len(order) - batch + 1, batch):  # drop_last
            ids = [train_ids[i] for i in order[k:k + batch]]
            xb = np.stack([crop(rec[h], int(rng.integers(0, max(rec[h].shape[1] - WIN, 0) + 1))) for h in ids])
            xb = torch.tensor(xb, device=device); yb = torch.tensor(y_all[order[k:k + batch]], dtype=torch.long, device=device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if ep in checkpoints:
            snaps[ep] = margins(model, rec, eval_ids, device)
        if progress is not None:
            progress.update(1)
    return model, snaps


@torch.no_grad()
def margins(model, rec, ids, device):
    """Patient logit margin (LVOT - RVOT), averaged over the evaluation crops."""
    model.eval()
    out = {}
    for h in ids:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            z = model(torch.tensor(eval_crops(rec[h]), device=device)).float()
        out[h] = float((z[:, 1] - z[:, 0]).mean())
    return pd.Series(out)


# ---------------------------------------------------------------- nested CV
def run(rec, patients, device, n_outer, quick, ensemble, batch):
    all_p = patients.index.to_numpy(); all_y = patients.to_numpy()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    ckpts = [1, 2] if quick else EPOCH_CHECKPOINTS
    lrs = LR_GRID[:1] if quick else LR_GRID
    rows, folds = [], []
    for fo, (tr_p, te_p) in enumerate(outer.split(all_p, all_y)):
        if fo >= n_outer:
            break
        t0 = time.time()
        tr_pat, te_pat = all_p[tr_p], all_p[te_p]
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fo)
        bar = tqdm(total=3 * len(lrs) * max(ckpts), desc=f"outer {fo + 1}/5", unit="epoch", leave=False)
        bar.set_postfix_str("1/2 inner grid")
        oof = {}
        for itr, iva in inner.split(tr_pat, patients[tr_pat].to_numpy()):
            for lr in lrs:
                _, snaps = train(rec, patients, list(tr_pat[itr]), lr, max(ckpts), device, eval_ids=list(tr_pat[iva]),
                                 checkpoints=ckpts, batch=batch, progress=bar)
                for ep, s in snaps.items():
                    oof.setdefault((lr, ep), []).append(s)
        best, best_c, best_thr = None, -1, 0.0
        for key, parts in oof.items():
            s = pd.concat(parts); y = patients[s.index].to_numpy()
            thr = n21.best_threshold(s.to_numpy(), y)
            c = n21.criterion_at(s.to_numpy(), y, thr)
            if c > best_c:
                best, best_c, best_thr = key, c, thr
        lr, ep = best
        bar.total += ensemble * ep
        bar.refresh()
        bar.set_postfix_str("2/2 outer ensemble")
        s_te = pd.concat([margins(train(rec, patients, list(tr_pat), lr, ep, device, seed=SEED + k, batch=batch,
                                        progress=bar)[0], rec, list(te_pat), device) for k in range(ensemble)],
                         axis=1).mean(axis=1)
        bar.close()
        y_te = patients[s_te.index].to_numpy()
        for h, sc in s_te.items():
            rows.append({"hospital_id": h, "label": int(patients[h]), "outer_fold": fo, "score": float(sc),
                         "threshold": best_thr, "pred_thr": int(sc > best_thr), "pred_zero": int(sc > 0)})
        fm = n21.metrics(y_te, s_te.to_numpy(), best_thr)
        folds.append({"outer_fold": fo, "learning_rate": lr, "epochs": ep, "threshold": best_thr,
                      "inner_criterion": best_c, "test_patients": int(len(te_pat)), "test_lvot": int(y_te.sum()),
                      **{f"test_{k}": v for k, v in fm.items()}, "minutes": round((time.time() - t0) / 60, 1)})
        tqdm.write(f"[founder] outer {fo + 1}/5: lr={lr} epochs={ep} thr={best_thr:.3f} inner={best_c:.3f} | "
                   f"test macro F1={fm['macro_f1']:.3f} bal. acc={fm['balanced_accuracy']:.3f} AUC={fm['auc']:.3f} "
                   f"({folds[-1]['minutes']} min)")
    df = pd.DataFrame(rows)
    y, sc, th = df.label.to_numpy(), df.score.to_numpy(), df.threshold.to_numpy()
    pooled = {"threshold_inner": n21.metrics(y, sc, th) | {"ci95": n21.bootstrap_ci(y, sc, th)},
              "threshold_zero": n21.metrics(y, sc, 0.0) | {"ci95": n21.bootstrap_ci(y, sc, 0.0)},
              "patients": int(len(df)), "lvot": int(y.sum())}
    return df, folds, pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", type=int, default=3, help="number of seeds of the outer model")
    ap.add_argument("--select", choices=["f1", "bacc"], default="f1", help="inner-loop criterion")
    ap.add_argument("--batch", type=int, default=16, help="batch size (lower it if the GPU runs out of memory)")
    ap.add_argument("--check", action="store_true", help="load the checkpoint and the data, one forward pass")
    ap.add_argument("--trust-checkpoint", action="store_true",
                    help="load the checkpoint with weights_only=False (only if the allow-listed load still fails)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 outer fold, 1 learning rate, 2 epochs")
    args = ap.parse_args()
    n21.OPTS.update(select=args.select)
    global TRUST_CHECKPOINT
    TRUST_CHECKPOINT = args.trust_checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if not CKPT.exists():
        sys.exit(f"checkpoint not found: {CKPT}\nsee the setup lines at the top of this script")
    rec, patients = load_recordings()
    lengths = np.array([x.shape[1] / FS_OUT for x in rec.values()])
    print(f"{len(rec)} recordings, duration median {np.median(lengths):.1f} s (min {lengths.min():.1f}, "
          f"max {lengths.max():.1f}); {int((lengths < 10).sum())} shorter than 10 s are zero-padded")
    if args.check:
        model, missing, unexpected = founder(device)
        print(f"ECGFounder loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters, "
              f"{len(missing)} missing keys {missing[:4]}, {len(unexpected)} unexpected keys {unexpected[:4]}")
        h = next(iter(rec))
        with torch.no_grad():
            z = model(torch.tensor(eval_crops(rec[h]), device=device))
        print("forward pass on", eval_crops(rec[h]).shape, "->", tuple(z.shape))
        if device == "cuda":
            print(f"GPU memory in use {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    df, folds, pooled = run(rec, patients, device, 1 if args.quick else 5, args.quick, args.ensemble, args.batch)
    name = "founder" + ("_bacc" if args.select == "bacc" else "") + f"_ens{args.ensemble}"
    tag = f"{name}_quick" if args.quick else name
    df.to_csv(OUT / f"{tag}_patients.csv", index=False)
    (OUT / f"{tag}_folds.json").write_text(json.dumps(folds, indent=1))
    summary_path = OUT / ("summary_quick.json" if args.quick else "summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary[name] = pooled
    summary_path.write_text(json.dumps(summary, indent=1))
    m, z = pooled["threshold_inner"], pooled["threshold_zero"]
    print(f"\n[{tag}] pooled over {pooled['patients']} patients ({pooled['lvot']} LVOT), inner-selected threshold: "
          f"macro F1 {m['macro_f1']:.3f} [{m['ci95']['macro_f1'][0]:.3f}, {m['ci95']['macro_f1'][1]:.3f}], "
          f"accuracy {m['accuracy']:.3f}, balanced acc {m['balanced_accuracy']:.3f}, sens {m['sensitivity']:.3f}, "
          f"spec {m['specificity']:.3f}, AUC {m['auc']:.3f}  (TP {m['tp']} FN {m['fn']} TN {m['tn']} FP {m['fp']})")
    print(f"[{tag}] threshold 0 (reference): macro F1 {z['macro_f1']:.3f}, accuracy {z['accuracy']:.3f}, "
          f"balanced acc {z['balanced_accuracy']:.3f}")
    print("written", OUT)


if __name__ == "__main__":
    main()
