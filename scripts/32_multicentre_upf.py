"""Two-centre development (Zheng + UPF) and local threshold recalibration, evaluated on held-out UPF patients.

Follows the external validation of scripts/31 (models from the Zheng database only, AUC kept but thresholds shifted).
Protocol, fixed and committed before any result of this script was computed:

Patients: the 106 UPF outflow-tract patients with at least one PVC beat (the primary set of scripts/31), with the
same strips and beats (scripts/31 external_inputs, seed 42). Outer split: 5 stratified folds of these patients
(seed 42); every UPF patient is tested exactly once, by models and rules that never saw it.

Arm A, local recalibration (no retraining): the scores of scripts/31 (results/external_upf/patients.csv, models
  from Zheng only, fusion weight and standardisation unchanged). On the four training folds of UPF patients, the
  thresholds (macro F1, ties -> 0), the referral band (target coverages 100-70 %, primary 90 %) and the Platt maps
  are re-estimated; they are applied to the test fold.

Arm B, two-centre development: per outer fold, the training part is all 329 Zheng patients plus the UPF training
  patients. Model settings are fixed to those scripts/31 chose on Zheng (results/external_upf/summary.json); no grid
  search. Three inner folds of the training part (stratified by side x centre, seed 42 + fold) give five-seed
  ensemble out-of-fold scores of ECGFounder and of the PSD + CWT CNN; five seeds trained on the whole training part
  score the test fold. ECGFounder trains on one randomly drawn UPF strip per patient and epoch (zero-padded to 10 s,
  as at scoring), so every patient of either centre has the same weight; the CNN trains on all PVC beats. The
  clinical-criteria model is refitted on the training part (C by inner AUC).
  Decision rules, primary: from the inner out-of-fold scores of the UPF training patients only (target centre):
  standardisation, fusion weight (inner AUC, ties -> 0.5), thresholds, referral band, Platt maps.
  Secondary: the same rules from the out-of-fold scores of the whole training part.

Reported, on the pooled held-out predictions of all 106 patients: macro F1, balanced accuracy, accuracy,
sensitivity, specificity and AUC (of the score minus its fold threshold) with patient bootstrap 95 % CIs; selective
classification of the fusion; calibration; and paired bootstrap differences of the fusion against the zero-shot
fusion of scripts/31. The zero-shot result of scripts/31 stays the external validation; this script answers how
much local data recover.

Run on the PC (GPU), after scripts/31; several hours. Finished outer folds are cached, so an interrupted run resumes.
Usage:
    python scripts/32_multicentre_upf.py data/external/otva_upf/all_points_may_2024.pkl
    python scripts/32_multicentre_upf.py <pkl> --quick      # smoke test (1 outer fold, 1 seed, 2-3 epochs)
Output: results/external_upf/multicentre_patients.csv, multicentre_summary.json
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n31 = _load("external31", "31_external_validation_upf.py")  # also loads scripts 21-24, 29, 30
n21, n22, n23, n24 = n31.n21, n31.n22, n31.n23, n31.n24

from pvc_localization import config  # noqa: E402
from pvc_localization.features.psd import flatten_psd_features  # noqa: E402
from pvc_localization.features.wavelet import extract_cwt_scalogram  # noqa: E402
from pvc_localization.preprocessing.normalization import zscore_normalize  # noqa: E402

SEED = n21.SEED
OUT = n31.OUT
CACHE = config.INTERIM_DIR / "external_multicentre"
COVERAGES = n31.COVERAGES
PRIMARY_COVERAGE = n31.PRIMARY_COVERAGE
MODELS = ("fusion", "founder", "cnn", "clinical")


def save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_cached(path):
    if path.exists():
        try:
            return torch.load(path, weights_only=False)
        except Exception as e:  # noqa: BLE001  (an interrupted save leaves an unreadable file)
            print(f"{path.name} unreadable ({e.__class__.__name__}); computing again")
            path.unlink()
    return None


# ---------------------------------------------------------------- data of both centres
def upf_inputs(pkl):
    """Strips, PVC-beat features and clinical features of the UPF patients with PVC beats (ids 'u<id>')."""
    f = CACHE / "upf_inputs.pt"
    out = load_cached(f)
    if out is not None:
        return out
    ext = n31.external_inputs(pkl, False)
    clin = n31.clinical_features(ext)
    hs = [h for h, e in ext.items() if e["pvc"]]
    psd, cwt, pid = [], [], []
    for h in tqdm(hs, desc="PSD/CWT of UPF beats", unit="patient"):
        for b in ext[h]["pvc"]:
            zb = zscore_normalize(b)
            psd.append(flatten_psd_features(zb).astype(np.float32))
            cwt.append(extract_cwt_scalogram(zb).astype(np.float32))
            pid.append(f"u{h}")
    out = {"labels": pd.Series({f"u{h}": ext[h]["label"] for h in hs}),
           "windows": {f"u{h}": ext[h]["windows"] for h in hs},
           "psd": np.stack(psd), "cwt": np.stack(cwt), "pid": np.array(pid, dtype=object),
           "clinical": clin.loc[hs].set_axis([f"u{h}" for h in hs])}
    save(out, f)
    return out


def both_centres(upf, device):
    rec, patients = n23.load_recordings()
    rec = {f"z{h}": x for h, x in rec.items()} | upf["windows"]
    y = pd.concat([patients.set_axis([f"z{h}" for h in patients.index]), upf["labels"]]).astype(int)
    feats, pid, labels, _ = n21.load_features(device)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    feats = {"psd": torch.cat([feats["psd"], t(upf["psd"])]), "cwt": torch.cat([feats["cwt"], t(upf["cwt"])])}
    pid = np.concatenate([np.array([f"z{p}" for p in pid], dtype=object), upf["pid"]])
    labels = np.concatenate([labels, upf["labels"][upf["pid"]].to_numpy()])
    clinical = pd.read_csv(n21.OUT / "clinical_features.csv", index_col="hospital_id")
    clinical = pd.concat([clinical.set_axis([f"z{h}" for h in clinical.index]), upf["clinical"][clinical.columns]])
    return rec, y, feats, pid, labels, clinical


# ---------------------------------------------------------------- ECGFounder on strips of either centre
def draw(x, rng):
    """One 10-s training crop; a UPF patient (list of strips) contributes one random strip."""
    if isinstance(x, list):
        x = x[int(rng.integers(len(x)))]
    return n23.crop(x, int(rng.integers(0, max(x.shape[1] - n23.WIN, 0) + 1)))


def founder_train(rec, y, train_ids, lr, epochs, device, seed, batch, progress):
    """scripts/23 train() with the strip draw above."""
    n23.set_seed(seed)
    model, _, _ = n23.founder(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = (torch.amp.GradScaler("cuda", enabled=device == "cuda") if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=device == "cuda"))
    y_all = y[train_ids].to_numpy()
    counts = np.bincount(y_all, minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(y_all) / (2 * np.maximum(counts, 1)), dtype=torch.float32,
                                                      device=device))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_ids))
        for k in range(0, len(order) - batch + 1, batch):  # drop_last
            xb = torch.tensor(np.stack([draw(rec[train_ids[i]], rng) for i in order[k:k + batch]]), device=device)
            yb = torch.tensor(y_all[order[k:k + batch]], dtype=torch.long, device=device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        progress.update(1)
    return model


@torch.no_grad()
def founder_margins(model, rec, ids, device):
    """Patient margin: mean over the evaluation crops of the recording, or of all strips of a UPF patient."""
    model.eval()
    out = {}
    for h in ids:
        xs = rec[h] if isinstance(rec[h], list) else [rec[h]]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            z = model(torch.tensor(np.concatenate([n23.eval_crops(x) for x in xs]), device=device)).float()
        out[h] = float((z[:, 1] - z[:, 0]).mean())
    return pd.Series(out)


def founder_ensemble(rec, y, tr, ev, s, device, ensemble, batch, bar):
    per = []
    for k in range(ensemble):
        m = founder_train(rec, y, list(tr), s["learning_rate"], s["epochs"], device, SEED + k, batch, bar)
        per.append(founder_margins(m, rec, list(ev), device))
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    return pd.concat(per, axis=1).mean(axis=1)


def cnn_ensemble(feats, pid, labels, tr, ev, s, device, ensemble, bar):
    b_tr, b_ev = n24.beats(pid, tr), n24.beats(pid, ev)
    per = [n21.patient_scores(n21.margins(n21.train(feats, b_tr, labels, "pvc", s["learning_rate"], s["dropout"],
                                                    s["epochs"], device, seed=SEED + k, progress=bar)[0],
                                          feats, b_ev, "pvc"), pid[b_ev]) for k in range(ensemble)]
    return pd.concat(per, axis=1).mean(axis=1)


# ---------------------------------------------------------------- decision rules
def rules(f, c, cl, y):
    """scripts/31 decision rules on the given out-of-fold scores, plus the clinical threshold and Platt map."""
    r = n31.decision_rules({"f_oof": f, "c_oof": c}, y)
    ids = cl.index
    r["thr"]["clinical"] = n21.best_threshold(cl.to_numpy(), y[ids].to_numpy())
    r["platt"]["clinical"] = n31.LogisticRegression(C=1e6, max_iter=1000).fit(cl.to_numpy()[:, None], y[ids].to_numpy())
    return r


def apply(r, f, c, cl):
    """Test scores, thresholds, referral bands and probabilities of every model under rule set r."""
    s = {"founder": f.to_numpy(), "cnn": c.to_numpy(), "fusion": r["fuse"](r["w"], f, c), "clinical": cl.to_numpy()}
    out = pd.DataFrame(index=f.index)
    for m in MODELS:
        out[f"{m}_score"] = s[m]
        out[f"{m}_thr"] = r["thr"][m]
        out[f"{m}_prob"] = r["platt"][m].predict_proba(np.asarray(s[m])[:, None])[:, 1]
    for cov in COVERAGES:
        out[f"delta_{cov}"] = r["delta"][cov]
    out["fusion_weight"] = r["w"]
    return out


# ---------------------------------------------------------------- arms
def arm_recalibration(zero, outer, upf_ids):
    """Arm A: scripts/31 scores, thresholds and bands re-estimated on the UPF training folds."""
    parts = []
    w = float(json.loads((OUT / "summary.json").read_text())["fusion_weight"])
    for fo, (tr, te) in enumerate(outer):
        d_tr, d_te = zero.loc[upf_ids[tr]], zero.loc[upf_ids[te]]
        y_tr = d_tr.label
        out = pd.DataFrame(index=d_te.index)
        for m in MODELS:
            t = n21.best_threshold(d_tr[f"{m}_score"].to_numpy(), y_tr.to_numpy())
            platt = n31.LogisticRegression(C=1e6, max_iter=1000).fit(d_tr[f"{m}_score"].to_numpy()[:, None], y_tr)
            out[f"{m}_score"] = d_te[f"{m}_score"]; out[f"{m}_thr"] = t
            out[f"{m}_prob"] = platt.predict_proba(d_te[f"{m}_score"].to_numpy()[:, None])[:, 1]
            if m == "fusion":
                for cov in COVERAGES:
                    out[f"delta_{cov}"] = 0.0 if cov >= 1 else float(
                        np.quantile(np.abs(d_tr.fusion_score - t), 1 - cov))
        out["fusion_weight"] = w; out["fold"] = fo
        parts.append(out)
    return pd.concat(parts)


def arm_multicentre(rec, y, feats, pid, labels, clinical, outer, upf_ids, settings, device, ensemble, batch, quick):
    """Arm B: two-centre development per outer fold; returns the test predictions under both rule sets."""
    zheng = np.array([h for h in y.index if h.startswith("z")], dtype=object)
    f_s = dict(settings["founder"]); c_s = dict(settings["cnn"])
    if quick:
        f_s["epochs"], c_s["epochs"] = 2, 3
    per_fold = (3 * ensemble + ensemble) * (f_s["epochs"] + c_s["epochs"])
    todo = [fo for fo in range(len(outer)) if not (CACHE / f"fold{fo}{'_quick' if quick else ''}.pt").exists()]
    bar = tqdm(total=per_fold * len(todo), desc="two-centre development", unit="epoch")
    res = {"primary": [], "secondary": []}
    for fo, (tr, te) in enumerate(outer):
        f_cache = CACHE / f"fold{fo}{'_quick' if quick else ''}.pt"
        done = load_cached(f_cache)
        if done is None:
            t0 = time.time()
            train = np.concatenate([zheng, upf_ids[tr]]); test = upf_ids[te]
            strata = y[train].to_numpy() + 2 * np.array([h.startswith("u") for h in train])
            inner = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fo).split(train, strata))
            f_oof, c_oof, cl_oof = [], [], []
            for k, (itr, iva) in enumerate(inner):
                bar.set_postfix_str(f"fold {fo + 1}/{len(outer)} inner {k + 1}/3")
                f_oof.append(founder_ensemble(rec, y, train[itr], train[iva], f_s, device, ensemble, batch, bar))
                c_oof.append(cnn_ensemble(feats, pid, labels, train[itr], train[iva], c_s, device, ensemble, bar))
            bar.set_postfix_str(f"fold {fo + 1}/{len(outer)} final models")
            f_te = founder_ensemble(rec, y, train, test, f_s, device, ensemble, batch, bar)
            c_te = cnn_ensemble(feats, pid, labels, train, test, c_s, device, ensemble, bar)
            X = clinical.reindex(train).to_numpy(); yt = y[train].to_numpy()
            clin_model, clin_C, _, _ = n22.fit_logistic(X, yt, inner)
            oof = np.zeros(len(train))
            for itr, iva in inner:
                oof[iva] = n22.logistic(clin_C).fit(X[itr], yt[itr]).decision_function(X[iva])
            cl_oof = pd.Series(oof, index=train)
            cl_te = pd.Series(clin_model.decision_function(clinical.reindex(test).to_numpy()), index=test)
            done = {"f_oof": pd.concat(f_oof), "c_oof": pd.concat(c_oof), "cl_oof": cl_oof,
                    "f_te": f_te, "c_te": c_te, "cl_te": cl_te, "clin_C": clin_C, "seconds": time.time() - t0}
            save(done, f_cache)
        for name, keep in (("primary", lambda s: s[[h for h in s.index if h.startswith("u")]]),
                           ("secondary", lambda s: s)):
            r = rules(keep(done["f_oof"]), keep(done["c_oof"]), keep(done["cl_oof"]), y)
            out = apply(r, done["f_te"], done["c_te"][done["f_te"].index], done["cl_te"][done["f_te"].index])
            out["fold"] = fo
            res[name].append(out)
        if quick:
            break
    bar.close()
    return {k: pd.concat(v) for k, v in res.items()}


# ---------------------------------------------------------------- evaluation of pooled held-out predictions
def evaluate(pred, y):
    yy = y[pred.index].to_numpy()
    res = {"models": {}, "selective": {}, "calibration": {}}
    for m in MODELS:
        d = (pred[f"{m}_score"] - pred[f"{m}_thr"]).to_numpy()
        res["models"][m] = n21.metrics(yy, d, 0.0) | {"ci95": n21.bootstrap_ci(yy, d, 0.0)}
        res["calibration"][m] = n31.calibration(yy, pred[f"{m}_prob"].to_numpy())
    d = (pred.fusion_score - pred.fusion_thr).to_numpy()
    for cov in COVERAGES:
        keep = np.abs(d) >= pred[f"delta_{cov}"].to_numpy()
        if keep.sum() and len(set(yy[keep])) > 1:
            res["selective"][str(cov)] = n21.metrics(yy[keep], d[keep], 0.0) | {
                "coverage": float(keep.mean()), "referred": int((~keep).sum()), "referred_lvot": int(yy[~keep].sum()),
                "ci95": n21.bootstrap_ci(yy[keep], d[keep], 0.0)}
    return res


def paired_difference(d_a, d_b, y, n=2000):
    """Bootstrap difference (a - b) of macro F1, balanced accuracy and AUC of two decision values on the same patients."""
    rng = np.random.default_rng(SEED)
    keys = ("macro_f1", "balanced_accuracy", "auc")
    point = {k: n21.metrics(y, d_a, 0.0)[k] - n21.metrics(y, d_b, 0.0)[k] for k in keys}
    vals = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        a, b = n21.metrics(y[i], d_a[i], 0.0), n21.metrics(y[i], d_b[i], 0.0)
        for k in keys:
            vals[k].append(a[k] - b[k])
    return {k: {"difference": float(point[k]), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                "p_two_sided": float(min(1.0, 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0))))}
            for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", help="the extracted all_points_may_2024.pkl of the UPF dataset")
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16, help="ECGFounder batch size")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 outer fold, 1 seed, 2-3 epochs")
    args = ap.parse_args()
    n21.OPTS.update(psd12=True, augment=True, wd=1e-4, select="f1")
    ensemble = 1 if args.quick else args.ensemble
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    s31 = json.loads((OUT / "summary.json").read_text())
    zero = pd.read_csv(OUT / "patients.csv", index_col="patient")
    zero = zero[zero.cnn_score.notna()].set_axis([f"u{h}" for h in zero[zero.cnn_score.notna()].index])
    upf = upf_inputs(args.pkl)
    upf_ids = upf["labels"].index.to_numpy()
    assert set(upf_ids) == set(zero.index), "UPF patients differ from scripts/31"
    outer = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(upf_ids, upf["labels"].to_numpy()))
    print(f"{len(upf_ids)} UPF patients ({int(upf['labels'].sum())} LVOT) in 5 outer folds; settings from scripts/31:",
          s31["settings"])

    rec, y, feats, pid, labels, clinical = both_centres(upf, device)
    arms = {"A_recalibration": arm_recalibration(zero, outer, upf_ids)}
    mc = arm_multicentre(rec, y, feats, pid, labels, clinical, outer, upf_ids, s31["settings"], device, ensemble,
                         args.batch, args.quick)
    arms["B_two_centre"] = mc["primary"]; arms["B_two_centre_pooled_rules"] = mc["secondary"]
    if args.quick:
        arms["A_recalibration"] = arms["A_recalibration"].loc[mc["primary"].index]
    tested = arms["B_two_centre"].index
    zero_d = {m: (zero.loc[tested, f"{m}_score"] - s31["thresholds"][m]).to_numpy() for m in MODELS}

    rows = []
    for name, p in arms.items():
        rows.append(p.assign(arm=name, label=y[p.index]))
    pd.concat(rows).to_csv(OUT / f"multicentre_patients{'_quick' if args.quick else ''}.csv", index_label="patient")

    yy = y[tested].to_numpy()
    summary = {"protocol": "scripts/32 docstring", "patients": int(len(tested)), "lvot": int(yy.sum()),
               "settings": s31["settings"], "ensemble": ensemble,
               "zero_shot": {m: n21.metrics(yy, zero_d[m], 0.0) for m in MODELS}, "arms": {}, "vs_zero_shot": {}}
    for name, p in arms.items():
        summary["arms"][name] = evaluate(p.loc[tested], y) | {"fusion_weights": p.groupby("fold").fusion_weight.first().tolist()}
        d = (p.loc[tested].fusion_score - p.loc[tested].fusion_thr).to_numpy()
        summary["vs_zero_shot"][name] = paired_difference(d, zero_d["fusion"], yy)
    (OUT / f"multicentre_summary{'_quick' if args.quick else ''}.json").write_text(json.dumps(summary, indent=1, default=float))

    print(f"\nheld-out UPF patients: {len(tested)} ({int(yy.sum())} LVOT); {time.time() - t0:.0f} s")
    print("  zero-shot (scripts/31): " + "  ".join(f"{m} F1 {r['macro_f1']:.3f} AUC {r['auc']:.3f}"
                                                for m, r in summary["zero_shot"].items()))
    for name, r in summary["arms"].items():
        print(f"  {name}:")
        for m, x in r["models"].items():
            print(f"    {m:8s} macro F1 {x['macro_f1']:.3f} [{x['ci95']['macro_f1'][0]:.3f}, {x['ci95']['macro_f1'][1]:.3f}]"
                  f"  bal. acc {x['balanced_accuracy']:.3f}  sens {x['sensitivity']:.3f}  spec {x['specificity']:.3f}"
                  f"  AUC {x['auc']:.3f}")
        for c, x in r["selective"].items():
            mark = "  <- primary" if np.isclose(float(c), PRIMARY_COVERAGE) else ""
            print(f"    referral target {float(c):.0%}: coverage {x['coverage']:.1%}  macro F1 {x['macro_f1']:.3f}"
                  f"  bal. acc {x['balanced_accuracy']:.3f}{mark}")
        v = summary["vs_zero_shot"][name]
        print("    fusion vs zero-shot: " + "  ".join(f"{k} {x['difference']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
                                                   for k, x in v.items()))
    print("written", OUT)


if __name__ == "__main__":
    main()
