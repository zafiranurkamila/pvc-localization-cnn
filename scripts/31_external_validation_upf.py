"""External validation of the fusion model on the UPF OTVA dataset (mapping-point ECGs, another centre).

Protocol, fixed and committed before any external result was computed:

Development (Zheng database, all 329 patients; the external data are not used):
  As one outer fold of scripts/24_nested_cv_fusion.py with all 329 patients as the outer training part: three
  stratified inner folds (seed 42) select the ECGFounder setting (lr 3e-4/1e-4 x 20/30/45/60 epochs) and the
  PSD + CWT CNN setting (scenario pvc, --psd12 --augment --wd 1e-4); five-seed ensemble inner out-of-fold scores
  give the standardisation, the fusion weight (inner AUC), the decision thresholds (inner macro F1), the referral
  band (target coverage 90 % and the curve 100-70 %) and the Platt calibration; five seeds trained on all 329
  patients are the final models. The clinical-criteria model (scripts/22) is fitted on all 329 patients with the
  same inner folds.

External data (UPF OTVA dataset, patients with SOO_chamber RVOT or LVOT):
  Per patient, up to 30 mapping-point strips (2.5 s, 12 leads, 1000 Hz) drawn with seed 42 over all maps.
  Every strip is resampled to 2000 Hz and band-pass and notch filtered as the Zheng recordings.
  CNN and clinical model: the PVC beats are the wide beats of the largest-gap QRS-width rule on lead II, cut with
  the same 800-ms window; the patient's sinus template is the median of its z-scored narrow beats. Patients without
  a PVC beat get no CNN or clinical score.
  ECGFounder: every strip is resampled to 500 Hz, its baseline removed and z-scored, and zero-padded to 10 s (as the
  Zheng recordings shorter than 10 s); the patient's score is the mean margin over its strips.
  Primary analysis set: the patients with at least one PVC beat (all models available). ECGFounder is also
  reported on all outflow-tract patients.
Reported: macro F1, balanced accuracy, accuracy, sensitivity, specificity, AUC with patient-level bootstrap 95 % CIs;
selective classification of the fusion at the Zheng referral bands; calibration (Brier, slope, intercept).

Run on the PC (GPU), after scripts/29 has extracted the pickle; about 30-60 min.
Usage:
    python scripts/31_external_validation_upf.py data/external/otva_upf/all_points_may_2024.pkl
    python scripts/31_external_validation_upf.py <pkl> --quick      # smoke test (short grids, 2 seeds, 10 patients)
Output: results/external_upf/patients.csv, summary.json (final models cached in data/interim/external_final/)
"""
import argparse
import importlib.util
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import medfilt, resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n21 = _load("nested21", "21_nested_cv_dual_rhythm.py")  # also puts src/ on sys.path
n22 = _load("nested22", "22_nested_cv_clinical_stack.py")
n23 = _load("nested23", "23_nested_cv_ecgfounder.py")
n24 = _load("fusion24", "24_nested_cv_fusion.py")
n29 = _load("inspect29", "29_inspect_external_dataset.py")
n30 = _load("check30", "30_check_external_pvc.py")

from pvc_localization import config  # noqa: E402
from pvc_localization.features.psd import flatten_psd_features  # noqa: E402
from pvc_localization.features.wavelet import extract_cwt_scalogram  # noqa: E402
from pvc_localization.preprocessing.beat_detection import flag_wide_beats  # noqa: E402
from pvc_localization.preprocessing.normalization import zscore_normalize  # noqa: E402

SEED = n21.SEED
OUT = config.RESULTS_DIR / "external_upf"
CACHE = config.INTERIM_DIR / "external_final"
PRE, POST = config.BEAT_PRE_SAMPLES, config.BEAT_POST_SAMPLES
MAX_STRIPS = 30
COVERAGES = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
PRIMARY_COVERAGE = 0.90
W_GRID = n24.W_GRID


# ---------------------------------------------------------------- development on all Zheng patients
def develop(rec, feats, pid, labels, patients, clinical, device, ensemble, quick, batch):
    """Inner out-of-fold ensemble scores and final models of both networks, fitted on all 329 patients."""
    all_p = patients.index.to_numpy()
    inner = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(all_p, patients.to_numpy()))
    f_lrs, f_ckpts = (n24.FOUNDER_LRS[:1], [1, 2]) if quick else (n24.FOUNDER_LRS, n24.FOUNDER_EPOCHS)
    c_ckpts = [3, 5] if quick else n21.EPOCH_CHECKPOINTS
    grid = [dict(zip(n21.GRID, v)) for v in product(*n21.GRID.values())][: (1 if quick else None)]
    bar = tqdm(total=len(inner) * (len(f_lrs) * max(f_ckpts) + len(grid) * max(c_ckpts)), desc="development",
               unit="epoch")

    # ECGFounder: grid on the inner folds, then inner ensemble and final ensemble
    bar.set_postfix_str("founder inner grid")
    oof = {}
    for itr, iva in inner:
        for lr in f_lrs:
            _, snaps = n23.train(rec, patients, list(all_p[itr]), lr, max(f_ckpts), device, eval_ids=list(all_p[iva]),
                                 checkpoints=f_ckpts, batch=batch, progress=bar)
            for ep, s in snaps.items():
                oof.setdefault((lr, ep), []).append(s)
    (f_lr, f_ep), f_c = n24.select(oof, patients)
    bar.total += (len(inner) * (ensemble - 1) + ensemble) * f_ep; bar.refresh()
    bar.set_postfix_str("founder inner ensemble")
    parts = []
    for (itr, iva), seed0 in zip(inner, oof[(f_lr, f_ep)]):
        extra = [n23.margins(n23.train(rec, patients, list(all_p[itr]), f_lr, f_ep, device, seed=SEED + k, batch=batch,
                                       progress=bar)[0], rec, list(all_p[iva]), device) for k in range(1, ensemble)]
        parts.append(pd.concat([seed0] + extra, axis=1).mean(axis=1))
    f_oof = pd.concat(parts)
    bar.set_postfix_str("founder final models")
    f_models = []
    for k in range(ensemble):
        m, _ = n23.train(rec, patients, list(all_p), f_lr, f_ep, device, seed=SEED + k, batch=batch, progress=bar)
        f_models.append({n: t.detach().cpu() for n, t in m.state_dict().items()})
        del m
        if device == "cuda":
            torch.cuda.empty_cache()

    # CNN
    bar.set_postfix_str("cnn inner grid")
    oof = {}
    for itr, iva in inner:
        b_tr, b_va = n24.beats(pid, all_p[itr]), n24.beats(pid, all_p[iva])
        for hp in grid:
            _, snaps = n21.train(feats, b_tr, labels, "pvc", hp["learning_rate"], hp["dropout"], max(c_ckpts), device,
                                 eval_sets=[b_va], checkpoints=c_ckpts, progress=bar)
            for ep, (m_va,) in snaps.items():
                oof.setdefault((hp["learning_rate"], hp["dropout"], ep), []).append(n21.patient_scores(m_va, pid[b_va]))
    (c_lr, c_do, c_ep), c_c = n24.select(oof, patients)
    bar.total += (len(inner) * (ensemble - 1) + ensemble) * c_ep; bar.refresh()
    bar.set_postfix_str("cnn inner ensemble")
    parts = []
    for (itr, iva), seed0 in zip(inner, oof[(c_lr, c_do, c_ep)]):
        b_tr, b_va = n24.beats(pid, all_p[itr]), n24.beats(pid, all_p[iva])
        extra = [n21.patient_scores(n21.margins(n21.train(feats, b_tr, labels, "pvc", c_lr, c_do, c_ep, device,
                                                          seed=SEED + k, progress=bar)[0], feats, b_va, "pvc"),
                                    pid[b_va]) for k in range(1, ensemble)]
        parts.append(pd.concat([seed0] + extra, axis=1).mean(axis=1))
    c_oof = pd.concat(parts)
    bar.set_postfix_str("cnn final models")
    b_all = n24.beats(pid, all_p)
    # state dicts, not modules: the network class lives in a module loaded by path, which pickle cannot import
    c_models = [{n: t.detach().cpu() for n, t in
                 n21.train(feats, b_all, labels, "pvc", c_lr, c_do, c_ep, device, seed=SEED + k,
                           progress=bar)[0].state_dict().items()} for k in range(ensemble)]
    psd0, cwt0 = n21.inputs(feats, b_all[:1], "pvc")
    c_shape = {"psd_len": int(psd0.shape[1]), "cwt_channels": int(cwt0.shape[1]), "dropout": c_do}
    bar.close()

    # clinical-criteria model on all patients, same inner folds
    cols = list(clinical.columns)
    clin_model, clin_C, _, clin_thr = n22.fit_logistic(clinical.reindex(all_p)[cols].to_numpy(), patients.to_numpy(),
                                                       inner)
    return {"f_models": f_models, "c_models": c_models, "c_shape": c_shape, "f_oof": f_oof, "c_oof": c_oof, "clin_model": clin_model,
            "clin_thr": clin_thr, "clin_cols": cols,
            "settings": {"founder": {"learning_rate": f_lr, "epochs": f_ep, "inner_criterion_1seed": f_c},
                         "cnn": {"learning_rate": c_lr, "dropout": c_do, "epochs": c_ep, "inner_criterion_1seed": c_c},
                         "clinical_C": clin_C}}


def decision_rules(dev, patients):
    """Standardisers, fusion weight, thresholds, referral bands and Platt maps from the inner out-of-fold scores."""
    ids = dev["f_oof"].index.intersection(dev["c_oof"].index)
    y = patients[ids].to_numpy()
    zf, zc = n24.standardiser(dev["f_oof"][ids]), n24.standardiser(dev["c_oof"][ids])
    fuse = lambda w, f, c: w * zf(f) + (1 - w) * zc(c)
    aucs = {w: roc_auc_score(y, fuse(w, dev["f_oof"][ids], dev["c_oof"][ids])) for w in W_GRID}
    w = min((v for v in W_GRID if np.isclose(aucs[v], max(aucs.values()))), key=lambda v: abs(v - 0.5))
    inner = {"founder": dev["f_oof"][ids].to_numpy(), "cnn": dev["c_oof"][ids].to_numpy(),
             "fusion": fuse(w, dev["f_oof"][ids], dev["c_oof"][ids])}
    thr = {m: n21.best_threshold(s, y) for m, s in inner.items()}
    delta = {c: (0.0 if c >= 1 else float(np.quantile(np.abs(inner["fusion"] - thr["fusion"]), 1 - c)))
             for c in COVERAGES}
    platt = {m: LogisticRegression(C=1e6, max_iter=1000).fit(s[:, None], y) for m, s in inner.items()}
    return {"fuse": fuse, "w": w, "inner_auc_by_weight": aucs, "thr": thr, "delta": delta, "platt": platt}


# ---------------------------------------------------------------- external data
def external_inputs(pkl, quick):
    """Per outflow-tract patient: label, PVC beats (raw and features), sinus template and ECGFounder windows."""
    data = n29.load_pickle(pkl)
    ot = {h: p for h, p in data.items() if n30.first(p.get("SOO_chamber")) in ("RVOT", "LVOT")}
    if quick:
        ot = dict(list(ot.items())[:10])
    rng = np.random.default_rng(SEED)
    out = {}
    for h, p in tqdm(ot.items(), desc="external patients", unit="patient"):
        pvc, narrow, windows = [], [], []
        for _, _, x in n30.strips_of(p, MAX_STRIPS, rng):
            xf, peaks, widths = n30.process(x)  # (n, 12) at 2000 Hz, filtered; R peaks on lead II
            wide = flag_wide_beats(widths) if len(widths) >= 2 else np.zeros(len(widths), bool)
            for pk, wd in zip(peaks, wide):
                if pk - PRE >= 0 and pk + POST <= len(xf):
                    (pvc if wd else narrow).append(xf[pk - PRE:pk + POST].T)
            x500 = resample_poly(xf, 1, config.SAMPLING_RATE_HZ // n23.FS_OUT, axis=0).T
            x500 = x500 - np.stack([medfilt(lead, int(0.4 * n23.FS_OUT) + 1) for lead in x500])
            windows.append(x500.astype(np.float32))
        sinus = np.median(np.stack([zscore_normalize(b) for b in narrow]), axis=0) if narrow else None
        out[h] = {"label": int(n30.first(p["SOO_chamber"]) == "LVOT"), "sublocation": n30.first(p.get("SOO")),
                  "pvc": pvc, "sinus": sinus, "windows": windows}
    return out


def clinical_features(ext):
    rows = {}
    for h, e in ext.items():
        if not e["pvc"]:
            continue
        if e["sinus"] is not None:
            x, _ = n22.qrs_deflections(e["sinus"].astype(np.float64))
            srs = {lead: n22.rs_ratio(x[n22.L[lead]])[0] for lead in ("V2", "V3")}
        else:
            srs = {"V2": np.nan, "V3": np.nan}
        rows[h] = pd.DataFrame([n22.beat_features(b.astype(np.float64), srs) for b in e["pvc"]]).median()
    return pd.DataFrame(rows).T


@torch.no_grad()
def founder_scores(f_models, ext, device):
    rec = {(h, i): w for h, e in ext.items() for i, w in enumerate(e["windows"])}
    keys = list(rec)
    per = []
    for state in tqdm(f_models, desc="ECGFounder on external strips", unit="model"):
        model, _, _ = n23.founder(device)
        model.load_state_dict(state)
        per.append(n23.margins(model, rec, keys, device))
        del model
    s = pd.concat(per, axis=1).mean(axis=1)
    if not isinstance(s.index, pd.MultiIndex):
        s.index = pd.MultiIndex.from_tuples(list(s.index))
    return s.groupby(level=0).mean()


@torch.no_grad()
def cnn_scores(c_states, c_shape, ext, device):
    hs = [h for h, e in ext.items() if e["pvc"]]
    beats = [(h, zscore_normalize(b)) for h in hs for b in ext[h]["pvc"]]
    t = lambda a: torch.tensor(np.stack(a), dtype=torch.float32, device=device)
    feats = {"psd": t([flatten_psd_features(b) for _, b in tqdm(beats, desc="PSD/CWT of external beats", unit="beat")]),
             "cwt": t([extract_cwt_scalogram(b) for _, b in beats])}
    pid = np.array([h for h, _ in beats], dtype=object)
    idx = np.arange(len(beats))
    margins = []
    for state in c_states:
        model = n21.TwoBranchCNN(c_shape["psd_len"], c_shape["cwt_channels"], c_shape["dropout"],
                                 psd12=n21.OPTS["psd12"]).to(device)
        model.load_state_dict(state)
        margins.append(n21.margins(model, feats, idx, "pvc"))
    m = np.mean(margins, axis=0)
    return pd.Series(m).groupby(pid).mean()


def calibration(y, p):
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit[:, None], y)
    return {"brier": float(brier_score_loss(y, p)), "slope": float(fit.coef_[0][0]),
            "intercept": float(fit.intercept_[0]), "mean_predicted": float(p.mean()), "observed": float(y.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", help="the extracted all_points_may_2024.pkl of the UPF dataset")
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16, help="ECGFounder batch size")
    ap.add_argument("--quick", action="store_true", help="smoke test: short grids, 2 seeds, 10 external patients")
    args = ap.parse_args()
    n21.OPTS.update(psd12=True, augment=True, wd=1e-4, select="f1")
    ensemble = 2 if args.quick else args.ensemble
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cache = CACHE / ("development_quick.pt" if args.quick else f"development_ens{ensemble}.pt")
    rec, patients = n23.load_recordings()
    dev = None
    if cache.exists():
        try:
            dev = torch.load(cache, weights_only=False)
            print("reusing the development of", cache)
        except Exception as e:  # noqa: BLE001  (an interrupted save leaves an unreadable file)
            print(f"cached development unreadable ({e.__class__.__name__}); developing again")
            cache.unlink()
    if dev is None:
        feats, pid, labels, _ = n21.load_features(device)
        clinical = pd.read_csv(n21.OUT / "clinical_features.csv", index_col="hospital_id")
        dev = develop(rec, feats, pid, labels, patients, clinical, device, ensemble, args.quick, args.batch)
        tmp = cache.with_suffix(".tmp")
        torch.save(dev, tmp)
        tmp.replace(cache)  # only a complete file gets the cache name
    rules = decision_rules(dev, patients)
    print("development settings:", dev["settings"], "| fusion weight", rules["w"])

    ext = external_inputs(args.pkl, args.quick)
    f_s = founder_scores(dev["f_models"], ext, device)
    c_s = cnn_scores(dev["c_models"], dev["c_shape"], ext, device)
    clin = clinical_features(ext)
    cl_s = pd.Series(dev["clin_model"].decision_function(clin[dev["clin_cols"]].to_numpy()), index=clin.index)

    df = pd.DataFrame({"label": {h: e["label"] for h, e in ext.items()},
                       "sublocation": {h: e["sublocation"] for h, e in ext.items()},
                       "strips": {h: len(e["windows"]) for h, e in ext.items()},
                       "pvc_beats": {h: len(e["pvc"]) for h, e in ext.items()}})
    df["founder_score"] = f_s; df["cnn_score"] = c_s; df["clinical_score"] = cl_s
    has = df.cnn_score.notna()
    df.loc[has, "fusion_score"] = rules["fuse"](rules["w"], df.loc[has, "founder_score"], df.loc[has, "cnn_score"])
    df.to_csv(OUT / "patients.csv", index_label="patient")

    thr = rules["thr"] | {"clinical": dev["clin_thr"]}
    summary = {"settings": dev["settings"], "fusion_weight": rules["w"], "thresholds": thr,
               "patients_outflow_tract": int(len(df)), "patients_with_pvc": int(has.sum()),
               "lvot_with_pvc": int(df[has].label.sum()), "results": {}, "selective": {}, "calibration": {}}
    sets = {"primary (patients with PVC beats)": df[has], "all outflow-tract patients": df}
    for name, d in sets.items():
        y = d.label.to_numpy()
        res = {}
        for m in ("fusion", "founder", "cnn", "clinical"):
            if d[f"{m}_score"].isna().any():
                continue
            s = d[f"{m}_score"].to_numpy()
            res[m] = n21.metrics(y, s, thr[m]) | {"ci95": n21.bootstrap_ci(y, s, thr[m])}
        summary["results"][name] = res
    d = df[has]; y = d.label.to_numpy(); s = d.fusion_score.to_numpy()
    for c in COVERAGES:
        keep = np.abs(s - thr["fusion"]) >= rules["delta"][c]
        if keep.sum() and len(set(y[keep])) > 1:
            summary["selective"][str(c)] = n21.metrics(y[keep], s[keep], thr["fusion"]) | {
                "coverage": float(keep.mean()), "referred": int((~keep).sum()), "referred_lvot": int(y[~keep].sum()),
                "ci95": n21.bootstrap_ci(y[keep], s[keep], thr["fusion"])}
    for m in ("fusion", "founder", "cnn"):
        p = rules["platt"][m].predict_proba(d[f"{m}_score"].to_numpy()[:, None])[:, 1]
        summary["calibration"][m] = calibration(y, p)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    print(f"\nexternal validation, UPF: {len(df)} outflow-tract patients, {int(has.sum())} with PVC beats "
          f"({int(d.label.sum())} LVOT); {time.time() - t0:.0f} s")
    for name, res in summary["results"].items():
        print(f"  {name}:")
        for m, r in res.items():
            print(f"    {m:8s} macro F1 {r['macro_f1']:.3f} [{r['ci95']['macro_f1'][0]:.3f}, {r['ci95']['macro_f1'][1]:.3f}]"
                  f"  bal. acc {r['balanced_accuracy']:.3f}  acc {r['accuracy']:.3f}  sens {r['sensitivity']:.3f}"
                  f"  spec {r['specificity']:.3f}  AUC {r['auc']:.3f}  (TP {r['tp']} FN {r['fn']} TN {r['tn']} FP {r['fp']})")
    print("  selective classification (fusion, referral bands from the Zheng development):")
    for c, r in summary["selective"].items():
        mark = "  <- primary" if np.isclose(float(c), PRIMARY_COVERAGE) else ""
        print(f"    target {float(c):.0%}: coverage {r['coverage']:.1%} (referred {r['referred']}, LVOT "
              f"{r['referred_lvot']})  macro F1 {r['macro_f1']:.3f}  bal. acc {r['balanced_accuracy']:.3f}  "
              f"AUC {r['auc']:.3f}{mark}")
    print("  calibration:", {m: {k: round(v, 3) for k, v in c.items()} for m, c in summary["calibration"].items()})
    print("written", OUT)


if __name__ == "__main__":
    main()
