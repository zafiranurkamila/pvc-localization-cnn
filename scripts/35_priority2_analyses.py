"""Analyses for the second round of reviewer points (6-10); no model is trained, only saved predictions are used.

1. Direct clinical rules (point 6), with the cut-offs of the original publications, applied without any fitting:
     V2 transition ratio >= 0.60 -> LVOT (Betensky et al. 2011)
     V2S/V3R index      <= 1.5  -> LVOT (Yoshida et al. 2014)
   The criteria are the patient medians of scripts/22 (results/nested_cv/clinical_features.csv; stored as log1p, so
   expm1 recovers the ratio). Reported on all patients with a measurable criterion and on the patients whose PVC
   transition is in V3, with patient-bootstrap 95 % CI, next to the fusion on the same patients (mean of 3 splits).
2. Errors and referrals per 100 patients against the target coverage (point 7): LVOT called RVOT, RVOT called LVOT.
3. Error analysis of the first cohort (point 8): fusion errors without referral and referrals at the 90 % target, by
   recorded sub-location and by sex, summed over the three splits; characteristics of the referred patients.
4. Recording length (point 9): fusion and ECGFounder in the recordings shorter than 10 s (zero-padded for ECGFounder)
   and in the others.
5. Second cohort (point 8): errors by grouped sub-location for (i) the unchanged model and (iii) local model adaptation.

Usage:  python scripts/35_priority2_analyses.py
Output: results/priority2/*.csv and summary.json
"""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
NCV = ROOT / "results" / "nested_cv"
UPF = ROOT / "results" / "external_upf"
OUT = ROOT / "results" / "priority2"
COVERAGES = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
N_BOOT = 2000
SEED = 42

_spec = importlib.util.spec_from_file_location("n33", Path(__file__).resolve().parent / "33_fusion_vs_founder.py")
n33 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n33)


def metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max((y == 1).sum(), 1); spec = tn / max((y == 0).sum(), 1)
    return {"n": len(y), "lvot": int(y.sum()), "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "balanced_accuracy": (sens + spec) / 2, "sensitivity": sens, "specificity": spec,
            "accuracy": float((pred == y).mean())}


def boot_ci(y, pred, rng):
    v = {"macro_f1": [], "balanced_accuracy": []}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        m = metrics(y[i], pred[i])
        v["macro_f1"].append(m["macro_f1"]); v["balanced_accuracy"].append(m["balanced_accuracy"])
    return {f"{k}_ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] for k, a in v.items()}


def load_splits():
    """Outer-test predictions of the three splits, with the 90 % referral mask of the fusion."""
    out = {}
    for split, run in n33.RUNS.items():
        test = pd.read_csv(NCV / f"{run}_patients.csv")
        inner = pd.read_csv(NCV / f"{run}_inner_oof.csv")
        for m in ("fusion", "founder", "cnn"):
            test[f"{m}_pred"] = (test[f"{m}_score"] > test[f"{m}_threshold"]).astype(int)
        for c in COVERAGES:
            test[f"keep_{c}"] = n33.classified(test, inner, "fusion", c)
        out[split] = test
    return out


def clinical_rules(splits, rng):
    f = pd.read_csv(NCV / "clinical_features.csv").set_index("hospital_id")
    lab = splits["42"].set_index("hospital_id").label
    f = f.join(lab, how="inner")
    v2tr, v2s3r = np.expm1(f.tz_ratio_V2), np.expm1(f.v2s_v3r)
    rules = {"V2 transition ratio >= 0.60": (v2tr >= 0.60).astype(int).where(v2tr.notna()),
             "V2S/V3R index <= 1.5": (v2s3r <= 1.5).astype(int).where(v2s3r.notna())}
    v3 = (f.transition > 2) & (f.transition <= 3)
    rows = []
    for name, pred in rules.items():
        for group, mask in {"all measurable": pred.notna(), "PVC transition in V3": pred.notna() & v3}.items():
            ids = f.index[mask]
            y, p = f.label[ids].to_numpy(), pred[ids].to_numpy().astype(int)
            r = {"rule": name, "group": group, **metrics(y, p), **boot_ci(y, p, rng)}
            fus = [metrics(t.set_index("hospital_id").label[ids].to_numpy(),
                           t.set_index("hospital_id").fusion_pred[ids].to_numpy()) for t in splits.values()]
            r["fusion_macro_f1"] = float(np.mean([m["macro_f1"] for m in fus]))
            r["fusion_balanced_accuracy"] = float(np.mean([m["balanced_accuracy"] for m in fus]))
            rows.append(r)
    return pd.DataFrame(rows), {"no sinus beat (V2TR not measurable)": int(v2tr.isna().sum()),
                                "PVC transition in V3": int(v3.sum())}


def tradeoff(splits):
    rows = []
    for split, t in splits.items():
        y, p, n = t.label.to_numpy(), t.fusion_pred.to_numpy(), len(t)
        for c in COVERAGES:
            k = t[f"keep_{c}"].to_numpy()
            rows.append({"split": split, "target_coverage": c, "referred_per_100": 100 * (~k).sum() / n,
                         "lvot_called_rvot_per_100": 100 * ((y == 1) & (p == 0) & k).sum() / n,
                         "rvot_called_lvot_per_100": 100 * ((y == 0) & (p == 1) & k).sum() / n,
                         "errors_per_100": 100 * ((y != p) & k).sum() / n})
    return pd.DataFrame(rows).groupby("target_coverage").mean(numeric_only=True).reset_index() \
        .sort_values("target_coverage", ascending=False)


def error_analysis(splits):
    diag = pd.read_excel(ROOT / "data" / "external" / "Diagnosis.xlsx").set_index("HospitalID")
    stats = pd.read_csv(ROOT / "results" / "dataset_stats_per_patient.csv").set_index("hospital_id")
    feats = pd.read_csv(NCV / "clinical_features.csv").set_index("hospital_id")
    rows = []
    for split, t in splits.items():
        t = t.set_index("hospital_id")
        t["sublocation"] = diag.Sublocation.reindex(t.index).fillna("not recorded")
        t["sex"] = diag.Gender.reindex(t.index)
        t["error"] = t.fusion_pred != t.label
        t["referred"] = ~t["keep_0.9"]
        t["error_classified"] = t.error & ~t.referred
        t["split"] = split
        rows.append(t)
    a = pd.concat(rows)
    side = a.label.map({0: "RVOT", 1: "LVOT"})
    by = lambda key: a.groupby([side.rename("side"), key]).agg(
        predictions=("error", "size"), errors_no_referral=("error", "sum"), referred_90=("referred", "sum"),
        errors_after_referral_90=("error_classified", "sum")).reset_index()
    sub, sex = by("sublocation"), by("sex")
    for d in (sub, sex):
        d["error_rate_no_referral"] = d.errors_no_referral / d.predictions
        d["referral_rate_90"] = d.referred_90 / d.predictions
    # characteristics of referred vs classified (pooled predictions)
    a = a.join(stats[["duration_s", "pvc_beats_kept"]]).join(feats[["qrs_ms", "transition"]])
    char = a.groupby("referred").agg(predictions=("label", "size"), lvot_share=("label", "mean"),
                                     male_share=("sex", lambda s: (s == "male").mean()),
                                     median_duration_s=("duration_s", "median"),
                                     median_pvc_beats=("pvc_beats_kept", "median"),
                                     median_qrs_ms=("qrs_ms", "median"),
                                     median_transition=("transition", "median")).reset_index()
    # recording length (point 9)
    short = a.duration_s < 10
    rec = []
    for split in splits:
        s = a.split == split
        for g, m in {"< 10 s": short & s, ">= 10 s": ~short & s}.items():
            y = a.label[m].to_numpy()
            for model in ("fusion", "founder", "cnn"):
                rec.append({"split": split, "recording": g, "model": model,
                            **metrics(y, a[f"{model}_pred"][m].to_numpy())})
    rec = pd.DataFrame(rec).groupby(["recording", "model"]).agg(
        patients=("n", "mean"), lvot=("lvot", "mean"), macro_f1=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"), balanced_accuracy=("balanced_accuracy", "mean")).reset_index()
    return sub, sex, char, rec


def upf_group(label, text):
    t = str(text).lower()
    if label == 1:
        if re.search(r"intersen|intersinus|commiss", t):
            return "LVOT: L-R cusp commissure"
        if re.search(r"izdo|lcc|left coronary", t):
            return "LVOT: left coronary cusp"
        if re.search(r"dcho|derecho|rcc", t):
            return "LVOT: right coronary cusp"
        if "summit" in t:
            return "LVOT: summit"
        return "LVOT: other or not specified"
    if re.search(r"sept|s.eto", t):
        return "RVOT: septal"
    if re.search(r"libre|lateral", t):
        return "RVOT: free wall"
    return "RVOT: other or not specified"


def upf_errors():
    pat = pd.read_csv(UPF / "patients.csv")
    thr = json.loads((UPF / "summary.json").read_text())["thresholds"]["fusion"]
    pat = pat[pat.pvc_beats > 0].copy()
    pat["group"] = [upf_group(l, s) for l, s in zip(pat.label, pat.sublocation)]
    pat["error_unchanged"] = (pat.fusion_score > thr).astype(int) != pat.label
    mc = pd.read_csv(UPF / "multicentre_patients.csv")
    mc = mc[mc.arm == "B_two_centre"].assign(patient=lambda d: d.patient.str[1:]).set_index("patient")
    pred = (mc.fusion_score > mc.fusion_thr).astype(int)
    pat["error_adapted"] = (pred.reindex(pat.patient).to_numpy() != pat.label.to_numpy())
    pat["referred_adapted_90"] = ((mc.fusion_score - mc.fusion_thr).abs() < mc["delta_0.9"]).reindex(pat.patient).to_numpy()
    g = pat.groupby("group").agg(patients=("patient", "size"), errors_unchanged=("error_unchanged", "sum"),
                                 errors_adapted=("error_adapted", "sum"),
                                 referred_adapted_90=("referred_adapted_90", "sum")).reset_index()
    by_side = pat.groupby("label").agg(patients=("patient", "size"), errors_unchanged=("error_unchanged", "sum"),
                                       errors_adapted=("error_adapted", "sum")).reset_index()
    return g, by_side


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    splits = load_splits()
    rules, rule_info = clinical_rules(splits, rng)
    trade = tradeoff(splits)
    sub, sex, char, rec = error_analysis(splits)
    upf_sub, upf_side = upf_errors()
    for name, d in {"clinical_rules": rules, "tradeoff": trade, "errors_sublocation": sub, "errors_sex": sex,
                    "referred_characteristics": char, "recording_length": rec, "upf_errors_sublocation": upf_sub,
                    "upf_errors_side": upf_side}.items():
        d.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"clinical_rule_counts": rule_info,
                                                  "clinical_rules": rules.to_dict(orient="records")}, indent=1))
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("== direct clinical rules", rule_info); print(rules.round(3).to_string(index=False))
    print("\n== errors and referrals per 100 patients (fusion, mean of 3 splits)"); print(trade.round(2).to_string(index=False))
    print("\n== first cohort, by sub-location (3 splits pooled)"); print(sub.round(3).to_string(index=False))
    print("\n== by sex"); print(sex.round(3).to_string(index=False))
    print("\n== referred (True) vs classified at the 90 % target"); print(char.round(3).to_string(index=False))
    print("\n== recording length"); print(rec.round(3).to_string(index=False))
    print("\n== second cohort, fusion errors by sub-location"); print(upf_sub.to_string(index=False))
    print(upf_side.to_string(index=False))
    print("written", OUT)


if __name__ == "__main__":
    main()
