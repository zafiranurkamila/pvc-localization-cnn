"""Quality control of the PVC extraction on the UPF strips and its effect on the results (reviewer round 2, point 1).
No model is trained; the final models of scripts/31 are only applied again.

Problem. The largest-gap rule (flag_wide_beats) always splits the beats of a strip into two groups, so a strip
without a PVC still yields "wide" beats (its widest narrow beats).

Rule (fixed before looking at its effect, no UPF label used). A flagged beat is kept only if its width index is at
least W_MIN, the 1st percentile of the patient-median width index of the flagged beats in the first cohort
(results/pvc_check/auto_check_per_patient.csv, 329 patients): beats narrower than almost every PVC of the
development data are not PVCs. The same rule applies to every strip and both sides.

Reported:
  1. extraction before and after QC: strips with flagged beats, flagged beats, beats kept, patients with >= 1 kept
     beat, per side; the width index of the flagged beats;
  2. the 20 patients of the visual review (scripts/34, same strips): flagged beats kept by QC next to the reviewed
     counts (results/external_upf/visual_review/review_form.csv);
  3. sensitivity of (i) the unchanged models: CNN, fusion and ECG-criteria scores recomputed from the kept beats
     with the cached final models of scripts/31 (ECGFounder uses the whole strip and is unchanged);
  4. (i) and (iii) local model adaptation (saved predictions of scripts/32) in the patients with and without a
     kept beat; the models of (iii) were not stored, so their CNN input cannot be re-extracted without retraining.

Usage (on the PC, after scripts/31 and 32):
    python scripts/38_upf_extraction_qc.py data/external/otva_upf/all_points_may_2024.pkl
Output: results/external_upf/qc_patients.csv, qc_visual_review.csv, qc_summary.json
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n31 = _load("external31", "31_external_validation_upf.py")  # also loads scripts 21-24, 29, 30
n34 = _load("review34", "34_upf_visual_review.py")
n21, n23, n29, n30 = n31.n21, n31.n23, n31.n29, n31.n30

from pvc_localization import config  # noqa: E402
from pvc_localization.data.loader import load_patient_index  # noqa: E402
from pvc_localization.preprocessing.beat_detection import flag_wide_beats  # noqa: E402
from pvc_localization.preprocessing.normalization import zscore_normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UPF = ROOT / "results" / "external_upf"
FIRST = pd.read_csv(ROOT / "results" / "pvc_check" / "auto_check_per_patient.csv")
W_MIN = float(np.percentile(FIRST.wide_width_ms, 1))
PRE, POST = n31.PRE, n31.POST


def external_inputs_qc(data):
    """scripts/31 external_inputs (same strips, same order of random draws) with the QC rule applied."""
    ot = {h: p for h, p in data.items() if n30.first(p.get("SOO_chamber")) in ("RVOT", "LVOT")}
    rng = np.random.default_rng(n31.SEED)
    ext, stats = {}, []
    for h, p in tqdm(ot.items(), desc="UPF strips with QC", unit="patient"):
        pvc, narrow = [], []
        s = {"patient": h, "label": int(n30.first(p["SOO_chamber"]) == "LVOT"), "strips": 0, "strips_flagged": 0,
             "strips_kept": 0, "flagged": 0, "kept": 0, "flagged_widths": []}
        for _, _, x in n30.strips_of(p, n31.MAX_STRIPS, rng):
            xf, peaks, widths = n30.process(x)
            wide = flag_wide_beats(widths) if len(widths) >= 2 else np.zeros(len(widths), bool)
            ok = wide & (widths >= W_MIN)
            s["strips"] += 1
            s["strips_flagged"] += int(wide.any()); s["strips_kept"] += int(ok.any())
            for pk, wd, k, w in zip(peaks, wide, ok, widths):
                if pk - PRE >= 0 and pk + POST <= len(xf):
                    if wd:
                        s["flagged"] += 1; s["flagged_widths"].append(float(w))
                    if k:
                        s["kept"] += 1
                    # removed beats are not PVCs by the rule, so they join the sinus template as other beats
                    (pvc if k else narrow).append(xf[pk - PRE:pk + POST].T)
        sinus = np.median(np.stack([zscore_normalize(b) for b in narrow]), axis=0) if narrow else None
        ext[h] = {"label": s["label"], "pvc": pvc, "sinus": sinus}
        stats.append(s)
    return ext, pd.DataFrame(stats)


def visual_review_check(data):
    """Flagged and QC-kept beats of the 20 reviewed patients, on the strips of scripts/34."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for h in n34.sample_patients():
        flagged = kept = 0
        for _, _, x in n30.strips_of(data[h], n34.STRIPS, rng):
            _, _, widths = n30.process(x)
            wide = flag_wide_beats(widths) if len(widths) >= 2 else np.zeros(len(widths), bool)
            flagged += int(wide.sum()); kept += int((wide & (widths >= W_MIN)).sum())
        rows.append({"patient": h, "flagged_qc_input": flagged, "kept_by_qc": kept})
    form = pd.read_csv(UPF / "visual_review" / "review_form.csv")
    return form.merge(pd.DataFrame(rows), on="patient", how="left")


def evaluate(y, d):
    m = n21.metrics(y, d, 0.0)
    return {k: float(m[k]) for k in ("macro_f1", "balanced_accuracy", "sensitivity", "specificity", "auc")} | {
        "patients": int(len(y)), "lvot": int(y.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", help="the extracted all_points_may_2024.pkl")
    args = ap.parse_args()
    n21.OPTS.update(psd12=True, augment=True, wd=1e-4, select="f1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"QC rule: flagged beat kept if width index >= {W_MIN:.1f} ms (1st percentile, first cohort)")
    data = n29.load_pickle(args.pkl)

    ext, st = external_inputs_qc(data)
    st["median_flagged_width"] = st.flagged_widths.apply(lambda v: float(np.median(v)) if v else np.nan)
    st = st.drop(columns="flagged_widths")
    side = st.label.map({0: "RVOT", 1: "LVOT"})
    extraction = {}
    for name, g in [("all", st), ("RVOT", st[side == "RVOT"]), ("LVOT", st[side == "LVOT"])]:
        extraction[name] = {"patients": int(len(g)), "strips": int(g.strips.sum()),
                            "strips_with_flagged_beat": int(g.strips_flagged.sum()),
                            "strips_with_kept_beat": int(g.strips_kept.sum()),
                            "flagged_beats": int(g.flagged.sum()), "kept_beats": int(g.kept.sum()),
                            "patients_with_flagged_beat": int((g.flagged > 0).sum()),
                            "patients_with_kept_beat": int((g.kept > 0).sum())}

    review = visual_review_check(data)
    review.to_csv(UPF / "qc_visual_review.csv", index=False)

    # 3. (i) unchanged models with the QC beats
    dev = torch.load(n31.CACHE / "development_ens5.pt", weights_only=False)
    patients = load_patient_index().set_index("hospital_id").label.map(config.LABEL_TO_INT)
    rules = n31.decision_rules(dev, patients)
    orig = pd.read_csv(UPF / "patients.csv", index_col="patient")
    qc_ids = [h for h, e in ext.items() if e["pvc"]]
    c_qc = n31.cnn_scores(dev["c_models"], dev["c_shape"], {h: ext[h] for h in qc_ids}, device)
    clin = n31.clinical_features({h: ext[h] for h in qc_ids})
    cl_qc = pd.Series(dev["clin_model"].decision_function(clin[dev["clin_cols"]].to_numpy()), index=clin.index)
    q = orig.loc[qc_ids, ["label", "founder_score"]].copy()
    q["cnn_score"] = c_qc; q["clinical_score"] = cl_qc
    q["fusion_score"] = rules["fuse"](rules["w"], q.founder_score, q.cnn_score)
    thr = rules["thr"] | {"clinical": dev["clin_thr"]}
    base = orig[orig.cnn_score.notna()]
    sens = {}
    for m in ("fusion", "founder", "cnn", "clinical"):
        sens[m] = {"original_106": evaluate(base.label.to_numpy(), (base[f"{m}_score"] - thr[m]).to_numpy()),
                   "original_same_patients": evaluate(orig.loc[qc_ids, "label"].to_numpy(),
                                                      (orig.loc[qc_ids, f"{m}_score"] - thr[m]).to_numpy()),
                   "qc_beats": evaluate(q.label.to_numpy(), (q[f"{m}_score"] - thr[m]).to_numpy())}

    # 4. saved predictions of (i) and (iii) in the patients with and without a kept beat
    mc = pd.read_csv(UPF / "multicentre_patients.csv")
    mc = mc[mc.arm == "B_two_centre"].assign(patient=lambda d: d.patient.str[1:]).set_index("patient")
    with_kept = set(qc_ids)
    subsets = {}
    for name, ids in [("with_kept_beat", [h for h in base.index if h in with_kept]),
                      ("without_kept_beat", [h for h in base.index if h not in with_kept])]:
        if not ids:
            continue
        y = base.loc[ids, "label"].to_numpy()
        subsets[name] = {"unchanged_fusion": evaluate(y, (base.loc[ids, "fusion_score"] - thr["fusion"]).to_numpy())
                         if len(set(y)) > 1 else {"patients": len(ids), "lvot": int(y.sum())},
                         "adapted_fusion": evaluate(y, (mc.loc[ids, "fusion_score"] - mc.loc[ids, "fusion_thr"]).to_numpy())
                         if len(set(y)) > 1 else {"patients": len(ids), "lvot": int(y.sum())},
                         "errors_unchanged": int(((base.loc[ids, "fusion_score"] > thr["fusion"]).astype(int) != y).sum()),
                         "errors_adapted": int(((mc.loc[ids, "fusion_score"] > mc.loc[ids, "fusion_thr"]).astype(int) != y).sum())}

    st.merge(q[["cnn_score", "clinical_score", "fusion_score"]].add_suffix("_qc"), left_on="patient",
             right_index=True, how="left").to_csv(UPF / "qc_patients.csv", index=False)
    summary = {"rule": {"w_min_ms": W_MIN, "definition": "flagged and width index >= 1st percentile of the "
                        "patient-median flagged width of the first cohort"},
               "extraction": extraction, "visual_review": {
                   "patients": int(len(review)), "flagged": int(review.flagged_qc_input.sum()),
                   "kept_by_qc": int(review.kept_by_qc.sum()),
                   "reviewed_pvc": int(review.flagged_pvc.sum()), "reviewed_not_pvc": int(review.flagged_not_pvc.sum())},
               "sensitivity_unchanged": sens, "subsets": subsets}
    (UPF / "qc_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(review[["patient", "side", "flagged_beats", "flagged_pvc", "flagged_not_pvc", "flagged_qc_input",
                  "kept_by_qc"]].to_string(index=False))
    print("written", UPF)


if __name__ == "__main__":
    main()
