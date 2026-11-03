"""Sex and age of the second cohort (TRIPOD+AI item 20b) and the errors of the fusion by sex; no model is trained.

Uses the Sex and Age attributes of the UPF pickle for the 112 outflow-tract patients and the 106 with PVC beats
(results/external_upf/patients.csv), and the saved predictions of (i) the unchanged model (scripts/31) and (iii) local
model adaptation (scripts/32, arm B_two_centre).

Usage (on the PC):  python scripts/36_upf_demographics.py data/external/otva_upf/all_points_may_2024.pkl
Output: results/external_upf/demographics.json, demographics_errors_by_sex.csv
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("check30", Path(__file__).resolve().parent / "30_check_external_pvc.py")
n30 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n30)

ROOT = Path(__file__).resolve().parents[1]
UPF = ROOT / "results" / "external_upf"


def describe(d):
    age = d.age.dropna()
    return {"patients": int(len(d)), "men": int((d.sex == "Male").sum()), "women": int((d.sex == "Female").sum()),
            "sex_missing": int(d.sex.isna().sum()), "age_mean": round(float(age.mean()), 1),
            "age_sd": round(float(age.std()), 1), "age_missing": int(d.age.isna().sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", help="the extracted all_points_may_2024.pkl")
    args = ap.parse_args()
    data = n30.n29.load_pickle(args.pkl)
    pat = pd.read_csv(UPF / "patients.csv")
    sex = {h: n30.first(data[h].get("Sex")) for h in pat.patient}
    age = {h: n30.first(data[h].get("Age")) for h in pat.patient}
    pat["sex"] = pat.patient.map(sex).where(lambda s: s.isin(["Male", "Female"]))
    pat["age"] = pd.to_numeric(pat.patient.map(age), errors="coerce")
    side = pat.label.map({0: "RVOT", 1: "LVOT"})
    ana = pat[pat.pvc_beats > 0].copy()
    out = {"outflow_tract_112": describe(pat), "analysis_106": describe(ana),
           "analysis_106_by_side": {s: describe(ana[side[ana.index] == s]) for s in ("RVOT", "LVOT")}}
    # errors of the fusion by sex: (i) unchanged threshold, (iii) local model adaptation without referral
    thr = json.loads((UPF / "summary.json").read_text())["thresholds"]["fusion"]
    ana["error_unchanged"] = (ana.fusion_score > thr).astype(int) != ana.label
    mc = pd.read_csv(UPF / "multicentre_patients.csv")
    mc = mc[mc.arm == "B_two_centre"].assign(patient=lambda d: d.patient.str[1:]).set_index("patient")
    ana["error_adapted"] = ((mc.fusion_score > mc.fusion_thr).astype(int).reindex(ana.patient).to_numpy()
                            != ana.label.to_numpy())
    rows = ana.assign(side=side[ana.index]).groupby(["side", "sex"]).agg(
        patients=("patient", "size"), errors_unchanged=("error_unchanged", "sum"),
        errors_adapted=("error_adapted", "sum")).reset_index()
    rows.to_csv(UPF / "demographics_errors_by_sex.csv", index=False)
    (UPF / "demographics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(rows.to_string(index=False))
    print("written", UPF)


if __name__ == "__main__":
    main()
