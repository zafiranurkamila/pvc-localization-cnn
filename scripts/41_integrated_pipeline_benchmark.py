"""Benchmark of the integrated HyFusion pipeline with the batched CWT (supervisor's final review, JOS version).
Table 8 of the JOS version estimated the end-to-end time with the batched CWT from the times of its components
(scripts/39 and 40). Here one function runs the whole pipeline per patient, from the stored recording to the
HyFusion decision, and is timed as a whole. No model is trained; the stored final models of scripts/31 are used.

Pipeline per patient (one first-cohort recording, with all its PVC beats):
  load, filtering, R peaks and PVC-beat extraction (as scripts/39) -> z-scored beats -> PSD (CPU) and batched CWT
  (GPU, single precision; CPU, double precision) -> five-seed CNN ensemble -> resampling, baseline removal and
  evaluation crops -> five-seed ECGFounder ensemble (half precision on the GPU, as in the article) -> standardised
  weighted fusion, threshold and referral band of scripts/31.
ECGFounder alone runs the same pipeline without the CNN branch.

Reported per configuration (HyFusion and ECGFounder; GPU and CPU-only): median end-to-end time per patient (each
patient timed `--repeats` times, median; then median and interquartile range over patients), also for the
patients with three to five PVC beats; peak GPU memory of the whole pipeline (models loaded, all intermediate
tensors including the CWT) next to the peak of the models alone measured by scripts/39.
The HyFusion decisions of the GPU and the CPU pipelines are compared as a check.

Run on the PC (GPU, data/processed, cached development models of scripts/31); a few minutes.
Usage:
    python scripts/41_integrated_pipeline_benchmark.py
    python scripts/41_integrated_pipeline_benchmark.py --patients 50 --repeats 5
Output: results/efficiency/pipeline_benchmark.json and pipeline_benchmark_patients.csv
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import medfilt, resample_poly


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n18 = _load("check18", "18_pvc_extraction_check.py")
n31 = _load("external31", "31_external_validation_upf.py")  # also loads scripts 21-24
n21, n23 = n31.n21, n31.n23

from pvc_localization import config  # noqa: E402
from pvc_localization.data.loader import load_patient_index  # noqa: E402
from pvc_localization.features.psd import flatten_psd_features  # noqa: E402
from pvc_localization.features.wavelet_fast import FastCWT  # noqa: E402
from pvc_localization.preprocessing.normalization import zscore_normalize  # noqa: E402

OUT = config.RESULTS_DIR / "efficiency"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=50, help="random first-cohort recordings with PVC beats")
    ap.add_argument("--repeats", type=int, default=5, help="timed runs per patient and configuration")
    args = ap.parse_args()
    n21.OPTS.update(psd12=True, augment=True, wd=1e-4, select="f1")
    assert torch.cuda.is_available(), "the GPU pipeline needs CUDA"
    OUT.mkdir(parents=True, exist_ok=True)

    dev = torch.load(n31.CACHE / "development_ens5.pt", weights_only=False)
    labels = load_patient_index().set_index("hospital_id").label.map(config.LABEL_TO_INT)
    rules = n31.decision_rules(dev, labels)
    w, thr, delta = rules["w"], rules["thr"]["fusion"], rules["delta"][0.9]
    zf_thr = rules["thr"]["founder"]
    cs = dev["c_shape"]

    def load_models(device):
        cnns = []
        for s in dev["c_models"]:
            m = n21.TwoBranchCNN(cs["psd_len"], cs["cwt_channels"], cs["dropout"], psd12=True)
            m.load_state_dict(s)
            cnns.append(m.to(device).eval())
        fds = []
        for s in dev["f_models"]:
            m, _, _ = n23.founder(device)
            m.load_state_dict(s)
            fds.append(m.eval())
        return cnns, fds

    @torch.no_grad()
    def pipeline(row, device, cnns, fds, cwt, use_cnn):
        ecg, peaks, widths, wide, kept = n18.analyse(row)  # load, filter, R peaks, largest-gap PVC flags
        c_score = None
        if use_cnn:
            beats = np.stack([zscore_normalize(ecg[p - config.BEAT_PRE_SAMPLES:p + config.BEAT_POST_SAMPLES].T)
                              for p, wd, k in zip(peaks, wide, kept) if wd and k])
            psd = torch.as_tensor(np.stack([flatten_psd_features(b) for b in beats]), dtype=torch.float32,
                                  device=device)
            sc = cwt(beats).to(device, torch.float32)
            c_score = float(np.mean([(lambda z: (z[:, 1] - z[:, 0]).mean().item())(m(psd, sc)) for m in cnns]))
        x500 = resample_poly(ecg, 1, config.SAMPLING_RATE_HZ // n23.FS_OUT, axis=0).T
        x500 = x500 - np.stack([medfilt(lead, int(0.4 * n23.FS_OUT) + 1) for lead in x500])
        crops = torch.as_tensor(n23.eval_crops(x500.astype(np.float32)), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            f_score = float(np.mean([(lambda z: (z[:, 1] - z[:, 0]).float().mean().item())(m(crops)) for m in fds]))
        if device == "cuda":
            torch.cuda.synchronize()
        if not use_cnn:
            return {"founder": f_score, "decision": int(f_score > zf_thr)}
        s = float(rules["fuse"](w, np.array([f_score]), np.array([c_score]))[0])
        return {"founder": f_score, "cnn": c_score, "fusion": s, "decision": int(s > thr),
                "referred": int(abs(s - thr) < delta)}

    # patients: random first-cohort recordings with at least one PVC beat
    index = n18.load_patient_index().sample(frac=1, random_state=config.RANDOM_SEED)
    rows, n_beats = [], []
    for _, row in index.iterrows():
        _, peaks, _, wide, kept = n18.analyse(row)
        nb = int(np.sum(np.asarray(wide) & np.asarray(kept)))
        if nb:
            rows.append(row); n_beats.append(nb)
        if len(rows) == args.patients:
            break

    configs = {"HyFusion_gpu": ("cuda", True), "ECGFounder_gpu": ("cuda", False),
               "HyFusion_cpu": ("cpu", True), "ECGFounder_cpu": ("cpu", False)}
    cwt = {"cuda": FastCWT(device="cuda", dtype=torch.float32), "cpu": FastCWT(device="cpu", dtype=torch.float64)}
    res = {"configs": {}}
    per_patient = pd.DataFrame({"patient": [r["hospital_id"] for r in rows], "pvc_beats": n_beats})
    outputs = {}
    for device in ("cuda", "cpu"):
        cnns, fds = load_models(device)
        names = [n for n, (d, _) in configs.items() if d == device]
        for name in names:  # warm-up
            for r in rows[:3]:
                pipeline(r, device, cnns, fds, cwt[device], configs[name][1])
        peaks = {}
        if device == "cuda":  # peak GPU memory of the whole pipeline, all models of the configuration loaded
            for name in names:
                use_cnn = configs[name][1]
                cn = cnns if use_cnn else []
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                for r in rows:
                    pipeline(r, device, cn, fds, cwt[device], use_cnn)
                peaks[name] = {"peak_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                               "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20,
                               "resident_models_mb": base / 2 ** 20}
        times = {n: [] for n in names}
        for i, r in enumerate(rows):  # configurations alternate per patient
            for name in names:
                ts = []
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    o = pipeline(r, device, cnns, fds, cwt[device], configs[name][1])
                    ts.append((time.perf_counter() - t0) * 1000)
                times[name].append(float(np.median(ts)))
                outputs.setdefault(name, []).append(o)
        for name in names:
            t = np.array(times[name]); per_patient[f"{name}_ms"] = t
            mid = (per_patient.pvc_beats >= 3) & (per_patient.pvc_beats <= 5)
            res["configs"][name] = {"median_ms": float(np.median(t)), "q1_ms": float(np.percentile(t, 25)),
                                    "q3_ms": float(np.percentile(t, 75)),
                                    "median_ms_3_to_5_beats": float(np.median(t[mid.to_numpy()])),
                                    "patients_3_to_5_beats": int(mid.sum())} | peaks.get(name, {})
        del cnns, fds
        if device == "cuda":
            torch.cuda.empty_cache()

    hg, hc = pd.DataFrame(outputs["HyFusion_gpu"]), pd.DataFrame(outputs["HyFusion_cpu"])
    per_patient = pd.concat([per_patient, hg.add_prefix("gpu_"), hc.add_prefix("cpu_")], axis=1)
    eff = json.loads((OUT / "efficiency.json").read_text())
    pilot = json.loads((OUT / "cwt_pilot.json").read_text())
    res["check_gpu_vs_cpu"] = {"fusion_score_max_abs_diff": float((hg.fusion - hc.fusion).abs().max()),
                               "decisions_differing": int((hg.decision != hc.decision).sum()),
                               "referrals_differing": int((hg.referred != hc.referred).sum())}
    res["reference"] = {
        "estimated_hyfusion_gpu_ms": pilot["end_to_end"]["gpu_float32"]["hyfusion_end_to_end_ms_gpu_inference"],
        "estimated_hyfusion_cpu_ms": pilot["end_to_end"]["cpu_float64"]["hyfusion_end_to_end_ms_cpu_inference"],
        "estimated_founder_gpu_ms": eff["models"]["ECGFounder"]["end_to_end_ms_per_patient_cuda"],
        "estimated_founder_cpu_ms": eff["models"]["ECGFounder"]["end_to_end_ms_per_patient_cpu"],
        "models_only_peak_gpu_mb": {k: eff["models"][k]["peak_gpu_memory_mb"] for k in ("HyFusion", "ECGFounder")},
        "cwt_only_peak_gpu_mb_4_beats": pilot["peak_gpu_memory_mb"]["gpu_float32"]}
    res["setting"] = {"patients": len(rows), "median_pvc_beats": float(np.median(n_beats)),
                      "repeats": args.repeats, "gpu": torch.cuda.get_device_name(0),
                      "cpu_threads": torch.get_num_threads(), "torch": torch.__version__,
                      "cwt": {"cuda": "batched, float32", "cpu": "batched, float64"},
                      "founder_precision_gpu": "float16 autocast (as in the article)"}
    (OUT / "pipeline_benchmark.json").write_text(json.dumps(res, indent=1))
    per_patient.to_csv(OUT / "pipeline_benchmark_patients.csv", index=False)
    print(json.dumps(res, indent=1))
    print("written", OUT)


if __name__ == "__main__":
    main()
