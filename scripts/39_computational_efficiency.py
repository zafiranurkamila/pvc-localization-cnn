"""Computational cost of HyFusion and its components (PSD-CWT CNN, fine-tuned ECGFounder), measured on the stored
final models of the first cohort (data/interim/external_final/development_ens5.pt, written by scripts/31). No model is
trained.

Per model: parameters, model size, FLOPs per patient, inference time per patient (GPU and CPU) and peak GPU memory.
A patient is one 10-s recording with its median number of PVC beats in the first cohort (4); every model is the
five-seed ensemble used in the article (ECGFounder: the evaluation crops of the recording, first, middle and last
10 s, i.e. one crop for a recording of at most 10 s; CNN: one forward pass per PVC beat).
Preprocessing (filtering, R-peak detection and PVC-beat extraction; PSD and CWT of each beat; resampling and baseline
removal for ECGFounder) is timed on real recordings of the first cohort. FLOPs are counted with
torch.utils.flop_counter (one multiply-add = 2 FLOPs) on a single forward pass and scaled to the ensemble.

Run on the PC (GPU, data/processed, ECGFounder checkpoint, cached development models); a few minutes.
Usage:
    python scripts/39_computational_efficiency.py
    python scripts/39_computational_efficiency.py --patients 20 --repeats 50
Output: results/efficiency/efficiency.json and efficiency.csv
"""
import argparse
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import medfilt, resample_poly
from torch.utils.flop_counter import FlopCounterMode


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n18 = _load("check18", "18_pvc_extraction_check.py")
n21 = _load("nested21", "21_nested_cv_dual_rhythm.py")
n23 = _load("nested23", "23_nested_cv_ecgfounder.py")
n31 = _load("external31", "31_external_validation_upf.py")

from pvc_localization import config  # noqa: E402
from pvc_localization.features.psd import flatten_psd_features  # noqa: E402
from pvc_localization.features.wavelet import extract_cwt_scalogram  # noqa: E402
from pvc_localization.preprocessing.normalization import zscore_normalize  # noqa: E402

OUT = config.RESULTS_DIR / "efficiency"
BEATS_PER_PATIENT = 4  # median number of PVC beats per patient in the first cohort (Methods 2.1)


def timed(fn, repeats, device):
    """Median wall-clock time of fn() in ms after a warm-up."""
    for _ in range(3):
        fn()
    ts = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def flops(model, *x):
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            model(*x)
    return fc.get_total_flops()


def size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=20, help="first-cohort recordings used for the timings")
    ap.add_argument("--repeats", type=int, default=50, help="timed repetitions per measurement")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.is_available()

    # ---------------------------------------------------------------- real inputs and preprocessing time
    index = n18.load_patient_index().sample(args.patients, random_state=config.RANDOM_SEED)
    t_extract, t_psd, t_cwt, t_founder_pre, beats, crops = [], [], [], [], [], []
    for _, row in index.iterrows():
        t0 = time.perf_counter()
        ecg, peaks, widths, wide, kept = n18.analyse(row)  # load, filter, R peaks, largest-gap PVC flags
        t_extract.append((time.perf_counter() - t0) * 1000)
        for p, w, k in zip(peaks, wide, kept):
            if w and k:
                beats.append(zscore_normalize(ecg[p - config.BEAT_PRE_SAMPLES:p + config.BEAT_POST_SAMPLES].T))
        t0 = time.perf_counter()
        x500 = resample_poly(ecg, 1, config.SAMPLING_RATE_HZ // n23.FS_OUT, axis=0).T
        x500 = x500 - np.stack([medfilt(lead, int(0.4 * n23.FS_OUT) + 1) for lead in x500])
        c = n23.eval_crops(x500.astype(np.float32))
        t_founder_pre.append((time.perf_counter() - t0) * 1000)
        crops.append(c)
    for b in beats[:args.repeats]:
        t0 = time.perf_counter(); psd = flatten_psd_features(b); t_psd.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); cwt = extract_cwt_scalogram(b); t_cwt.append((time.perf_counter() - t0) * 1000)
    psd_x = torch.tensor(np.stack([flatten_psd_features(b) for b in beats[:BEATS_PER_PATIENT]]), dtype=torch.float32)
    cwt_x = torch.tensor(np.stack([extract_cwt_scalogram(b) for b in beats[:BEATS_PER_PATIENT]]), dtype=torch.float32)
    # evaluation crops (first, middle, last 10 s; one for a recording of at most 10 s) of a recording with the
    # median number of crops
    n_crops = int(np.median([len(c) for c in crops]))
    crop_x = torch.tensor(next(c for c in crops if len(c) == n_crops))
    pre = {"filter_rpeaks_pvc_extraction_ms_per_recording": float(np.median(t_extract)),
           "psd_ms_per_beat": float(np.median(t_psd)), "cwt_ms_per_beat": float(np.median(t_cwt)),
           "founder_resample_baseline_crops_ms_per_recording": float(np.median(t_founder_pre))}
    pre["cnn_features_ms_per_patient"] = BEATS_PER_PATIENT * (pre["psd_ms_per_beat"] + pre["cwt_ms_per_beat"])

    # ---------------------------------------------------------------- stored final models
    dev = torch.load(n31.CACHE / "development_ens5.pt", weights_only=False)
    cs = dev["c_shape"]

    def cnn(state, device):
        m = n21.TwoBranchCNN(cs["psd_len"], cs["cwt_channels"], cs["dropout"], psd12=True)
        m.load_state_dict(state)
        return m.to(device).eval()

    def founder(state, device):
        m, _, _ = n23.founder(device)
        m.load_state_dict(state)
        return m.eval()

    res = {"preprocessing_cpu": pre, "models": {}}
    c0, f0 = cnn(dev["c_models"][0], "cpu"), founder(dev["f_models"][0], "cpu")
    for name, m, x, n_pass in [("PSD-CWT CNN", c0, (psd_x[:1], cwt_x[:1]), BEATS_PER_PATIENT),
                               ("ECGFounder", f0, (crop_x[:1],), n_crops)]:
        n_par = sum(p.numel() for p in m.parameters())
        f_one = flops(m, *x)
        res["models"][name] = {"parameters": int(n_par), "parameters_ensemble": int(5 * n_par),
                               "size_mb": size_mb(m), "size_mb_ensemble": 5 * size_mb(m),
                               "gflops_per_pass": f_one / 1e9, "passes_per_patient": n_pass,
                               "gflops_per_patient_ensemble": 5 * n_pass * f_one / 1e9}
    del c0, f0

    for device in (["cuda"] if gpu else []) + ["cpu"]:
        cnns = [cnn(s, device) for s in dev["c_models"]]
        fds = [founder(s, device) for s in dev["f_models"]]
        p, c, w = psd_x.to(device), cwt_x.to(device), crop_x.to(device)

        @torch.no_grad()
        def run_cnn():
            return np.mean([(lambda z: (z[:, 1] - z[:, 0]).mean().item())(m(p, c)) for m in cnns])

        @torch.no_grad()
        def run_founder():  # as in the article: half precision on the GPU (scripts/23)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                return np.mean([(lambda z: (z[:, 1] - z[:, 0]).float().mean().item())(m(w)) for m in fds])

        def run_hyfusion():  # the weighted average of two standardized scores is negligible
            return 0.5 * run_founder() + 0.5 * run_cnn()

        reps = args.repeats if device == "cuda" else max(5, args.repeats // 5)
        for name, fn in [("PSD-CWT CNN", run_cnn), ("ECGFounder", run_founder), ("HyFusion", run_hyfusion)]:
            if device == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            ms = timed(fn, reps, device)
            r = res["models"].setdefault(name, {})
            r[f"inference_ms_per_patient_{device}"] = ms
            if device == "cuda":
                r["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        del cnns, fds
        if device == "cuda":
            torch.cuda.empty_cache()

    cn, fo, hy = res["models"]["PSD-CWT CNN"], res["models"]["ECGFounder"], res["models"]["HyFusion"]
    for k in ("parameters", "parameters_ensemble", "size_mb", "size_mb_ensemble", "gflops_per_patient_ensemble"):
        hy[k] = cn[k] + fo[k]
    for device in (["cuda"] if gpu else []) + ["cpu"]:
        cn[f"end_to_end_ms_per_patient_{device}"] = (pre["filter_rpeaks_pvc_extraction_ms_per_recording"]
                                                     + pre["cnn_features_ms_per_patient"]
                                                     + cn[f"inference_ms_per_patient_{device}"])
        fo[f"end_to_end_ms_per_patient_{device}"] = (pre["filter_rpeaks_pvc_extraction_ms_per_recording"]
                                                     + pre["founder_resample_baseline_crops_ms_per_recording"]
                                                     + fo[f"inference_ms_per_patient_{device}"])
        hy[f"end_to_end_ms_per_patient_{device}"] = (pre["filter_rpeaks_pvc_extraction_ms_per_recording"]
                                                     + pre["cnn_features_ms_per_patient"]
                                                     + pre["founder_resample_baseline_crops_ms_per_recording"]
                                                     + hy[f"inference_ms_per_patient_{device}"])
    res["setting"] = {"beats_per_patient": BEATS_PER_PATIENT, "founder_crops_per_patient": n_crops, "ensemble": 5,
                      "timing_patients": args.patients, "repeats": args.repeats,
                      "gpu": torch.cuda.get_device_name(0) if gpu else None, "cpu": platform.processor(),
                      "torch": torch.__version__, "cpu_threads": torch.get_num_threads()}
    (OUT / "efficiency.json").write_text(json.dumps(res, indent=1))
    pd.DataFrame(res["models"]).T.to_csv(OUT / "efficiency.csv")
    print(json.dumps(res, indent=1))
    print("written", OUT)


if __name__ == "__main__":
    main()
