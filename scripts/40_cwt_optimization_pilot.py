"""Pilot optimisation of the CWT, the term that dominated the end-to-end time of HyFusion (Table 8 of the JOS
version: 163 ms per PVC beat with pywt on the CPU). No model is trained.

The optimised transform (src/pvc_localization/features/wavelet_fast.py, FastCWT) computes the same scalogram as
extract_cwt_scalogram, with (1) each wavelet kernel cut exactly to the samples that reach the kept outputs and
(2) all beats, leads and scales of a patient batched into one FFT call, on the CPU or the GPU.

Reported:
  1. time of the CWT per patient (four PVC beats, the first-cohort median): pywt per beat (as in Table 8), the
     batched transform on the CPU (float64, float32) and on the GPU (float64, float32; raw beats copied to the GPU
     and the scalograms left there for the CNN); throughput per beat for a batch of 64 beats on the GPU;
  2. agreement with pywt on the PVC beats of all first-cohort patients: largest absolute difference of the
     scalograms, of the beat logit margins and of the patient scores of the stored final CNN ensemble, and the
     number of CNN decisions that change at the stored threshold;
  3. re-evaluation of the classification on the second cohort (unchanged final models, same strips and beats as
     scripts/31): CNN and HyFusion scores from the optimised CWT against the saved scores of
     results/external_upf/patients.csv, with macro F1, balanced accuracy, AUC, and coverage / macro F1 at the 90%
     referral target;
  4. end-to-end time per patient with the optimised CWT, the other terms taken from results/efficiency/
     efficiency.json (scripts/39).

Run on the PC (GPU, data/processed, cached development models of scripts/31, the UPF pickle); about 10 minutes.
Usage:
    python scripts/40_cwt_optimization_pilot.py data/external/otva_upf/all_points_may_2024.pkl
    python scripts/40_cwt_optimization_pilot.py            # without the second-cohort re-evaluation
Output: results/efficiency/cwt_pilot.json
"""
import argparse
import importlib.util
import json
import time
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


n18 = _load("check18", "18_pvc_extraction_check.py")
n31 = _load("external31", "31_external_validation_upf.py")  # also loads scripts 21-24, 29, 30
n21 = n31.n21

from pvc_localization import config  # noqa: E402
from pvc_localization.data.loader import load_patient_index  # noqa: E402
from pvc_localization.features.psd import flatten_psd_features  # noqa: E402
from pvc_localization.features.wavelet import extract_cwt_scalogram  # noqa: E402
from pvc_localization.features.wavelet_fast import FastCWT  # noqa: E402
from pvc_localization.preprocessing.normalization import zscore_normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = config.RESULTS_DIR / "efficiency"
UPF = ROOT / "results" / "external_upf"
BEATS_PER_PATIENT = 4


def timed(fn, repeats, cuda):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def evaluate(y, s, thr, delta):
    m = n21.metrics(y, s, thr)
    keep = np.abs(s - thr) >= delta
    r = n21.metrics(y[keep], s[keep], thr)
    return {"macro_f1": float(m["macro_f1"]), "balanced_accuracy": float(m["balanced_accuracy"]),
            "auc": float(m["auc"]), "referral_coverage": float(keep.mean()), "referral_macro_f1": float(r["macro_f1"]),
            "referral_balanced_accuracy": float(r["balanced_accuracy"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", nargs="?", help="the UPF all_points_may_2024.pkl (for the second-cohort re-evaluation)")
    ap.add_argument("--timing-patients", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    n21.OPTS.update(psd12=True, augment=True, wd=1e-4, select="f1")
    cuda = torch.cuda.is_available()
    dev_name = "cuda" if cuda else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    dev = torch.load(n31.CACHE / "development_ens5.pt", weights_only=False)
    patients = load_patient_index().set_index("hospital_id").label.map(config.LABEL_TO_INT)
    rules = n31.decision_rules(dev, patients)
    thr, delta = rules["thr"], rules["delta"][0.9]
    cs = dev["c_shape"]
    cnns = []
    for s in dev["c_models"]:
        m = n21.TwoBranchCNN(cs["psd_len"], cs["cwt_channels"], cs["dropout"], psd12=True)
        m.load_state_dict(s)
        cnns.append(m.to(dev_name).eval())

    @torch.no_grad()
    def beat_margins(psd, cwt):  # mean beat logit margin of the five-seed ensemble, per beat
        return np.mean([(lambda z: (z[:, 1] - z[:, 0]).double().cpu().numpy())(m(psd, cwt)) for m in cnns], axis=0)

    # ---------------------------------------------------------------- first-cohort PVC beats
    index = n18.load_patient_index()
    beats_by_patient = {}
    for _, row in tqdm(index.iterrows(), total=len(index), desc="first-cohort PVC beats", unit="patient"):
        ecg, peaks, widths, wide, kept = n18.analyse(row)
        b = [zscore_normalize(ecg[p - config.BEAT_PRE_SAMPLES:p + config.BEAT_POST_SAMPLES].T)
             for p, w, k in zip(peaks, wide, kept) if w and k]
        if b:
            beats_by_patient[row["hospital_id"]] = np.stack(b)
    in_dtype = next(iter(beats_by_patient.values())).dtype
    variants = {"cpu_float64": FastCWT(device="cpu", dtype=torch.float64, input_dtype=in_dtype),
                "cpu_float32": FastCWT(device="cpu", dtype=torch.float32, input_dtype=in_dtype)}
    if cuda:
        variants["gpu_float64"] = FastCWT(device="cuda", dtype=torch.float64, input_dtype=in_dtype)
        variants["gpu_float32"] = FastCWT(device="cuda", dtype=torch.float32, input_dtype=in_dtype)

    # ---------------------------------------------------------------- 2. agreement with pywt (all first-cohort beats)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev_name)
    diff = {v: {"scalogram_max_abs": 0.0, "beat_margin_max_abs": 0.0} for v in variants}
    p_ref, p_fast = {}, {v: {} for v in variants}
    n_beats = 0
    for h, b in tqdm(beats_by_patient.items(), desc="pywt vs optimised CWT", unit="patient"):
        psd = t([flatten_psd_features(x) for x in b])
        ref = np.stack([extract_cwt_scalogram(x) for x in b])
        m_ref = beat_margins(psd, t(ref))
        p_ref[h] = m_ref.mean()
        n_beats += len(b)
        for v, f in variants.items():
            fast = f(b)
            diff[v]["scalogram_max_abs"] = max(diff[v]["scalogram_max_abs"],
                                               float(np.abs(fast.cpu().numpy() - ref).max()))
            m_fast = beat_margins(psd, fast.to(dev_name, torch.float32))
            diff[v]["beat_margin_max_abs"] = max(diff[v]["beat_margin_max_abs"], float(np.abs(m_fast - m_ref).max()))
            p_fast[v][h] = m_fast.mean()
    ref_s = pd.Series(p_ref)
    agreement = {"patients": int(len(ref_s)), "beats": int(n_beats), "variants": {}}
    for v in variants:
        fs = pd.Series(p_fast[v])[ref_s.index]
        agreement["variants"][v] = diff[v] | {
            "patient_score_max_abs": float((fs - ref_s).abs().max()),
            "cnn_decisions_changed": int(((fs > thr["cnn"]) != (ref_s > thr["cnn"])).sum())}
    print(json.dumps(agreement, indent=1))
    for v, a in agreement["variants"].items():
        assert a["scalogram_max_abs"] < 1e-3, f"{v}: optimised CWT differs from pywt ({a['scalogram_max_abs']:.2e})"

    # ---------------------------------------------------------------- 1. timing per patient (four beats)
    rng = np.random.default_rng(config.RANDOM_SEED)
    pool = np.concatenate(list(beats_by_patient.values()))
    sets = [pool[rng.choice(len(pool), BEATS_PER_PATIENT, replace=False)] for _ in range(args.timing_patients)]
    big = pool[rng.choice(len(pool), 64, replace=False)]
    k = {"i": 0}

    def nxt():
        k["i"] = (k["i"] + 1) % len(sets)
        return sets[k["i"]]

    methods = {"pywt_per_beat_cpu": (lambda: [extract_cwt_scalogram(x) for x in nxt()], False)}
    for v, f in variants.items():
        methods[v] = ((lambda f=f: f(nxt())), v.startswith("gpu"))
    rounds = {m: [] for m in methods}
    for _ in range(args.rounds):
        for m, (fn, on_gpu) in methods.items():
            reps = max(3, args.repeats // 4) if m == "pywt_per_beat_cpu" else args.repeats
            rounds[m].append(timed(fn, reps, on_gpu))
    timing = {m: float(np.median(r)) for m, r in rounds.items()}
    throughput = {}
    for v, f in variants.items():
        throughput[v] = timed(lambda f=f: f(big, chunk=32 if v.startswith("gpu") else 8), 5,
                              v.startswith("gpu")) / len(big)
    peak = {}
    if cuda:
        for v in ("gpu_float64", "gpu_float32"):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            variants[v](sets[0])
            torch.cuda.synchronize()
            peak[v] = torch.cuda.max_memory_allocated() / 2 ** 20

    # ---------------------------------------------------------------- 4. end to end (other terms from scripts/39)
    eff = json.loads((OUT / "efficiency.json").read_text())
    pre, hy = eff["preprocessing_cpu"], eff["models"]["HyFusion"]
    rest_gpu = (pre["filter_rpeaks_pvc_extraction_ms_per_recording"] + BEATS_PER_PATIENT * pre["psd_ms_per_beat"]
                + pre["founder_resample_baseline_crops_ms_per_recording"])
    end_to_end = {}
    for m, ms in timing.items():
        end_to_end[m] = {"cwt_ms_per_patient": ms, "hyfusion_end_to_end_ms_gpu_inference": rest_gpu + ms
                         + hy["inference_ms_per_patient_cuda"], "hyfusion_end_to_end_ms_cpu_inference": rest_gpu + ms
                         + hy["inference_ms_per_patient_cpu"]}
    end_to_end["table8_reference"] = {"cwt_ms_per_patient": BEATS_PER_PATIENT * pre["cwt_ms_per_beat"],
                                      "hyfusion_end_to_end_ms_gpu_inference": hy["end_to_end_ms_per_patient_cuda"],
                                      "hyfusion_end_to_end_ms_cpu_inference": hy["end_to_end_ms_per_patient_cpu"]}
    end_to_end["founder_end_to_end_ms_gpu"] = eff["models"]["ECGFounder"]["end_to_end_ms_per_patient_cuda"]

    # ---------------------------------------------------------------- 3. second cohort, unchanged final models
    external = None
    if args.pkl:
        main_v = "gpu_float32" if cuda else "cpu_float32"
        ext = n31.external_inputs(args.pkl, quick=False)
        orig = pd.read_csv(UPF / "patients.csv", index_col="patient")
        orig.index = orig.index.astype(str)
        hs = [h for h, e in ext.items() if e["pvc"]]
        scores = {}
        for h in tqdm(hs, desc="second cohort, optimised CWT", unit="patient"):
            b = np.stack([zscore_normalize(x) for x in ext[h]["pvc"]])
            psd = t([flatten_psd_features(x) for x in b])
            f = variants[main_v] if b.dtype == in_dtype else FastCWT(device=variants[main_v].device,
                                                                   dtype=variants[main_v].dtype, input_dtype=b.dtype)
            scores[str(h)] = beat_margins(psd, f(b).to(dev_name, torch.float32)).mean()
        d = orig[orig.cnn_score.notna()].copy()
        d["cnn_fast"] = pd.Series(scores)[d.index]
        d["fusion_fast"] = rules["fuse"](rules["w"], d.founder_score, d.cnn_fast)
        y = d.label.to_numpy()
        external = {"variant": main_v, "patients": int(len(d)), "lvot": int(y.sum()),
                    "cnn_score_max_abs_diff": float((d.cnn_fast - d.cnn_score).abs().max()),
                    "fusion_score_max_abs_diff": float((d.fusion_fast - d.fusion_score).abs().max()),
                    "cnn_decisions_changed": int(((d.cnn_fast > thr["cnn"]) != (d.cnn_score > thr["cnn"])).sum()),
                    "fusion_decisions_changed": int(((d.fusion_fast > thr["fusion"])
                                                     != (d.fusion_score > thr["fusion"])).sum()),
                    "referral_changed": int(((np.abs(d.fusion_fast - thr["fusion"]) >= delta)
                                             != (np.abs(d.fusion_score - thr["fusion"]) >= delta)).sum()),
                    "cnn_pywt": evaluate(y, d.cnn_score.to_numpy(), thr["cnn"], 0.0),
                    "cnn_fast": evaluate(y, d.cnn_fast.to_numpy(), thr["cnn"], 0.0),
                    "hyfusion_pywt": evaluate(y, d.fusion_score.to_numpy(), thr["fusion"], delta),
                    "hyfusion_fast": evaluate(y, d.fusion_fast.to_numpy(), thr["fusion"], delta)}

    res = {"setting": {"beats_per_patient": BEATS_PER_PATIENT, "timing_sets": args.timing_patients,
                       "repeats": args.repeats, "rounds": args.rounds, "beat_dtype": str(in_dtype),
                       "fft_length": variants["cpu_float64"].m,
                       "gpu": torch.cuda.get_device_name(0) if cuda else None, "cpu_threads": torch.get_num_threads(),
                       "torch": torch.__version__},
           "cwt_ms_per_patient": timing, "cwt_ms_per_beat_batch64": throughput, "peak_gpu_memory_mb": peak,
           "end_to_end": end_to_end, "agreement_first_cohort": agreement, "second_cohort": external}
    (OUT / "cwt_pilot.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "agreement_first_cohort"}, indent=1, default=float))
    print("written", OUT / "cwt_pilot.json")


if __name__ == "__main__":
    main()
