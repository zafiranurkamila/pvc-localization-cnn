"""End-to-end latency of the pipeline and the software environment (reviewer comments 8 and minor 8).

A synthetic 10-s, 12-lead recording at 2000 Hz with one wide-QRS beat per second runs through the real
functions: band-pass and notch filtering, PVC-beat extraction, per-beat z-score, PSD, CWT and HOS
extraction, and the CNN forward pass of one beat (batch size 1) on the CPU and, if available, the GPU.
Timings do not depend on the signal values or on trained weights, so untrained networks are used.
Also records the CPU/GPU names and the versions of the main packages.

No training, no patient data needed.

Usage:
    python scripts/14_latency.py
Output: results/latency.json
"""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config  # noqa: E402
from pvc_localization.features.hos import flatten_hos_features  # noqa: E402
from pvc_localization.features.psd import flatten_psd_features  # noqa: E402
from pvc_localization.features.wavelet import extract_cwt_scalogram  # noqa: E402
from pvc_localization.models.fusion import BaselineCNN, FusionCNN  # noqa: E402
from pvc_localization.preprocessing.beat_detection import extract_pvc_beats  # noqa: E402
from pvc_localization.preprocessing.filtering import clean_signal  # noqa: E402
from pvc_localization.preprocessing.normalization import zscore_normalize  # noqa: E402

FS = config.SAMPLING_RATE_HZ
REPEATS = 30


def synthetic_recording(seconds=10):
    t = np.arange(0, seconds, 1 / FS)
    rng = np.random.default_rng(0)
    sig = np.zeros((t.size, 12))
    g = lambda mu, s, a: a * np.exp(-0.5 * ((t - mu) / s) ** 2)
    for k in range(1, seconds):
        narrow = g(k - 0.45, 0.012, 1.0)
        wide = g(k, 0.03, 1.2) + g(k + 0.06, 0.03, -0.6) - g(k + 0.3, 0.06, 0.35)
        for lead in range(12):
            sig[:, lead] += (0.6 + 0.05 * lead) * (narrow + wide)
    return sig + 0.02 * rng.standard_normal(sig.shape)


def timed(fn, repeats=REPEATS, sync=None):
    fn()
    if sync:
        sync()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def environment():
    def version(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    cpu = platform.processor()
    try:
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True).stdout
        cpu = [l.strip() for l in out.splitlines() if l.strip() and l.strip() != "Name"][0]
    except Exception:
        pass
    env = {"cpu": cpu, "os": platform.platform(), "python": platform.python_version(),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cpu_threads": torch.get_num_threads()}
    for mod in ("numpy", "scipy", "sklearn", "pywt", "shap", "neurokit2"):
        env[mod] = version(mod)
    return env


def main():
    torch.set_grad_enabled(False)
    rec = synthetic_recording()
    filtered = clean_signal(rec)
    beats, _ = extract_pvc_beats(filtered)
    if len(beats) == 0:
        raise SystemExit("no PVC beats found in the synthetic recording")
    beat = zscore_normalize(beats[0])

    res = {"environment": environment(), "n_beats_in_recording": int(len(beats)),
           "per_recording_ms": {"filtering_10s": timed(lambda: clean_signal(rec)),
                                "pvc_beat_extraction_10s": timed(lambda: extract_pvc_beats(filtered))},
           "per_beat_cpu_ms": {"zscore": timed(lambda: zscore_normalize(beats[0])),
                               "psd": timed(lambda: flatten_psd_features(beat)),
                               "cwt": timed(lambda: extract_cwt_scalogram(beat), repeats=10),
                               "hos": timed(lambda: flatten_hos_features(beat), repeats=10)}}

    inputs_np = {"psd": flatten_psd_features(beat), "wavelet": extract_cwt_scalogram(beat),
                 "hos": flatten_hos_features(beat)}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    res["cnn_forward_ms_batch1"] = {}
    for dev in devices:
        sync = torch.cuda.synchronize if dev == "cuda" else None
        x = {k: torch.tensor(v, dtype=torch.float32)[None].to(dev) for k, v in inputs_np.items()}
        raw = torch.tensor(beat, dtype=torch.float32)[None].to(dev)
        models = {"baseline": (BaselineCNN().to(dev).eval(), lambda m: m(raw)),
                  "psd_wavelet": (FusionCNN(["psd", "wavelet"]).to(dev).eval(), lambda m: m(x)),
                  "psd_wavelet_hos": (FusionCNN(["psd", "wavelet", "hos"]).to(dev).eval(), lambda m: m(x))}
        res["cnn_forward_ms_batch1"][dev] = {k: timed(lambda m=m, f=f: f(m), sync=sync) for k, (m, f) in models.items()}

    pb = res["per_beat_cpu_ms"]
    feats = {"baseline": pb["zscore"], "psd_wavelet": pb["zscore"] + pb["psd"] + pb["cwt"],
             "psd_wavelet_hos": pb["zscore"] + pb["psd"] + pb["cwt"] + pb["hos"]}
    res["end_to_end_per_beat_ms"] = {dev: {k: feats[k] + v for k, v in cnn.items()}
                                     for dev, cnn in res["cnn_forward_ms_batch1"].items()}
    out = config.RESULTS_DIR / "latency.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
