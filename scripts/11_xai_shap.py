"""Phase 9 - Explainable AI on the saved final models (Proposal Bab 3.1.13).

For each model:
  1. Branch permutation importance: shuffle one feature input across test beats and
     measure the drop in AUC and macro F1 (30 repeats). Answers which feature the model relies on.
  2. SHAP (GradientExplainer) on the LVOT-vs-RVOT logit margin, with 100 training beats as
     background, explaining all test beats. Aggregated as mean |SHAP| per branch, per lead,
     per PSD frequency bin, per bispectrum bin / moment, and per wavelet (frequency x time) cell.

No training. Run on the PC (needs data/cache and results/models).

Usage:
    python scripts/11_xai_shap.py
    python scripts/11_xai_shap.py --models psd_wavelet_hos_tuned psd_hos_tuned psd_wavelet_tuned
Output: results/xai/<model>_*.csv and *.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvc_localization import config
from pvc_localization.data.dataset import PVCBeatsDataset, create_train_val_test_split
from pvc_localization.evaluation.metrics import compute_metrics
from pvc_localization.models.fusion import FusionCNN
from pvc_localization.training.trainer import CACHE_DIR, set_seed

LEADS = config.LEADS
N_PSD_BINS, N_BISP = 32, 8


class MarginWrapper(nn.Module):
    """Takes the branch inputs positionally and returns logit_LVOT - logit_RVOT, shape (N, 1)."""

    def __init__(self, model, features):
        super().__init__()
        self.model, self.features = model, features

    def forward(self, *xs):
        logits = self.model(dict(zip(self.features, xs)))
        return (logits[:, 1] - logits[:, 0]).unsqueeze(1)


def load_model(name, device):
    ckpt = torch.load(config.RESULTS_DIR / "models" / f"{name}.pt", map_location=device)
    features, hp = ckpt["features"], ckpt["hyperparameters"]
    if not features:
        sys.exit(f"{name} is the raw-signal baseline; XAI here is for feature-based models.")
    model = FusionCNN(features, n_filters=hp["n_filters"], kernel_size=hp["kernel_size"], dropout=hp["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), features


def stack(dataset, features, indices=None):
    indices = range(len(dataset)) if indices is None else indices
    items = [dataset[i] for i in indices]
    x = {f: torch.stack([it[f] for it in items]) for f in features}
    y = np.array([it["label"] for it in items])
    return x, y


def predict_margin(model, x, device, batch=64):
    n = len(next(iter(x.values())))
    out = []
    with torch.no_grad():
        for s in range(0, n, batch):
            b = {k: v[s:s + batch].to(device) for k, v in x.items()}
            logits = model(b)
            out.append((logits[:, 1] - logits[:, 0]).cpu())
    return torch.cat(out).numpy()


def scores(y, margin):
    m = compute_metrics(y, (margin > 0).astype(int), margin)
    return m["auc"], m["macro_f1"]


def branch_permutation(model, x, y, features, device, repeats=30, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    base_auc, base_f1 = scores(y, predict_margin(model, x, device))
    rows = []
    for f in features:
        d_auc, d_f1 = [], []
        for _ in range(repeats):
            xp = dict(x)
            xp[f] = x[f][torch.from_numpy(rng.permutation(len(y)))]
            a, f1 = scores(y, predict_margin(model, xp, device))
            d_auc.append(base_auc - a)
            d_f1.append(base_f1 - f1)
        rows.append({"branch": f, "auc_drop_mean": float(np.mean(d_auc)), "auc_drop_std": float(np.std(d_auc)),
                     "macro_f1_drop_mean": float(np.mean(d_f1)), "macro_f1_drop_std": float(np.std(d_f1))})
    return {"baseline_auc": base_auc, "baseline_macro_f1": base_f1, "branches": rows}


def shap_values(model, features, x_bg, x_test, device, nsamples=200):
    wrapper = MarginWrapper(model, features).to(device).eval()
    explainer = shap.GradientExplainer(wrapper, [x_bg[f].to(device) for f in features])
    vals = explainer.shap_values([x_test[f].to(device) for f in features], nsamples=nsamples)
    if not isinstance(vals, list) or len(vals) != len(features):
        vals = vals[0] if isinstance(vals, list) and len(vals) == 1 else vals
    out = {}
    for f, v in zip(features, vals):
        v = np.asarray(v)
        if v.ndim == x_test[f].ndim + 1 and v.shape[-1] == 1:
            v = v[..., 0]
        out[f] = v
    return out


def save_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r) + "\n")


def analyse(name, device, train_set, test_set, out_dir, n_background):
    model, features = load_model(name, device)
    print(f"\n=== {name}  features={features} ===")

    x_test, y_test = stack(test_set, features)
    rng = np.random.default_rng(config.RANDOM_SEED)
    bg_idx = rng.choice(len(train_set), size=min(n_background, len(train_set)), replace=False)
    x_bg, _ = stack(train_set, features, bg_idx)

    # 1. Branch permutation importance
    perm = branch_permutation(model, x_test, y_test, features, device)
    (out_dir / f"{name}_branch_permutation.json").write_text(json.dumps(perm, indent=2))
    for r in perm["branches"]:
        print(f"  permutation {r['branch']:<8} AUC drop {r['auc_drop_mean']:+.3f} ± {r['auc_drop_std']:.3f} | "
              f"macro F1 drop {r['macro_f1_drop_mean']:+.3f} ± {r['macro_f1_drop_std']:.3f}")

    # 2. SHAP
    sv = shap_values(model, features, x_bg, x_test, device)
    branch_total = {f: float(np.abs(sv[f]).sum(axis=tuple(range(1, sv[f].ndim))).mean()) for f in features}
    total = sum(branch_total.values())
    save_csv(out_dir / f"{name}_shap_branch.csv", ["branch", "mean_abs_shap_per_beat", "share"],
             [(f, f"{branch_total[f]:.6f}", f"{branch_total[f] / total:.4f}") for f in features])
    for f in features:
        print(f"  SHAP share  {f:<8} {branch_total[f] / total:.1%}")

    lead_imp = np.zeros(len(LEADS))
    fig_rows = len(features)
    fig, axes = plt.subplots(fig_rows, 1, figsize=(8, 3.2 * fig_rows))
    axes = np.atleast_1d(axes)

    for ax, f in zip(axes, features):
        a = np.abs(sv[f]).mean(axis=0)
        if f == "psd":
            a = a.reshape(len(LEADS), N_PSD_BINS)
            lead_imp += a.sum(axis=1)
            edges = np.linspace(config.FEATURE_FMIN_HZ, config.FEATURE_FMAX_HZ, N_PSD_BINS + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            freq = a.mean(axis=0)
            save_csv(out_dir / f"{name}_shap_psd_freq.csv", ["freq_hz", "mean_abs_shap"],
                     [(f"{c:.2f}", f"{v:.6f}") for c, v in zip(centers, freq)])
            ax.bar(centers, freq, width=(edges[1] - edges[0]) * 0.9, color="#028090")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_title("PSD: mean |SHAP| per frequency bin (averaged over leads)")
        elif f == "hos":
            a = a.reshape(len(LEADS), N_BISP * N_BISP + 2)
            lead_imp += a.sum(axis=1)
            bisp = a[:, :N_BISP * N_BISP].mean(axis=0).reshape(N_BISP, N_BISP)
            moments = a[:, N_BISP * N_BISP:].mean(axis=0)
            save_csv(out_dir / f"{name}_shap_hos_bispectrum.csv", ["f1_block"] + [f"f2_block_{j}" for j in range(N_BISP)],
                     [[i] + [f"{v:.6f}" for v in bisp[i]] for i in range(N_BISP)])
            save_csv(out_dir / f"{name}_shap_hos_moments.csv", ["moment", "mean_abs_shap"],
                     [("skewness", f"{moments[0]:.6f}"), ("kurtosis", f"{moments[1]:.6f}"),
                      ("bispectrum_mean_per_bin", f"{bisp.mean():.6f}")])
            im = ax.imshow(bisp, origin="lower", cmap="viridis",
                           extent=[0, config.FEATURE_FMAX_HZ, 0, config.FEATURE_FMAX_HZ])
            ax.set_xlabel("f2 (Hz)")
            ax.set_ylabel("f1 (Hz)")
            ax.set_title("HOS: mean |SHAP| per bispectrum block (averaged over leads)")
            fig.colorbar(im, ax=ax)
        elif f == "wavelet":
            lead_imp += a.sum(axis=(1, 2))
            tf = a.mean(axis=0)
            np.savetxt(out_dir / f"{name}_shap_wavelet_map.csv", tf, delimiter=",", fmt="%.6f")
            n_time = tf.shape[1]
            t_ms = np.linspace(-config.BEAT_PRE_MS, config.BEAT_POST_MS, n_time)
            freqs = np.geomspace(config.CWT_FMIN_HZ, config.FEATURE_FMAX_HZ, tf.shape[0])
            im = ax.pcolormesh(t_ms, freqs, tf, shading="auto", cmap="magma")
            ax.set_yscale("log")
            ax.axvline(0, color="white", lw=0.8, ls="--")
            ax.set_xlabel("Time relative to R peak (ms)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title("Wavelet: mean |SHAP| per time-frequency cell (averaged over leads)")
            fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_shap_maps.png", dpi=150)
    plt.close(fig)

    save_csv(out_dir / f"{name}_shap_leads.csv", ["lead", "mean_abs_shap", "share"],
             [(l, f"{v:.6f}", f"{v / lead_imp.sum():.4f}") for l, v in zip(LEADS, lead_imp)])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(LEADS, lead_imp / lead_imp.sum(), color="#E4572E")
    ax.set_ylabel("Share of |SHAP|")
    ax.set_title(f"Lead importance ({name})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_shap_leads.png", dpi=150)
    plt.close(fig)
    top = np.argsort(lead_imp)[::-1][:4]
    print("  top leads   " + ", ".join(f"{LEADS[i]} ({lead_imp[i] / lead_imp.sum():.1%})" for i in top))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["psd_hos_tuned", "psd_wavelet_hos_tuned"])
    parser.add_argument("--background", type=int, default=100)
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = config.RESULTS_DIR / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids, test_ids = create_train_val_test_split()

    for name in args.models:
        features = torch.load(config.RESULTS_DIR / "models" / f"{name}.pt", map_location="cpu")["features"]
        train_set = PVCBeatsDataset(train_ids, features, cache_dir=CACHE_DIR)
        test_set = PVCBeatsDataset(test_ids, features, cache_dir=CACHE_DIR)
        analyse(name, device, train_set, test_set, out_dir, args.background)

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
