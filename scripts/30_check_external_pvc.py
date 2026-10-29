"""Check whether the mapping-point ECGs of the UPF OTVA dataset show the patient's spontaneous PVC (step 1 of the
external validation; no training).

Each patient of all_points_may_2024.pkl has electroanatomical maps (e.g. 1-AO, 2-RV) with mapping points (P1, P2,
...), and every point stores a 2.5-s, 12-lead surface ECG at 1000 Hz. In activation mapping during the clinical
arrhythmia, the surface ECG of every point shows the same spontaneous PVC; in pace mapping it shows paced beats
whose morphology changes with the catheter position. For the outflow-tract patients (SOO_chamber RVOT or LVOT), up
to --points points per patient spread over all maps are processed exactly like the main pipeline (resampled to
2000 Hz, band-pass 0.5-40 Hz and 50 Hz notch, R peaks on lead II, QRS width at 25 % of the peak amplitude):

  per strip     number of beats, QRS width of every beat, which beat lies closest to the strip centre
  per patient   median width of the beats flagged wide by the largest-gap rule and of the other beats, the
                fraction of strips that contain a wide beat, and the morphological consistency of the wide beats
                (median correlation of their lead-II and V1 QRS, 100 ms around the R peak, with the patient's
                median wide beat): close to 1 for one spontaneous PVC morphology, lower for pace maps from
                different sites
  figure        lead II and V1 of six strips each for three RVOT and three LVOT patients

Usage (on the PC, after scripts/29 has extracted the pickle):
    python scripts/30_check_external_pvc.py data/external/otva_upf/all_points_may_2024.pkl
Output: data/external/otva_upf/inspect/patients.csv, strips.csv, strips_example.png (inside the ignored data folder)
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
from tqdm import tqdm  # noqa: E402

_spec = importlib.util.spec_from_file_location("inspect29", Path(__file__).resolve().parent / "29_inspect_external_dataset.py")
n29 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n29)

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvc_localization import config  # noqa: E402
from pvc_localization.preprocessing.beat_detection import detect_r_peaks, flag_wide_beats, qrs_width_ms  # noqa: E402
from pvc_localization.preprocessing.filtering import clean_signal  # noqa: E402

LEADS_UPF = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS_UPF = 1000
FS = config.SAMPLING_RATE_HZ  # 2000, as the main pipeline
HALF = int(0.1 * FS)


def first(v):
    while isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    return v


def strips_of(patient, n_points, rng):
    """Up to n_points (map, point, (2500, 12) array) spread over all maps of the patient."""
    items = [(m, p, pts) for m, maps in (patient.get("Structures") or {}).items() if isinstance(maps, dict)
             for p, pts in maps.items() if isinstance(pts, dict)]
    if len(items) > n_points:
        items = [items[i] for i in sorted(rng.choice(len(items), n_points, replace=False))]
    out = []
    for m, p, pts in items:
        try:
            x = np.stack([np.asarray(pts[k], dtype=np.float64) for k in LEADS_UPF], axis=1)
        except KeyError:
            continue
        out.append((m, p, x))
    return out


def process(x):
    """Filtered (n, 12) strip at 2000 Hz, R peaks on lead II and their QRS widths (ms)."""
    x = clean_signal(resample_poly(x, FS // FS_UPF, 1, axis=0))
    lead2 = x[:, 1]
    peaks = detect_r_peaks(lead2, fs=FS)
    peaks = peaks[(peaks > HALF) & (peaks < len(lead2) - HALF)]
    widths = np.array([qrs_width_ms(lead2, p, fs=FS) for p in peaks])
    return x, peaks, widths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", help="the extracted all_points_may_2024.pkl")
    ap.add_argument("--points", type=int, default=30, help="mapping points per patient")
    args = ap.parse_args()
    out = Path(args.pkl).parent / "inspect"
    out.mkdir(exist_ok=True)
    print("loading (restricted unpickler) ...")
    with open(args.pkl, "rb") as f:
        data = n29.RestrictedUnpickler(f).load()
    ot = {h: p for h, p in data.items() if first(p.get("SOO_chamber")) in ("RVOT", "LVOT")}
    print(f"{len(ot)} outflow-tract patients: "
          f"{sum(first(p['SOO_chamber']) == 'RVOT' for p in ot.values())} RVOT, "
          f"{sum(first(p['SOO_chamber']) == 'LVOT' for p in ot.values())} LVOT")
    rng = np.random.default_rng(config.RANDOM_SEED)
    strip_rows, pat_rows, examples = [], [], {}
    for h, p in tqdm(ot.items(), desc="patients", unit="patient"):
        side = first(p["SOO_chamber"])
        templates, n_wide_strips, wide_w, narrow_w, n_strips = [], 0, [], [], 0
        for m, pt, x in strips_of(p, args.points, rng):
            xf, peaks, widths = process(x)
            n_strips += 1
            wide = flag_wide_beats(widths) if len(widths) >= 2 else np.zeros(len(widths), bool)
            centre = int(np.argmin(np.abs(peaks - len(xf) / 2))) if len(peaks) else -1
            strip_rows.append({"patient": h, "side": side, "map": m, "point": pt, "beats": len(peaks),
                               "widths_ms": ";".join(f"{w:.0f}" for w in widths), "wide_beats": int(wide.sum()),
                               "centre_width_ms": float(widths[centre]) if centre >= 0 else np.nan,
                               "centre_is_wide": bool(wide[centre]) if centre >= 0 else False})
            if wide.any():
                n_wide_strips += 1
                wide_w += list(widths[wide]); narrow_w += list(widths[~wide])
                for k in np.where(wide)[0]:
                    templates.append(np.concatenate([xf[peaks[k] - HALF:peaks[k] + HALF, 1],
                                                     xf[peaks[k] - HALF:peaks[k] + HALF, 6]]))
            if len(examples.get(side, {})) < 3 and n_strips <= 6:
                examples.setdefault(side, {}).setdefault(h, []).append((m, pt, xf, peaks, wide))
        corr = np.nan
        if len(templates) >= 3:
            t = np.array(templates); med = np.median(t, axis=0)
            corr = float(np.median([np.corrcoef(r, med)[0, 1] for r in t]))
        pat_rows.append({"patient": h, "side": side, "pvc_transition": first(p.get("PVC_transition")),
                         "strips": n_strips, "strips_with_wide_beat": n_wide_strips,
                         "median_wide_ms": float(np.median(wide_w)) if wide_w else np.nan,
                         "median_other_ms": float(np.median(narrow_w)) if narrow_w else np.nan,
                         "wide_beat_morphology_corr": corr})
    pats = pd.DataFrame(pat_rows); strips = pd.DataFrame(strip_rows)
    pats.to_csv(out / "patients.csv", index=False); strips.to_csv(out / "strips.csv", index=False)

    print(f"\nstrips analysed: {len(strips)}; beats per strip: median {strips.beats.median():.0f} "
          f"(range {strips.beats.min()}-{strips.beats.max()})")
    print(f"strips with a wide beat: {strips.wide_beats.gt(0).mean():.1%}; strips whose centre beat is wide: "
          f"{strips.centre_is_wide.mean():.1%}")
    print(f"QRS width (25 % amplitude; the Zheng data gave a median of 28 ms narrow and 108 ms wide): centre beats "
          f"median {strips.centre_width_ms.median():.0f} ms")
    for side, g in pats.groupby("side"):
        print(f"{side}: {len(g)} patients; strips with a wide beat per patient median "
              f"{(g.strips_with_wide_beat / g.strips).median():.0%}; wide beats median {g.median_wide_ms.median():.0f} ms, "
              f"other beats {g.median_other_ms.median():.0f} ms; morphology correlation of wide beats median "
              f"{g.wide_beat_morphology_corr.median():.3f} (IQR {g.wide_beat_morphology_corr.quantile(.25):.3f}-"
              f"{g.wide_beat_morphology_corr.quantile(.75):.3f})")
    print(f"patients without any wide beat in the analysed strips: {int((pats.strips_with_wide_beat == 0).sum())}")

    fig, axes = plt.subplots(6, 2, figsize=(11, 12), sharex=True)
    rows = [(s, h, v) for s in ("RVOT", "LVOT") for h, v in examples.get(s, {}).items()][:6]
    t = np.arange(int(2.5 * FS)) / FS
    for r, (side, h, strips_h) in enumerate(rows):
        for c, lead in enumerate((1, 6)):
            ax = axes[r, c]
            for k, (m, pt, xf, peaks, wide) in enumerate(strips_h):
                y = xf[:len(t), lead]
                ax.plot(t[:len(y)], (y - np.median(y)) / (np.ptp(y) + 1e-9) - 1.2 * k, lw=0.6, color="#333333")
                for pk, w in zip(peaks, wide):
                    ax.plot(pk / FS, (y[pk] - np.median(y)) / (np.ptp(y) + 1e-9) - 1.2 * k, "v" if w else "o",
                            color="#C0392B" if w else "#2E86C1", ms=3)
            ax.set_title(f"{h} ({side}) lead {'II' if lead == 1 else 'V1'}, {len(strips_h)} points", fontsize=8)
            ax.set_yticks([])
    for ax in axes[-1]:
        ax.set_xlabel("s")
    fig.tight_layout()
    fig.savefig(out / "strips_example.png", dpi=110)
    print("written", out)


if __name__ == "__main__":
    main()
