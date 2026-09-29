# PVC Localization — HOS vs Wavelet vs PSD Feature Comparison (CNN)

TA: *Analisis dan Perbandingan Efektivitas Fitur Higher-Order Statistics, Wavelet,
dan Power Spectrum Density dalam Model CNN untuk Lokalisasi PVC (RVOT vs LVOT)*.

Dataset: Zheng, J. — *A 12-Lead ECG Database to Identify Origins of Idiopathic
Ventricular Arrhythmia* (334 patients), via Figshare.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Development workflow (2 mesin)

- **Laptop (tanpa GPU):** menulis & debug kode, mengolah hasil (tabel, grafik, statistik ringan).
- **PC (GPU, data asli):** semua training, tuning, multi-seed CV, dan XAI.
  Hasil di-push dari PC, lalu di-pull di laptop.
- Device dipilih otomatis (`cuda` jika ada, selain itu `cpu`), jadi kode sama di kedua mesin.

## Project layout

```
data/
  raw/            raw 12-lead ECG CSVs (per patient, one file per HospitalID)
  processed/      noise-reduced ECG CSVs (input utama sesuai proposal)
  external/       Diagnosis.xlsx (labels)
  interim/        patient_index.csv, segmented beats
  cache/          cached features
src/pvc_localization/
  data/           loading ECG + labels, PyTorch Dataset, split train/test per pasien
  preprocessing/  filtering, R-peak/QRS detection, PVC beat segmentation, normalization
  features/       psd.py, wavelet.py (CWT scalogram), hos.py (bispectrum)
  models/         CNN branch per fitur + multi-branch fusion, BaselineCNN (sinyal mentah)
  training/       patient-level k-fold CV, class weight, final fit + evaluasi test
  evaluation/     accuracy/precision/recall/F1/macro-F1/balanced-acc/AUC/confusion matrix
scripts/          entry point bernomor (pipeline, berurutan)
results/          hasil per skenario, model, prediksi test, tuning, multi-seed, XAI
docs/reference/   proposal PDF
```

## Pipeline

| Script | Fungsi | Output |
|---|---|---|
| `01_build_label_index.py` | index pasien + label RVOT/LVOT | `data/interim/patient_index.csv` |
| `03_full_segmentation.py` | segmentasi beat PVC | `data/interim/beats/` |
| `04_train_scenario.py` | 1 skenario: 5-fold CV per pasien + final fit, diuji di data test | `results/scenarios/`, `results/models/`, `results/predictions/` |
| `06_run_all_scenarios.py` | 8 skenario sekaligus (resumable); `--tuned` untuk hasil tuning | `results/scenarios/` |
| `08_tune_scenarios.py` | tuning bertahap Tabel 3.5, hanya CV di data train | `results/tuning/` |
| `10_export_predictions.py` | ekspor ulang prediksi test model tuned | `results/predictions/` |
| `11_xai_shap.py` | permutation importance per cabang + SHAP per lead | `results/xai/` |
| `12_stats_fixed_setting.py` | bootstrap per pasien (2000×): CI 95% + uji selisih berpasangan di data test | `results/stats_fixed_setting.json` |
| `13_multiseed_cv.py` | CV setting tetap dengan 5 seed (data test tidak dipakai) | `results/multiseed/` |

Contoh (di PC):

```bash
python scripts/06_run_all_scenarios.py --epochs 50 --batch-size 64 --skip-beats
python scripts/12_stats_fixed_setting.py
python scripts/13_multiseed_cv.py
python scripts/11_xai_shap.py --models psd_wavelet_tuned
```

## Desain eksperimen

- **Data:** 329 pasien PVC (RVOT 254, LVOT 75), 1.767 beat. Split per pasien, stratified:
  263 pasien train (60 LVOT), 66 pasien test (15 LVOT).
- **8 skenario:** baseline (CNN sinyal mentah 12-lead), PSD, Wavelet, HOS, PSD+Wavelet,
  PSD+HOS, Wavelet+HOS, PSD+Wavelet+HOS.
- **Setting tetap** (dipakai untuk membandingkan fitur): Adam, lr 0,001, batch 64, 50 epoch,
  64 filter, dropout 0,3, class weight (termasuk baseline).
- **Validasi:** 5-fold CV per pasien (StratifiedGroupKFold); beat satu pasien tidak pernah
  ada di train dan validasi sekaligus. Metrik utama: Macro F1.
- **Tuning:** per skenario, bertahap (91 run), dipilih dari CV di data train saja.
- **XAI:** dijalankan setelah model selesai dilatih, hanya untuk penjelasan.

## Hasil

### Multi-seed CV (setting tetap, 5 seed × 5 fold, data train saja)

| Skenario | Macro F1 | sd antar seed | AUC | Peringkat per seed | Rata-rata peringkat |
|---|---|---|---|---|---|
| PSD+Wavelet+HOS | 0,820 | 0,015 | 0,886 | 4 2 1 2 1 | 2,0 |
| **PSD+Wavelet** | **0,818** | 0,012 | 0,886 | 1 1 2 3 3 | 2,0 |
| Baseline (acuan) | 0,815 | 0,013 | 0,893 | – | – |
| PSD+HOS | 0,803 | 0,014 | 0,890 | 2 5 3 4 4 | 3,6 |
| Wavelet+HOS | 0,798 | 0,025 | 0,875 | 3 3 6 1 2 | 3,0 |
| PSD | 0,789 | 0,008 | 0,884 | 6 6 4 5 5 | 5,2 |
| HOS | 0,781 | 0,009 | 0,870 | 5 7 5 6 6 | 5,8 |
| Wavelet | 0,753 | 0,025 | 0,855 | 7 4 7 7 7 | 6,4 |

Perbandingan berpasangan PSD+Wavelet pada 25 fold yang sama (Wilcoxon):
- signifikan lebih baik dari PSD (p = 0,007), HOS (p = 0,007), dan Wavelet (p < 0,001);
- tidak berbeda dari PSD+Wavelet+HOS (p = 0,458) dan baseline (p = 0,812).

### Data test (66 pasien), Macro F1

| Skenario | Setting tetap run 1 | Setting tetap run 2 | Tuned |
|---|---|---|---|
| **PSD+Wavelet** | **0,894** | **0,873** | **0,876** |
| PSD+Wavelet+HOS | 0,861 | 0,873 | 0,864 |
| PSD | 0,846 | 0,846 | 0,854 |
| Wavelet | 0,879 | 0,812 | 0,817 |
| PSD+HOS | 0,836 | 0,802 | 0,801 |
| Wavelet+HOS | 0,730 | 0,756 | 0,770 |
| HOS | 0,642 | 0,644 | 0,725 |
| Baseline | 0,876 | 0,856 | 0,870* |

\* baseline tuned = tanpa class weight dan tanpa tuning (versi proposal awal).

### XAI (penurunan AUC saat input satu cabang diacak)

| Model | Wavelet | PSD | HOS |
|---|---|---|---|
| PSD+Wavelet (tuned) | 0,308 | 0,081 | – |
| PSD+Wavelet+HOS (tuned) | 0,204 | 0,090 | 0,013 |

Lead prekordial V1–V6 menyumbang sekitar 62% pengaruh (SHAP) pada kedua model.

## Kesimpulan & keputusan (bimbingan 29 Sep 2026)

- **Model utama: PSD+Wavelet.** Di CV multi-seed (data train saja) berada di kelompok teratas,
  signifikan lebih baik dari setiap fitur tunggal, dan setara dengan fusi tiga fitur.
  Dipilih daripada fusi tiga fitur karena lebih sederhana dan HOS hampir tidak dipakai model (XAI).
  Di data test selalu tertinggi pada tiga eksperimen.
- **HOS fitur terlemah**, baik sendiri maupun di dalam fusi.
- **Baseline** (CNN sinyal mentah + class weight) setara dengan PSD+Wavelet; keunggulan model
  berbasis fitur ada pada interpretasi (frekuensi, waktu-frekuensi, lead).
- Hasil utama laporan disusun per kombinasi fitur; setting tetap dan tuning adalah metode pengujian.
