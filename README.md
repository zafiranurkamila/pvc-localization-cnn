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

- **Laptop ini (Intel HD, no GPU):** tempat menulis & debug kode. Jalankan
  semuanya dengan subset kecil data (mis. 20-30 pasien) supaya cepat dites.
  Tidak untuk training penuh / hyperparameter search.
- **PC rumah (ada GPU):** tempat training penuh — 5-fold CV x 8 skenario x
  hyperparameter tuning. Pindahkan project (git push/pull, atau copy folder
  `src/`, `scripts/`, `configs/`, dan `data/`) ke sana untuk run berat.
- Supaya kode jalan di kedua mesin tanpa diubah, semua modul training/model
  harus pilih device otomatis, bukan hardcode:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
  Ini akan dipakai di `src/pvc_localization/training/trainer.py` saat fase itu dikerjakan.

## Project layout

```
data/
  raw/            raw 12-lead ECG CSVs (per patient, one file per HospitalID)
  processed/      noise-reduced ECG CSVs (primary input per proposal Bab III.3.1.5)
  external/       Diagnosis.xlsx (labels)
  interim/        generated artifacts: patient_index.csv, segmented beats, cached features
src/pvc_localization/
  data/           loading raw/processed ECG + labels, PyTorch Dataset
  preprocessing/  filtering, R-peak/QRS detection, PVC beat segmentation, normalization
  features/       psd.py, wavelet.py, hos.py — one feature extractor per method
  models/         CNN branch per feature type + multi-branch fusion model
  training/       k-fold CV loop, class-weighting, hyperparameter search
  evaluation/     accuracy/precision/recall/F1/macro-F1/balanced-acc/AUC/confusion matrix
  xai/            feature importance / SHAP on the best model per scenario
scripts/          numbered, runnable entry points (the actual pipeline, in order)
notebooks/        exploration/sanity-check notebooks, not the source of truth
configs/          scenario definitions (which features go into which CNN run)
results/          figures, metrics, saved model checkpoints (gitignored)
docs/reference/   proposal PDF for reference while implementing
```

## Status

**Fase 1-5 selesai (laptop, CPU):**
- ✅ Data: 329 PVC, RVOT 254, LVOT 75 (imbalance 3.4:1)
- ✅ Preprocessing: bandpass+notch (SOS-stable), QRS-width PVC detection, beat segmentation (300-500ms)
- ✅ Features: PSD (384D), Wavelet CWT scalogram (12×64×1600), HOS bispectrum (792D)
- ✅ Dataset: PyTorch lazy-loading, stratified split (no patient leakage), support fusion
- ✅ Models: 3-branch CNN (PSD 1D-Conv, Wavelet 2D-Conv, HOS 1D-Conv), baseline
- ✅ Training: 5-fold CV loop, class weighting, device-agnostic (GPU-ready)

**Fase 6-10 ready untuk GPU:**
- 8 skenario fitur (baseline + 3 individual + 4 kombinasi) `scripts/04_train_scenario.py`
- Metrics: accuracy, precision, recall, F1, macro-F1, balanced-acc, AUC
- Next: XAI + analysis setelah training di rumah

See [ROADMAP.md](ROADMAP.md) and [GPU_TRAINING.md](GPU_TRAINING.md) for details.
