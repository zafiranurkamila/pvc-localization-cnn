# GPU Training Instructions

Kode di project ini sudah siap untuk training di PC rumah (dengan GPU). Berikut langkah-langkah:

## Persiapan di Laptop (sudah selesai)

- [x] Data: ECG raw & processed (334 patients × 2 file = 668 CSVs) + diagnosis label
- [x] Feature extraction: PSD, Wavelet (CWT scalogram), HOS (bispectrum + moments)
- [x] Beat detection & segmentation: QRS-width classifier (bukan RR-interval) + sliding window 300ms-500ms
- [x] PyTorch Dataset: lazy loading, on-the-fly feature extraction, support feature fusion
- [x] CNN architectures: 3-branch (PSD 1D-Conv, Wavelet 2D-Conv, HOS 1D-Conv) + fusion classifier
- [x] Training loop: 5-fold CV, stratified split at patient level (no data leakage)
- [x] Metrics: accuracy, precision, recall, F1, macro-F1, balanced-accuracy, AUC, confusion matrix

## Pindah ke GPU Rumah

1. **Copy folder ke sana:**
   ```
   src/                          ← semua feature/model/training code (device-agnostic)
   scripts/                      ← semua orchestration scripts
   data/external/Diagnosis.xlsx  ← label file (copy saja, kecil)
   data/raw/ & data/processed/   ← ECG CSV files (besar, ~2GB total)
   requirements.txt
   pyproject.toml
   ```
   (Data sudah ada di `data/interim/patient_index.csv` dari script `01_build_label_index.py`)

2. **Install dependencies di GPU machine:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # atau .venv\Scripts\activate di Windows
   pip install -r requirements.txt
   ```
   **Note:** di GPU machine, gunakan PyTorch versi terbaru (dengan CUDA support) dan numpy yang compatible. Laptop saat ini punya numpy/PyTorch mismatch minor yang tidak perlu di-fix (bukan critical path untuk training).

3. **Jalankan training untuk satu skenario** (test dulu):
   ```bash
   python scripts/04_train_scenario.py --scenario psd --epochs 20 --batch-size 32
   ```

4. **Jalankan semua 8 skenario** (dalam urutan):
   ```bash
   for scenario in baseline psd wavelet hos psd_wavelet psd_hos wavelet_hos psd_wavelet_hos; do
     python scripts/04_train_scenario.py --scenario $scenario --epochs 50 --batch-size 64
   done
   ```

5. **Hasil tersimpan di:**
   ```
   results/scenarios/  ← CV metrics per skenario (JSON)
   results/models/     ← best model checkpoints (plan untuk fase berikutnya)
   results/metrics/    ← detailed eval per fold (plan untuk fase berikutnya)
   results/figures/    ← confusion matrix, learning curve (plan untuk fase berikutnya)
   ```

## Konfigurasi untuk Tuning

Edit `src/pvc_localization/config.py` untuk adjust:
- `SAMPLING_RATE_HZ = 2000` (confirmed dari paper)
- `BEAT_PRE_MS = 300, BEAT_POST_MS = 500` (window segmentasi)
- `CWT_WAVELET = "cmor1.5-1.0"` (mother wavelet)
- `CWT_NUM_SCALES = 64` (jumlah skala CWT)
- `RANDOM_SEED = 42` (reproducibility)
- `TEST_SIZE = 0.2` (test fraction)
- `N_FOLDS = 5` (K-fold)

## Device Handling

Semua code sudah pakai auto-detection:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Jadi tidak perlu ubah kode sama sekali untuk pindah ke GPU.

## Estimated Runtime (GPU RTX 3070 / A100)

- 1 skenario, 50 epoch, 5-fold CV: ~2-5 menit
- 8 skenario semua: ~20-40 menit total

## Next Phase (Fase 9-10)

Setelah training selesai di GPU rumah:
1. **Fase 9 (XAI):** SHAP feature importance pada best model per skenario
2. **Fase 10 (Analysis):** Kumpulkan semua metrik jadi comparison table, plot confusion matrix, tulis kesimpulan

Kode untuk fase ini belum dibuat (masih placeholder di `src/pvc_localization/xai/`).

---

**Catatan:** Semua fitur extraction happen lazily (on-the-fly per batch) untuk minimize memory. Kalau mau cache feature dulu (faster epoch tapi gede disk), uncomment parameter `cache_dir` di `PVCBeatsDataset`.
