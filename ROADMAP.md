# Roadmap — urutan ngoding

Setiap fase = 1 sesi kerja yang bisa diselesaikan lalu di-checkpoint (commit / catat hasil)
sebelum lanjut ke fase berikutnya. Jangan loncat ke feature extraction sebelum
segmentasi beat (Fase 2) beres dan tervalidasi visual — kalau segmentasi salah,
semua fitur & model di atasnya ikut salah.

## Fakta data yang sudah diverifikasi (`scripts/01_build_label_index.py`)

- 334 pasien total, 329 setelah difilter `Type == PVC` (cocok Tabel 3.1 proposal).
- Label: RVOT 254, LVOT 75 (cocok Tabel 3.2) — **imbalance ratio ~3.4:1**.
- Tiap pasien = 1 file CSV, 12 kolom (lead), **panjang sinyal berbeda-beda per
  pasien** (contoh: pasien 1000364 = 11.865 sample; pasien 1003838 = 60.276
  sample). Ini strip ECG kontinu, bukan beat yang sudah dipotong → **wajib ada tahap deteksi & segmentasi beat PVC sebelum ekstraksi fitur.**
- File `raw/<id>.csv` dan `processed/<id>.csv` **panjangnya tidak sama** untuk
  pasien yang sama (11.865 vs 8.690 baris pada contoh di atas). Jangan asumsikan
  keduanya sample-aligned satu-satu — perlu dicek ulang di Fase 1 apakah ini
  beda durasi rekaman, beda resampling, atau efek edge dari filter.
- **Sampling rate terkonfirmasi: 2000 Hz** — dari paper Zheng et al., *Scientific Data* (2020), https://doi.org/10.1038/s41597-020-0440-8 ("EP WorkMate System,
  Abbott, sampling rate 2000 Hz"). Sudah diset di `config.SAMPLING_RATE_HZ`.
  Paper juga menyebut 257 RVOT / 77 LVOT total (termasuk VT) — cocok dengan
  254/75 hasil filter PVC kita.

## Fase 0 — Environment (sudah bisa langsung dikerjakan)

- [ ] `python -m venv .venv` lalu `pip install -r requirements.txt` dan `pip install -e .`
- [ ] Jalankan ulang `python scripts/01_build_label_index.py` untuk pastikan environment baru menghasilkan output yang sama.

## Fase 1 — Eksplorasi Data (`notebooks/01_data_exploration.ipynb`)

Tujuan: kenal bentuk sinyal sebelum menulis pipeline otomatis.

- Plot 12-lead raw vs processed untuk beberapa pasien RVOT dan LVOT berdampingan.
- Cari tahu kenapa panjang raw vs processed berbeda (lihat catatan di atas).
- Cek satuan/skala amplitudo (raw terlihat skala kecil ~ratusan, processed
  skala ribuan — kemungkinan unit ADC vs µV berbeda; perlu diseragamkan).
- Identifikasi visual bentuk kompleks QRS lebar (ciri PVC) di sinyal supaya
  Fase 2 bisa divalidasi "matanya" sesuai teori di Bab 2.2.1 proposal.

## Fase 2 — Preprocessing & Segmentasi Beat (`src/pvc_localization/preprocessing/`)

Ini fase paling krusial dan paling belum terdefinisi di proposal — perlu keputusan desain:

1. **Filtering** (`filtering.py`): bandpass (mis. 0.5–40 Hz) + notch 50/60 Hz
   untuk baseline wander & powerline noise (pakai `scipy.signal`). Berlaku ke
   data `raw/` kalau mau bikin skenario pembanding "efek reduksi noise".
2. **Deteksi beat** (`beat_detection.py`): pakai `neurokit2.ecg_process` /
   `ecg_peaks` untuk deteksi R-peak per lead (biasanya lead II). PVC punya QRS
   lebar (>120ms) dan tanpa gelombang P — bisa dipakai neurokit2's built-in
   ectopic detection atau kriteria lebar QRS manual.
3. **Segmentasi**: potong window tetap di sekitar tiap R-peak (mis. -200ms
   sampai +400ms) untuk membentuk satu "beat" per lead x 12 lead. Keputusan
   panjang window ini menentukan bentuk input semua fitur di fase berikutnya.
4. **Normalisasi** (`normalization.py`): per-beat z-score atau min-max, biar
   skala amplitudo raw vs processed tidak jadi confound.
5. Output: simpan beat-beat yang sudah tersegmentasi ke `data/interim/beats/`
   (mis. `.npy` per pasien, shape `(n_beats, 12, window_len)`), plus tabel
   metadata (pasien, index beat, label) — ini yang jadi satuan sampel untuk
   training model, **bukan** 1 pasien = 1 sampel.

Validasi fase ini dengan plot overlay beat-beat hasil segmentasi sebelum lanjut.

## Fase 3 — Ekstraksi Fitur (`src/pvc_localization/features/`)

Satu modul per metode, dengan fungsi `extract(beat: np.ndarray) -> np.ndarray`
yang konsisten sinyaturnya supaya bisa dipanggil seragam dari `configs/scenarios.yaml`.

- `psd.py` — Welch's method (`scipy.signal.welch`) → vektor 1D per lead, sesuai Persamaan 2.1/3.1.
- `wavelet.py` — DWT (PyWavelets, mother wavelet db4/db6 sesuai Bab 2.4.2)
  untuk koefisien 1D, ATAU CWT → scalogram 2D untuk cabang 2D-Conv sesuai
  Bab 2.6.2 poin (b). Proposal menyebut scalogram untuk arsitektur multi-branch,
  jadi condong ke CWT scalogram (`pywt.cwt`).
- `hos.py` — bispectrum (Persamaan 2.2/3.4) via estimasi langsung (FFT-based
  triple product) atau library seperti `spectrum`/`stingray`, dan/atau momen
  orde-3 (skewness) sebagai fitur pelengkap yang lebih murah dihitung.

Validasi: bandingkan fitur pasien RVOT vs LVOT secara visual (apakah ada
perbedaan pola yang masuk akal) sebelum masuk ke model.

## Fase 4 — Dataset & Split (`src/pvc_localization/data/dataset.py`)

- Stratified hold-out 80/20 di level **pasien** (bukan di level beat) supaya
  tidak ada data leakage — beat dari pasien yang sama tidak boleh tersebar ke
  train dan test sekaligus.
- 5-Fold CV di dalam 80% data training (juga stratified per pasien).
- `torch.utils.data.Dataset` yang menerima daftar fitur mana yang aktif sesuai
  skenario (baseline/psd/wavelet/hos/kombinasi).

## Fase 5 — Model CNN (`src/pvc_localization/models/`)

- `baseline_cnn.py`: CNN sederhana 1D-conv langsung di atas 1 fitur (skenario 0).
- `branches.py`: satu class per cabang — PSD branch (1D-Conv), Wavelet branch
  (2D-Conv untuk scalogram), HOS branch (Conv untuk bispectrum/dense untuk momen).
- `fusion_cnn.py`: model yang menerima daftar cabang aktif, concat sebelum
  dense layer terakhir — ini yang dipakai untuk skenario 4-7 (kombinasi fitur).

## Fase 6 — Training Loop (`src/pvc_localization/training/trainer.py`)

- Loop generik: terima (model_fn, dataset, config) → jalankan 5-fold CV,
  hitung class weight *hanya dari fold training* (Persamaan 3.14), simpan
  metric per fold, retrain di full 80% data, evaluasi di 20% test.
- `hyperparam_search.py`: grid/random search sesuai Tabel 3.5 (learning rate,
  batch size, epoch, filter, kernel, dropout, optimizer).

## Fase 7 — Evaluasi (`src/pvc_localization/evaluation/metrics.py`)

Implementasikan sekali, pakai di semua skenario: accuracy, precision, recall,
specificity, F1, macro-F1, balanced accuracy, AUC, confusion matrix (Bab 3.1.9)
— gunakan `sklearn.metrics` untuk sebagian besar, macro-F1 & balanced accuracy
sudah tersedia langsung di sklearn.

## Fase 8 — Jalankan 8 Skenario (`scripts/06_train_scenario.py` + `configs/scenarios.yaml`)

Loop resmi sesuai Bab 3.1.14: Skenario 0 (baseline) s.d. Skenario 7 (fusi
ketiganya). `config.FEATURE_SCENARIOS` sudah berisi nama & anggota tiap skenario.

## Fase 9 — XAI (`src/pvc_localization/xai/explain.py`)

SHAP atau feature importance pada model dengan performa terbaik, sesuai Bab 3.1.13.

## Fase 10 — Analisis & Laporan (`notebooks/04_results_analysis.ipynb`)

Kumpulkan metrik semua skenario jadi satu tabel pembanding (mirip Tabel 3.8),
plot confusion matrix per skenario, dan tulis kesimpulan fitur/kombinasi mana
yang paling optimal untuk BAB IV/V laporan TA.

---

**Keputusan yang masih perlu kamu ambil sebelum Fase 2-3 jalan:**
1. ~~Sampling rate~~ — sudah confirmed 2000 Hz.
2. Definisi window segmentasi beat (berapa ms sebelum/sesudah R-peak).
3. CWT vs DWT untuk cabang wavelet (proposal condong ke scalogram 2D → CWT).
4. Library/metode estimasi bispectrum HOS yang dipakai.

Diskusikan tiap keputusan ini di sesi berikutnya sebelum saya bantu implementasi
modulnya — supaya kodenya tidak perlu dirombak ulang.
