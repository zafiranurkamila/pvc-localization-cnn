# Reproducing the Journal of Medical Systems article

Article: *Premature Ventricular Contraction Origin Localization with a Fine-Tuned ECG Foundation Model and Referral of
Uncertain Cases* (Kamila and Mandala). This file lists the data, the setup and the commands behind every number of the
article and its Supplementary Information. The rest of the README describes earlier work (feature comparison).

## Data and permissions

| Cohort | Source | Access |
|---|---|---|
| First (development): 334 patients, 12-lead ECG, 2000 Hz | Zheng et al., Sci Data 2020, <https://doi.org/10.6084/m9.figshare.11926506> | Public, de-identified. Put the raw and noise-reduced CSV folders and `Diagnosis.xlsx` under `data/raw`, `data/processed` and `data/external` |
| Second (independent): 190 patients with mapping-point strips, 2.5 s, 1000 Hz | Casaú, Mallol, Pairés, Pont (UPF), <https://github.com/uripont/arrhythmia-origin-predictor>, release `dataset` | De-identified; used with the permission of its authors. Put `dataset_arrhythmias.zip` in `data/external/otva_upf/` |

Eligibility of the first cohort follows the `Type` column of `Diagnosis.xlsx` (329 PVC, 5 VT excluded); see
`scripts/01_build_label_index.py`. No data are stored in this repository.

## Setup

Python 3.11, PyTorch with CUDA (the article used one NVIDIA GeForce RTX 5060 Ti GPU).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pip freeze > results/environment.txt
```

`requirements.txt` lists the packages without pinned versions; `results/environment.txt` records the exact versions of
the machine that runs the commands.

ECGFounder checkpoint (once):

```bash
git clone https://github.com/PKUDigitalHealth/ECGFounder.git external/ECGFounder
curl -L -o external/ECGFounder/checkpoint/12_lead_ECGFounder.pth --create-dirs https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth
python scripts/23_nested_cv_ecgfounder.py --check
```

## Seeds

- Patient folds: `--split-seed` 42 (default), 7 and 123; inner folds use `split_seed + outer fold`.
- Training seeds of the five-seed ensembles are the same in every split.
- Visual review sample of the second cohort: seed 42 (`config.RANDOM_SEED`).

## Commands (in order)

| Step | Command | Article |
|---|---|---|
| Label index | `python scripts/01_build_label_index.py` | Methods 2.1, Table 2 |
| Beat extraction and PSD/CWT | `python scripts/03_full_segmentation.py` | Methods 2.1 |
| Extraction example | `python scripts/19_paper_fig_extraction.py` | Figure 1 |
| Sinus templates | `python scripts/20_sinus_beats.py` | ECG criteria |
| ECG criteria (features) | `python scripts/22_nested_cv_clinical_stack.py --features-only` | Methods 2.2 |
| Nested CV, split 42 | `python scripts/24_nested_cv_fusion.py --psd12 --augment --wd 0.0001 --ensemble 5` | Tables 4, 5 |
| Nested CV, splits 7 and 123 | the same with `--split-seed 7` and `--split-seed 123` | Tables 4, 5 |
| Number of PVC beats | `python scripts/25_min_beats_subgroup.py` | Table S6 |
| Referral | `python scripts/26_selective_classification.py` | Table 6, Figure 4 |
| Statistics, calibration, ECG-criteria model on the three splits | `python scripts/27_paper_statistics.py` | Tables 4--6, Figure 3 |
| Figures | `python scripts/28_paper_figures.py` | Figures 3, 4 |
| Second cohort: inspection and PVC check | `python scripts/29_inspect_external_dataset.py data/external/otva_upf/dataset_arrhythmias.zip` and `python scripts/30_check_external_pvc.py data/external/otva_upf/all_points_may_2024.pkl` | Methods 2.5 |
| (i) Unchanged external evaluation | `python scripts/31_external_validation_upf.py data/external/otva_upf/all_points_may_2024.pkl` | Table 7 |
| (ii) Local recalibration, (iii) local model adaptation | `python scripts/32_multicentre_upf.py data/external/otva_upf/all_points_may_2024.pkl` | Table 7 |
| Fusion vs ECGFounder at equal coverage | `python scripts/33_fusion_vs_founder.py` | Results 3.2, Table S3 |
| Visual review of the second cohort | `python scripts/34_upf_visual_review.py data/external/otva_upf/all_points_may_2024.pkl`, then `--summarize` | Methods 2.5 |
| Sex and age of the second cohort, errors by sex | `python scripts/36_upf_demographics.py data/external/otva_upf/all_points_may_2024.pkl` | Methods 2.5, Supplementary results |
| Criteria applied directly, errors, recording length, trade-off | `python scripts/35_priority2_analyses.py` | Tables S2, S4, S5, S7--S9 |
| Fusion vs ECGFounder at equal realized coverage | `python scripts/37_fusion_vs_founder_realized.py` | Results 3.2, Table S3 (B) |
| QC of the second-cohort extraction (sensitivity analysis) | `python scripts/38_upf_extraction_qc.py data/external/otva_upf/all_points_may_2024.pkl` | Methods 2.5, Results 3.4, Supplementary results |

Scripts 33, 35 and 37 use only saved predictions (no GPU); script 38 re-scores the second cohort with the stored
development models of (i) (`data/interim/external_final/development_ens5.pt`, written by script 31), without training. The development runs of Table S1 are the earlier result files
in `results/nested_cv/` (split 42).

## Pre-specified analyses (commits made before the results were computed)

| Analysis | Commit | Date |
|---|---|---|
| Repetition on new patient splits (`--split-seed`) | `3a8edda` | 2026-10-28 |
| Number of PVC beats, primary cutoff N >= 3 | `b38b06e` | 2026-10-28 |
| Referral, primary target coverage 90 % | `0cc6625` | 2026-10-28 |
| Second cohort, unchanged external evaluation | `8ceffd9` | 2026-10-29 |
| Second cohort, local recalibration and local model adaptation | `faaa8d7` | 2026-10-29 |

Analyses added after the review of the draft (no model retrained): `scripts/33` (commit `3e46563`), `scripts/34`
and its completed form (`a44eac5`), `scripts/35` (`fc14b5e`), `scripts/36` (`2309566`), `scripts/37` and `scripts/38`
(`f57f6bc`, results `b80fe01`).

## Outputs

`results/nested_cv/` (first cohort), `results/external_upf/` (second cohort), `results/priority2/` (analyses of
Tables S2, S4, S5, S7--S9). The Supplementary tables are built from these CSV files by `make_supplementary.py` of the
manuscript folder.
