# AKI and subsequent CKD Modeling

This repo builds an **AKI/CKD prediction cohort** from MIMIC-III/IV, engineers features, and trains multiple machine learning and survival models.  

It supports:  

* Cohort construction with admission-level and patient-level de-duplication  
* Exclusion of primary diagnoses and lab-based removal of pre-admission AKI  
* Feature extraction of demographics, insurance, comorbidities, medications/procedures
* Feature engineering of **pre-ICU vitals (48h)**, **pre-ICU labs (7d)**, **fluids (48h)** , and **clinical notes (7d)** for AKI
* Feature engineering of **pre-event labs (90d)** and **per-event procedures (90d)** for CKD and death 
* Flexible imputation (3-tier by version, gender, age ±2; adds *_missing flags)  
* Training/evaluation of classical ML, deep models, and survival models (XGBoost-Cox, DeepSurv)  
* External validation: train on MIMIC-IV, test on MIMIC-III  

---

## 0) Requirements

* Python 3.10+
* `data.pkl` (output of `src/compile_data.py`, or a symlink to an existing one) present at the repo root
* Packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Both `run_aki_pipeline.sh` and `run_ckd_survival.sh` invoke `python3` by default. Activate your environment first, or override with `PYTHON=/path/to/python3 bash run_aki_pipeline.sh`.

---

## 1) Data Compilation

**Script:** `src/compile_data.py`  

Loads raw MIMIC tables and outputs a single `data.pkl` bundle.  

**Tables loaded:**
- `patients`, `admissions`, `icustays`, `services`, `transfers`  
- Diagnoses (`diagnosis`), procedures, prescriptions  
- `labitems`, `labevents`  
- `d_items` (for CHARTEVENTS itemid mapping)  
- `chartevents` (vitals + ABG pH & lactate, extendable)  
- `inputevents`, `outputevents` (urine subset)  

---

## 2) Target Construction

**Scripts:**  
- `src/compute_target_admissions.py`  
- `src/incident_aki_target.py`  
- `src/dedup_patient_level.py`  

Rules:  
1. Adults only (≥18).  
2. Exclude **primary AKI** admissions (`seq_num==1 & is_aki==1`).  
   *Primary CKD admissions are retained for AKI→CKD trajectory tracking.*  
3. Flag admission-level AKI/CKD if codes appear in later admissions.  
4. Provide `ckd_only_flag` and `ckd_admission_flag`.  
5. Flag **pre-AKI** using KDIGO labs in prior 7d/48h (not excluded, only flagged).  
6. Exclude AutoCar admissions by default (`--keep_autocar` to override).  
7. Patient-level deduplication via `dedup_patient_level.py`.  

---

## 3) Feature Engineering

**Script:** `src/build_features.py`  

Feature modules under `src/features/`:  

- `demographics.py` – age, sex, race, marital status  
- `insurance.py` – insurance categories  
- `comorbidities.py` – Charlson/Elixhauser features  
- `meds_procedures_history.py` – medication/procedure history  
- `vitals_preicu_48h.py` – vitals aggregation  
- `labs_preicu_7d.py` – lab values aggregation  
- `fluids_preicu_48h.py` – fluid balance  
- `notes.py` – clinical text features (optional)  

Features for CKD survival modeling are built in a similar way, under the
`ckd_survival/` directory (see §5b below).

---

## 4) Imputation

**Script:** `src/imputation.py`  

Fast 3-tier strategy (MIMIC-III/IV):  
1. version + gender + age±2  
2. version + age±2 (ignore gender)  
3. version-specific medians  

Adds `_missing` flags by default so models can capture missingness.  

---

## 5) AKI Model Training

- **Classical ML:** `train_logreg.py`, `train_rf.py`, `train_xgboost.py`  
- **Deep models:** `train_dnn.py`, `train_selfattn.py`, `train_dcn.py`  

Each `train_*.py` script excludes patients with `cmb_ckd==1` or `renal_impaired_at_adm==1`
before splitting (this is on top of the CONSORT-level exclusions already applied in target
construction), trains on MIMIC-IV (`GroupShuffleSplit` on `subject_id`; default
`--test-size 0.2 --val-size 0.1`, i.e. 70/10/20 train/val/test), and externally validates
on all of MIMIC-III. Each writes `.joblib`, `.features.txt`,
`.splits.csv`, `.meta.json`, `.predictions.csv`, `.metrics.json`, and `.summary.txt` to
`--out-prefix`.

---

## 5b) CKD / Death Survival Pipeline (`ckd_survival/`)

Runs on the post-AKI cohort, with `cmb_ckd`/`renal_impaired_at_adm` patients excluded
**upstream of target construction** (via `filter_upstream_exclusions.py`), not just at
training time.

- **Targets:** `compute_target_admissions.py` → `filter_upstream_exclusions.py` →
  `incident_ckd_target.py`
- **Features:** `features/demographics_comorbidity_ckd.py`, `features/labs_preckd.py`,
  `features/meds_procedures_history.py` → merged by `features/merge_ckd_features.py`.
  `features/build_sex_interaction_features.py` additionally builds a 67-feature
  variable×sex interaction set (34 base features + 33 interaction terms) from the merged
  CKD feature table.
- **Models:** `train_xgboost_time_to_ckd.py` / `train_xgboost_time_to_death.py` (XGBoost-Cox),
  `train_deepsurv_time_to_ckd.py` / `train_deepsurv_time_to_death.py` (DeepSurv). Each takes
  `--input <features_parquet> --out-prefix <path>`, so the same training scripts run on
  either the native or sex-interaction feature set.
- **Evaluation:** Harrell's C-index, time-dependent AUROC/AUPRC, Brier score, iAUC/iAUPRC
  (test split + MIMIC-III external).

Run end-to-end with `run_ckd_survival.sh` (see Quickstart) — it runs both feature sets
(native + sex-interaction) for time-to-CKD, and the native feature set only for
time-to-death.

---

## 6) Evaluation

- **Classification models:** AUROC, AUPRC, calibration.  
- **Survival models:** Harrell’s C-index, time-dependent AUROC/AUPRC, mean AUC, iAUC/iAUPRC.  
- **External validation:** train on MIMIC-IV, test on MIMIC-III.  

---

## 7) Quickstart

Place (or symlink) `data.pkl` at the repo root, then:

```bash
# 1) AKI pipeline: targets -> features -> imputation -> 6 AKI models -> artifacts/
bash run_aki_pipeline.sh

# 2) CKD/death survival pipeline: needs step 1's features/features_all*.parquet
#    -> ckd_survival_run/artifacts{,_sex_interaction}/
bash run_ckd_survival.sh
```

Both scripts are idempotent per stage via flags — e.g. re-run just training after
changing a model:

```bash
bash run_aki_pipeline.sh --skip-targets --skip-features
bash run_ckd_survival.sh --skip-targets --skip-features
```

To run the underlying steps by hand instead (what the scripts above wrap):

```bash
# Targets
python -m src.compute_target_admissions --data-pkl data.pkl --keep_autocar
python -m src.incident_aki_target
python -m src.dedup_patient_level

# Features
python -m src.features.demographics            --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.insurance                --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.comorbidities             --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.meds_procedures_history   --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.vitals_preicu_48h         --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.labs_preicu_7d            --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.fluids_preicu_48h         --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.notes                     --data-pkl data.pkl --target target.parquet --outdir .  # optional, needs sentence-transformers

# Merge + impute
python -m src.build_features --target target.parquet --features-dir . --outdir ./features
python -m src.imputation

# Train one AKI model (example: XGBoost)
python -m src.train_xgboost --input features/features_all.parquet --out-prefix artifacts/xgboost_model/xgb
```

See `run_ckd_survival.sh` for the equivalent CKD/death survival steps
(`ckd_survival.src.*` modules).

---

## 8) Notes & Caveats

* **Education** not in MIMIC → omitted  
* **Pre-AKI removal** uses KDIGO labs in prior 7d window (requires `labevents`)  
* **Regex safety**: meds regex uses non-capturing groups to avoid `str.contains` warnings  
* **Memory**: CHARTEVENTS and LABEVENTS are large → use chunked I/O for extensions  
* **Advanced ICU signals** (CO, PAP, VT, VE, GCS) not currently included; can be added by extending `itemids.py`, `compile_data.py`, and `vitals_preicu_48h.py`  
* **H-statistics** (Friedman-Popescu, sex × predictor interaction) are computed in Python
  in each `train_*.py`, via `sklearn.inspection.partial_dependence` anchored on
  `dem_sex_F`/`dem_sex_M` — see `compute_hstats_with_anchor()` — writing
  `<out-prefix>.hstats_gender.csv`. No external R step is required.
* **DeepSurv device**: `ckd_survival/src/train_deepsurv_time_to_{ckd,death}.py` auto-select
  `mps` > `cuda` > `cpu` at runtime; there's no CLI flag to override this.
---

## 9) 📂 Source Tree (abridged)
```bash
src/
  compile_data.py
  compute_target_admissions.py
  incident_aki_target.py
  dedup_patient_level.py
  pre_aki_lab_flags.py
  build_features.py
  imputation.py
  train_logreg.py / train_rf.py / train_xgboost.py
  train_dnn.py / train_selfattn.py / train_dcn.py
  features/
    demographics.py, insurance.py, comorbidities.py,
    meds_procedures_history.py, vitals_preicu_48h.py,
    labs_preicu_7d.py, fluids_preicu_48h.py, notes.py
  utils/
    io.py, itemids.py, time_windows.py, meds_maps.py,
    dx_maps.py, agg.py

ckd_survival/src/
  compute_target_admissions.py
  filter_upstream_exclusions.py
  incident_ckd_target.py
  pre_aki_lab_flags.py
  train_xgboost_time_to_ckd.py / train_xgboost_time_to_death.py
  train_deepsurv_time_to_ckd.py / train_deepsurv_time_to_death.py
  features/
    demographics_comorbidity_ckd.py, labs_preckd.py,
    meds_procedures_history.py, merge_ckd_features.py,
    build_sex_interaction_features.py
  utils/
    io.py, agg.py, time_windows.py, meds_maps.py

run_aki_pipeline.sh    # AKI end-to-end
run_ckd_survival.sh    # CKD/death survival end-to-end
requirements.txt
```
