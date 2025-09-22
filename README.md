# Incident AKI / CKD Survival Modeling

This repo builds an **incident AKI/CKD prediction cohort** from MIMIC-III/IV, engineers features, and trains multiple machine learning and survival models.  

It supports:  

* Cohort construction with admission-level and patient-level de-duplication  
* Exclusion of primary diagnoses and lab-based removal of pre-admission AKI  
* Feature engineering across demographics, insurance, comorbidities, medications/procedures, **pre-ICU vitals (48h)**, **pre-ICU labs (7d)**, and **fluids (48h)**  
* Flexible imputation (3-tier by version, gender, age ±2; adds *_missing flags)  
* Training/evaluation of classical ML, deep models, and survival models (CoxPH, DeepSurv)  
* External validation: train on MIMIC-IV, test on MIMIC-III  

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

Features for CKD survival modeling are built in a similar way, under
ckd_survivial directory.

---

## 4) Imputation

**Script:** `src/imputation.py`  

Fast 3-tier strategy (MIMIC-III/IV):  
1. version + gender + age±2  
2. version + age±2 (ignore gender)  
3. version-specific medians  

Adds `_missing` flags by default so models can capture missingness.  

---

## 5) Model Training

- **Classical ML:** `train_logreg.py`, `train_rf.py`, `train_xgboost.py`  
- **Deep models:** `train_dnn.py`, `train_selfattn.py`, `train_dcn.py`  
- **Survival:** `train_deepsurv_time_to_ckd.py` (DeepSurv Cox model)  

---

## 6) Evaluation

- **Classification models:** AUROC, AUPRC, calibration.  
- **Survival models:** Harrell’s C-index, time-dependent AUROC/AUPRC, mean AUC, iAUC/iAUPRC.  
- **External validation:** train on MIMIC-IV, test on MIMIC-III.  

---

## 7) Quickstart

```bash
# 1) Compile data
python -m src.compile_data --output_pickle data.pkl

# 2) Build targets
python -m src.compute_target_admissions --keep_autocar
python -m src.incident_aki_target
python -m src.dedup_patient_level

# 3) Engineer features
python -m src.features.demographics        --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.insurance           --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.comorbidities       --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.meds_procedures_history --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.vitals_preicu_48h   --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.labs_preicu_7d      --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.fluids_preicu_48h   --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.notes               --data-pkl data.pkl --target target.parquet --outdir .

# 4) Merge all features
python -m src.build_features --target target.parquet --features-dir . --outdir ./features

# 5) Imputation
python -m src.imputation

# 6) Train model (example: XGBoost)
python -m src.train_xgboost --input features_ckd.parquet --out-prefix artifacts/xgb_ckd/xgb_ckd
```
---

## 8) Notes & Caveats

* **Education** not in MIMIC → omitted  
* **Pre-AKI removal** uses KDIGO labs in prior 7d window (requires `labevents`)  
* **Regex safety**: meds regex uses non-capturing groups to avoid `str.contains` warnings  
* **Memory**: CHARTEVENTS and LABEVENTS are large → use chunked I/O for extensions  
* **Advanced ICU signals** (CO, PAP, VT, VE, GCS) not currently included; can be added by extending `itemids.py`, `compile_data.py`, and `vitals_preicu_48h.py`  

---

## 9) 📂 Source Tree (abridged)
```bash
src/
  compile_data.py
  compute_target_admissions.py
  incident_aki_target.py
  dedup_patient_level.py
  build_features.py
  imputation.py
  train_logreg.py / train_rf.py / train_xgboost.py
  train_dnn.py / train_selfattn.py / train_dcn.py
  train_deepsurv_time_to_ckd.py
  features/
    demographics.py, insurance.py, comorbidities.py,
    meds_procedures_history.py, vitals_preicu_48h.py,
    labs_preicu_7d.py, fluids_preicu_48h.py, notes.py
  utils/
    io.py, itemids.py, time_windows.py, meds_maps.py,
    dx_maps.py, agg.py
```
