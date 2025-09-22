# AKI Modeling – Project README

This repo builds a **secondary AKI/CKD prediction cohort** from MIMIC-III/IV and engineers features for modeling. It:

* Compiles raw MIMIC tables into a single `data.pkl` bundle
* Creates a **target** of secondary AKI/CKD (excludes primary diagnoses and **removes pre-admission AKI by labs**) with patient-level dedup
* Engineers features across demographics, insurance, comorbidities, meds/procedures history, **pre‑ICU vitals (48h)**, **pre‑ICU labs (7d)**, and **fluids (48h)**
* Provides utilities for QC and cohort analysis (incl. by gender & age buckets)

---

## 1) Data compilation

**Script:** `src/compile_data.py`

**What it loads (no row de-dup):**

* `patients`, `admissions`, `icustays`, `services`, `transfers`
* `diagnosis` (III/IV ICD), `procedures` (III/IV), `prescriptions`
* `labitems`, `labevents`
* `d_items` (for CHARTEVENTS label→itemid mapping)
* `chartevents` (filtered to vitals + ABG pH & lactate; easily extendable)
* `inputevents`, `outputevents` (urine subset)

**Output:** `data.pkl`

**Run:**

```bash
python -m src.compile_data --output_pickle data.pkl
```

---

## 2) Target construction

**Script:** `src/compute_target.py`

**Definition (secondary AKI/CKD):**

1. Adults (≥18)
2. **Exclude primary** (admitting) AKI/CKD (seq\_num == 1)
3. Mark **secondary AKI/CKD** if codes appear at any later diagnosis position within the index admission
4. Drop CKD‑only (to keep clean negatives when desired)
5. **Remove pre‑AKI** based on **lab KDIGO** in the **7 days prior to hospital admission** (calls `src/pre_aki_lab_flags.py`)
6. **Patient-level dedup** (configurable):

   * Default `--dedup_mode patient`: keep **first** admission for AKI=1, keep **last** admission for AKI=0
   * `--dedup_mode none`: skip dedup

**Outputs:**

* `target.parquet` (final)
* `cohort_counts.csv` (row counts by step)
* `pivot_overall.csv`, `pivot_mimic3.csv`, `pivot_mimic4.csv`
* `aki_ckd_death_target_first.csv` (first admission per subject; continuity aid)

**Run:**

```bash
python -m src.compute_target --data-pkl data.pkl --outdir . --dedup_mode patient
```

---

## 3) Feature engineering

All features are keyed by `version, subject_id, hadm_id` and merged later into `features_all.parquet`.

### 3.1 Demographics

**Script:** `src/features/demographics.py`

* **Includes:** sex, **age**, race (WHITE/BLACK/ASIAN/HISPANIC/OTHER/UNK), marital status (MARRIED/SINGLE/…)
* **Prefixes/columns:** `dem_sex_*`, `dem_race_*`, `dem_mar_*`, plus age retained from target
* **Note:** Education isn’t present in MIMIC

### 3.2 Insurance

**Script:** `src/features/insurance.py`

* **Includes:** `PRIVATE`, `MEDICARE`, `ASSISTANT` (Medicaid, government, self‑pay), `UNKNOWN`
* **Prefix:** `ins_*`

### 3.3 Comorbidities

**Script:** `src/features/comorbidities.py`

* **Includes:** HTN, DM, CAD, HF, COPD, cancer, infection, CKD, liver disease, obesity, etc.
* **Prefix:** `cmb_*`
* **Source:** Diagnosis tables (ICD-9/10)

### 3.4 Medications & Procedures history (lookback windows)

**Script:** `src/features/meds_procedures_history.py`

* **Windows:** e.g., `--windows 30d,6m` → `hx30d_*`, `hx180d_*`
* **Med classes (from PRESCRIPTIONS):** NSAIDs, diuretics, vasopressors, aminoglycosides, mannitol, colloid bolus

  * Columns: `hx30d_<class>_any`, `hx180d_<class>_any`
* **Procedures (from PROCEDURES ICD):** RRT (`rrt_any`), CRRT (`crrt_any`), mechanical ventilation (`mech_vent_any`)

  * Columns: `hx30d_rrt_any`, `hx180d_rrt_any`, `hx30d_crrt_any`, …

**Run:**

```bash
python -m src.features.meds_procedures_history --data-pkl data.pkl \
  --target target.parquet --outdir . --windows 30d,6m
```

### 3.5 Pre‑ICU labs (7 days)

**Script:** `src/features/labs_preicu_7d.py`

* **Analytes:** WBC, HGB, Platelets, Creatinine, BUN, Albumin, Sodium, Potassium, Bicarbonate, **Lactate**, **pH**
* **Stats:** min/max/mean/median/std/last/count → `lab7d_<analyte>_*`
* **Missingness filter:** drop analytes with coverage < threshold (e.g., `--drop-thresh 0.7`)

**Run:**

```bash
python -m src.features.labs_preicu_7d --data-pkl data.pkl \
  --target target.parquet --outdir . --drop-thresh 0.7
```

### 3.6 Pre‑ICU vitals (48h)

**Script:** `src/features/vitals_preicu_48h.py`

* **Signals:** heart rate, SBP/DBP/MAP, respiratory rate, SpO₂, temperature, CVP
* **Derived:** pulse pressure (= SBP−DBP), shock index (= HR/MAP)
* **Stats:** min/max/mean/std/last/count → `vital48_<signal>_*`

**Run:**

```bash
python -m src.features.vitals_preicu_48h --data-pkl data.pkl \
  --target target.parquet --outdir .
```

### 3.7 Pre‑ICU fluids (48h)

**Script:** `src/features/fluids_preicu_48h.py`

* **Outputs:** `fluid48_input_ml`, `fluid48_output_ml`, `fluid48_balance_ml`, `fluid48_input_count`, `fluid48_output_count`
* **Resuscitation rate:** 6h early rate (configurable) `fluid_resus6h_ml_per_hr`

**Run:**

```bash
python -m src.features.fluids_preicu_48h --data-pkl data.pkl \
  --target target.parquet --outdir .
```

### 3.8 Merge all features

**Script:** `src/build_features.py`

* Merges any `features_*.parquet` + `target.parquet`
* Writes `features_all.parquet` & `features/missingness.csv`

**Run:**

```bash
python -m src.build_features --target target.parquet --features-dir . --outdir ./features
```

---

## 4) QC & analysis

* **Smoke test:** `src/checks/smoke_test.py` – validates table presence, key columns, quick coverage, and ensures feature files merge with target
* **Cohort stats by gender/age:** `src/analyze_by_gender_age.py` – pivots by sex & age buckets; can plot if `--plot`

---

## 5) Folder layout (key files)

```
src/
  compile_data.py
  compute_target.py
  pre_aki_lab_flags.py
  build_features.py
  checks/smoke_test.py
  features/
    demographics.py
    insurance.py
    comorbidities.py
    meds_procedures_history.py
    vitals_preicu_48h.py
    labs_preicu_7d.py
    fluids_preicu_48h.py
  utils/
    io.py, itemids.py, time_windows.py, meds_maps.py, dx_maps.py, agg.py
```

---

## 6) Repro quickstart

```bash
# 1) Compile data
python -m src.compile_data --output_pickle data.pkl

# 2) Build target (removes pre‑AKI by labs, dedups at patient level)
python -m src.compute_target_admissions --keep_autocar
python -m src.incident_aki_target
python -m src.dedup_patient_level

# 3) Engineer features
python -m src.features.demographics        --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.insurance           --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.comorbidities       --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.meds_procedures_history --data-pkl data.pkl --target target.parquet --outdir . --windows 30d,6m
python -m src.features.vitals_preicu_48h   --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.labs_preicu_7d      --data-pkl data.pkl --target target.parquet --outdir . --drop-thresh 0.7
python -m src.features.fluids_preicu_48h   --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.meds_procedures_history --data-pkl data.pkl --target target.parquet --outdir ./features
python -m src.features.notes --data-pkl data.pkl --target target.parquet --out 

# 4) Merge all features & audit missingness
python -m src.build_features --target target.parquet --features-dir . --outdir ./features

# 5) Imputation (optional)
python -m src.imputation
```

---

## 7) Notes & caveats

* **Education** not available in MIMIC → omitted
* **Pre‑AKI removal** uses lab KDIGO prior‑7d window; make sure `labevents` exists
* **Regex safety** for meds: we combine patterns as **non‑capturing groups** to prevent `str.contains` warnings
* **Memory**: large tables (CHARTEVENTS, LABEVENTS). Use chunked processing when extending
* **Advanced ICU signals** (CO/PAP/VT/VE/GCS) are not included in vitals features by design right now; easily added by extending `itemids.py` + `compile_data.py` + `vitals_preicu_48h.py`

---


