#!/usr/bin/env bash
# CKD/death survival pipeline: upstream cmb_ckd/renal_impaired exclusion,
# corrected Harrell c-index, plus the sex-interaction feature set (CKD only).
#
# Prerequisites: the AKI pipeline has already been run (see README Quickstart
# steps 1-5) so that data.pkl, features/features_all.parquet, and
# features/features_all_imputed.parquet exist.
#
# Run from project root:
#   bash run_ckd_survival.sh [--skip-targets] [--skip-features] [--skip-training]
#
# DeepSurv scripts auto-detect device (mps > cuda > cpu); there is no device flag here.

set -euo pipefail
cd "$(dirname "$0")"

SKIP_TARGETS=0; SKIP_FEATURES=0; SKIP_TRAINING=0

for arg in "$@"; do
  case $arg in
    --skip-targets)  SKIP_TARGETS=1 ;;
    --skip-features) SKIP_FEATURES=1 ;;
    --skip-training) SKIP_TRAINING=1 ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

OUTDIR="ckd_survival_run"
AKI_FEAT_RAW="features/features_all.parquet"
AKI_FEAT_IMPUTED="features/features_all_imputed.parquet"

log() { echo; echo "==> $*"; }
# Override with: PYTHON=/path/to/python3 bash run_ckd_survival.sh
# Must have the packages in requirements.txt installed (activate your env first).
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUTDIR"

# ── Step 1: Build CKD targets, with cmb_ckd/renal_impaired excluded upstream ──
if [ "$SKIP_TARGETS" -eq 0 ]; then
  log "Computing CKD target admissions -> $OUTDIR/"
  $PYTHON -m ckd_survival.src.compute_target_admissions \
    --data-pkl data.pkl \
    --keep_autocar \
    --outdir "$OUTDIR"

  log "Filtering cmb_ckd/renal_impaired patients upstream of target construction"
  $PYTHON -m ckd_survival.src.filter_upstream_exclusions \
    --aki-features "$AKI_FEAT_RAW" \
    --admissions "$OUTDIR/incident_ckd_admission.csv" \
    --out "$OUTDIR/incident_ckd_admission.csv"

  log "Building patient-level CKD survival targets -> $OUTDIR/"
  $PYTHON -m ckd_survival.src.incident_ckd_target \
    --adm "$OUTDIR/incident_ckd_admission.csv" \
    --outdir "$OUTDIR"
else
  log "Skipping target construction"
fi

# ── Step 2: Feature engineering ───────────────────────────────────────────────
if [ "$SKIP_FEATURES" -eq 0 ]; then
  log "Building CKD features -> $OUTDIR/"

  $PYTHON -m ckd_survival.src.features.demographics_comorbidity_ckd \
    --aki-feat "$AKI_FEAT_IMPUTED" \
    --outdir "$OUTDIR"

  $PYTHON -m ckd_survival.src.features.labs_preckd \
    --data-pkl data.pkl \
    --target "$OUTDIR/incident_ckd_patient.csv" \
    --outdir "$OUTDIR"

  $PYTHON -m ckd_survival.src.features.meds_procedures_history \
    --data-pkl data.pkl \
    --target "$OUTDIR/incident_ckd_patient.csv" \
    --outdir "$OUTDIR"

  log "Merging CKD features -> $OUTDIR/features_ckd.parquet"
  $PYTHON -m ckd_survival.src.features.merge_ckd_features \
    --targets "$OUTDIR/incident_ckd_patient.csv" \
    --dem-cmb "$OUTDIR/features_dem_cmb_ckd.parquet" \
    --labs "$OUTDIR/features_labs_preckd.parquet" \
    --medsprocs "$OUTDIR/features_medsprocs_preckd.parquet" \
    --outdir "$OUTDIR"

  log "Building sex-interaction feature set (CKD only) -> $OUTDIR/features_ckd_sex_interaction.parquet"
  $PYTHON -m ckd_survival.src.features.build_sex_interaction_features \
    --src "$OUTDIR/features_ckd.parquet" \
    --out "$OUTDIR/features_ckd_sex_interaction.parquet"
else
  log "Skipping feature engineering"
fi

# ── Step 3: Train CKD/death models ────────────────────────────────────────────
if [ "$SKIP_TRAINING" -eq 0 ]; then
  log "Training XGBoost time-to-CKD"
  $PYTHON -m ckd_survival.src.train_xgboost_time_to_ckd \
    --input "$OUTDIR/features_ckd.parquet" \
    --out-prefix "$OUTDIR/artifacts/xgb_ckd"

  log "Training XGBoost time-to-death"
  $PYTHON -m ckd_survival.src.train_xgboost_time_to_death \
    --input "$OUTDIR/features_ckd.parquet" \
    --out-prefix "$OUTDIR/artifacts/xgb_death"

  log "Training DeepSurv time-to-CKD"
  $PYTHON -m ckd_survival.src.train_deepsurv_time_to_ckd \
    --input "$OUTDIR/features_ckd.parquet" \
    --out-prefix "$OUTDIR/artifacts/deepsurv_ckd"

  log "Training DeepSurv time-to-death"
  $PYTHON -m ckd_survival.src.train_deepsurv_time_to_death \
    --input "$OUTDIR/features_ckd.parquet" \
    --out-prefix "$OUTDIR/artifacts/deepsurv_death"

  log "Training XGBoost time-to-CKD (sex-interaction feature set)"
  $PYTHON -m ckd_survival.src.train_xgboost_time_to_ckd \
    --input "$OUTDIR/features_ckd_sex_interaction.parquet" \
    --out-prefix "$OUTDIR/artifacts_sex_interaction/xgb_ckd"

  log "Training DeepSurv time-to-CKD (sex-interaction feature set)"
  $PYTHON -m ckd_survival.src.train_deepsurv_time_to_ckd \
    --input "$OUTDIR/features_ckd_sex_interaction.parquet" \
    --out-prefix "$OUTDIR/artifacts_sex_interaction/deepsurv_ckd"
else
  log "Skipping model training"
fi

log "CKD pipeline complete. Outputs in $OUTDIR/"
