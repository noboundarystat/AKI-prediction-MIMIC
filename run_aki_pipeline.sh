#!/usr/bin/env bash
# Full AKI pipeline: targets -> features -> merge -> imputation -> model training.
# Assumes data.pkl already exists (compile_data.py NOT re-run -- underlying
# MIMIC extract hasn't changed, no need to reprocess raw tables).
#
# Run from project root:
#   bash run_aki_pipeline.sh [--skip-targets] [--skip-features] [--skip-training]

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

log() { echo; echo "==> $*"; }
# Override with: PYTHON=/path/to/python3 bash run_aki_pipeline.sh
# Must have the packages in requirements.txt installed (activate your env first).
PYTHON="${PYTHON:-python3}"

# ── Step 1: Targets ───────────────────────────────────────────────────────────
if [ "$SKIP_TARGETS" -eq 0 ]; then
  log "compute_target_admissions"
  $PYTHON -m src.compute_target_admissions --data-pkl data.pkl --keep_autocar

  log "incident_aki_target"
  $PYTHON -m src.incident_aki_target

  log "dedup_patient_level"
  $PYTHON -m src.dedup_patient_level
else
  log "Skipping target construction"
fi

# ── Step 2: Feature engineering ───────────────────────────────────────────────
if [ "$SKIP_FEATURES" -eq 0 ]; then
  log "features.demographics"
  $PYTHON -m src.features.demographics --data-pkl data.pkl --target target.parquet --outdir .

  log "features.insurance"
  $PYTHON -m src.features.insurance --data-pkl data.pkl --target target.parquet --outdir .

  log "features.comorbidities"
  $PYTHON -m src.features.comorbidities --data-pkl data.pkl --target target.parquet --outdir .

  log "features.meds_procedures_history"
  $PYTHON -m src.features.meds_procedures_history --data-pkl data.pkl --target target.parquet --outdir .

  log "features.vitals_preicu_48h"
  $PYTHON -m src.features.vitals_preicu_48h --data-pkl data.pkl --target target.parquet --outdir .

  log "features.labs_preicu_7d"
  $PYTHON -m src.features.labs_preicu_7d --data-pkl data.pkl --target target.parquet --outdir .

  log "features.fluids_preicu_48h"
  $PYTHON -m src.features.fluids_preicu_48h --data-pkl data.pkl --target target.parquet --outdir .

  log "features.notes (optional -- text embedding, skip if unavailable)"
  $PYTHON -m src.features.notes --data-pkl data.pkl --target target.parquet --outdir . || log "notes.py failed/skipped (sentence-transformers likely unused elsewhere -- non-fatal)"

  log "build_features (merge)"
  $PYTHON -m src.build_features --target target.parquet --features-dir . --outdir ./features

  log "imputation"
  $PYTHON -m src.imputation
else
  log "Skipping feature engineering"
fi

# ── Step 3: Train AKI models ──────────────────────────────────────────────────
if [ "$SKIP_TRAINING" -eq 0 ]; then
  log "train_logreg"
  $PYTHON -m src.train_logreg

  log "train_rf"
  $PYTHON -m src.train_rf

  log "train_xgboost"
  $PYTHON -m src.train_xgboost

  log "train_dnn"
  $PYTHON -m src.train_dnn

  log "train_selfattn"
  $PYTHON -m src.train_selfattn

  log "train_dcn"
  $PYTHON -m src.train_dcn
else
  log "Skipping model training"
fi

log "AKI pipeline complete."
