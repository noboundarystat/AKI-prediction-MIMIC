#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_sex_interaction_features.py

Builds the co-author-specified reduced feature set for the CKD survival
model (time-to-CKD only, not death): reference-category dummy coding for
categorical variables, plus an explicit variable x dem_sex_M interaction
term for every feature except sex itself.

Rules:
  - Categorical variables use k-1 (reference-category) dummy coding:
      dem_race: WHITE dropped (reference)
      dem_mar:  MARRIED dropped (reference); OTHER/SEPARATED rows excluded
                from the cohort entirely (not folded into MARRIED)
      ins:      MEDICARE dropped (reference); UNKNOWN rows excluded
  - Every variable except dem_sex_M is interacted with dem_sex_M (male
    indicator): 34 base features x 33 interaction terms (dem_sex_M has no
    self-interaction) = 67 total features.
  - Labs restricted to the 90-day pre-event window only (labs90d_*_mean_90d,
    11 labs) -- no 180d/365d/lastobs windows.
  - Native (raw) values throughout -- no imputation applied.

Input:  features_ckd.parquet -- the merged CKD-survival feature file produced
        by merge_ckd_features.py, run on the upstream-filtered admission list
        (see ../filter_upstream_exclusions.py). Native/raw values, not imputed.
Output: features_ckd_sex_interaction.parquet

Usage:
  python3 -m ckd_survival.src.features.build_sex_interaction_features \
    --src features_ckd.parquet --out features_ckd_sex_interaction.parquet
"""

import argparse
from pathlib import Path
import pandas as pd

DEFAULT_SRC = "features_ckd.parquet"
DEFAULT_OUT = "features_ckd_sex_interaction.parquet"

BASE_FEATURES = [
    "age",
    "cmb_htn", "cmb_dm", "cmb_cad", "cmb_hf", "cmb_copd", "cmb_cancer",
    "cmb_infection", "cmb_liver", "cmb_cerebrovasc", "cmb_obesity",
    "dem_sex_M",
    "dem_race_ASIAN", "dem_race_BLACK", "dem_race_HISPANIC", "dem_race_OTHER", "dem_race_UNK",
    "dem_mar_DIVORCED", "dem_mar_SINGLE", "dem_mar_UNK", "dem_mar_WIDOWED",
    "ins_ASSISTANT", "ins_PRIVATE",
    "labs90d_wbc_mean_90d", "labs90d_hgb_mean_90d", "labs90d_plt_mean_90d",
    "labs90d_creatinine_mean_90d", "labs90d_bun_mean_90d", "labs90d_albumin_mean_90d",
    "labs90d_sodium_mean_90d", "labs90d_potassium_mean_90d", "labs90d_bicarbonate_mean_90d",
    "labs90d_lactate_mean_90d", "labs90d_ph_mean_90d",
]

PASSTHROUGH_COLUMNS = [
    "subject_id", "version", "time_days", "is_ckd",
    "is_postckd", "is_censored", "cmb_ckd",
]


def build(src_parquet: Path, out_parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(src_parquet)
    print(f"input shape: {df.shape}")

    excl = (df["dem_mar_OTHER"] == 1) | (df["dem_mar_SEPARATED"] == 1) | (df["ins_UNKNOWN"] == 1)
    print(f"excluding {excl.sum()} rows (dem_mar OTHER/SEPARATED or ins UNKNOWN)")
    df = df[~excl].copy()
    print(f"after exclusion: {df.shape}")

    missing_base = [c for c in BASE_FEATURES if c not in df.columns]
    assert not missing_base, f"missing base columns: {missing_base}"
    missing_pt = [c for c in PASSTHROUGH_COLUMNS if c not in df.columns]
    assert not missing_pt, f"missing passthrough columns: {missing_pt}"

    out = df[PASSTHROUGH_COLUMNS + BASE_FEATURES].copy()

    interaction_cols = []
    for c in BASE_FEATURES:
        if c == "dem_sex_M":
            continue
        newc = f"{c}M"
        out[newc] = out[c] * out["dem_sex_M"]
        interaction_cols.append(newc)

    print(f"n base features: {len(BASE_FEATURES)}")
    print(f"n interaction features: {len(interaction_cols)}")
    print(f"total features: {len(BASE_FEATURES) + len(interaction_cols)}")
    print(f"total output columns (incl. passthrough): {out.shape[1]}")

    out.to_parquet(out_parquet, index=False)
    print(f"\nwrote {out_parquet}  shape={out.shape}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="Source native features_ckd.parquet")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output path")
    args = ap.parse_args()
    build(Path(args.src), Path(args.out))


if __name__ == "__main__":
    main()
