#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge CKD event targets with demographics, comorbidity,
labs, and meds/procs features into a single cleaned modeling dataset.

- Keeps age (from baseline features).
- Keeps gender (string).
- Cleans duplicated suffixes from lab/med/proc features.
"""

import argparse, os
import pandas as pd

def safe_merge(left, right, keys, how="left"):
    keys = [k for k in keys if k in left.columns and k in right.columns]
    return left.merge(right, on=keys, how=how)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        newc = c
        # normalize meds/procs window suffixes
        newc = newc.replace("_hx90d", "_90d")
        newc = newc.replace("_hx90to180d", "_90to180d")
        newc = newc.replace("_hx180to365d", "_180to365d")
        newc = newc.replace("_lastobs", "_lastobs")

        # remove duplicated window tags (labs)
        newc = newc.replace("_90d_90d", "_90d")
        newc = newc.replace("_180d_180d", "_180d")
        newc = newc.replace("_365d_365d", "_365d")
        newc = newc.replace("_lastobs_lastobs", "_lastobs")

        cols.append(newc)
    df.columns = cols
    return df

def main():
    ap = argparse.ArgumentParser(description="Merge and clean CKD features for modeling")
    ap.add_argument("--targets", default="incident_ckd_patient.csv")
    ap.add_argument("--dem-cmb", default="features_dem_cmb_ckd.parquet")
    ap.add_argument("--labs", default="features_labs_preckd.parquet")
    ap.add_argument("--medsprocs", default="features_medsprocs_preckd.parquet")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    # Load
    df_target = pd.read_csv(args.targets)
    df_demcmb = pd.read_parquet(args.dem_cmb)
    df_labs = pd.read_parquet(args.labs)
    df_medsprocs = pd.read_parquet(args.medsprocs)

    # Merge
    df = df_target.merge(df_demcmb, on=["version","subject_id"], how="left")
    df = safe_merge(df, df_labs, ["version","subject_id","event_type","event_hadm_id"])
    df = safe_merge(df, df_medsprocs, ["version","subject_id","event_type","event_hadm_id"])

    # Fix age columns
    if "age_x" in df.columns and "age_y" in df.columns:
        df = df.drop(columns=["age_x"]).rename(columns={"age_y": "age"})

    # Clean column names
    df = clean_columns(df)

    # Save outputs
    os.makedirs(args.outdir, exist_ok=True)
    out_parquet = os.path.join(args.outdir, "features_ckd.parquet")
    out_csv = os.path.join(args.outdir, "features_ckd_modeling_preview.csv")

    df.to_parquet(out_parquet, index=False)
    df.head(200).to_csv(out_csv, index=False)

    print(f"✓ wrote {out_parquet}  shape={df.shape}")
    print(f"✓ wrote preview {out_csv}")
    print("  #cols:", len(df.columns))
    print("  Events counts:")
    print(df["event_type"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
