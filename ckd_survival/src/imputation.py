#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast 3-tier imputation (MIMIC3/4):
  1. version + gender + age±2
  2. version + age±2 (ignore gender)
  3. version-specific medians

Adds *_missing flags by default so neural nets can capture missingness.
"""

import os
import argparse
import pandas as pd

# ----------------------------
# Core imputation
# ----------------------------
def impute_with_tiers(df, feature_cols, add_missing_flags=True):
    df = df.copy()
    df_out = df.copy()
    df_out['age_bin'] = df_out['age'].round().astype(int)

    # Add missingness flags before filling
    if add_missing_flags:
        for col in feature_cols:
            df_out[f"{col}_missing"] = df_out[col].isna().astype(int)

    # -----------------
    # Tier 1: version + sex + age±2
    # -----------------
    tier1_maps = {}
    for (ver, f, m), sub in df.groupby(['version','dem_sex_F','dem_sex_M']):
        age_map = {}
        for age in range(int(sub['age'].min()), int(sub['age'].max())+1):
            mask = sub['age'].between(age-2, age+2)
            if mask.any():
                age_map[age] = sub.loc[mask, feature_cols].median(numeric_only=True)
        tier1_maps[(ver,f,m)] = age_map

    for idx, row in df_out[df_out[feature_cols].isna().any(axis=1)].iterrows():
        key = (row['version'], row['dem_sex_F'], row['dem_sex_M'])
        age = row['age_bin']
        if key in tier1_maps and age in tier1_maps[key]:
            for col in feature_cols:
                if pd.isna(row[col]):
                    df_out.at[idx, col] = tier1_maps[key][age][col]

    # -----------------
    # Tier 2: version + age±2 (ignore sex)
    # -----------------
    tier2_maps = {}
    for ver, sub in df.groupby('version'):
        age_map = {}
        for age in range(int(sub['age'].min()), int(sub['age'].max())+1):
            mask = sub['age'].between(age-2, age+2)
            if mask.any():
                age_map[age] = sub.loc[mask, feature_cols].median(numeric_only=True)
        tier2_maps[ver] = age_map

    for idx, row in df_out[df_out[feature_cols].isna().any(axis=1)].iterrows():
        ver = row['version']
        age = row['age_bin']
        if ver in tier2_maps and age in tier2_maps[ver]:
            for col in feature_cols:
                if pd.isna(row[col]):
                    df_out.at[idx, col] = tier2_maps[ver][age][col]

    # -----------------
    # Tier 3: version-specific global medians
    # -----------------
    # medians_by_version = df.groupby('version')[feature_cols].median(numeric_only=True)
    # for ver, meds in medians_by_version.iterrows():
    #     mask = (df_out['version'] == ver)
    #     for col in feature_cols:
    #         df_out.loc[mask, col] = df_out.loc[mask, col].fillna(meds[col])

    # -----------------
    # Tier 3: version-specific medians (vectorized)
    # -----------------
    version_medians = df_out.groupby("version")[feature_cols].transform("median")
    df_out[feature_cols] = df_out[feature_cols].fillna(version_medians)

    df_out.drop(columns=['age_bin'], inplace=True)

    # After Tier 3
    still_na = df_out[feature_cols].isna().all(axis=0)
    if still_na.any():
        for col in still_na.index[still_na]:
            df_out[col] = 0.0  

    return df_out

import pandas.api.types as ptypes

def get_feature_cols(df, spec="labs", exclude=("censor_time","is_ckd")):
    if spec == "all":
        return [
            c for c in df.columns
            if c not in exclude and ptypes.is_numeric_dtype(df[c])
        ]
    else:
        cols = []
        for prefix in spec.split(","):
            cols.extend([c for c in df.columns if c.startswith(prefix)])
        return list(set(cols))

# ----------------------------
# CLI entrypoint
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description="Fast 3-tier imputation (gender+age±2, age±2, version medians)"
    )
    parser.add_argument("mode", choices=["train","test"], nargs="?", default="train")
    parser.add_argument("infile", nargs="?", default="features_ckd.parquet")
    parser.add_argument("outfile", nargs="?", default="features_ckd_imputed.parquet")
    parser.add_argument("--feature-cols", default="labs",
                        help="Comma-separated prefixes OR 'all'")
    parser.add_argument("--no-missing-flags", action="store_true")
    parser.add_argument("--exclude", default="censor_time,is_ckd")
    args = parser.parse_args()

    df = pd.read_parquet(args.infile)
    exclude = args.exclude.split(",")
    feature_cols = get_feature_cols(df, args.feature_cols, exclude=exclude)

    df_out = impute_with_tiers(
        df, feature_cols,
        add_missing_flags=not args.no_missing_flags
    )

    # Guarantee all feature columns exist
    for col in feature_cols:
        if col not in df_out.columns:
            df_out[col] = 0.0

    df_out.to_parquet(args.outfile, index=False)
    print(f"✓ {args.mode} imputation complete: {len(feature_cols)} features")

if __name__ == "__main__":
    main()
