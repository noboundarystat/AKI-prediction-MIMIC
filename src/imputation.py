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
    return df_out

# ----------------------------
# CLI entrypoint
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Fast 3-tier imputation (gender+age±2, age±2, version medians)")
    parser.add_argument("mode", choices=["train","test"], nargs="?", default="train")
    parser.add_argument("infile", nargs="?", default="features/features_all.parquet")
    parser.add_argument("outfile", nargs="?", default="features/features_all_imputed.parquet")
    parser.add_argument("--feature-cols", default="labs,vital,fluid,vital48h",
                        help="Comma-separated list of feature columns or prefixes (default: labs,vital,fluid)")
    parser.add_argument("--no-missing-flags", action="store_true",
                        help="Disable adding *_missing indicator columns (default: False)")
    args = parser.parse_args()

    df = pd.read_parquet(args.infile)

    # Expand default prefixes
    feature_cols = []
    for prefix in args.feature_cols.split(","):
        feature_cols.extend([c for c in df.columns if c.startswith(prefix)])
    feature_cols = list(set(feature_cols))
    # print(feature_cols)
    df_out = impute_with_tiers(df, feature_cols, add_missing_flags=not args.no_missing_flags)

    df_out.to_parquet(args.outfile, index=False)
    print(f"✓ {args.mode} imputation complete: wrote {args.outfile} with {len(feature_cols)} features "
          f"(missing flags={'on' if not args.no_missing_flags else 'off'})")

if __name__ == "__main__":
    main()
