#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract demographics + comorbidity features from AKI features
for reuse in CKD modeling.

Inputs:
  - features_all_imputed.parquet (from AKI pipeline)

Outputs:
  - features_dem_cmb_ckd.parquet
"""

import argparse
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Extract demographics + comorbidity features for CKD modeling")
    ap.add_argument("--aki-feat", default="../features/features_all_imputed.parquet",
                    help="Input AKI feature file")
    ap.add_argument("--outdir", default=".", help="Output directory")
    args = ap.parse_args()

    # Load AKI features
    df = pd.read_parquet(args.aki_feat)

    # Select baseline demographics + comorbidity columns
    base_cols = ["version","subject_id","age"] \
              + [c for c in df.columns if c.startswith("cmb_")] \
              + [c for c in df.columns if c.startswith("dem_")] \
              + [c for c in df.columns if c.startswith("ins_")]

    df_out = df[base_cols].drop_duplicates()

    # Save
    os.makedirs(args.outdir, exist_ok=True)
    outp = os.path.join(args.outdir, "features_dem_cmb_ckd.parquet")
    df_out.to_parquet(outp, index=False)

    print(f"✓ wrote {outp}  shape={df_out.shape}")
    print(f"  columns: {df_out.columns.tolist()}")

if __name__ == "__main__":
    main()
