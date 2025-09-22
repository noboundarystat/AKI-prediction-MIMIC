#!/usr/bin/env python3
import argparse, os
import pandas as pd
from src.utils.io import load_pickle, load_target, save_parquet
from src.utils.dx_maps import diagnosis_to_hadm_flags

KEYS = ["version","subject_id","hadm_id"]
FEAT_NAME = "features_comorbidities"

CATS = ["htn","dm","cad","hf","copd","cancer","infection","ckd","liver","cerebrovasc","obesity"]

def build(df_dict, df_target, exclude_primary: bool = True):
    dx = df_dict["diagnosis"].copy()

    # normalize seq_num & filter if excluding primary
    if "seq_num" in dx.columns and exclude_primary:
        dx["seq_num"] = pd.to_numeric(dx["seq_num"], errors="coerce")
        dx = dx[(dx["seq_num"].isna()) | (dx["seq_num"] > 1.0)]

    hadm_flags = diagnosis_to_hadm_flags(
        dx.rename(columns={"icd_code":"icd_code", "icd_version":"icd_version"}),
        categories=CATS,
        subject_col="subject_id",
        hadm_col="hadm_id",
        icd_col="icd_code",
        ver_col="icd_version",
    )

    # merge onto target keys
    out = df_target[KEYS].merge(hadm_flags, on=["subject_id","hadm_id"], how="left")
    for c in CATS:
        out[c] = out[c].fillna(0).astype(int)

    # prefix for clarity
    out = out.rename(columns={c: f"cmb_{c}" for c in CATS})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-pkl", default="data.pkl")
    ap.add_argument("--target", default="target.parquet")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--include-primary", action="store_true",
                    help="If set, includes primary diagnosis (seq_num==1) when computing comorbidity flags.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df_dict = load_pickle(args.data_pkl)
    df_target = load_target(args.target)
    feats = build(df_dict, df_target, exclude_primary=not args.include_primary)
    save_parquet(feats, os.path.join(args.outdir, f"{FEAT_NAME}.parquet"))
    print(f"✓ wrote {os.path.join(args.outdir, f'{FEAT_NAME}.parquet')}  shape={feats.shape}")

if __name__ == "__main__":
    main()
