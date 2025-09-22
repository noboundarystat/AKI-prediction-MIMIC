#!/usr/bin/env python3
import argparse, os
import pandas as pd
from src.utils.io import load_pickle, load_target, save_parquet

KEYS = ["version","subject_id","hadm_id"]
FEAT_NAME = "features_insurance"

def normalize_ins(ins):
    if pd.isna(ins): return "UNKNOWN"
    s = str(ins).strip().upper()
    if "MEDICARE" in s: return "MEDICARE"
    if "PRIVATE" in s or "BCBS" in s or "HMO" in s: return "PRIVATE"
    # assistant bucket
    if any(k in s for k in ["MEDICAID","GOVERNMENT","SELF PAY","SELF-PAY","NO CHARGE","CHARITY","PUBLIC"]):
        return "ASSISTANT"
    return "UNKNOWN"

def build(df_dict, df_target):
    adm = df_dict["admissions"].copy()
    use = adm[KEYS + ["insurance"]].copy()
    use["ins_norm"] = use["insurance"].apply(normalize_ins)
    out = df_target[KEYS].merge(use[KEYS + ["ins_norm"]], on=KEYS, how="left", validate="one_to_one")
    out["ins_norm"] = out["ins_norm"].fillna("UNKNOWN")
    out = pd.get_dummies(out, columns=["ins_norm"], prefix="ins", dtype=int)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-pkl", default="data.pkl")
    ap.add_argument("--target", default="target.parquet")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df_dict = load_pickle(args.data_pkl)
    df_target = load_target(args.target)
    feats = build(df_dict, df_target)
    save_parquet(feats, os.path.join(args.outdir, f"{FEAT_NAME}.parquet"))
    print(f"✓ wrote {os.path.join(args.outdir, f'{FEAT_NAME}.parquet')}  shape={feats.shape}")

if __name__ == "__main__":
    main()
