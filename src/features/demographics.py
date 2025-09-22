#!/usr/bin/env python3
import argparse, os
import pandas as pd
from src.utils.io import load_pickle, load_target, save_parquet

KEYS = ["version","subject_id","hadm_id"]
FEAT_NAME = "features_demographics"

# ----- canonicalizers -----
def clean_str(x):
    if pd.isna(x): return "UNK"
    return str(x).strip().upper()

def normalize_gender(g):
    g = clean_str(g)
    if g in {"M","MALE"}: return "M"
    if g in {"F","FEMALE"}: return "F"
    return "UNK"

def normalize_race(row_race, row_ethnicity=None):
    r = clean_str(row_race) if row_race is not None else "UNK"
    if r == "UNK" and row_ethnicity is not None:
        r = clean_str(row_ethnicity)  # fallback (common in MIMIC-III)
    # coarse buckets
    if "WHITE" in r: return "WHITE"
    if "BLACK" in r or "AFRICAN" in r: return "BLACK"
    if "ASIAN" in r: return "ASIAN"
    if "HISPANIC" in r or "LATINO" in r: return "HISPANIC"
    if r in {"UNK","OTHER","PATIENT DECLINED TO ANSWER"}: return "UNK"
    return "OTHER"

def normalize_marital(m):
    m = clean_str(m)
    if "MARRIED" in m: return "MARRIED"
    if "SINGLE" in m or "NEVER MARRIED" in m: return "SINGLE"
    if "DIVORCED" in m: return "DIVORCED"
    if "WIDOW" in m: return "WIDOWED"
    if m in {"SEPARATED"}: return "SEPARATED"
    if m == "UNK" or "UNKNOWN" in m: return "UNK"
    return "OTHER"

def build(df_dict, df_target):
    adm = df_dict["admissions"].copy()
    pat = df_dict["patients"].copy()

    # keys + age already in target
    base = df_target[KEYS + ["age","gender"]].merge(
        adm[KEYS + [c for c in ["race","ethnicity","marital_status"] if c in adm.columns]],
        on=KEYS, how="left", validate="one_to_one"
    ).merge(
        pat[["version","subject_id","gender"]].rename(columns={"gender":"gender_pat"}),
        on=["version","subject_id"], how="left", validate="many_to_one"
    )

    # prefer target/admissions gender; fallback to patient table
    base["gender"] = base["gender"].fillna(base["gender_pat"])
    base.drop(columns=["gender_pat"], inplace=True, errors="ignore")

    base["gender"] = base["gender"].map(normalize_gender)
    race_src = base["race"] if "race" in base.columns else None
    eth_src  = base["ethnicity"] if "ethnicity" in base.columns else None
    base["race_norm"] = [normalize_race(r, e) for r, e in zip(race_src, eth_src)] if race_src is not None or eth_src is not None else "UNK"
    base["marital_norm"] = base["marital_status"].map(normalize_marital) if "marital_status" in base.columns else "UNK"

    # one-hots (keep UNK)
    out = base[KEYS + ["age","gender","race_norm","marital_norm"]].copy()
    out = pd.get_dummies(out, columns=["gender","race_norm","marital_norm"],
                         prefix=["dem_sex","dem_race","dem_mar"], dtype=int)

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
