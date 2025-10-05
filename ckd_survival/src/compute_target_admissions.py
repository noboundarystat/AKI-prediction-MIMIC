#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute secondary AKI/CKD admission-level targets with pre-AKI flags and optional AutoCar exclusion.

Pipeline:
  1) Adults only (>=18).
  2) Exclude PRIMARY AKI admissions (seq_num == 1 & is_aki==1).
     - Primary CKD admissions are retained (to enable AKI -> CKD trajectory tracking).
  3) For remaining admissions, flag hadm-level AKI/CKD if codes appear in any later seq.
  4) Mark CKD cases (is_ckd==1) with `ckd_only_flag` for AKI -> CKD tracking.
     Also provide `ckd_admission_flag` (1 for any CKD, whether CKD-only or AKI+CKD).
  5) Flag pre-AKI using lab-based KDIGO criteria within 7 days / 48 hours prior to admission
     (via pre_aki_lab_flags.attach_lab_aki_prior7d_flags). These admissions are not excluded,
     only flagged.
  6) Exclude AutoCar admissions (E810–E819, E820–E825, E826–E829, E830–E838, V00–V89)
     by default; override with --keep_autocar to retain them.
  7) Add admission-level event flags:
       - event_ckd_flag: CKD diagnosis
       - event_postckd_flag: ESRD / dialysis / transplant diagnosis
       - event_death_flag: death by hospital_expire_flag or dod
  8) No patient-level deduplication (admission-level dataset only).
     For deduplication see dedup_target.py.

Outputs (to --outdir):
  - target_admissions.parquet
  - incident_ckd_admission.csv (enriched with CKD/PostCKD/Death event flags)
  - aki_ckd_death_target_first.csv
  - pivot_overall.csv, pivot_mimic3.csv, pivot_mimic4.csv
  - cohort_counts.csv
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

# import the shared pre-AKI function 
from src.pre_aki_lab_flags import attach_lab_aki_prior7d_flags

# -----------------------
# Logging helpers
# -----------------------

def log(msg: str, verbose: bool = True):
    if verbose:
        print(msg, flush=True)

class CounterLog:
    def __init__(self):
        self.rows: List[Dict] = []

    def add(self, step: str, df: pd.DataFrame = None, **kwargs):
        entry = {"step": step}
        if df is not None:
            entry["n_rows"] = int(len(df))
            if {"subject_id", "hadm_id"}.issubset(df.columns):
                entry["n_admissions"] = int(df[["subject_id", "hadm_id"]].drop_duplicates().shape[0])
        entry.update({k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in kwargs.items()})
        self.rows.append(entry)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

# -----------------------
# Domain helpers
# -----------------------

AKI_PREFIXES = ["584", "3995", "5498", "N17"]                           # AKI + dialysis procedures 
CKD_PREFIXES = ["5851","5852","5853","5854","5855","5859","N181","N182","N183","N184","N185","N189"] 
POSTCKD_PREFIXES = ["5856","N186","Z992","Z940"]

def startswith_any(code: str, prefixes: List[str]) -> bool:
    return any(str(code).startswith(p) for p in prefixes)

def get_age_row_ts(row) -> float:
    if row["version"] == "mimic3":
        admittime = row["admittime"].to_pydatetime()
        dob = row["dob"].to_pydatetime()
        age = (admittime - dob).days / 365
        return 90 if age > 150 else age
    elif row["version"] == "mimic4":
        return row["anchor_age"]
    return np.nan

def extract_kidney_flags(df_diagnosis: pd.DataFrame,
                         subject_col: str = "subject_id",
                         hadm_col: str = "hadm_id",
                         icd_col: str = "icd_code") -> pd.DataFrame:
    df = df_diagnosis.copy()
    df[icd_col] = df[icd_col].astype(str)
    df["is_aki"] = df[icd_col].apply(lambda x: int(startswith_any(x, AKI_PREFIXES)))
    df["is_ckd"] = df[icd_col].apply(lambda x: int(startswith_any(x, CKD_PREFIXES)))
    if "seq_num" not in df.columns:
        df["seq_num"] = np.nan
    df["seq_num"] = pd.to_numeric(df["seq_num"], errors="coerce")
    out = (df.groupby([subject_col, hadm_col, "seq_num"])[["is_aki", "is_ckd"]]
             .max()
             .reset_index())
    return out

def extract_postckd_flags(df_diagnosis: pd.DataFrame,
                          subject_col: str = "subject_id",
                          hadm_col: str = "hadm_id",
                          icd_col: str = "icd_code") -> pd.DataFrame:
    df = df_diagnosis.copy()
    df[icd_col] = df[icd_col].astype(str)
    df["is_postckd"] = df[icd_col].apply(lambda x: int(startswith_any(x, POSTCKD_PREFIXES)))
    out = (df.groupby([subject_col, hadm_col])[["is_postckd"]]
             .max()
             .reset_index())
    return out

AUTOCAR_PREFIXES = [
    *[f"E{c}" for c in range(810, 820)],
    *[f"E{c}" for c in range(820, 826)],
    *[f"E{c}" for c in range(826, 830)],
    *[f"E{c}" for c in range(830, 839)],
    *[f"V{c:02d}" for c in range(0, 90)]
]

def extract_autocar_flags(df_diagnosis: pd.DataFrame,
                          subject_col: str = "subject_id",
                          hadm_col: str = "hadm_id",
                          icd_col: str = "icd_code") -> pd.DataFrame:
    df = df_diagnosis.copy()
    df[icd_col] = df[icd_col].astype(str)
    df["is_autocar"] = df[icd_col].str.startswith(tuple(AUTOCAR_PREFIXES)).astype(int)
    out = (df.groupby([subject_col, hadm_col])[["is_autocar"]]
             .max()
             .reset_index())
    return out

# -----------------------
# Core pipeline
# -----------------------

def compute_targets(data_dict: Dict, outdir: str, dedup_mode: str = "patient", keep_autocar: bool = False, verbose: bool = True):
    os.makedirs(outdir, exist_ok=True)
    C = CounterLog()

    # ---- Load required tables
    df_patients = data_dict["patients"].copy()
    df_admissions = data_dict["admissions"].copy()
    df_diagnosis = data_dict["diagnosis"].copy()
    df_edstays = pd.read_csv('../../data/mimic-iv-ed-2.2/ed/edstays.csv.gz').copy()

    # patch admissions with ED deaths
    death_hadm_ids = df_edstays[df_edstays['disposition'] == 'EXPIRED']['hadm_id'].unique()
    df_admissions['hospital_expire_flag'] = df_admissions.apply(
        lambda row: 1 if row['hadm_id'] in death_hadm_ids else row['hospital_expire_flag'], axis=1)

    # ---- Age merge + adults
    df_adm_pat = pd.merge(
        df_admissions,
        df_patients[["subject_id", "version", "gender", "dob", "dod", "anchor_age"]],
        on=["subject_id", "version"],
        how="left",
        validate="many_to_one",
    )
    for c in ("admittime", "dischtime", "deathtime", "dob", "dod"):
        if c in df_adm_pat.columns:
            df_adm_pat[c] = pd.to_datetime(df_adm_pat[c], errors="coerce")

    df_adm_pat["death_flag"] = df_adm_pat["dod"].notna().astype(int)
    df_adm_pat["age"] = df_adm_pat[["version", "admittime", "dob", "anchor_age"]].apply(get_age_row_ts, axis=1)
    df_adults = df_adm_pat[df_adm_pat["age"] >= 18.0].drop(columns=["dob", "anchor_age"])

    # ---- ICD flags
    df_kdx = extract_kidney_flags(df_diagnosis)
    df_postckd = extract_postckd_flags(df_diagnosis)
    df_autocar = extract_autocar_flags(df_diagnosis)

    # Primary AKI (exclude), retain CKD primary
    df_aki_primary = df_kdx[(df_kdx["is_aki"] == 1) & (df_kdx["seq_num"] == 1.0)][["subject_id", "hadm_id"]].drop_duplicates()
    primary_hadm_ids = df_aki_primary

    # Non-primary hadm flags
    not_primary_mask = ~df_kdx.set_index(["subject_id", "hadm_id"]).index.isin(
        primary_hadm_ids.set_index(["subject_id", "hadm_id"]).index
    )
    hadm_flags = (df_kdx[not_primary_mask]
                  .groupby(["subject_id", "hadm_id"])[["is_aki", "is_ckd"]]
                  .max()
                  .reset_index())

    # hadm_flags["ckd_only_flag"] = ((hadm_flags["is_aki"] == 0) & (hadm_flags["is_ckd"] == 1)).astype(int)
    hadm_flags["ckd_only_flag"] = (hadm_flags["is_ckd"] == 1).astype(int)
    hadm_flags["ckd_admission_flag"] = (hadm_flags["is_ckd"] == 1).astype(int)

    # Adults without primary AKI
    adults_no_primary = pd.merge(
        df_adults,
        primary_hadm_ids.assign(primary_flag=1),
        on=["subject_id", "hadm_id"],
        how="left",
    )
    adults_no_primary = adults_no_primary[adults_no_primary["primary_flag"].isna()].drop(columns=["primary_flag"])

    # Merge
    target_cols = [
        "version", "subject_id", "hadm_id",
        "admittime", "dischtime", "dod", 
        "age", "gender",
        "hospital_expire_flag", "death_flag"
    ]

    df_target0 = (adults_no_primary[target_cols]
                  .merge(hadm_flags, on=["subject_id", "hadm_id"], how="inner")
                  .merge(df_postckd, on=["subject_id", "hadm_id"], how="left")
                  .merge(df_autocar, on=["subject_id", "hadm_id"], how="left")
                  .fillna({"is_postckd":0,"is_autocar":0})
                  .drop_duplicates())

    df_target = df_target0 if keep_autocar else df_target0[df_target0['is_autocar']==0].copy()

    # Flag pre-AKI labs (if available)
    if ("labevents" in data_dict) and ("labitems" in data_dict):
        df_target = attach_lab_aki_prior7d_flags(
            target=df_target,
            labevents=data_dict["labevents"],
            labitems=data_dict["labitems"],
        )

    # -----------------------
    # New: admission-level event flags
    # -----------------------
    df_target["event_ckd_flag"] = df_target["is_ckd"]
    df_target["event_postckd_flag"] = df_target["is_postckd"]
    df_target["event_death_flag"] = (
        (df_target["hospital_expire_flag"]==1) | (df_target["death_flag"]==1)
    ).astype(int)

    # ---- Save outputs
    out_final = os.path.join(outdir, "target_admissions.parquet")
    out_adm   = os.path.join(outdir, "incident_ckd_admission.csv")
    out_first = os.path.join(outdir, "aki_ckd_death_target_first.csv")

    df_target.to_parquet(out_final, index=False)
    df_target.to_csv(out_adm, index=False)
    (df_target.groupby(["version", "subject_id", "age", "gender"])
              .first()
              .reset_index()
              .to_csv(out_first, index=False))

    log(f"target_admissions.parquet -> {out_final}", verbose)
    log(f"incident_ckd_admission.csv -> {out_adm}", verbose)

def main():
    ap = argparse.ArgumentParser(description="Compute secondary AKI/CKD targets with pre-AKI removal and optional patient-level dedup.")
    ap.add_argument("--data-pkl", type=str, default="../data.pkl", help="Path to pickle containing data_dict.")
    ap.add_argument("--outdir", type=str, default=".", help="Directory to write outputs.")
    ap.add_argument("--dedup_mode", type=str, default="patient", choices=["patient", "none"], help="Deduplication: 'patient' keeps first-AKI/last-nonAKI per patient; 'none' keeps all admissions.")
    ap.add_argument("--keep_autocar", action="store_true", help="Keep AutoCar admissions instead of excluding them.")
    ap.set_defaults(keep_autocar=True)
    ap.add_argument("--quiet", action="store_true", help="Silence console logs.")
    args = ap.parse_args()

    if not os.path.exists(args.data_pkl):
        print(f"ERROR: pickle not found at {args.data_pkl}", file=sys.stderr); sys.exit(1)
    if os.path.getsize(args.data_pkl) == 0:
        print(f"ERROR: pickle at {args.data_pkl} is empty.", file=sys.stderr); sys.exit(1)

    with open(args.data_pkl, "rb") as f:
        data_dict = pd.read_pickle(f)

    compute_targets(data_dict=data_dict, outdir=args.outdir, dedup_mode=args.dedup_mode, keep_autocar=args.keep_autocar, verbose=not args.quiet)

if __name__ == "__main__":
    main()
