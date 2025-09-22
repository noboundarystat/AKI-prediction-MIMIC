#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute secondary AKI/CKD admission-level targets with pre-AKI flags and optional AutoCar exclusion.

Pipeline:
  1) Adults only (>=18).
  2) Exclude PRIMARY AKI admissions (seq_num == 1 & is_aki==1).
     - Primary CKD admissions are retained (to enable AKI -> CKD trajectory tracking).
  3) For remaining admissions, flag hadm-level AKI/CKD if codes appear in any later seq.
  4) Mark CKD-only cases (is_aki==0 & is_ckd==1) with `ckd_only_flag` for AKI -> CKD tracking.
     Also provide `ckd_admission_flag` (1 for any CKD, whether CKD-only or AKI+CKD).
  5) Flag pre-AKI using lab-based KDIGO criteria within 7 days / 48 hours prior to admission
     (via pre_aki_lab_flags.attach_lab_aki_prior7d_flags). These admissions are not excluded,
     only flagged.
  6) Exclude AutoCar admissions (E810–E819, E820–E825, E826–E829, E830–E838, V00–V89)
     by default; override with --keep_autocar to retain them.
  7) No patient-level deduplication (admission-level dataset only).
     For deduplication see dedup_target.py.

Outputs (to --outdir):
  - target.parquet
  - aki_ckd_death_target_first.csv  (convenience single-row-per-patient view)
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
        # cast numpy ints for nice CSV
        entry.update({k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in kwargs.items()})
        self.rows.append(entry)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# -----------------------
# Domain helpers
# -----------------------

AKI_PREFIXES = ["584", "3995", "5498", "N17"]                           # AKI + dialysis procedures
CKD_PREFIXES = ["585", "V451", "V560", "V5631", "V5632", "V568", "N18"] # CKD

def startswith_any(code: str, prefixes: List[str]) -> bool:
    return any(str(code).startswith(p) for p in prefixes)

def get_age_row(row) -> float:
    """Age at admission. For MIMIC-III compute from DOB; for MIMIC-IV use anchor_age."""
    if row["version"] == "mimic3":
        admittime = datetime.strptime(row["admittime"], "%Y-%m-%d %H:%M:%S")
        dob = datetime.strptime(row["dob"], "%Y-%m-%d %H:%M:%S")
        age = (admittime - dob).days // 365
        return 90 if age > 150 else age  # de-id capping
    elif row["version"] == "mimic4":
        return row["anchor_age"]
    return np.nan

def get_age_row_ts(row) -> float:
    """Age at admission. For MIMIC-III compute from DOB; for MIMIC-IV use anchor_age."""
    if row["version"] == "mimic3":
        # admittime = datetime.strptime(row["admittime"], "%Y-%m-%d %H:%M:%S")
        # dob = datetime.strptime(row["dob"], "%Y-%m-%d %H:%M:%S")
        admittime = row["admittime"].to_pydatetime()
        dob = row["dob"].to_pydatetime()
        age = (admittime - dob).days / 365
        return 90 if age > 150 else age  # de-id capping
    elif row["version"] == "mimic4":
        return row["anchor_age"]
    return np.nan

def extract_kidney_flags(df_diagnosis: pd.DataFrame,
                         subject_col: str = "subject_id",
                         hadm_col: str = "hadm_id",
                         icd_col: str = "icd_code") -> pd.DataFrame:
    """
    Create binary flags for AKI and CKD for each (subject_id, hadm_id, seq_num).
    Returns one row per (subject, hadm, seq_num) with is_aki, is_ckd.
    """
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

# On-renal exclusion ICD ranges
AUTOCAR_PREFIXES = [
    # E810–E819
    *[f"E{c}" for c in range(810, 820)],
    # E820–E825
    *[f"E{c}" for c in range(820, 826)],
    # E826–E829
    *[f"E{c}" for c in range(826, 830)],
    # E830–E838
    *[f"E{c}" for c in range(830, 839)],
    # V00–V89
    *[f"V{c:02d}" for c in range(0, 90)]
]

def extract_autocar_flags(df_diagnosis: pd.DataFrame,
                          subject_col: str = "subject_id",
                          hadm_col: str = "hadm_id",
                          icd_col: str = "icd_code") -> pd.DataFrame:
    """
    Create binary flag for auto/transport-related external causes per admission.
    """
    df = df_diagnosis.copy()
    df[icd_col] = df[icd_col].astype(str)
    df["is_autocar"] = df[icd_col].str.startswith(tuple(AUTOCAR_PREFIXES)).astype(int)

    out = (df.groupby([subject_col, hadm_col])[["is_autocar"]]
             .max()
             .reset_index())
    return out

def dedup_patients_by_target(df: pd.DataFrame, mode: str = "patient") -> pd.DataFrame:
    """
    Deduplicate admissions based on target, or return unchanged.

    mode:
      - "patient": For target=1 (AKI), keep first admission chronologically.
                   For target=0, keep last admission chronologically.
      - "none": Return df unchanged.
    """
    if mode == "none":
        return df

    if mode == "patient":
        df = df.copy()
        df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")

        def pick_rows(g: pd.DataFrame) -> pd.DataFrame:
            g1 = g[g["is_aki"] == 1].sort_values("admittime", ascending=True)
            g0 = g[g["is_aki"] == 0].sort_values("admittime", ascending=False)
            keep = []
            if not g1.empty:
                keep.append(g1.iloc[0])
            if not g0.empty:
                keep.append(g0.iloc[0])
            return pd.DataFrame(keep) if keep else g.iloc[0:0]

        return (df.groupby(["version", "subject_id"], group_keys=False)
                  .apply(pick_rows)
                  .reset_index(drop=True))

    raise ValueError(f"Unknown deduplication mode: {mode}")


# -----------------------
# Core pipeline
# -----------------------

def compute_targets(data_dict: Dict, outdir: str, dedup_mode: str = "patient", keep_autocar: bool = False, verbose: bool = True):
    os.makedirs(outdir, exist_ok=True)
    C = CounterLog()

    # ---- Load required tables
    try:
        df_patients = data_dict["patients"].copy()
        df_admissions = data_dict["admissions"].copy()
        df_diagnosis = data_dict["diagnosis"].copy()
    except KeyError as e:
        raise KeyError(f"Missing key in data_dict: {e}. Required: 'patients','admissions','diagnosis'")

    C.add("loaded_patients", df_patients)
    C.add("loaded_admissions", df_admissions)
    C.add("loaded_diagnosis", df_diagnosis)

    # Optional labs for pre-AKI removal
    has_labs = ("labevents" in data_dict) and ("labitems" in data_dict)

    # Input checks
    need_cols_pat = {"subject_id", "version", "gender", "dob", "anchor_age"}
    need_cols_adm = {"subject_id", "version", "hadm_id", "admittime", "hospital_expire_flag"}
    need_cols_dx  = {"subject_id", "hadm_id", "seq_num", "icd_code"}
    missing = []
    for cols, dfname in [(need_cols_pat, "patients"), (need_cols_adm, "admissions"), (need_cols_dx, "diagnosis")]:
        have = set(locals()[f"df_{dfname}"].columns)
        miss = cols - have
        if miss:
            missing.append((dfname, sorted(list(miss))))
    if missing:
        raise ValueError(f"Input columns missing: {missing}")

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

    if "death_flag" not in df_adm_pat.columns:
        df_adm_pat["death_flag"] = df_adm_pat["dod"].notna().astype(int)

    df_adm_pat['time_to_death_from_adm_hr'] = (df_adm_pat['dod'] - df_adm_pat['admittime']).dt.total_seconds() / 3600
    df_adm_pat['time_to_death_from_disch_hr'] = (df_adm_pat['dod'] - df_adm_pat['dischtime']).dt.total_seconds() / 3600

    C.add("admissions_join_patients", df_adm_pat)

    df_adm_pat["age"] = df_adm_pat[["version", "admittime", "dob", "anchor_age"]].apply(get_age_row_ts, axis=1)
    df_adults = df_adm_pat[df_adm_pat["age"] >= 18.0].drop(columns=["dob", "anchor_age"])
    C.add("adult_admissions", df_adults)

    # ---- ICD flags at seq-level
    df_kdx = extract_kidney_flags(df_diagnosis)
    C.add("kidney_flags_seq", df_kdx)

    # Primary/admitting dx: seq_num == 1
    df_aki_primary = df_kdx[(df_kdx["is_aki"] == 1) & (df_kdx["seq_num"] == 1.0)][["subject_id", "hadm_id"]].drop_duplicates()
    # df_ckd_primary = df_kdx[(df_kdx["is_ckd"] == 1) & (df_kdx["seq_num"] == 1.0)][["subject_id", "hadm_id"]].drop_duplicates()
    df_ckd_primary = pd.DataFrame()
    C.add("aki_primary_admissions", df_aki_primary)
    C.add("ckd_primary_admissions", df_ckd_primary)

    primary_hadm_ids = pd.concat([df_aki_primary, df_ckd_primary]).drop_duplicates()
    C.add("aki_or_ckd_primary_union", primary_hadm_ids)

    # Non-primary hadm flags (any seq)
    not_primary_mask = ~df_kdx.set_index(["subject_id", "hadm_id"]).index.isin(
        primary_hadm_ids.set_index(["subject_id", "hadm_id"]).index
    )
    not_primary_kdx = df_kdx[not_primary_mask]
    hadm_flags = (not_primary_kdx.groupby(["subject_id", "hadm_id"])[["is_aki", "is_ckd"]]
                  .max()
                  .reset_index())
    C.add("hadm_flags_nonprimary_anyseq", hadm_flags)

    # Keep all CKD admissions; add CKD-only flag (no AKI)
    hadm_flags_clean = hadm_flags.copy()
    hadm_flags_clean["ckd_only_flag"] = (
        (hadm_flags_clean["is_aki"] == 0) & (hadm_flags_clean["is_ckd"] == 1)
    ).astype(int)

    # Admission-level CKD flag (includes AKI+CKD and CKD-only)
    hadm_flags_clean["ckd_admission_flag"] = (hadm_flags_clean["is_ckd"] == 1).astype(int)
    C.add("hadm_flags_with_ckd_only_and_any_ckd", hadm_flags_clean)

    # Adults without primary AKI/CKD
    adults_no_primary = pd.merge(
        df_adults,
        primary_hadm_ids.assign(primary_flag=1),
        on=["subject_id", "hadm_id"],
        how="left",
    )
    adults_no_primary = adults_no_primary[adults_no_primary["primary_flag"].isna()].drop(columns=["primary_flag"])
    C.add("adults_excluding_primary_aki_ckd", adults_no_primary)

    # Build target with pre-AKI info and Autocar flag
    target_cols = [
        "version", "subject_id", "hadm_id", "admittime", "age", "gender",
        "hospital_expire_flag", "death_flag",
        "time_to_death_from_adm_hr", "time_to_death_from_disch_hr"
    ]

    # Autocar exclusion flags
    df_autocar = extract_autocar_flags(df_diagnosis)
    C.add("autocar_flags", df_autocar)

    # Merge both hadm flags and autocar into target
    df_target0 = (adults_no_primary[target_cols]
                  .merge(hadm_flags_clean, on=["subject_id", "hadm_id"], how="inner", validate="one_to_one")
                  .merge(df_autocar, on=["subject_id", "hadm_id"], how="left")
                  .fillna({"is_autocar": 0})
                  .drop_duplicates())
    C.add("target_initial", df_target0)

    n_autocar_before = int((df_target0["is_autocar"] == 1).sum())   
    n_total_before = len(df_target0)
    C.add("autocar_before_exclusion",
          n_autocar=n_autocar_before,
        n_total=n_total_before,
        pct_autocar=n_autocar_before / max(n_total_before, 1))

    # df_target_noauto = df_target0[df_target0['is_autocar']==0].copy()
    if keep_autocar:
        df_target_noauto = df_target0.copy()
    else:
        df_target_noauto = df_target0[df_target0['is_autocar'] == 0].copy()


    # -----------------------
    # Mark pre-AKI using labs (prior 7d/48h KDIGO) via shared helper
    # -----------------------
    if has_labs:
        df_with_flags = attach_lab_aki_prior7d_flags(
            target=df_target_noauto,
            labevents=data_dict["labevents"],
            labitems=data_dict["labitems"],
        )

    # -----------------------
    # Optional patient-level de-dup
    # -----------------------
    # df_target = dedup_patients_by_target(df_with_flags, mode=dedup_mode)
    df_target = df_with_flags # no dedup
    C.add("target_final", df_target)

    n_autocar_after = int((df_target["is_autocar"] == 1).sum())
    n_total_after = len(df_target)

    # -----------------------
    # Summaries & saves (on final cohort)
    # -----------------------
    # Overall pivot by version/expire/flags
    df_summary = (df_target.groupby(["version", "hospital_expire_flag", "is_aki", "is_ckd"])
                  .agg(age_mean=("age", "mean"), count=("age", "count"))
                  .reset_index())
    df_pivot = (df_summary.pivot_table(index=["hospital_expire_flag", "is_aki", "is_ckd"],
                                       columns="version",
                                       values=["age_mean", "count"])
                           .reset_index())
    df_pivot.columns = ["_".join(c).strip() if isinstance(c, tuple) else c for c in df_pivot.columns]

    total_m3 = int(df_target[df_target["version"] == "mimic3"].shape[0])
    total_m4 = int(df_target[df_target["version"] == "mimic4"].shape[0])
    if "count_mimic3" in df_pivot.columns:
        df_pivot["pct_mimic3"] = (df_pivot["count_mimic3"] / max(total_m3, 1) * 100).round(1)
    if "count_mimic4" in df_pivot.columns:
        df_pivot["pct_mimic4"] = (df_pivot["count_mimic4"] / max(total_m4, 1) * 100).round(1)

    # By gender, per version
    def piv_by_gender(df_v):
        g = (df_v.groupby(["hospital_expire_flag", "is_aki", "is_ckd", "gender"])
                .agg(age_mean=("age", "mean"), count=("age", "count"))
                .reset_index())
        p = (g.pivot_table(index=["hospital_expire_flag", "is_aki", "is_ckd"],
                           columns="gender",
                           values=["age_mean", "count"])
               .reset_index())
        p.columns = ["_".join(c).strip() if isinstance(c, tuple) else c for c in p.columns]
        for col in ["age_mean_F", "age_mean_M"]:
            if col in p.columns:
                p[col] = p[col].round(1)
        return p

    df_pivot_m3 = piv_by_gender(df_target[df_target["version"] == "mimic3"])
    df_pivot_m4 = piv_by_gender(df_target[df_target["version"] == "mimic4"])

    # Target rates
    n_aki = int((df_target["is_aki"] == 1).sum())
    n_nonaki = int((df_target["is_aki"] == 0).sum())
    n_ckd = int((df_target["is_ckd"] == 1).sum())
    n_nonckd = int((df_target["is_ckd"] == 0).sum())

    aki_rate = n_aki / max(n_aki + n_nonaki, 1)
    ckd_rate = n_ckd / max(n_ckd + n_nonckd, 1)

    df_aki = df_target[df_target["is_aki"] == 1]
    aki_deaths = int((df_aki["hospital_expire_flag"] == 1).sum())
    aki_survive = int((df_aki["hospital_expire_flag"] == 0).sum())
    aki_mortality = aki_deaths / max(aki_deaths + aki_survive, 1)

    df_ckd = df_target[df_target["is_ckd"] == 1]
    ckd_deaths = int((df_ckd["hospital_expire_flag"] == 1).sum())
    ckd_survive = int((df_ckd["hospital_expire_flag"] == 0).sum())
    ckd_mortality = ckd_deaths / max(ckd_deaths + ckd_survive, 1)

    C.add("target_rates_final",
          aki_rate=aki_rate, ckd_rate=ckd_rate,
          aki_deaths=aki_deaths, aki_survive=aki_survive, aki_mortality=aki_mortality,
          ckd_deaths=ckd_deaths, ckd_survive=ckd_survive, ckd_mortality=ckd_mortality,
          total_mimic3=total_m3, total_mimic4=total_m4)

    # AutoCar summary
    n_autocar = int((df_target["is_autocar"] == 1).sum())
    n_non_autocar = int((df_target["is_autocar"] == 0).sum())
    autocar_rate = n_autocar / max(n_autocar + n_non_autocar, 1)

    C.add("autocar_rates_final",
          autocar_rate=autocar_rate,
        autocar_n=n_autocar,
        non_autocar_n=n_non_autocar)


    # ---- Save outputs
    out_final = os.path.join(outdir, "target_admissions.parquet")
    out_first = os.path.join(outdir, "aki_ckd_death_target_first.csv")

    df_target.to_parquet(out_final, index=False)
    (df_target.groupby(["version", "subject_id", "age", "gender"])
              .first()
              .reset_index()
              .to_csv(out_first, index=False))

    df_pivot.to_csv(os.path.join(outdir, "pivot_overall.csv"), index=False)
    df_pivot_m3.to_csv(os.path.join(outdir, "pivot_mimic3.csv"), index=False)
    df_pivot_m4.to_csv(os.path.join(outdir, "pivot_mimic4.csv"), index=False)

    df_counts = C.to_df()
    df_counts.to_csv(os.path.join(outdir, "cohort_counts.csv"), index=False)

    # ---- Console summary
    log(f"\n=== OUTPUTS ===", verbose)
    log(f"target.parquet -> {out_final}", verbose)
    log(f"target_first  -> {out_first}", verbose)

    log("\n=== AUTOCAR EXCLUSION ===", verbose)
    log(f"Before exclusion: {n_autocar_before}/{n_total_before} ({n_autocar_before/n_total_before:.2%})", verbose)
    log(f"After exclusion : {n_autocar_after}/{n_total_after} ({n_autocar_after/n_total_after:.2%})", verbose)

    log("\n=== TARGETS (final) ===", verbose)
    log(f"AKI (secondary) rate: {aki_rate:.4f}", verbose)
    log(f"CKD rate: {ckd_rate:.4f}", verbose)
    log(f"AKI mortality among secondary AKI: {aki_mortality:.4f}  ({aki_deaths}/{aki_deaths+aki_survive})", verbose)
    log(f"CKD mortality among CKD: {ckd_mortality:.4f}  ({ckd_deaths}/{ckd_deaths+ckd_survive})", verbose)

    log("\n=== STEP COUNTS (saved to cohort_counts.csv) ===", verbose)
    log(df_counts.to_string(index=False), verbose)


def main():
    ap = argparse.ArgumentParser(description="Compute secondary AKI/CKD targets with pre-AKI removal and optional patient-level dedup.")
    ap.add_argument("--data-pkl", type=str, default="data.pkl", help="Path to pickle containing data_dict.")
    ap.add_argument("--outdir", type=str, default=".", help="Directory to write outputs.")
    ap.add_argument("--dedup_mode", type=str, default="patient",
                    choices=["patient", "none"],
                    help="Deduplication: 'patient' keeps first-AKI/last-nonAKI per patient; 'none' keeps all admissions.")
    ap.add_argument("--keep_autocar", action="store_true", help="Keep AutoCar admissions instead of excluding them.")
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
