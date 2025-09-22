#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build incident AKI target from admission-level target.parquet.

Assumptions:
  - Input (target.parquet) already excludes:
      * <18 yrs
      * Primary AKI (seq_num == 1)

AutoCar accident admissions marked (--keep_autocar set upstream)

Adds patient-level renal history per admission:
  - renal_impaired_at_adm: 1 if patient had CKD before this admission
  - aki_history_flag: 1 if patient had AKI before this admission
  - onset_aki_flag: 1 if this is first-ever AKI admission

Labels:
  - incident_aki_label = 1 if onset_aki_flag == 1
  - incident_aki_label = 0 if is_aki == 0 & is_ckd == 0 & renal_impaired_at_adm == 0
  - Otherwise left as NA (e.g. renal impaired or CKD-only)

Outputs:
  - incident_aki_target.parquet (admission-level labels)
  - incident_aki_summary.csv (admission-level summary counts)
  - patient_incident_aki.csv (patient-level summary)
"""

import argparse
import os
import pandas as pd


def add_history_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df = df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    df["renal_impaired_at_adm"] = 0
    df["aki_history_flag"] = 0
    df["onset_aki_flag"] = 0

    out = []
    for sid, g in df.groupby("subject_id", sort=False):
        g = g.sort_values("admittime").copy()
        had_prior_ckd = False
        had_prior_aki = False
        onset_assigned = False

        rows = []
        for _, row in g.iterrows():
            r = row.to_dict()

            # flags based on history
            r["renal_impaired_at_adm"] = 1 if had_prior_ckd else 0
            r["aki_history_flag"] = 1 if had_prior_aki else 0

            # onset AKI = first-ever AKI admission
            if r["is_aki"] == 1 and not onset_assigned and not had_prior_aki:
                r["onset_aki_flag"] = 1
                onset_assigned = True
            else:
                r["onset_aki_flag"] = 0

            # update trackers
            if r["is_ckd"] == 1:
                had_prior_ckd = True
            if r["is_aki"] == 1:
                had_prior_aki = True

            rows.append(r)

        out.append(pd.DataFrame(rows))
    return pd.concat(out, ignore_index=True)


def label_incident_aki(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["incident_aki_label"] = pd.NA

    # positives: first in-hospital AKI
    df.loc[df["onset_aki_flag"] == 1, "incident_aki_label"] = 1

    # negatives: no AKI, no CKD, renal healthy
    mask_neg = (
        (df["is_aki"] == 0) &
        (df["is_ckd"] == 0) &
        (df["renal_impaired_at_adm"] == 0)
    )
    df.loc[mask_neg, "incident_aki_label"] = 0

    return df


def patient_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize patient-level outcomes."""
    summary = []
    for sid, g in df.groupby("subject_id", sort=False):
        g = g.sort_values("admittime")

        ever_incident_aki = int((g["incident_aki_label"] == 1).any())
        ever_any_aki = int((g["is_aki"] == 1).any())
        ever_ckd = int((g["is_ckd"] == 1).any())
        ever_renal_impair = int((g["renal_impaired_at_adm"] == 1).any())

        first_incident_time = (
            g.loc[g["incident_aki_label"] == 1, "admittime"].min()
            if ever_incident_aki else pd.NaT
        )

        summary.append({
            "subject_id": sid,
            "ever_incident_aki": ever_incident_aki,
            "ever_any_aki": ever_any_aki,
            "ever_ckd": ever_ckd,
            "ever_renal_impair": ever_renal_impair,
            "first_incident_aki_time": first_incident_time,
        })

    return pd.DataFrame(summary)


def main():
    ap = argparse.ArgumentParser(description="Build incident AKI target from target_admissions.parquet")
    ap.add_argument("--input", type=str, default="target_admissions.parquet", help="Path to target.parquet")
    ap.add_argument("--outdir", type=str, default=".", help="Directory to save outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.input} ...")
    df = pd.read_parquet(args.input)

    print("Adding renal history flags ...")
    df_hist = add_history_flags(df)

    print("Labeling incident AKI ...")
    df_labeled = label_incident_aki(df_hist)

    # Save admission-level
    out_parquet = os.path.join(args.outdir, "incident_aki_target.parquet")
    df_labeled.to_parquet(out_parquet, index=False)

    # Admission-level summary
    n_total = len(df_labeled)
    n_pos = int((df_labeled["incident_aki_label"] == 1).sum())
    n_neg = int((df_labeled["incident_aki_label"] == 0).sum())
    n_na = int(df_labeled["incident_aki_label"].isna().sum())
    summary_adm = {
        "n_total": n_total,
        "n_incident_aki": n_pos,
        "n_negatives": n_neg,
        "n_excluded_NA": n_na,
    }
    pd.DataFrame([summary_adm]).to_csv(
        os.path.join(args.outdir, "incident_aki_summary.csv"), index=False
    )

    # Patient-level summary
    df_patient = patient_summary(df_labeled)
    df_patient.to_csv(os.path.join(args.outdir, "patient_incident_aki.csv"), index=False)

    print("\n=== INCIDENT AKI ADMISSION SUMMARY ===")
    for k, v in summary_adm.items():
        print(f"{k}: {v}")

    print("\n=== PATIENT-LEVEL SUMMARY ===")
    print(df_patient[["ever_incident_aki", "ever_any_aki", "ever_ckd", "ever_renal_impair"]].sum())

    print("\nOutputs written:")
    print(f"- {out_parquet}")
    print(f"- incident_aki_summary.csv")
    print(f"- patient_incident_aki.csv")


if __name__ == "__main__":
    main()
