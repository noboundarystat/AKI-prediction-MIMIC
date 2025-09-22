#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deduplicate to patient-level incident AKI target.

Input:
  - incident_aki_target.parquet (admission-level, with incident_aki_label)

Logic:
  - For each patient:
      * If any admission has incident_aki_label == 1 → keep the FIRST such admission.
      * Else if no positives but at least one incident_aki_label == 0 → keep the LAST admission.
      * Else (all NA): drop (default) or keep if --keep_na flag set.

Outputs:
  - incident_aki_patient_level.parquet (one row per patient)
  - dedup_summary.csv (counts of patients kept by label)
"""

import argparse
import os
import pandas as pd


def dedup_patient_level(df: pd.DataFrame, keep_na: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")

    out = []
    for sid, g in df.groupby("subject_id", sort=False):
        g = g.sort_values("admittime")

        if (g["incident_aki_label"] == 1).any():
            # keep first positive
            keep = g[g["incident_aki_label"] == 1].iloc[0:1]
        elif (g["incident_aki_label"] == 0).any():
            # keep last negative
            keep = g[g["incident_aki_label"] == 0].iloc[-1:]
        else:
            # all NA
            if keep_na:
                keep = g.iloc[-1:]  # keep last admission by default
                keep = keep.assign(incident_aki_label=pd.NA)
            else:
                continue

        out.append(keep)

    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Deduplicate to patient-level incident AKI target")
    ap.add_argument("--input", type=str, default="incident_aki_target.parquet",
                    help="Path to incident_aki_target.parquet (default: incident_aki_target.parquet)")
    ap.add_argument("--outdir", type=str, default=".",
                    help="Directory to save outputs")
    ap.add_argument("--keep_na", action="store_true",
                    help="Keep patients with only NA labels instead of dropping them")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.input} ...")
    df = pd.read_parquet(args.input)

    print("Deduplicating to patient level ...")
    df_dedup = dedup_patient_level(df, keep_na=args.keep_na)

    # Save outputs
    out_parquet = os.path.join(args.outdir, "target.parquet")
    out_summary = os.path.join(args.outdir, "dedup_summary.csv")

    df_dedup.to_parquet(out_parquet, index=False)

    # Summary counts
    n_total = df_dedup["subject_id"].nunique()
    n_pos = int((df_dedup["incident_aki_label"] == 1).sum())
    n_neg = int((df_dedup["incident_aki_label"] == 0).sum())
    n_na = int(df_dedup["incident_aki_label"].isna().sum())

    summary = {
        "n_total_patients": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_na": n_na,
    }
    pd.DataFrame([summary]).to_csv(out_summary, index=False)

    print("\n=== PATIENT DEDUP SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nOutputs written:")
    print(f"- {out_parquet}")
    print(f"- {out_summary}")


if __name__ == "__main__":
    main()
