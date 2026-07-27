#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
filter_upstream_exclusions.py

Excludes patients with pre-existing CKD comorbidity (cmb_ckd==1) or impaired
renal function at admission (renal_impaired_at_adm==1) from the CKD-survival
admission list, BEFORE CKD targets are built -- i.e. moved upstream of target
construction, rather than filtered post-hoc after the survival cohort/labels
already exist.

Rationale: "incident CKD" should mean genuinely new-onset CKD after AKI, not
a continuation/recurrence of prior kidney disease. Excluding these patients
after target construction risks the target itself having already encoded
their prior CKD status (e.g. via how "incident" was determined); filtering
upstream guarantees they never influence target construction at all.

Usage:
  python3 -m ckd_survival.src.filter_upstream_exclusions \
    --aki-features features/features_all.parquet \
    --admissions ckd_survival/incident_ckd_admission.csv \
    --out ckd_survival/incident_ckd_admission.csv
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aki-features", default="features/features_all.parquet",
                     help="AKI feature file carrying cmb_ckd / renal_impaired_at_adm flags")
    ap.add_argument("--admissions", default="ckd_survival/incident_ckd_admission.csv",
                     help="Candidate CKD-survival admissions, pre-exclusion")
    ap.add_argument("--out", default="ckd_survival/incident_ckd_admission.csv",
                     help="Output path (defaults to overwriting --admissions in place)")
    args = ap.parse_args()

    feat = pd.read_parquet(
        args.aki_features,
        columns=["subject_id", "cmb_ckd", "renal_impaired_at_adm"],
        engine="pyarrow",
    )
    excl_sids = set(
        feat.loc[(feat["cmb_ckd"] == 1) | (feat["renal_impaired_at_adm"] == 1), "subject_id"]
    )
    print(f"Excluding {len(excl_sids):,} subjects with cmb_ckd==1 or renal_impaired_at_adm==1")

    adm = pd.read_csv(args.admissions)
    print(f"incident_ckd_admission before: {adm['subject_id'].nunique():,} patients, {len(adm):,} rows")

    adm_filtered = adm[~adm["subject_id"].isin(excl_sids)]
    print(f"incident_ckd_admission after:  {adm_filtered['subject_id'].nunique():,} patients, {len(adm_filtered):,} rows")

    adm_filtered.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
