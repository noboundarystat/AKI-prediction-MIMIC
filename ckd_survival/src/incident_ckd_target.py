#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patient-level incident CKD after AKI.

Inputs:
  - incident_ckd_admission.csv (admission-level with event flags)

Logic:
  * For each subject, anchor at FIRST AKI admission.
  * Collect candidate events:
      - CKD (any later admission with CKD flag)
      - PostCKD (any later admission with ESRD/dialysis/transplant flag)
      - Death (dod or hospital_expire_flag after index)
      - Censor (last observed discharge if no events)
  * Compare event times relative to index admission.
  * Choose the earliest event (event_type, time_days).
  * Preserve all event times for QC (time_to_ckd, time_to_postckd, time_to_death, censor_time).
  * Filtering (<90d, <60d) is applied downstream, not here.

Outputs:
  - incident_ckd_patient.csv (wide format, 1 row per patient with flags & times)
  - incident_ckd_survival.csv (tidy survival format for modeling)
"""

import argparse, os
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Collapse admission-level CKD targets to patient-level")
    ap.add_argument("--adm", type=str, default="incident_ckd_admission.csv")
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    adm = pd.read_csv(args.adm)

    # Convert date columns if present
    for col in ["admittime", "dischtime", "dod"]:
        if col in adm.columns:
            adm[col] = pd.to_datetime(adm[col], errors="coerce")

    rows = []
    surv = []

    for sid, g in adm.groupby("subject_id"):
        g = g.sort_values("admittime")

        # anchor at first AKI admission
        g_aki = g[g["is_aki"] == 1]
        if g_aki.empty:
            continue
        index_row = g_aki.iloc[0]
        index_admit = index_row["admittime"]

        version = index_row["version"]
        age     = index_row["age"]
        gender  = index_row["gender"]

        # --- candidate events ---
        events = []
        time_to_ckd = np.nan
        time_to_postckd = np.nan
        time_to_death = np.nan
        censor_time = np.nan

        # CKD after index
        g_ckd = g[(g["event_ckd_flag"] == 1) & (g["admittime"] > index_admit)]
        if not g_ckd.empty:
            ckd_row = g_ckd.iloc[0]
            time_to_ckd = (ckd_row["admittime"] - index_admit).days
            events.append(("CKD", time_to_ckd, ckd_row["hadm_id"]))

        # PostCKD after index
        g_post = g[(g["event_postckd_flag"] == 1) & (g["admittime"] > index_admit)]
        if not g_post.empty:
            post_row = g_post.iloc[0]
            time_to_postckd = (post_row["admittime"] - index_admit).days
            events.append(("PostCKD", time_to_postckd, post_row["hadm_id"]))

        # Death after index
        g_death = g[(g["event_death_flag"] == 1) & (g["dischtime"] >= index_admit)]
        if not g_death.empty:
            death_row = g_death.iloc[0]
            time_to_death = (death_row["dischtime"] - index_admit).days
            events.append(("Death", time_to_death, death_row["hadm_id"]))

        # Censor if no events
        if not events:
            last_disch = g["dischtime"].max()
            censor_time = (last_disch - index_admit).days
            events.append(("Censor", censor_time, np.nan))

        # --- choose earliest event ---
        event_type, time_days, event_hadm_id = min(events, key=lambda x: x[1])

        # --- record outputs ---
        rows.append(dict(
            version=version, subject_id=sid, age=age, gender=gender,
            index_admit=index_admit, event_type=event_type, time_days=time_days,
            ckd_event_hadm_id=(g_ckd.iloc[0]["hadm_id"] if not g_ckd.empty else np.nan),
            postckd_event_hadm_id=(g_post.iloc[0]["hadm_id"] if not g_post.empty else np.nan),
            death_event_hadm_id=(g_death.iloc[0]["hadm_id"] if not g_death.empty else np.nan),
            time_to_ckd=time_to_ckd, time_to_postckd=time_to_postckd,
            time_to_death=time_to_death, censor_time=censor_time,
            is_ckd=int(event_type=="CKD"),
            is_postckd=int(event_type=="PostCKD"),
            is_death=int(event_type=="Death"),
            is_censored=int(event_type=="Censor")
        ))

        surv.append(dict(
            version=version, subject_id=sid, age=age, gender=gender,
            index_admit=index_admit, event_type=event_type,
            time_days=time_days, event_hadm_id=event_hadm_id
        ))

    df_wide = pd.DataFrame(rows)
    df_surv = pd.DataFrame(surv)

    # Save
    os.makedirs(args.outdir, exist_ok=True)
    out_wide = os.path.join(args.outdir,"incident_ckd_patient.csv")
    out_surv = os.path.join(args.outdir,"incident_ckd_survival.csv")
    df_wide.to_csv(out_wide,index=False)
    df_surv.to_csv(out_surv,index=False)

    print(f"Wrote {out_wide} ({len(df_wide)} patients)")
    print(f"Wrote {out_surv} ({len(df_surv)} patients)")

if __name__=="__main__":
    main()
