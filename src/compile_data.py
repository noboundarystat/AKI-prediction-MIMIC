#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compile MIMIC-III/IV into a single pickle (no dedup).
Includes:
  - patients, admissions, icustays, services, transfers
  - diagnosis, prescriptions, procedures
  - labitems, labevents
  - d_items (III + IV)
  - chartevents filtered to vitals/ABG (base + advanced)
  - inputevents (aligned), outputevents (urine subset)

Requires:
  src/utils/itemids.py  with get_vital_itemids() and get_bp_itemids()
"""

import argparse
import os
import pickle
import pandas as pd
from glob import glob

from src.utils.itemids import get_vital_itemids, get_bp_itemids

def log(msg: str) -> None:
    print(msg, flush=True)

# -----------------------------
# Helpers
# -----------------------------
def abg_ids(d_items_df: pd.DataFrame):
    """ABG pH (avoid 'phosph') + lactate by label from D_ITEMS."""
    di = d_items_df.copy()
    di.columns = [c.lower() for c in di.columns]
    lbl = di["label"].astype(str)

    ph_mask = lbl.str.contains(r"\bph\b", case=False, regex=True, na=False) & \
              ~lbl.str.contains(r"phosph", case=False, regex=True, na=False)
    ph_ids = di.loc[ph_mask, "itemid"].dropna().astype(int).unique().tolist()

    lact_mask = lbl.str.contains(r"lactate", case=False, regex=True, na=False)
    lact_ids = di.loc[lact_mask, "itemid"].dropna().astype(int).unique().tolist()

    return sorted(ph_ids), sorted(lact_ids)

def collect_vital_ids(d_items: pd.DataFrame, base_labels, adv_labels):
    """Resolve itemids for all requested signals via get_vital_itemids()."""
    ids = []
    for lab in base_labels + adv_labels:
        try:
            ids += get_vital_itemids(d_items, lab)
        except Exception:
            pass
    return sorted(set(int(x) for x in ids if pd.notna(x)))

def load_filtered_chartevents(file_path, usecols, itemids, version, rename_icustay=True, chunksize=1_000_000):
    """Stream CHARTEVENTS and keep only itemids in allow-list."""
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=[c.lower() for c in usecols] + ["version"])
    out = []
    for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=chunksize, low_memory=True):
        chunk.columns = [c.lower() for c in chunk.columns]
        if rename_icustay and "icustay_id" in chunk.columns and "stay_id" not in chunk.columns:
            chunk = chunk.rename(columns={"icustay_id": "stay_id"})
        if itemids:
            chunk = chunk[chunk["itemid"].isin(itemids)]
        if not chunk.empty:
            out.append(chunk[["subject_id","hadm_id","stay_id","charttime","itemid","valuenum","valueuom"]])
    if out:
        df = pd.concat(out, ignore_index=True)
        df["version"] = version
        return df
    return pd.DataFrame(columns=["subject_id","hadm_id","stay_id","charttime","itemid","valuenum","valueuom","version"])

def load_filtered_outputevents(file_path, itemids, usecols, version, chunksize=500_000):
    """Stream OUTPUTEVENTS and keep only urinary itemids."""
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=[c.lower() for c in usecols] + ["version"])
    frames = []
    for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk.columns = [c.lower() for c in chunk.columns]
        if itemids:
            chunk = chunk[chunk["itemid"].isin(itemids)]
        if not chunk.empty:
            frames.append(chunk)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[c.lower() for c in usecols])
    df["version"] = version
    return df


def _read_csv_keep(path, usecols=None, low_memory=False):
    df = pd.read_csv(path, usecols=usecols, low_memory=low_memory)
    df.columns = [c.lower() for c in df.columns]
    return df

def _first_existing(*patterns):
    for pat in patterns:
        hits = glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None

def load_notes(
    data_dict: dict,
    path_mimic3="../data/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv.gz",
    path_mimic4_notes_root="../data/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note",
):
    """
    Load clinical notes into data_dict:
      - MIMIC-III NOTEEVENTS → data_dict['noteevents']
      - MIMIC-IV notes module (if present):
          radiology.csv.gz        → data_dict['radiology']
          radiology_detail.csv.gz → data_dict['radiology_detail']
          discharge.csv.gz        → data_dict['discharge']
          discharge_detail.csv.gz → data_dict['discharge_detail']
    Safe to call even if some files are missing.
    """
    # ---------- MIMIC-III NOTEEVENTS ----------
    try:
        df3 = _read_csv_keep(
            path_mimic3,
            usecols=["SUBJECT_ID", "HADM_ID", "CHARTTIME", "CATEGORY", "TEXT"],
        )
        df3 = df3.rename(columns={"charttime": "note_time"})
        df3["version"] = "mimic3"
        if "category" in df3.columns:
            df3["category"] = df3["category"].astype(str).str.lower()
        data_dict["noteevents"] = df3[["subject_id", "hadm_id", "note_time", "category", "text", "version"]]
        print(f"Loaded MIMIC-III NOTEEVENTS: {len(df3):,} rows")
    except FileNotFoundError:
        print(f"⚠️  MIMIC-III NOTEEVENTS not found at {path_mimic3} — skipping.")

    # ---------- MIMIC-IV notes module (optional separate download) ----------
    # Build absolute paths if the root exists
    if os.path.isdir(path_mimic4_notes_root):
        # Radiology
        p_rad = _first_existing(
            os.path.join(path_mimic4_notes_root, "radiology.csv.gz")
        )
        if p_rad:
            rad = _read_csv_keep(p_rad)
            # choose a reasonable time column
            for cand in ["report_time", "charttime", "note_time", "chartdate", "study_time"]:
                if cand in rad.columns:
                    rad["note_time"] = pd.to_datetime(rad[cand], errors="coerce")
                    break
            if "note_time" not in rad.columns:
                rad["note_time"] = pd.NaT
            # choose text column
            text_col = next((c for c in ["text", "note", "report", "impression"] if c in rad.columns), None)
            if text_col is None:
                rad["text"] = ""
                text_col = "text"
            rad["category"] = "radiology"
            rad["version"] = "mimic4"
            rad = rad.rename(columns={text_col: "text"})
            keep = ["subject_id", "hadm_id", "note_time", "category", "text", "version"]
            data_dict["radiology"] = rad[[c for c in keep if c in rad.columns]]
            print(f"Loaded MIMIC-IV radiology: {len(data_dict['radiology']):,} rows")

        # Radiology detail (metadata/sections)
        p_radd = _first_existing(
            os.path.join(path_mimic4_notes_root, "radiology_detail.csv.gz")
        )
        if p_radd:
            radd = _read_csv_keep(p_radd)
            data_dict["radiology_detail"] = radd
            print(f"Loaded MIMIC-IV radiology_detail: {len(radd):,} rows")

        # Discharge (leakage risk if used naively)
        p_dis = _first_existing(
            os.path.join(path_mimic4_notes_root, "discharge.csv.gz")
        )
        if p_dis:
            dis = _read_csv_keep(p_dis)
            for cand in ["charttime", "report_time", "note_time", "chartdate"]:
                if cand in dis.columns:
                    dis["note_time"] = pd.to_datetime(dis[cand], errors="coerce")
                    break
            if "note_time" not in dis.columns:
                dis["note_time"] = pd.NaT
            text_col = next((c for c in ["text", "note", "discharge_summary", "impression"] if c in dis.columns), None)
            if text_col is None:
                dis["text"] = ""
                text_col = "text"
            dis["category"] = "discharge"
            dis["version"] = "mimic4"
            dis = dis.rename(columns={text_col: "text"})
            keep = ["subject_id", "hadm_id", "note_time", "category", "text", "version"]
            data_dict["discharge"] = dis[[c for c in keep if c in dis.columns]]
            print(f"Loaded MIMIC-IV discharge: {len(data_dict['discharge']):,} rows")

        # Discharge detail
        p_disd = _first_existing(
            os.path.join(path_mimic4_notes_root, "discharge_detail.csv.gz")
        )
        if p_disd:
            disd = _read_csv_keep(p_disd)
            data_dict["discharge_detail"] = disd
            print(f"Loaded MIMIC-IV discharge_detail: {len(disd):,} rows")
    else:
        print(f"ℹ️  MIMIC-IV notes root not found: {path_mimic4_notes_root} — skipping radiology/discharge.")

    return data_dict

# -----------------------------
# Main compiler
# -----------------------------
def compile_data(output_pickle="data.pkl"):
    data = {}

    # ---- patients
    p3 = pd.read_csv("../data/mimic-iii-clinical-database-1.4/PATIENTS.csv.gz", low_memory=False)
    p3.columns = [c.lower() for c in p3.columns]; p3["version"] = "mimic3"
    p4 = pd.read_csv("../data/mimic-iv-3.1/hosp/patients.csv.gz", low_memory=False)
    p4.columns = [c.lower() for c in p4.columns]; p4["version"] = "mimic4"
    data["patients"] = pd.concat([p3, p4], ignore_index=True, sort=False)

    # ---- admissions
    a3 = pd.read_csv("../data/mimic-iii-clinical-database-1.4/ADMISSIONS.csv.gz", low_memory=False)
    a3.columns = [c.lower() for c in a3.columns]; a3["version"] = "mimic3"
    a4 = pd.read_csv("../data/mimic-iv-3.1/hosp/admissions.csv.gz", low_memory=False)
    a4.columns = [c.lower() for c in a4.columns]; a4["version"] = "mimic4"
    data["admissions"] = pd.concat([a3, a4], ignore_index=True, sort=False)

    # ---- icustays
    i3 = pd.read_csv("../data/mimic-iii-clinical-database-1.4/ICUSTAYS.csv.gz", low_memory=False)
    i3.columns = [c.lower() for c in i3.columns]; i3 = i3.rename(columns={"icustay_id": "stay_id"}); i3["version"] = "mimic3"
    i4 = pd.read_csv("../data/mimic-iv-3.1/icu/icustays.csv.gz", low_memory=False)
    i4.columns = [c.lower() for c in i4.columns]; i4["version"] = "mimic4"
    data["icustays"] = pd.concat([i3, i4], ignore_index=True, sort=False)

    # ---- services
    s3 = pd.read_csv("../data/mimic-iii-clinical-database-1.4/SERVICES.csv.gz", low_memory=False)
    s3.columns = [c.lower() for c in s3.columns]; s3["version"] = "mimic3"
    s4 = pd.read_csv("../data/mimic-iv-3.1/hosp/services.csv.gz", low_memory=False)
    s4.columns = [c.lower() for c in s4.columns]; s4["version"] = "mimic4"
    data["services"] = pd.concat([s3, s4], ignore_index=True, sort=False)

    # ---- transfers
    t3 = pd.read_csv("../data/mimic-iii-clinical-database-1.4/TRANSFERS.csv.gz", low_memory=False)
    t3.columns = [c.lower() for c in t3.columns]; t3["version"] = "mimic3"
    t4 = pd.read_csv("../data/mimic-iv-3.1/hosp/transfers.csv.gz", low_memory=False)
    t4.columns = [c.lower() for c in t4.columns]; t4["version"] = "mimic4"
    data["transfers"] = pd.concat([t3, t4], ignore_index=True, sort=False)

    # ---- diagnosis
    d3 = pd.read_csv("../data/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv.gz", low_memory=False)
    d3.columns = [c.lower() for c in d3.columns]; d3 = d3.rename(columns={"icd9_code": "icd_code"}); d3["icd_version"] = 9; d3["version"] = "mimic3"
    d4 = pd.read_csv("../data/mimic-iv-3.1/hosp/diagnoses_icd.csv.gz", low_memory=False)
    d4.columns = [c.lower() for c in d4.columns]; d4["version"] = "mimic4"
    data["diagnosis"] = pd.concat([d3, d4], ignore_index=True, sort=False)

    # ---- prescriptions (align)
    rx3 = pd.read_csv("../data/mimic-iii-clinical-database-1.4/PRESCRIPTIONS.csv.gz", low_memory=False)\
            .rename(columns={"STARTDATE": "starttime", "ENDDATE": "stoptime"})
    rx3.columns = [c.lower() for c in rx3.columns]; rx3["version"] = "mimic3"
    keep_cols = [c for c in rx3.columns if c not in ["row_id","icustay_id","drug_name_poe","drug_name_generic","version"]]
    rx4 = pd.read_csv("../data/mimic-iv-3.1/hosp/prescriptions.csv.gz", low_memory=False)
    rx4.columns = [c.lower() for c in rx4.columns]; rx4 = rx4[keep_cols]; rx4["version"] = "mimic4"
    data["prescriptions"] = pd.concat([rx3, rx4], ignore_index=True, sort=False)[
        ["subject_id","hadm_id","version","starttime","stoptime","drug"]
    ]

    # ---- procedures (align)
    pr3 = pd.read_csv("../data/mimic-iii-clinical-database-1.4/PROCEDURES_ICD.csv.gz", low_memory=False)
    pr3.columns = [c.lower() for c in pr3.columns]; pr3 = pr3.rename(columns={"icd9_code": "icd_code"}); pr3["icd_version"] = 9; pr3["chartdate"] = pd.NaT; pr3["version"] = "mimic3"
    pr4 = pd.read_csv("../data/mimic-iv-3.1/hosp/procedures_icd.csv.gz", low_memory=False)
    pr4.columns = [c.lower() for c in pr4.columns]; pr4["version"] = "mimic4"
    proc_cols = ["subject_id","hadm_id","seq_num","chartdate","icd_code","icd_version","version"]
    data["procedures"] = pd.concat([pr3[proc_cols], pr4[proc_cols]], ignore_index=True, sort=False)

    # ---- labitems
    li3 = pd.read_csv("../data/mimic-iii-clinical-database-1.4/D_LABITEMS.csv.gz", low_memory=False)
    li3 = li3.drop(columns=[c for c in ["ROW_ID","LOINC_CODE"] if c in li3.columns], errors="ignore")
    li3.columns = [c.lower() for c in li3.columns]; li3["version"] = "mimic3"
    li4 = pd.read_csv("../data/mimic-iv-3.1/hosp/d_labitems.csv.gz", low_memory=False)
    li4.columns = [c.lower() for c in li4.columns]; li4["version"] = "mimic4"
    data["labitems"] = pd.concat([li3, li4], ignore_index=True, sort=False)

    # ---- labevents
    le3 = pd.read_csv("../data/mimic-iii-clinical-database-1.4/LABEVENTS.csv.gz", low_memory=False)
    le3 = le3.drop(columns=[c for c in ["ROW_ID"] if c in le3.columns], errors="ignore")
    le3.columns = [c.lower() for c in le3.columns]; le3["version"] = "mimic3"
    le4 = pd.read_csv("../data/mimic-iv-3.1/hosp/labevents.csv.gz", low_memory=False)
    le4 = le4.drop(columns=[c for c in ["labevent_id","specimen_id","order_provider_id","storetime"] if c in le4.columns], errors="ignore")
    le4.columns = [c.lower() for c in le4.columns]; le4["version"] = "mimic4"
    data["labevents"] = pd.concat([le3, le4], ignore_index=True, sort=False)

    # ---- d_items (III + IV)
    di3 = pd.read_csv("../data/mimic-iii-clinical-database-1.4/D_ITEMS.csv.gz", low_memory=False)
    di3.columns = [c.lower() for c in di3.columns]; di3["version"] = "mimic3"
    di4 = pd.read_csv("../data/mimic-iv-3.1/icu/d_items.csv.gz", low_memory=False)
    di4.columns = [c.lower() for c in di4.columns]; di4["version"] = "mimic4"
    data["d_items"] = pd.concat([di3, di4], ignore_index=True, sort=False)

    # ---- chartevents (allow-list base + advanced vitals + ABG pH/lactate)
    base_labels = ["heart_rate","sbp","dbp","mbp","resp_rate","spo2","temperature","cvp"]
    adv_labels  = ["cardiac_output","pap_sbp","pap_dbp","pap_mbp","tidal_volume","minute_ventilation","gcs"]

    vital_ids3 = collect_vital_ids(di3, base_labels, adv_labels)
    vital_ids4 = collect_vital_ids(di4, base_labels, adv_labels)

    ph3, lact3 = abg_ids(di3)
    ph4, lact4 = abg_ids(di4)
    vital_ids3 = sorted(set(vital_ids3 + ph3 + lact3))
    vital_ids4 = sorted(set(vital_ids4 + ph4 + lact4))

    # (optional) add BP ids from helper (already covered by patterns, but harmless to union)
    bp3 = get_bp_itemids(di3); bp4 = get_bp_itemids(di4)
    vital_ids3 = sorted(set(vital_ids3 + bp3["map_ids"] + bp3["sbp_ids"] + bp3["dbp_ids"]))
    vital_ids4 = sorted(set(vital_ids4 + bp4["map_ids"] + bp4["sbp_ids"] + bp4["dbp_ids"]))

    ce3_path = "../data/mimic-iii-clinical-database-1.4/CHARTEVENTS.csv.gz"
    ce3_use  = ["SUBJECT_ID","HADM_ID","ICUSTAY_ID","CHARTTIME","ITEMID","VALUENUM","VALUEUOM"]
    ce4_path = "../data/mimic-iv-3.1/icu/chartevents.csv.gz"
    ce4_use  = ["subject_id","hadm_id","stay_id","charttime","itemid","valuenum","valueuom"]

    ce3 = load_filtered_chartevents(ce3_path, ce3_use, vital_ids3, "mimic3", rename_icustay=True)
    ce4 = load_filtered_chartevents(ce4_path, ce4_use, vital_ids4, "mimic4", rename_icustay=False)
    data["chartevents"] = pd.concat([ce3, ce4], ignore_index=True, sort=False)

    # ---- inputevents (align)
    in3 = pd.read_csv("../data/mimic-iii-clinical-database-1.4/INPUTEVENTS_CV.csv.gz",
                      usecols=["SUBJECT_ID","HADM_ID","ICUSTAY_ID","CHARTTIME","ITEMID","AMOUNT"],
                      low_memory=False)
    in3.columns = [c.lower() for c in in3.columns]; in3 = in3.rename(columns={"icustay_id": "stay_id"}); in3["version"] = "mimic3"

    in4 = pd.read_csv("../data/mimic-iv-3.1/icu/inputevents.csv.gz",
                      usecols=["subject_id","hadm_id","stay_id","starttime","itemid","amount"],
                      low_memory=False)
    in4.columns = [c.lower() for c in in4.columns]; in4["charttime"] = in4["starttime"]; in4["version"] = "mimic4"

    data["inputevents"] = pd.concat([
        in3[["subject_id","hadm_id","stay_id","charttime","itemid","amount","version"]],
        in4[["subject_id","hadm_id","stay_id","charttime","itemid","amount","version"]],
    ], ignore_index=True, sort=False)

    # ---- outputevents (urine subset)
    urine_ids3 = [40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405]
    out3 = load_filtered_outputevents("../data/mimic-iii-clinical-database-1.4/OUTPUTEVENTS.csv.gz",
                                      urine_ids3,
                                      ["SUBJECT_ID","HADM_ID","ICUSTAY_ID","CHARTTIME","ITEMID","VALUE","VALUEUOM"],
                                      "mimic3").rename(columns={"icustay_id": "stay_id"})

    urine_ids4 = [220739, 226559, 227488, 226627]
    out4 = load_filtered_outputevents("../data/mimic-iv-3.1/icu/outputevents.csv.gz",
                                      urine_ids4,
                                      ["subject_id","hadm_id","stay_id","charttime","itemid","value","valueuom"],
                                      "mimic4")

    data["outputevents"] = pd.concat([out3, out4], ignore_index=True, sort=False)

    data = load_notes(
    data,
    path_mimic3="../data/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv.gz",
    path_mimic4_notes_root="../data/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note",
    )

    # ---- save
    with open(output_pickle, "wb") as f:
        pickle.dump(data, f)

    log(f"✅ Saved compiled data to {output_pickle}")
    log("Tables:")
    for k, v in data.items():
        log(f"  - {k:12s}: {len(v):,} rows")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Compile MIMIC-III/IV into one data_dict pickle (no dedup; vitals+ABG CE).")
    ap.add_argument("--output_pickle", type=str, default="data.pkl", help="Output pickle filename.")
    args = ap.parse_args()
    compile_data(output_pickle=args.output_pickle)

if __name__ == "__main__":
    main()
