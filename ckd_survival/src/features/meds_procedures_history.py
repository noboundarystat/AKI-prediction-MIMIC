# src/features/medsprocs_preckd.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Medication & Procedure exposures before CKD/PostCKD/Death/Censor events.

Windows:
  - 0–90d        → [event_time - 90d, event_time)
  - 90–180d      → [event_time - 180d, event_time - 90d)
  - 180–365d     → [event_time - 365d, event_time - 180d)
  - lastobs (censor only): [admit, last discharge]
"""

import argparse, os, re
import numpy as np
import pandas as pd

from src.utils.io import load_pickle, save_parquet
from src.utils.time_windows import to_datetime
from src.utils.meds_maps import DRUG_CLASS_PATTERNS  # dict: {class: [regex,...]}

KEYS = ["version","subject_id","event_type","event_hadm_id"]
FEAT_NAME = "features_medsprocs_preckd"

WINDOWS = {
    "hx90d":        (0, 90),
    "hx90to180d":   (90, 180),
    "hx180to365d":  (180, 365),
}

PROC_PREFIXES = {
    "rrt_any": {"icd9":["3995","5498"], "icd10":["5A1D"]},
    "crrt_any":{"icd9":[], "icd10":["5A1D8"]},
    "mech_vent_any":{"icd9":["967"], "icd10":["5A19"]},
}

# -----------------------
# Helpers
# -----------------------
def _combine_patterns_non_capturing(patterns):
    pats = []
    for p in patterns:
        pp = p.strip()
        if pp.startswith("(") and pp.endswith(")"):
            pp = pp[1:-1]
        pats.append(f"(?:{pp})")
    return re.compile("|".join(pats) if pats else r"^\b$", flags=re.I)

def _normalize_icd(code): return str(code).strip().upper().replace(".","")
def _match_icd_prefix(s, prefixes): 
    ss = _normalize_icd(s)
    return any(ss.startswith(pref.upper().replace(".","")) for pref in prefixes)

def make_event_windows(df_target, admissions):
    to_datetime(admissions, ["admittime","dischtime"])
    out = []
    for _, row in df_target.iterrows():
        sid, version = row["subject_id"], row["version"]
        for etype,col in [("CKD","ckd_event_hadm_id"),
                          ("PostCKD","postckd_event_hadm_id"),
                          ("Death","death_event_hadm_id")]:
            hadm_id = row.get(col)
            if pd.notna(hadm_id):
                adm_row = admissions[(admissions.subject_id==sid)&(admissions.hadm_id==hadm_id)]
                if adm_row.empty: continue
                event_time = pd.to_datetime(adm_row.iloc[0]["admittime"])
                for wname,(lo,hi) in WINDOWS.items():
                    out.append(dict(version=version,subject_id=sid,event_type=etype,
                                    event_hadm_id=hadm_id,
                                    window=wname,
                                    window_start=event_time - pd.Timedelta(days=hi),
                                    window_end=event_time - pd.Timedelta(days=lo)))
        if row["event_type"]=="Censored":
            hadms = admissions[admissions.subject_id==sid]
            if hadms.empty: continue
            last_row = hadms.sort_values("dischtime").iloc[-1]
            hadm_id = last_row["hadm_id"]; event_time = last_row["dischtime"]
            out.append(dict(version=version,subject_id=sid,event_type="Censored",
                            event_hadm_id=hadm_id,window="lastobs",
                            window_start=pd.NaT,window_end=event_time))
    return pd.DataFrame(out)

def build_meds_exposures(df_rx, win, class_patterns):
    if df_rx is None or df_rx.empty: return pd.DataFrame()
    to_datetime(df_rx,["starttime"])
    df_rx["drug_norm"] = df_rx["drug"].astype(str).str.lower()
    compiled = {cls:_combine_patterns_non_capturing(pats) for cls,pats in class_patterns.items()}
    for cls,creg in compiled.items():
        df_rx[f"class__{cls}"] = df_rx["drug_norm"].str.contains(creg,regex=True,na=False)
    joined = df_rx.merge(win,on=["version","subject_id"],how="inner")
    out=[]
    for keys,g in joined.groupby(["version","subject_id","event_type","event_hadm_id","window"]):
        flags={k:0 for k in class_patterns.keys()}
        for cls in class_patterns.keys():
            in_win=(g["starttime"]>=g["window_start"].iloc[0])&(g["starttime"]<g["window_end"].iloc[0])
            if in_win.any() and g.loc[in_win,f"class__{cls}"].any(): flags[cls]=1
        row=dict(zip(["version","subject_id","event_type","event_hadm_id","window"],keys))
        for cls,val in flags.items():
            row[f"{keys[-1]}_{cls}_any"]=val
        out.append(row)
    return pd.DataFrame(out)

def build_proc_exposures(df_proc, admissions, win):
    if df_proc is None or df_proc.empty: return pd.DataFrame()
    df=df_proc.copy(); df["icd_version"]=pd.to_numeric(df.get("icd_version",np.nan),errors="coerce")
    df["icd_code"]=df["icd_code"].astype(str).str.upper()
    adm_times=admissions[["version","subject_id","hadm_id","admittime","dischtime"]].copy()
    to_datetime(adm_times,["admittime","dischtime"])
    df=df.merge(adm_times,on=["version","subject_id","hadm_id"],how="left")
    df["event_time"]=pd.to_datetime(df.get("chartdate",pd.NaT),errors="coerce").fillna(df["dischtime"])
    for cls,pref in PROC_PREFIXES.items():
        icd9_p,icd10_p=pref.get("icd9",[]),pref.get("icd10",[])
        df[f"class__{cls}"]=((df["icd_version"]==9)&df["icd_code"].apply(lambda s:_match_icd_prefix(s,icd9_p)))|((df["icd_version"]==10)&df["icd_code"].apply(lambda s:_match_icd_prefix(s,icd10_p)))
    joined=df.merge(win,on=["version","subject_id"],how="inner")
    out=[]
    for keys,g in joined.groupby(["version","subject_id","event_type","event_hadm_id","window"]):
        flags={k:0 for k in PROC_PREFIXES.keys()}
        for cls in PROC_PREFIXES.keys():
            in_win=(g["event_time"]>=g["window_start"].iloc[0])&(g["event_time"]<g["window_end"].iloc[0])
            if in_win.any() and g.loc[in_win,f"class__{cls}"].any(): flags[cls]=1
        row=dict(zip(["version","subject_id","event_type","event_hadm_id","window"],keys))
        for cls,val in flags.items():
            row[f"{keys[-1]}_{cls}"]=val
        out.append(row)
    return pd.DataFrame(out)


def build_features(df_dict, df_target):
    admissions = df_dict.get("admissions", pd.DataFrame()).copy()
    rx = df_dict.get("prescriptions", pd.DataFrame()).copy()
    procedures = df_dict.get("procedures", pd.DataFrame()).copy()

    # make event windows (includes window col)
    win = make_event_windows(df_target, admissions)

    meds_feat = build_meds_exposures(rx, win, DRUG_CLASS_PATTERNS)
    proc_feat = build_proc_exposures(procedures, admissions, win)

    # keep window in the join key
    feats = win[["version","subject_id","event_type","event_hadm_id","window"]].drop_duplicates().copy()

    for f in [meds_feat, proc_feat]:
        if f is not None and not f.empty:
            feats = feats.merge(f, on=["version","subject_id","event_type","event_hadm_id","window"], how="left")

    # pivot window → wide columns
    feats = feats.pivot(index=["version","subject_id","event_type","event_hadm_id"], columns="window").reset_index()
    feats.columns = ["_".join([c for c in col if c]) for col in feats.columns.to_flat_index()]

    # fill missing flags with 0
    feats = feats.fillna(0).astype(int, errors="ignore")

    return feats


def main():
    ap=argparse.ArgumentParser(description="Meds/Procedures before CKD/PostCKD/Death/Censor events")
    ap.add_argument("--data-pkl",default="../data.pkl")
    ap.add_argument("--target",default="incident_ckd_patient.csv")
    ap.add_argument("--outdir",default=".")
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)
    df_dict=load_pickle(args.data_pkl); df_target=pd.read_csv(args.target)
    feats=build_features(df_dict,df_target)
    outp=os.path.join(args.outdir,f"{FEAT_NAME}.parquet"); save_parquet(feats,outp)
    print(f"✓ wrote {outp} shape={feats.shape}")

if __name__=="__main__": 
    main()
