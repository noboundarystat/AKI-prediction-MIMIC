# src/features/labs_preckd.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
import numpy as np
import pandas as pd

from src.utils.io import load_pickle, save_parquet
from src.utils.time_windows import to_datetime
from src.features.labs_preicu_7d import select_lab_itemids  # reuse helper

KEYS = ["version","subject_id"]
FEAT_NAME = "features_labs_preckd"

WINDOWS = {
    "90d":  (0, 90),
    "180d": (90, 180),
    "365d": (180, 365),
}

# -----------------------
# Core
# -----------------------

def make_event_windows(df_target, admissions):
    """
    For each event (CKD, PostCKD, Death, Censored),
    generate [start,end) windows counting backwards from event admit time.
    """
    to_datetime(admissions, ["admittime","dischtime"])
    out = []

    for _, row in df_target.iterrows():
        sid = row["subject_id"]
        version = row["version"]

        # Check each event type explicitly
        for etype, col in [("CKD","ckd_event_hadm_id"),
                           ("PostCKD","postckd_event_hadm_id"),
                           ("Death","death_event_hadm_id")]:
            hadm_id = row.get(col)
            if pd.notna(hadm_id):
                adm_row = admissions[(admissions.subject_id==sid)&(admissions.hadm_id==hadm_id)]
                if adm_row.empty: 
                    continue
                event_time = pd.to_datetime(adm_row.iloc[0]["admittime"])
                for win_name,(lo,hi) in WINDOWS.items():
                    out.append(dict(version=version, subject_id=sid, event_type=etype,
                                    event_hadm_id=hadm_id,
                                    window=win_name,
                                    window_start=event_time - pd.Timedelta(days=hi),
                                    window_end=event_time - pd.Timedelta(days=lo)))

        # Special: censor → last discharge
        if row["event_type"]=="Censored":
            hadms = admissions[admissions.subject_id==sid]
            if hadms.empty:
                continue
            last_row = hadms.sort_values("dischtime").iloc[-1]
            hadm_id = last_row["hadm_id"]
            event_time = last_row["dischtime"]
            out.append(dict(version=version, subject_id=sid, event_type="Censored",
                            event_hadm_id=hadm_id,
                            window="lastobs",
                            window_start=pd.NaT, window_end=event_time))

    return pd.DataFrame(out)

def clip_to_window(df, win, time_col="charttime", keep_cols=None):
    keep_cols = keep_cols or []
    # bring in event_type and event_hadm_id from win
    out = (df.merge(win[["version","subject_id","event_type","event_hadm_id","window","window_start","window_end"]],
                    on=["version","subject_id"], how="inner")
             .loc[lambda d: d[time_col].notna()
                           & d["window_end"].notna()
                           & ((d["window_start"].isna()) | (d[time_col] >= d["window_start"]))
                           & (d[time_col] < d["window_end"]),
                  ["version","subject_id","event_type","event_hadm_id",time_col]+keep_cols+["window"]]
             .copy())
    return out.loc[:, ~out.columns.duplicated()]

def aggregate_signal(df, analyte, value_col="valuenum"):
    """
    Compute only the mean for each analyte per window.
    """
    if df.empty:
        return pd.DataFrame()

    out = []
    for keys, g in df.groupby(["version","subject_id","event_type","event_hadm_id","window"], as_index=False):
        vals = pd.to_numeric(g[value_col], errors="coerce")
        if vals.notna().any():
            row = dict(zip(["version","subject_id","event_type","event_hadm_id","window"], keys))
            row[f"labs{keys[-1]}_{analyte}_mean"] = vals.mean()
            out.append(row)
    return pd.DataFrame(out)

def build_features(df_dict, df_target):
    admissions = df_dict.get("admissions",pd.DataFrame()).copy()
    labevents  = df_dict.get("labevents", pd.DataFrame()).copy()
    labitems   = df_dict.get("labitems", pd.DataFrame()).copy()

    to_datetime(labevents,["charttime"])

    win = make_event_windows(df_target, admissions)
    if win.empty: 
        return pd.DataFrame(columns=["version","subject_id","event_type","event_hadm_id"])

    # select itemids
    idmap = select_lab_itemids(labitems)
    wanted = sorted({iid for ids in idmap.values() for iid in ids})
    labs = labevents[labevents["itemid"].isin(wanted)][["subject_id","hadm_id","charttime","valuenum","itemid"]].copy()
    # join version from admissions
    labs = labs.merge(admissions[["subject_id","hadm_id","version"]], on=["subject_id","hadm_id"], how="left")

    labs = clip_to_window(labs, win, keep_cols=["valuenum","itemid","window"])

    feats = win[["version","subject_id","event_type","event_hadm_id","window"]].drop_duplicates().copy()
    for analyte, ids in idmap.items():
        sub = labs[labs["itemid"].isin(ids)][["version","subject_id","event_type","event_hadm_id","charttime","valuenum","window"]]
        agg = aggregate_signal(sub, analyte)
        feats = feats.merge(agg, on=["version","subject_id","event_type","event_hadm_id","window"], how="left")

    # pivot window dimension wide
    feats = feats.pivot(index=["version","subject_id","event_type","event_hadm_id"], columns="window").reset_index()
    feats.columns = ["_".join([c for c in col if c]) for col in feats.columns.to_flat_index()]

    return feats

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Labs before CKD/PostCKD/Death/Censor events (0–90,90–180,180–365d, lastobs).")
    ap.add_argument("--data-pkl", default="../data.pkl")
    ap.add_argument("--target",   default="incident_ckd_patient.csv")
    ap.add_argument("--outdir",   default=".")
    ap.add_argument("--drop-thresh", type=float, default=1.01)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df_dict   = load_pickle(args.data_pkl)
    df_target = pd.read_csv(args.target)

    feats_all = build_features(df_dict, df_target)

    # Missingness filter
    miss_rows, drop_list = [], []
    for c in feats_all.columns:
        if c in ["version","subject_id","event_type","event_hadm_id"]: 
            miss_rows.append({"column": c, "missing_frac": 0.0}); continue
        mf = feats_all[c].isna().mean()
        miss_rows.append({"column": c, "missing_frac": mf})
        if mf > args.drop_thresh: drop_list.append(c)

    pd.DataFrame(miss_rows).to_csv(os.path.join(args.outdir,"missingness_labs_preckd.csv"),index=False)

    feats = feats_all.drop(columns=drop_list) if drop_list else feats_all

    outp = os.path.join(args.outdir,f"{FEAT_NAME}.parquet")
    save_parquet(feats,outp)

    print(f"✓ wrote {outp} shape={feats.shape}")
    print(f"  dropped {len(drop_list)} cols for missingness > {args.drop_thresh}")
    print(f"  missingness report: {os.path.join(args.outdir,'missingness_labs_preckd.csv')}")

if __name__=="__main__":
    main()
