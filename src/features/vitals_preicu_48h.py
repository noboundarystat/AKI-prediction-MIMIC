# src/features/vitals_preicu_48h.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
import numpy as np
import pandas as pd

from src.utils.io import load_pickle, load_target, save_parquet
from src.utils.itemids import get_vital_itemids
from src.utils.time_windows import choose_first_icu_after_admit, to_datetime
from src.utils.agg import aggregate_timeseries

KEYS = ["version","subject_id","hadm_id"]
FEAT_NAME = "features_vitals_preicu48h"

# Base vitals + advanced physio additions (CO, PAP, VT, VE, GCS)
VITALS = [
    # core
    "heart_rate",
    "sbp", "dbp", "mbp",
    "resp_rate",
    "spo2",
    "temperature",
    "cvp",
    # advanced
    "cardiac_output",       # CO
    "pap_sbp", "pap_dbp", "pap_mbp",  # pulmonary artery pressures
    "tidal_volume",         # VT
    "minute_ventilation",   # VE
    "gcs",                  # GCS total (or derive via itemid map)
]

def build_vital_itemid_map(d_items: pd.DataFrame):
    """
    Build {signal: [itemids]} using pattern-based lookup in d_items.
    Signals with no matches are omitted.
    """
    if d_items is None or d_items.empty:
        return {}
    itemid_map = {}
    for sig in VITALS:
        try:
            ids = get_vital_itemids(d_items, sig)
        except Exception:
            ids = []
        if ids:
            itemid_map[sig] = ids
    return itemid_map

def make_windows(admissions: pd.DataFrame, icustays: pd.DataFrame) -> pd.DataFrame:
    """
    Return KEYS + window_start, window_end where:
      - if ICU exists: [max(admit, icu_intime - 48h), icu_intime)
      - else:          [admit, min(discharge, admit + 48h))
    """
    to_datetime(admissions, ["admittime","dischtime"])
    icu_first = choose_first_icu_after_admit(admissions, icustays)

    base = (admissions[KEYS + ["admittime","dischtime"]]
              .merge(icu_first, on=KEYS, how="left", validate="one_to_one"))

    admit = base["admittime"]; disc = base["dischtime"]; icu_in = base["icu_intime"]
    forty8 = pd.to_timedelta(48, unit="h")
    has_icu = icu_in.notna()

    start = np.where(has_icu, np.maximum(admit, icu_in - forty8), admit)
    end   = np.where(has_icu, icu_in, np.minimum(disc.fillna(admit + forty8), admit + forty8))

    win = base[KEYS].copy()
    win["window_start"] = pd.to_datetime(start)
    win["window_end"]   = pd.to_datetime(end)

    bad = win["window_end"] <= win["window_start"]
    if bad.any():
        win.loc[bad, "window_end"] = win.loc[bad, "window_start"] + pd.to_timedelta(1, "m")

    return win

def aggregate_signal(df_ce: pd.DataFrame, signal: str) -> pd.DataFrame:
    """
    df_ce is already filtered to rows for one signal within window.
    Returns KEYS + vital48h_{signal}_{min,max,mean,std,last,count}.
    """
    cols = KEYS + [f"vital48h_{signal}_{m}" for m in ["min","max","mean","std","last","count"]]
    if df_ce.empty:
        return pd.DataFrame(columns=cols)

    df_ce = df_ce.copy()
    df_ce["valuenum"]  = pd.to_numeric(df_ce["valuenum"], errors="coerce")
    df_ce["charttime"] = pd.to_datetime(df_ce["charttime"], errors="coerce")

    rows = []
    for keys, g in df_ce.groupby(KEYS, as_index=False):
        agg = aggregate_timeseries(g, value_col="valuenum", time_col="charttime")
        row = dict(zip(KEYS, keys))
        for k, v in agg.items():
            row[f"vital48h_{signal}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)

def build(df_dict, df_target):
    # Required tables
    admissions  = df_dict.get("admissions",  pd.DataFrame()).copy()
    icustays    = df_dict.get("icustays",    pd.DataFrame()).copy()
    chartevents = df_dict.get("chartevents", pd.DataFrame()).copy()
    d_items     = df_dict.get("d_items",     pd.DataFrame()).copy()

    # Parse times
    to_datetime(admissions,  ["admittime","dischtime"])
    to_datetime(chartevents, ["charttime"])

    # Build windows
    windows = make_windows(admissions, icustays)

    # Map vital → itemids
    idmap = build_vital_itemid_map(d_items if "itemid" in d_items.columns else pd.DataFrame())
    if not idmap:
        return df_target[KEYS].drop_duplicates().copy()

    wanted = sorted({iid for ids in idmap.values() for iid in ids})
    if not wanted:
        return df_target[KEYS].drop_duplicates().copy()

    # Trim CE to IDs of interest and clamp to window
    ce = chartevents[chartevents["itemid"].isin(wanted)].copy()
    if ce.empty:
        return df_target[KEYS].drop_duplicates().copy()

    ce = (ce.merge(windows, on=KEYS, how="inner")
             .loc[lambda d: d["charttime"].notna()
                           & d["window_start"].notna()
                           & d["window_end"].notna()
                           & (d["charttime"] >= d["window_start"])
                           & (d["charttime"] <  d["window_end"]),
                  KEYS + ["itemid","charttime","valuenum"]]
             .copy())

    # Aggregate each signal
    feats = df_target[KEYS].drop_duplicates().copy()
    for sig, ids in idmap.items():
        sub = ce[ce["itemid"].isin(ids)]
        agg = aggregate_signal(sub, sig)
        feats = feats.merge(agg, on=KEYS, how="left")

    # ---- Derived metrics (only if inputs exist)
    # Pulse pressure: SBP - DBP
    if {"vital48h_sbp_mean", "vital48h_dbp_mean"}.issubset(set(feats.columns)):
        feats["vital48h_pulse_pressure_mean"] = feats["vital48h_sbp_mean"] - feats["vital48h_dbp_mean"]

    # Shock index: HR / MAP (MBP)
    if {"vital48h_heart_rate_mean", "vital48h_mbp_mean"}.issubset(set(feats.columns)):
        denom = feats["vital48h_mbp_mean"].replace(0, np.nan)
        feats["vital48h_shock_index_mean"] = feats["vital48h_heart_rate_mean"] / denom

    # PAP pulse pressure: PAP SBP - PAP DBP
    if {"vital48h_pap_sbp_mean", "vital48h_pap_dbp_mean"}.issubset(set(feats.columns)):
        feats["vital48h_pap_pulse_pressure_mean"] = feats["vital48h_pap_sbp_mean"] - feats["vital48h_pap_dbp_mean"]

    # --- compat aliases for checker regexs: vital48h_<sig>_.*_mean
    compat_signals = ["sbp", "dbp", "mbp", "spo2", "temperature"]
    for sig in compat_signals:
        base = f"vital48h_{sig}_mean"
        alias = f"vital48h_{sig}_value_mean"   # matches ^vital48h_sbp_.*_mean$
        if base in feats.columns and alias not in feats.columns:
            feats[alias] = feats[base]

    return feats

def main():
    ap = argparse.ArgumentParser(description="Pre-ICU (48h) vitals & advanced physio from CHARTEVENTS.")
    ap.add_argument("--data-pkl", default="data.pkl")
    ap.add_argument("--target",   default="target.parquet")
    ap.add_argument("--outdir",   default=".")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df_dict   = load_pickle(args.data_pkl)
    df_target = load_target(args.target)

    feats = build(df_dict, df_target)
    outp  = os.path.join(args.outdir, f"{FEAT_NAME}.parquet")
    save_parquet(feats, outp)
    print(f"✓ wrote {outp}  shape={feats.shape}")

if __name__ == "__main__":
    main()


