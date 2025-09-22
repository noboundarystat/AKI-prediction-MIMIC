# src/features/labs_preicu_7d.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd

from src.utils.io import load_pickle, load_target, save_parquet
from src.utils.time_windows import choose_first_icu_after_admit, to_datetime
from src.utils.agg import aggregate_timeseries

KEYS = ["version","subject_id","hadm_id"]
FEAT_NAME = "features_labs_preicu7d"

# -----------------------
# Helpers
# -----------------------

def _parse_td(s: str) -> pd.Timedelta:
    s = str(s).strip().lower()
    if s.endswith("h"): return pd.to_timedelta(int(s[:-1]), unit="h")
    if s.endswith("d"): return pd.to_timedelta(int(s[:-1]), unit="d")
    return pd.to_timedelta(s)

def make_windows(admissions: pd.DataFrame,
                 icustays: pd.DataFrame,
                 window_td: pd.Timedelta) -> pd.DataFrame:
    """
    7d prior to ICU (or admit if no ICU):
      if ICU exists: [max(admit, icu_in - window_td), icu_in)
      else:          [admit, min(discharge, admit + window_td))
    """
    to_datetime(admissions, ["admittime","dischtime"])
    icu_first = choose_first_icu_after_admit(admissions, icustays)

    base = (admissions[KEYS + ["admittime","dischtime"]]
              .merge(icu_first, on=KEYS, how="left", validate="one_to_one"))

    admit = base["admittime"]; disc = base["dischtime"]; icu_in = base["icu_intime"]
    has_icu = icu_in.notna()

    start = np.where(has_icu, np.maximum(admit, icu_in - window_td), admit)
    end   = np.where(has_icu, icu_in, np.minimum(disc.fillna(admit + window_td), admit + window_td))

    win = base[KEYS].copy()
    win["window_start"] = pd.to_datetime(start)
    win["window_end"]   = pd.to_datetime(end)
    bad = win["window_end"] <= win["window_start"]
    if bad.any():
        win.loc[bad, "window_end"] = win.loc[bad, "window_start"] + pd.to_timedelta(1, "m")
    return win

def clip_to_window(df: pd.DataFrame, win: pd.DataFrame, time_col: str, keep_cols: Iterable[str]) -> pd.DataFrame:
    out = (df.merge(win, on=KEYS, how="inner")
             .loc[lambda d: d[time_col].notna()
                           & d["window_start"].notna()
                           & d["window_end"].notna()
                           & (d[time_col] >= d["window_start"])
                           & (d[time_col] <  d["window_end"]),
                  KEYS + [time_col] + list(keep_cols) + ["window_end"]]
             .copy())
    return out.loc[:, ~out.columns.duplicated()]

def _ids_like(df: pd.DataFrame, include: str, exclude: str = None) -> List[int]:
    if df is None or df.empty or "label" not in df.columns: return []
    lab = df["label"].astype(str)
    m = lab.str.contains(include, case=False, na=False)
    if exclude:
        m &= ~lab.str.contains(exclude, case=False, na=False)
    return df.loc[m, "itemid"].dropna().astype(int).unique().tolist()

def select_lab_itemids(labitems: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Map analyte name -> list[itemid] using label heuristics.
    Excludes obvious non-blood contexts for ABG-like analytes.
    """
    if labitems is None or labitems.empty: return {}

    excludes = "urine|csf|stool|gastric|aspirate|fluid"

    ids = {
        "wbc":        _ids_like(labitems, r"\bwbc\b|white blood"),
        "hgb":        _ids_like(labitems, r"\bhgb\b|hemoglobin"),
        "plt":        _ids_like(labitems, r"platelet"),
        "creatinine": _ids_like(labitems, r"creatinine"),
        "bun":        _ids_like(labitems, r"\bbun\b|urea"),
        "albumin":    _ids_like(labitems, r"\balbumin\b"),
        "sodium":     _ids_like(labitems, r"\bsodium\b|\bna\b(?!\w)"),
        "potassium":  _ids_like(labitems, r"\bpotassium\b|\bk\+\b|\bk\b(?!\w)"),
        "bicarbonate":_ids_like(labitems, r"bicarbonate|hco3|co2 \(bicarb\)"),
        "lactate":    _ids_like(labitems, r"lactate", exclude=excludes),
        "ph":         _ids_like(labitems, r"\bph\b", exclude=f"(phosph|{excludes})"),
    }
    # drop empties
    return {k: sorted(set(v)) for k, v in ids.items() if v}

def aggregate_signal(df: pd.DataFrame, analyte: str, value_col: str = "valuenum") -> pd.DataFrame:
    """
    df: KEYS + charttime + value_col
    Returns KEYS + labs7d_{analyte}_{min,max,mean,std,last,count}
    """
    prefix = f"labs7d_{analyte}"
    cols = KEYS + [f"{prefix}_{m}" for m in ["min","max","mean","std","last","count"]]
    if df.empty:
        return pd.DataFrame(columns=cols)

    out = []
    for keys, g in df.groupby(KEYS, as_index=False):
        vals = pd.to_numeric(g[value_col], errors="coerce")
        tt   = pd.to_datetime(g["charttime"], errors="coerce")
        ok = vals.notna() & tt.notna()
        if not ok.any(): 
            continue
        gg = g.loc[ok].copy()
        gg[value_col] = vals[ok]
        gg["charttime"] = tt[ok]
        agg = aggregate_timeseries(gg, value_col=value_col, time_col="charttime")
        row = dict(zip(KEYS, keys))
        for k, v in agg.items():
            row[f"{prefix}_{k}"] = v
        out.append(row)
    return pd.DataFrame(out, columns=cols)

# -----------------------
# Build
# -----------------------

def build_features(df_dict: dict,
                   df_target: pd.DataFrame,
                   window_str: str = "7d") -> pd.DataFrame:
    admissions = df_dict.get("admissions",  pd.DataFrame()).copy()
    icustays   = df_dict.get("icustays",    pd.DataFrame()).copy()
    labevents  = df_dict.get("labevents",   pd.DataFrame()).copy()
    labitems   = df_dict.get("labitems",    pd.DataFrame()).copy()

    to_datetime(admissions, ["admittime","dischtime"])
    to_datetime(labevents,  ["charttime"])

    win = make_windows(admissions, icustays, _parse_td(window_str))
    idx = df_target[KEYS].drop_duplicates()
    win = win.merge(idx, on=KEYS, how="inner")

    # select itemids
    idmap = select_lab_itemids(labitems)
    if not idmap:
        return idx.copy()

    wanted = sorted({iid for ids in idmap.values() for iid in ids})
    labs = labevents[labevents["itemid"].isin(wanted)][KEYS + ["itemid","charttime","valuenum"]].copy()
    if labs.empty:
        return idx.copy()

    labs = clip_to_window(labs, win, time_col="charttime", keep_cols=["valuenum","itemid"])

    feats = idx.copy()
    for analyte, ids in idmap.items():
        sub = labs[labs["itemid"].isin(ids)][KEYS + ["charttime","valuenum"]]
        agg = aggregate_signal(sub, analyte)
        feats = feats.merge(agg, on=KEYS, how="left")

    feats = feats.loc[:, ~feats.columns.duplicated()].copy()
    
    return feats

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Labs in 7 days prior to ICU/admit; drop features by missingness.")
    ap.add_argument("--data-pkl", default="data.pkl")
    ap.add_argument("--target",   default="target.parquet")
    ap.add_argument("--outdir",   default=".")
    ap.add_argument("--window",   default="7d", help="Lookback window (e.g., 7d).")
    ap.add_argument("--drop-thresh", type=float, default=1.01,
                    help="Drop feature columns with missing fraction > this threshold (0..1). Default 0.7")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df_dict   = load_pickle(args.data_pkl)
    df_target = load_target(args.target)

    feats_all = build_features(df_dict, df_target, window_str=args.window)

    # Missingness filter
    keep = feats_all.columns.tolist()
    drop_list = []
    miss_rows = []
    for c in feats_all.columns:
        if c in KEYS: 
            miss_rows.append({"column": c, "missing_frac": 0.0})
            continue
        mf = feats_all[c].isna().mean()
        miss_rows.append({"column": c, "missing_frac": mf})
        if mf > args.drop_thresh:
            drop_list.append(c)

    missing_df = pd.DataFrame(miss_rows)
    missing_df.to_csv(os.path.join(args.outdir, "missingness_labs_preicu7d.csv"), index=False)

    if drop_list:
        feats = feats_all.drop(columns=drop_list)
    else:
        feats = feats_all

    outp = os.path.join(args.outdir, f"{FEAT_NAME}.parquet")
    save_parquet(feats, outp)

    print(f"✓ wrote {outp}  shape={feats.shape}")
    print(f"  dropped {len(drop_list)} columns for missingness > {args.drop_thresh:.2f}")
    print(f"  missingness report: {os.path.join(args.outdir, 'missingness_labs_preicu7d.csv')}")

if __name__ == "__main__":
    main()
