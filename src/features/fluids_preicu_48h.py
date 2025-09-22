# src/features/fluids_preicu_48h.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
import numpy as np
import pandas as pd

from src.utils.io import load_pickle, load_target, save_parquet
from src.utils.time_windows import choose_first_icu_after_admit, to_datetime

FEAT_NAME = "features_fluids_preicu48h"
KEYS = ["version", "subject_id", "hadm_id"]

def _parse_td(s: str) -> pd.Timedelta:
    """Parse '48h', '6h', '2d' etc. into Timedelta."""
    try:
        # allow h/H/d/D/m/M (minutes)
        if s.lower().endswith("h"):
            return pd.to_timedelta(int(s[:-1]), unit="h")
        if s.lower().endswith("d"):
            return pd.to_timedelta(int(s[:-1]), unit="d")
        if s.lower().endswith("m"):
            return pd.to_timedelta(int(s[:-1]), unit="m")
        # fallback to pandas parser
        return pd.to_timedelta(s)
    except Exception:
        return pd.to_timedelta(48, unit="h")

def make_windows(admissions: pd.DataFrame,
                 icustays: pd.DataFrame,
                 window_td: pd.Timedelta) -> pd.DataFrame:
    """
    Build per-admission window:
      - if ICU exists: [max(admit, icu_in - window_td), icu_in)
      - else:          [admit, min(discharge, admit + window_td))
    Returns KEYS + window_start, window_end, window_hours.
    """
    to_datetime(admissions, ["admittime", "dischtime"])
    icu_first = choose_first_icu_after_admit(admissions, icustays)

    base = (admissions[KEYS + ["admittime", "dischtime"]]
            .merge(icu_first, on=KEYS, how="left", validate="one_to_one"))

    admit = base["admittime"]
    disc  = base["dischtime"]
    icu_in = base["icu_intime"]
    has_icu = icu_in.notna()

    start = np.where(has_icu, np.maximum(admit, icu_in - window_td), admit)
    end   = np.where(has_icu, icu_in, np.minimum(disc.fillna(admit + window_td), admit + window_td))

    win = base[KEYS].copy()
    win["window_start"] = pd.to_datetime(start)
    win["window_end"]   = pd.to_datetime(end)
    # ensure non-empty intervals
    bad = win["window_end"] <= win["window_start"]
    if bad.any():
        win.loc[bad, "window_end"] = win.loc[bad, "window_start"] + pd.to_timedelta(1, "m")

    # hours length
    win["window_hours"] = (win["window_end"] - win["window_start"]).dt.total_seconds() / 3600.0
    return win

def clip_to_window(df: pd.DataFrame, win: pd.DataFrame, time_col: str, value_cols: list) -> pd.DataFrame:
    """
    Attach windows and keep rows inside [start, end). Returns KEYS + time_col + value_cols.
    """
    out = (df.merge(win, on=KEYS, how="inner")
             .loc[lambda d: d[time_col].notna()
                           & d["window_start"].notna()
                           & d["window_end"].notna()
                           & (d[time_col] >= d["window_start"])
                           & (d[time_col] <  d["window_end"]),
                  KEYS + [time_col] + value_cols + ["window_start","window_end","window_hours"]]
             .copy())
    return out

def sum_and_rate(df: pd.DataFrame, value_col: str, suffix: str) -> pd.DataFrame:
    """
    Aggregate totals and per-hour rate per admission.
    Outputs columns:
      {suffix}_total_ml, {suffix}_count, {suffix}_rate_ml_per_hr
    """
    if df.empty:
        cols = KEYS + [f"{suffix}_total_ml", f"{suffix}_count", f"{suffix}_rate_ml_per_hr"]
        return pd.DataFrame(columns=cols)

    g = df.groupby(KEYS, as_index=False).agg(
        total_ml=(value_col, "sum"),
        n=(value_col, "count"),
        hours=("window_hours", "first")  # same per admission
    )
    g[f"{suffix}_total_ml"] = g["total_ml"]
    g[f"{suffix}_count"] = g["n"]
    # avoid div-by-zero
    g[f"{suffix}_rate_ml_per_hr"] = np.where(g["hours"] > 0, g["total_ml"] / g["hours"], np.nan)
    return g[KEYS + [f"{suffix}_total_ml", f"{suffix}_count", f"{suffix}_rate_ml_per_hr"]]

def last_hours_slice(df: pd.DataFrame, hours: float, time_col: str) -> pd.DataFrame:
    """
    Filter to last `hours` of the window (ending at window_end).
    """
    if df.empty:
        return df
    cut = df["window_end"] - pd.to_timedelta(hours, unit="h")
    return df[df[time_col] >= cut]

def build(df_dict: dict, df_target: pd.DataFrame, window_str: str = "48h", resus_str: str = "6h") -> pd.DataFrame:
    """
    Build fluid features:
      - inputs/outputs totals & per-hour rates over pre-ICU window
      - "resuscitation"  last X hours of the window (default 6h)
      - net = input - output for both full window and last-X
    """
    admissions   = df_dict.get("admissions", pd.DataFrame()).copy()
    icustays     = df_dict.get("icustays",   pd.DataFrame()).copy()
    inputevents  = df_dict.get("inputevents", pd.DataFrame()).copy()
    outputevents = df_dict.get("outputevents", pd.DataFrame()).copy()

    # Parse times
    to_datetime(admissions,   ["admittime", "dischtime"])
    to_datetime(inputevents,  ["charttime"])
    to_datetime(outputevents, ["charttime"])

    window_td = _parse_td(window_str)
    resus_td  = _parse_td(resus_str)

    # Windows
    win = make_windows(admissions, icustays, window_td)

    # Keep only target admissions
    idx = df_target[KEYS].drop_duplicates()
    win = win.merge(idx, on=KEYS, how="inner")

    # -------- Inputs --------
    inp = inputevents.copy()
    if "amount" not in inp.columns:
        # nothing to compute
        inp = pd.DataFrame(columns=KEYS + ["charttime", "amount", "window_start","window_end","window_hours"])
    else:
        inp["amount"] = pd.to_numeric(inp["amount"], errors="coerce")
        # drop non-sensical negatives
        inp = inp[inp["amount"].notna() & (inp["amount"] >= 0)]
        inp = clip_to_window(inp, win, time_col="charttime", value_cols=["amount"])

    agg_inp = sum_and_rate(inp, "amount", "fluid48h_input")

    # last-X hours slice
    inp6 = last_hours_slice(inp, resus_td.total_seconds()/3600.0, time_col="charttime")
    agg_inp6 = sum_and_rate(inp6, "amount", "fluid6h_input")

    # -------- Outputs (urine) --------
    out = outputevents.copy()
    if "value" not in out.columns:
        out = pd.DataFrame(columns=KEYS + ["charttime", "value", "window_start","window_end","window_hours"])
    else:
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out[out["value"].notna() & (out["value"] >= 0)]
        # units in compile_data are already urine subset; value is typically ml
        out = clip_to_window(out, win, time_col="charttime", value_cols=["value"])

    agg_out = sum_and_rate(out, "value", "fluid48h_output_urine")

    out6 = last_hours_slice(out, resus_td.total_seconds()/3600.0, time_col="charttime")
    agg_out6 = sum_and_rate(out6, "value", "fluid6h_output_urine")

    # -------- Combine + nets --------
    feats = idx.copy()
    for piece in (agg_inp, agg_out, agg_inp6, agg_out6):
        feats = feats.merge(piece, on=KEYS, how="left")

    # compute nets
    feats["fluid48h_net_ml"] = feats["fluid48h_input_total_ml"] - feats["fluid48h_output_urine_total_ml"]
    # per-hour net (use window hours from win; safer to recompute via merge)
    feats = feats.merge(win[KEYS + ["window_hours"]], on=KEYS, how="left")
    feats["fluid48h_net_rate_ml_per_hr"] = np.where(
        feats["window_hours"] > 0, feats["fluid48h_net_ml"] / feats["window_hours"], np.nan
    )

    # resus net (last X hours) — rate per hour uses X hours (not window length)
    resus_hours = resus_td.total_seconds()/3600.0
    feats["fluid6h_net_ml"] = feats["fluid6h_input_total_ml"] - feats["fluid6h_output_urine_total_ml"]
    feats["fluid6h_net_rate_ml_per_hr"] = np.where(
        resus_hours > 0, feats["fluid6h_net_ml"] / resus_hours, np.nan
    )

    # tidy
    feats = feats.drop(columns=["window_hours"])

    return feats

def main():
    ap = argparse.ArgumentParser(description="Pre-ICU fluids (inputs + urine outputs) over 48h window and last-6h slice.")
    ap.add_argument("--data-pkl", default="data.pkl", help="Path to compiled data pickle.")
    ap.add_argument("--target",   default="target.parquet", help="Target parquet.")
    ap.add_argument("--outdir",   default=".", help="Output directory.")
    ap.add_argument("--window",   default="48h", help="Total window length (e.g., 48h, 2d).")
    ap.add_argument("--resus",    default="6h",  help="Last-X hours slice inside window (e.g., 6h).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df_dict   = load_pickle(args.data_pkl)
    df_target = load_target(args.target)

    feats = build(df_dict, df_target, window_str=args.window, resus_str=args.resus)
    outp  = os.path.join(args.outdir, f"{FEAT_NAME}.parquet")
    save_parquet(feats, outp)
    print(f"✓ wrote {outp}  shape={feats.shape}")

    # quick summary
    num_with_any_input  = (feats["fluid48h_input_count"].fillna(0)  > 0).sum()
    num_with_any_output = (feats["fluid48h_output_urine_count"].fillna(0) > 0).sum()
    print(f"  admissions with any inputs:  {num_with_any_input:,}")
    print(f"  admissions with any outputs: {num_with_any_output:,}")

if __name__ == "__main__":
    main()
