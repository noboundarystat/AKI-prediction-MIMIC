# src/features/meds_procedures_history.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Medication & Procedure history features prior to the index admission.

Outputs one row per (version, subject_id, hadm_id) with flags like:
  - hx30d_nsaids_any, hx31to180d_vasopressors_any, ...
  - hx30d_rrt_any, hx31to180d_crrt_any, hx31to180d_mech_vent_any

Windows are relative to the index admission's admittime:
  hx30d        -> [admit - 30d, admit)
  hx31to180d   -> [admit - 180d, admit - 30d)
"""

import argparse
import os
import re
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.utils.io import load_pickle, load_target, save_parquet
from src.utils.time_windows import to_datetime
from src.utils.meds_maps import DRUG_CLASS_PATTERNS  # dict: {class_name: [regex,...]}

KEYS = ["version", "subject_id", "hadm_id"]
FEAT_NAME = "features_meds_procedures_history"

# ----------------------------
# Procedure classes → ICD prefixes
# ----------------------------
PROC_PREFIXES: Dict[str, Dict[str, List[str]]] = {
    # Renal replacement therapy (dialysis)
    "rrt_any": {
        "icd9":  ["3995", "5498"],
        "icd10": ["5A1D"],
    },
    "crrt_any": {
        "icd9":  [],
        "icd10": ["5A1D8"],
    },
    "mech_vent_any": {
        "icd9":  ["967"],
        "icd10": ["5A19"],
    },
}

# ----------------------------
# Utilities
# ----------------------------
def _combine_patterns_non_capturing(patterns: Iterable[str]) -> re.Pattern:
    pats = []
    for p in patterns:
        pp = p.strip()
        if pp.startswith("(") and pp.endswith(")"):
            pp = pp[1:-1]
        pats.append(f"(?:{pp})")
    combined = "|".join(pats) if pats else r"^\b$"
    return re.compile(combined, flags=re.I)

def _normalize_icd(code: str) -> str:
    s = str(code).strip().upper()
    return s.replace(".", "")

def _match_icd_prefix(s: str, prefixes: List[str]) -> bool:
    ss = _normalize_icd(s)
    return any(ss.startswith(pref.upper().replace(".", "")) for pref in prefixes)

def _mk_index(df_target: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    adm = admissions.copy()
    to_datetime(adm, ["admittime"])
    idx = df_target[KEYS].drop_duplicates().merge(
        adm[KEYS + ["admittime"]], on=KEYS, how="left", validate="one_to_one"
    )
    return idx

# ----------------------------
# Medications exposure
# ----------------------------
def build_meds_exposures(
    df_rx: pd.DataFrame,
    idx: pd.DataFrame,
    windows: List[tuple],
    class_patterns: Dict[str, Iterable[str]],
) -> pd.DataFrame:
    if df_rx is None or df_rx.empty:
        return idx[KEYS].copy()

    df = df_rx.copy()
    to_datetime(df, ["starttime"])
    if "drug" not in df.columns:
        df["drug"] = ""
    df["drug_norm"] = df["drug"].astype(str).str.lower()

    compiled: Dict[str, re.Pattern] = {cls: _combine_patterns_non_capturing(pats)
                                       for cls, pats in class_patterns.items()}

    if idx["admittime"].notna().any():
        t_min = idx["admittime"].min() - pd.to_timedelta(max(hi for _, hi in windows), unit="d")
        t_max = idx["admittime"].max()
        df = df[(df["starttime"] >= t_min) & (df["starttime"] < t_max)]

    if df.empty:
        return idx[KEYS].copy()

    for cls, creg in compiled.items():
        df[f"class__{cls}"] = df["drug_norm"].str.contains(creg, regex=True, na=False)

    joined = df.merge(
        idx[KEYS + ["admittime"]],
        on=["version", "subject_id"],
        how="inner",
        suffixes=("", "_index"),
        validate="many_to_many",
    )

    out = idx[KEYS].drop_duplicates().copy()
    for lo, hi in windows:
        wname = f"hx{lo}to{hi}d" if lo > 0 else f"hx{hi}d"
        win_start = joined["admittime"] - pd.to_timedelta(hi, unit="d")
        win_end   = joined["admittime"] - pd.to_timedelta(lo, unit="d")
        in_win = (joined["starttime"] >= win_start) & (joined["starttime"] < win_end)

        sub = joined.loc[in_win, KEYS + [c for c in joined.columns if c.startswith("class__")]].copy()
        if sub.empty:
            for cls in class_patterns.keys():
                out[f"{wname}_{cls}_any"] = 0
            continue

        group_keys = [k for k in KEYS if k in sub.columns]
        g = (
            sub.groupby(group_keys)[[c for c in sub.columns if c.startswith("class__")]]
               .any().astype(int).reset_index()
               .rename(columns=lambda c: f"{wname}_{c.replace('class__','')}_any" if c.startswith("class__") else c)
        )
        out = out.merge(g, on=group_keys, how="left")

    for c in out.columns:
        if c not in KEYS:
            out[c] = out[c].fillna(0).astype(int)
    return out

# ----------------------------
# Procedures exposure
# ----------------------------
def build_proc_exposures(
    df_proc: pd.DataFrame,
    admissions: pd.DataFrame,
    idx: pd.DataFrame,
    windows: List[tuple],
) -> pd.DataFrame:
    if df_proc is None or df_proc.empty:
        return idx[KEYS].copy()

    proc = df_proc.copy()
    proc["icd_version"] = pd.to_numeric(proc.get("icd_version", np.nan), errors="coerce")
    proc["icd_code"] = proc["icd_code"].astype(str).str.upper()

    adm_times = admissions[["version", "subject_id", "hadm_id", "admittime", "dischtime"]].copy()
    to_datetime(adm_times, ["admittime", "dischtime"])
    proc = proc.merge(adm_times, on=["version", "subject_id", "hadm_id"], how="left", validate="many_to_one")

    evt_time = pd.to_datetime(proc.get("chartdate", pd.NaT), errors="coerce")
    evt_time = evt_time.fillna(proc["dischtime"])
    proc["event_time"] = evt_time

    for cls, pref in PROC_PREFIXES.items():
        icd9_p = pref.get("icd9", [])
        icd10_p = pref.get("icd10", [])
        proc[f"class__{cls}"] = (
            ((proc["icd_version"] == 9)  & proc["icd_code"].apply(lambda s: _match_icd_prefix(s, icd9_p))) |
            ((proc["icd_version"] == 10) & proc["icd_code"].apply(lambda s: _match_icd_prefix(s, icd10_p)))
        )

    if idx["admittime"].notna().any():
        t_min = idx["admittime"].min() - pd.to_timedelta(max(hi for _, hi in windows), unit="d")
        t_max = idx["admittime"].max()
        proc = proc[(proc["event_time"] >= t_min) & (proc["event_time"] < t_max)]

    if proc.empty:
        return idx[KEYS].copy()

    joined = proc.merge(
        idx[KEYS + ["admittime"]],
        on=["version", "subject_id"],
        how="inner",
        suffixes=("", "_index"),
        validate="many_to_many",
    )

    out = idx[KEYS].drop_duplicates().copy()
    for lo, hi in windows:
        wname = f"hx{lo}to{hi}d" if lo > 0 else f"hx{hi}d"
        win_start = joined["admittime"] - pd.to_timedelta(hi, unit="d")
        win_end   = joined["admittime"] - pd.to_timedelta(lo, unit="d")
        in_win = (joined["event_time"] >= win_start) & (joined["event_time"] < win_end)

        sub = joined.loc[in_win, KEYS + [c for c in joined.columns if c.startswith("class__")]].copy()
        if sub.empty:
            for cls in PROC_PREFIXES.keys():
                out[f"{wname}_{cls}"] = 0
            continue

        group_keys = [k for k in KEYS if k in sub.columns]
        g = (
            sub.groupby(group_keys)[[c for c in sub.columns if c.startswith("class__")]]
               .any().astype(int).reset_index()
               .rename(columns=lambda c: f"{wname}_{c.replace('class__','')}" if c.startswith("class__") else c)
        )
        out = out.merge(g, on=group_keys, how="left")

    for c in out.columns:
        if c not in KEYS:
            out[c] = out[c].fillna(0).astype(int)
    return out

# ----------------------------
# Orchestration
# ----------------------------
def build_features(df_dict: dict, df_target: pd.DataFrame, windows=None) -> pd.DataFrame:
    if windows is None:
        windows = [(0,30), (31,180)]  # default non-overlapping windows

    admissions  = df_dict.get("admissions",    pd.DataFrame()).copy()
    rx          = df_dict.get("prescriptions", pd.DataFrame()).copy()
    procedures  = df_dict.get("procedures",    pd.DataFrame()).copy()

    idx = _mk_index(df_target, admissions)

    meds_feat = build_meds_exposures(rx, idx, windows, DRUG_CLASS_PATTERNS)
    proc_feat = build_proc_exposures(procedures, admissions, idx, windows)

    feats = idx[KEYS].drop_duplicates()
    feats = feats.merge(meds_feat, on=[k for k in KEYS if k in meds_feat.columns], how="left")
    feats = feats.merge(proc_feat, on=[k for k in KEYS if k in proc_feat.columns], how="left")

    for c in feats.columns:
        if c not in KEYS:
            feats[c] = feats[c].fillna(0).astype(int)

    return feats

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Medication & Procedure exposures in prior non-overlapping windows")
    ap.add_argument("--data-pkl", default="data.pkl")
    ap.add_argument("--target",   default="target.parquet")
    ap.add_argument("--outdir",   default=".")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df_dict   = load_pickle(args.data_pkl)
    df_target = load_target(args.target)

    feats = build_features(df_dict, df_target)
    outp  = os.path.join(args.outdir, f"{FEAT_NAME}.parquet")
    save_parquet(feats, outp)

    # quick recap
    flag_cols = [c for c in feats.columns if c not in KEYS]
    print(f"✓ wrote {outp}  shape={feats.shape}")
    by_win = {}
    for c in flag_cols:
        w = c.split("_", 1)[0]  # hx30d / hx31to180d
        by_win.setdefault(w, 0)
        by_win[w] += feats[c].sum()
    for w, s in sorted(by_win.items()):
        print(f"  total exposures in {w}: {s:,}")

if __name__ == "__main__":
    main()
