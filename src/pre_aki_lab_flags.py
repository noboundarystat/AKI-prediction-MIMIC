#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pre_aki_lab_flags.py

Create a lab-based pre-admission AKI flag (KDIGO creatinine criteria) in the
7 days prior to the index hospital admission time.

Exports two main entry points:
  - attach_lab_aki_prior7d_flags(target, labevents, labitems) -> target_with_flag
  - compute_lab_aki_prior7d(data_pkl, target_path, outdir)    -> writes artifacts

Artifacts (when run via CLI):
  - outdir/lab_aki_prior7d.parquet              # per-admission lab flag
  - outdir/false_negatives_prior7d.csv          # ICD is_aki==0 but lab prior-7d==1
  - outdir/lab_aki_summary.csv                  # counts by version
"""

import os
import argparse
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def to_datetime(df: pd.DataFrame, cols) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

# ---------------------------------------------------------------------
# Creatinine itemid selection (pattern-based, no dependency on utils)
# ---------------------------------------------------------------------

def select_creatinine_itemids(labitems: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with columns ['itemid'] for creatinine analytes.
    Uses label-based matching from D_LABITEMS.
    """
    li = labitems.copy()
    li.columns = [c.lower() for c in li.columns]
    if not {"itemid", "label"}.issubset(li.columns):
        raise ValueError("labitems must include columns: itemid, label")
    mask = li["label"].astype(str).str.contains(r"\bcreatinine\b", case=False, na=False)
    # Optional: try to exclude urine creatinine if you wish:
    # mask &= ~li["label"].astype(str).str.contains(r"urine", case=False, na=False)
    return li.loc[mask, ["itemid"]].drop_duplicates()

# ---------------------------------------------------------------------
# Prep labs & attach window relative to admissions
# ---------------------------------------------------------------------

def prep_labs(labevents: pd.DataFrame, creat_items: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only creatinine rows with numeric valuenum and charttime.
    Return standardized columns.
    """
    labs = labevents.copy()
    labs.columns = [c.lower() for c in labs.columns]
    to_datetime(labs, ["charttime"])
    labs["valuenum"] = pd.to_numeric(labs.get("valuenum", np.nan), errors="coerce")

    itemids = set(creat_items["itemid"].astype(int).unique().tolist())
    labs = labs[labs["itemid"].isin(itemids)]
    labs = labs[labs["valuenum"].notna() & labs["charttime"].notna()]

    keep = ["version", "subject_id", "hadm_id", "charttime", "valuenum"]
    keep = [c for c in keep if c in labs.columns]
    return labs[keep].copy()

def build_window(admissions: pd.DataFrame, labs: pd.DataFrame, days_prior: int = 7) -> pd.DataFrame:
    """
    Create a Cartesian join of labs to *each* admission for that subject/version,
    then flag labs that fall in [admit - days_prior, admit).
    """
    adm = admissions.copy()
    adm.columns = [c.lower() for c in adm.columns]
    to_datetime(adm, ["admittime"])

    idx = adm[["version", "subject_id", "hadm_id", "admittime"]].drop_duplicates()
    # Cartesian on (version, subject_id), then time filter
    labs_ext = labs.merge(idx[["version", "subject_id", "hadm_id", "admittime"]],
                          on=["version", "subject_id"], how="inner", suffixes=("", "_index"))

    win_start = labs_ext["admittime"] - pd.to_timedelta(days_prior, unit="d")
    in_prior7d = (labs_ext["charttime"] >= win_start) & (labs_ext["charttime"] < labs_ext["admittime"])
    labs_ext["in_prior7d"] = in_prior7d.astype(bool)

    # Keep standardized columns for downstream groupby
    out = labs_ext.rename(columns={"hadm_id_y": "hadm_id"})
    if "hadm_id_x" in out.columns:
        out = out.drop(columns=["hadm_id_x"])
    return out[["version", "subject_id", "hadm_id", "charttime", "valuenum", "in_prior7d", "admittime"]].copy()

# ---------------------------------------------------------------------
# KDIGO detection from a time series of creatinine
# ---------------------------------------------------------------------

def kdigo_from_series(group: pd.DataFrame) -> int:
    """
    KDIGO creatinine-based AKI detection within the provided time series.
    Criteria:
      (A) Absolute increase >= 0.3 mg/dL within 48 hours, OR
      (B) Increase to >= 1.5x baseline within 7 days.

    Input is the subset of labs already restricted to the prior-7d window for one admission.
    Returns 1 if AKI criteria satisfied, else 0.
    """
    if group.empty:
        return 0
    g = group.sort_values("charttime")
    s = g.set_index("charttime")["valuenum"].dropna()
    if s.empty:
        return 0

    # rolling windows (note lowercase 'h' to avoid FutureWarning)
    # For within-48h rise, compare current value to min in prior 48h
    min_48h = s.rolling("48h").min()
    abs_rise = (s - min_48h) >= 0.3

    # For 7-day baseline, use prior-7d min
    min_7d = s.rolling("7d").min()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_rise = (s / min_7d) >= 1.5

    flag = bool(abs_rise.fillna(False).any() or rel_rise.fillna(False).any())
    return int(flag)

# ---------------------------------------------------------------------
# Public helper: annotate target with lab_aki_prior7d
# ---------------------------------------------------------------------

def attach_lab_aki_prior7d_flags(
    target: pd.DataFrame,
    labevents: pd.DataFrame,
    labitems: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns target with a new column 'lab_aki_prior7d' (0/1) indicating KDIGO-based
    AKI in the 7 days prior to the admission time.
    """
    admissions = target[["version", "subject_id", "hadm_id", "admittime"]].drop_duplicates().copy()
    to_datetime(admissions, ["admittime"])

    creat_items = select_creatinine_itemids(labitems)
    labs = prep_labs(labevents, creat_items)
    labs_w = build_window(admissions, labs, days_prior=7)

    prior = labs_w[labs_w["in_prior7d"]].copy()
    grp_cols = ["version", "subject_id", "hadm_id"]

    # pandas 2.2+ supports include_groups=False; keep a fallback for older versions
    try:
        series = prior.groupby(grp_cols, group_keys=False).apply(kdigo_from_series, include_groups=False)
    except TypeError:
        series = prior.groupby(grp_cols, group_keys=False).apply(kdigo_from_series)
    lab_flags = series.rename("kdigo-aki").reset_index()
    lab_flags["kdigo-aki"] = lab_flags["kdigo-aki"].astype(int)

    out = target.merge(lab_flags, on=grp_cols, how="left")
    # out["lab_aki_prior7d"] = out["lab_aki_prior7d"].fillna(0).astype(int)
    return out

# ---------------------------------------------------------------------
# CLI: compute & write artifacts
# ---------------------------------------------------------------------

def load_tables(data_pkl: str) -> dict:
    with open(data_pkl, "rb") as f:
        return pd.read_pickle(f)

def compute_lab_aki_prior7d(data_pkl: str, target_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    data = load_tables(data_pkl)
    admissions = data["admissions"].copy()
    labevents  = data["labevents"].copy()
    labitems   = data["labitems"].copy()

    # normalize admissions columns
    admissions.columns = [c.lower() for c in admissions.columns]
    if "admittime" not in admissions.columns:
        raise ValueError("admittime missing in admissions.")
    to_datetime(admissions, ["admittime"])

    # subset target and attach flag
    target = pd.read_parquet(target_path)
    need = {"version", "subject_id", "hadm_id"}
    if not need.issubset(target.columns):
        miss = sorted(list(need - set(target.columns)))
        raise ValueError(f"Missing keys in target: {miss}")

    target_plus = attach_lab_aki_prior7d_flags(
        target=target,
        labevents=labevents,
        labitems=labitems,
    )

    # Save per-admission flags
    out_flags = os.path.join(outdir, "lab_aki_prior7d.parquet")
    target_plus[["version", "subject_id", "hadm_id", "lab_aki_prior7d"]].to_parquet(out_flags, index=False)

    # False negatives: ICD says no AKI, labs say AKI prior 7d
    if {"is_aki"}.issubset(target_plus.columns):
        fn = target_plus[(target_plus["is_aki"] == 0) & (target_plus["lab_aki_prior7d"] == 1)].copy()
        keep_cols = [c for c in [
            "version", "subject_id", "hadm_id", "admittime", "age", "gender",
            "hospital_expire_flag", "is_aki", "is_ckd", "lab_aki_prior7d"
        ] if c in target_plus.columns]
        fn_out = fn[keep_cols] if keep_cols else fn
        out_fn = os.path.join(outdir, "false_negatives_prior7d.csv")
        fn_out.to_csv(out_fn, index=False)
    else:
        fn = pd.DataFrame()
        out_fn = None

    # Summary
    grp = target_plus.groupby(["version"], dropna=False)
    summary = grp.agg(
        n_adm     = ("hadm_id", "nunique"),
        n_lab_aki = ("lab_aki_prior7d", "sum"),
    ).reset_index()
    summary["n_fn"] = 0
    if not fn.empty:
        n_fn = (fn.groupby("version")["hadm_id"].nunique()).rename("n_fn")
        summary = summary.merge(n_fn, on="version", how="left")
        summary["n_fn"] = summary["n_fn"].fillna(0).astype(int)

    out_summary = os.path.join(outdir, "lab_aki_summary.csv")
    summary.to_csv(out_summary, index=False)

    # Console recap
    prior_counts = 0
    try:
        # recompute "any creat in prior7d" count for display
        creat_items = select_creatinine_itemids(labitems)
        labs = prep_labs(labevents, creat_items)
        labs_w = build_window(admissions, labs, days_prior=7)
        prior_counts = labs_w[labs_w["in_prior7d"]][["version","subject_id","hadm_id"]].drop_duplicates().shape[0]
    except Exception:
        pass

    log("\n=== LAB AKI (prior 7 days) ===")
    log(f"Admissions with any creatinine in prior 7d: {prior_counts:,}")
    log(f"Admissions flagged lab_aki_prior7d: {int(summary['n_lab_aki'].sum()):,}")
    log("\n=== FALSE NEGATIVES (ICD is_aki==0 but lab prior7d==1) ===")
    if out_fn:
        log(f"Count: {fn.shape[0]:,}")
        log(f"Wrote: {out_fn}")
    log(f"Wrote: {out_flags}")
    log(f"Wrote: {out_summary}")

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute lab-based pre-admission AKI (prior 7 days) flags.")
    ap.add_argument("--data-pkl", type=str, default="data.pkl", help="Pickle with data_dict (admissions, labevents, labitems).")
    ap.add_argument("--target",   type=str, default="target.parquet", help="Target parquet to annotate.")
    ap.add_argument("--outdir",   type=str, default=".", help="Output directory.")
    args = ap.parse_args()
    compute_lab_aki_prior7d(args.data_pkl, args.target, args.outdir)

if __name__ == "__main__":
    main()
