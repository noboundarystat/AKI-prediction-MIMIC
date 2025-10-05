#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train XGBoost CoxPH Survival model for time-to-CKD prediction.
Evaluations:
  - Harrell's C-index (train/val/test + external MIMIC-III)
  - AUROC & AUPRC for event-by-horizon classification (default 365d)

Artifacts:
  .joblib, .features.txt, .metrics.json
"""

import argparse, json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_timedelta64_dtype
)

from xgboost import XGBModel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib


# ---------- helpers ----------

def select_feature_columns(df: pd.DataFrame, id_cols, target: str, dataset_col: Optional[str], max_null_rate=0.9):
    drop = set((id_cols or []) + ([target] if target else []) + ([dataset_col] if dataset_col else []))
    keep = []
    n = len(df)
    for c in df.columns:
        if c in drop:
            continue
        s = df[c]
        if is_datetime64_any_dtype(s) or is_timedelta64_dtype(s):
            continue
        if is_numeric_dtype(s) or is_bool_dtype(s):
            null_rate = (s.isna().sum() / float(n)) if n else 1.0
            if null_rate <= max_null_rate:
                keep.append(c)
    return keep


def groups_split_masks(n: int, groups: np.ndarray, test_size: float, val_size: float, seed: int):
    idx = np.arange(n)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(idx, groups=groups))

    rel_val = val_size / (1.0 - test_size + 1e-12)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
    tr_idx_rel, val_idx_rel = next(gss2.split(trainval_idx, groups=groups[trainval_idx]))

    train_idx = trainval_idx[tr_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    train_mask = np.zeros(n, dtype=bool); train_mask[train_idx] = True
    val_mask   = np.zeros(n, dtype=bool); val_mask[val_idx] = True
    test_mask  = np.zeros(n, dtype=bool); test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def concordance_index(event_times, predicted_scores, event_observed):
    """Compute C-index (Harrell’s concordance index)."""
    n = 0
    n_conc = 0.0
    for i in range(len(event_times)):
        for j in range(i + 1, len(event_times)):
            if event_observed[i] == 1 or event_observed[j] == 1:
                t_i, t_j = event_times[i], event_times[j]
                s_i, s_j = predicted_scores[i], predicted_scores[j]
                if t_i != t_j:
                    n += 1
                    if (t_i < t_j and s_i > s_j) or (t_j < t_i and s_j > s_i):
                        n_conc += 1
                    elif s_i == s_j:
                        n_conc += 0.5
    return n_conc / n if n > 0 else np.nan


def horizon_binary_eval(time, event, score, horizon_days: float):
    """
    AUROC/AUPRC for 'event by horizon' classification.
    - Positive = event==1 AND time <= horizon
    - Negative = time > horizon
    - Drop censored before horizon (event==0 AND time <= horizon)
    """
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    score = np.asarray(score, float)

    pos = (event == 1) & (time <= horizon_days)
    neg = (time > horizon_days)
    use = pos | neg
    if use.sum() == 0:
        return {"auroc": np.nan, "auprc": np.nan, "n_used": 0, "pos_rate": np.nan}

    y_true = pos[use].astype(int)
    y_score = score[use]  # higher risk should imply more likely event by horizon
    try:
        auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
        auprc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        auroc, auprc = np.nan, np.nan

    return {"auroc": float(auroc), "auprc": float(auprc), "n_used": int(use.sum()), "pos_rate": float(y_true.mean())}


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="features_ckd.parquet")
    ap.add_argument("--id-cols", nargs="*", default=["subject_id"])
    ap.add_argument("--time-col", default="time_days")
    ap.add_argument("--event-col", default="is_ckd")
    ap.add_argument("--dataset-col", default="version")
    ap.add_argument("--mimic3-value", default="mimic3")
    ap.add_argument("--mimic4-value", default="mimic4")
    ap.add_argument("--mode", choices=["mimic4_only","both"], default="mimic4_only")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    # Horizon eval
    ap.add_argument("--horizon-days", type=float, default=365.0)

    # Feature selection
    ap.add_argument("--max-null-rate", type=float, default=0.9)
    ap.add_argument("--features-list", default=None)
    ap.add_argument("--exclude-cols", nargs="*", default=[
        "version", "subject_id", "event_hadm_id", "index_admit",
        "ckd_event_hadm_id", "postckd_event_hadm_id", "death_event_hadm_id",
        "time_to_ckd", "time_to_postckd", "time_to_death",
        "is_ckd", "is_postckd", "is_death", "is_censored", "time_days"
    ])

    # XGB hyperparams
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)

    # Null handling
    ap.add_argument("--impute", action="store_true")

    ap.add_argument("--out-prefix", default="artifacts/xgb_ckd_survival/xgb_ckd")
    args = ap.parse_args()

    # --- Load data ---
    p = Path(args.input)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df = df.copy()

    time_col = args.time_col
    event_col = args.event_col

    # Feature list
    if args.features_list:
        feature_cols = [x.strip() for x in Path(args.features_list).read_text().splitlines() if x.strip()]
    else:
        scope = df[df[args.dataset_col] == args.mimic4_value] if (args.mode == "mimic4_only" and args.dataset_col in df.columns) else df
        feature_cols = select_feature_columns(scope, id_cols=args.id_cols, target=None,
                                              dataset_col=args.dataset_col, max_null_rate=args.max_null_rate)

    # Auto-exclude leakage
    auto_exclude = {"time_days", "cmb_ckd"}
    exclude = set(args.exclude_cols) | auto_exclude
    feature_cols = [c for c in feature_cols if c not in exclude]

    # Split dataset (train/val/test on MIMIC-IV)
    dsplit = df[df[args.dataset_col] == args.mimic4_value].copy() if args.mode == "mimic4_only" else df.copy()
    X = dsplit[feature_cols].copy()
    y_time = dsplit[time_col].astype(float).values
    y_event = dsplit[event_col].astype(int).values
    groups = dsplit[args.id_cols[0]].values

    train_m, val_m, test_m = groups_split_masks(len(dsplit), groups, args.test_size, args.val_size, args.seed)

    # Optional imputation
    imputer = None
    if args.impute:
        imputer = SimpleImputer(strategy="median")
        X.iloc[train_m, :] = imputer.fit_transform(X.loc[train_m])
        X.iloc[val_m,   :] = imputer.transform(X.loc[val_m])
        X.iloc[test_m,  :] = imputer.transform(X.loc[test_m])

    # --- Train Cox survival model ---
    clf = XGBModel(
        objective="survival:cox",
        eval_metric="cox-nloglik",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X[train_m], y_time[train_m], sample_weight=y_event[train_m])

    # --- Eval helpers ---
    def eval_cindex(Xp, time, event, mask=None):
        if mask is not None:
            Xp, time, event = Xp[mask], time[mask], event[mask]
        scores = clf.predict(Xp)
        return {"cindex": float(concordance_index(time, scores, event)), "n": int(len(time))}, scores

    def eval_all(name, Xp, time, event, mask=None):
        ci, scores = eval_cindex(Xp, time, event, mask)
        he = horizon_binary_eval(
            time[mask] if mask is not None else time,
            event[mask] if mask is not None else event,
            scores,
            args.horizon_days
        )
        out = {"cindex": ci["cindex"], "n": ci["n"], "horizon_days": args.horizon_days,
               "auroc_at_horizon": he["auroc"], "auprc_at_horizon": he["auprc"],
               "horizon_eval_n": he["n_used"], "horizon_pos_rate": he["pos_rate"]}
        return out

    results: Dict[str, dict] = {
        "train": eval_all("train", X, y_time, y_event, train_m),
        "val":   eval_all("val",   X, y_time, y_event, val_m),
        "test":  eval_all("test",  X, y_time, y_event, test_m),
    }

    # --- External evaluation on MIMIC-III ---
    if args.dataset_col in df.columns:
        df_m3 = df[df[args.dataset_col] == args.mimic3_value].copy()
        if len(df_m3):
            Xm3 = df_m3[feature_cols].copy()
            ym3_time = df_m3[time_col].astype(float).values
            ym3_event = df_m3[event_col].astype(int).values
            if args.impute and imputer is not None:
                Xm3 = pd.DataFrame(imputer.transform(Xm3), columns=feature_cols, index=Xm3.index)
            results["mimic3_full"] = eval_all("mimic3_full", Xm3, ym3_time, ym3_event, mask=None)

    # Outputs
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": clf, "imputer": imputer, "features": feature_cols}, f"{args.out_prefix}.joblib")
    Path(f"{args.out_prefix}.features.txt").write_text("\n".join(feature_cols))
    Path(f"{args.out_prefix}.metrics.json").write_text(json.dumps(results, indent=2))

    print("XGBoost Cox survival results:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
