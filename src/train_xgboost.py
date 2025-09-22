#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train XGBoost Classifier with group-aware splits and optional val-based hyperparameter tuning.
Artifacts: .joblib, .features.txt, .splits.csv, .meta.json, .predictions.csv,
.metrics.json, .summary.txt, .feature_importance.csv, .hstats_gender.csv
"""

import argparse, json, itertools
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from pandas.api.types import (
    is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_timedelta64_dtype
)

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
import joblib

# ------------------------
# H-statistics (optional)
# ------------------------
from sklearn.inspection import partial_dependence

def h_statistic_tree(model, X, f1, f2):
    """Compute Friedman's H-statistic using partial dependence (tree-based models)."""
    # Use feature names directly instead of indices
    pd_joint = partial_dependence(model, X, [(f1, f2)], kind="average")["average"][0].ravel()
    pd_f1    = partial_dependence(model, X, [f1], kind="average")["average"][0].ravel()
    pd_f2    = partial_dependence(model, X, [f2], kind="average")["average"][0].ravel()

    if hasattr(model, "predict_proba"):
        f_x = model.predict_proba(X)[:, 1]
    else:
        f_x = model.predict(X)

    numerator = np.var(f_x - pd_f1 - pd_f2)
    denominator = np.var(f_x)
    return np.sqrt(numerator / denominator) if denominator > 0 else 0.0


def compute_hstats_with_anchor(model, X, anchor="dem_sex_F", max_rows=2000):
    """Compute H-statistic of anchor vs all other features."""
    if anchor not in X.columns:
        print(f"⚠️ Anchor feature {anchor} not in dataframe, skipping H-stats.")
        return None

    if len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=42)

    results = []
    for f in X.columns:
        if f == anchor:
            continue
        try:
            h = h_statistic_tree(model, X, anchor, f)
            results.append({"feature": f, "H": h})
        except Exception as e:
            print(f"⚠️ Skipping pair ({anchor}, {f}): {e}")
    return pd.DataFrame(results).sort_values("H", ascending=False).reset_index(drop=True)


# ---------- helpers ----------
def choose_threshold(y_true, y_score, objective="specificity", target=0.90):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if objective == "fixed":
        return float(target)

    fpr, tpr, thr = roc_curve(y_true, y_score)
    specificity = 1.0 - fpr

    if objective == "specificity":
        ok = np.where(specificity >= target)[0]
        if len(ok) == 0: return 0.5
        best = ok[np.argmax(tpr[ok])]
        return float(thr[best])

    if objective == "sensitivity":
        ok = np.where(tpr >= target)[0]
        if len(ok) == 0: return 0.5
        best = ok[np.argmax(specificity[ok])]
        return float(thr[best])

    if objective == "youden":
        j = tpr - fpr
        return float(thr[np.argmax(j)])

    # PR-based
    prec, rec, thr_pr = precision_recall_curve(y_true, y_score)
    if len(thr_pr) == 0: return 0.5
    if objective == "f1":
        f1 = 2 * prec[1:] * rec[1:] / (prec[1:] + rec[1:] + 1e-12)
        return float(thr_pr[np.nanargmax(f1)])
    if objective == "precision_at_recall":
        ok = np.where(rec[1:] >= target)[0]
        if len(ok) == 0: return 0.5
        best = ok[np.nanargmax(prec[1:][ok])]
        return float(thr_pr[best])

    return 0.5


def select_feature_columns(df: pd.DataFrame, id_cols, target: str, dataset_col: Optional[str], max_null_rate=0.9):
    drop = set((id_cols or []) + ([target] if target else []) + ([dataset_col] if dataset_col else []))
    keep = []
    n = len(df)
    for c in df.columns:
        if c in drop: continue
        s = df[c]
        if is_datetime64_any_dtype(s) or is_timedelta64_dtype(s): continue
        if is_numeric_dtype(s) or is_bool_dtype(s):
            null_rate = (s.isna().sum() / float(n)) if n else 1.0
            if null_rate <= max_null_rate:
                keep.append(c)
    return keep


def groups_split_masks(n: int, y: np.ndarray, groups: np.ndarray, test_size: float, val_size: float, seed: int):
    idx = np.arange(n)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(idx, y, groups=groups))

    rel_val = val_size / (1.0 - test_size + 1e-12)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
    tr_idx_rel, val_idx_rel = next(gss2.split(trainval_idx, y[trainval_idx], groups=groups[trainval_idx]))
    train_idx = trainval_idx[tr_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    train_mask = np.zeros(n, dtype=bool); train_mask[train_idx] = True
    val_mask   = np.zeros(n, dtype=bool); val_mask[val_idx] = True
    test_mask  = np.zeros(n, dtype=bool); test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def prevalence(y):
    s = pd.Series(y).dropna().astype(int)
    return float((s == 1).mean())


def eval_split(y_true, y_score, threshold: float):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    auprc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan

    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    precision = tp / (tp + fp + 1e-12)

    pr_prec, pr_rec, pr_thr = precision_recall_curve(y_true, y_score)
    roc_fpr, roc_tpr, roc_thr = roc_curve(y_true, y_score)

    cls_report_text = classification_report(y_true, y_pred, digits=4, zero_division=0)
    cls_report_dict = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)

    return {
        "prevalence": prevalence(y_true),
        "auroc": auroc,
        "auprc": auprc,
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": {"labels": [0, 1], "matrix": cm.tolist()},
        "classification_report": cls_report_dict,
        "classification_report_text": cls_report_text,
        "pr_curve": {
            "precision": pr_prec.tolist(),
            "recall": pr_rec.tolist(),
            "thresholds": pr_thr.tolist() if len(pr_thr) > 0 else [],
        },
        "roc_curve": {"fpr": roc_fpr.tolist(), "tpr": roc_tpr.tolist(), "thresholds": roc_thr.tolist()},
    }


def small_param_grid() -> List[dict]:
    grid = {
        "learning_rate":[0.05,0.1],
        "max_depth":[4,6],
        "subsample":[0.8,1.0],
    }
    keys = sorted(grid.keys())
    combos = []
    for values in itertools.product(*[grid[k] for k in keys]):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="features/features_all.parquet")
    ap.add_argument("--id-cols", nargs="*", default=["subject_id","hadm_id"])
    ap.add_argument("--target", default="incident_aki_label")
    ap.add_argument("--dataset-col", default="version")
    ap.add_argument("--mimic3-value", default="mimic3")
    ap.add_argument("--mimic4-value", default="mimic4")
    ap.add_argument("--mode", choices=["mimic4_only","both"], default="mimic4_only")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-null-rate", type=float, default=0.9)
    ap.add_argument("--features-list", default=None)
    ap.add_argument("--exclude-cols", nargs="*", default=['version','subject_id','hadm_id','admittime',
        'target_meta__age','gender','hospital_expire_flag','death_flag',
        'time_to_death_from_adm_hr','time_to_death_from_disch_hr','cmb_ckd',
        'is_aki','is_ckd','ckd_only_flag','ckd_admission_flag','is_autocar',
        'kdigo-aki','renal_impaired_at_adm','aki_history_flag','onset_aki_flag'])
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--colsample-bytree", type=float, default=1.0)
    ap.add_argument("--scale-pos-weight", type=float, default=1.0)
    ap.add_argument("--impute", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tune-metric", choices=["auprc","auroc"], default="auprc")
    ap.add_argument("--threshold-objective", default="specificity",
        choices=["specificity","sensitivity","youden","f1","precision_at_recall","fixed"])
    ap.add_argument("--threshold-target", type=float, default=0.90)
    ap.add_argument("--out-prefix", default="artifacts/xgboost_model/xgb")
    args = ap.parse_args()

    # --- Load data ---
    p = Path(args.input)
    df = pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)
    df = df[~((df['cmb_ckd']==1)|(df['renal_impaired_at_adm']==1))]

    # Target prep
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found.")
    if is_bool_dtype(df[args.target]):
        df[args.target] = df[args.target].astype(int)
    elif is_numeric_dtype(df[args.target]):
        df[args.target] = (df[args.target].astype(float) > 0).astype(int)
    else:
        m = {"1":1,"0":0,"true":1,"false":0,"t":1,"f":0,"yes":1,"no":0,"y":1,"n":0}
        df[args.target] = df[args.target].astype(str).str.lower().map(m).fillna(0).astype(int)

    # Feature list
    if args.features_list:
        feature_cols = [x.strip() for x in Path(args.features_list).read_text().splitlines() if x.strip()]
    else:
        scope = df[df[args.dataset_col]==args.mimic4_value] if (args.mode=="mimic4_only" and args.dataset_col in df.columns) else df
        feature_cols = select_feature_columns(scope, id_cols=args.id_cols, target=args.target,
                                              dataset_col=args.dataset_col, max_null_rate=args.max_null_rate)

    exclude = set(args.exclude_cols)
    feature_cols = [c for c in feature_cols if c not in exclude and not c.startswith("emb_")]

    # Split
    if args.mode == "mimic4_only":
        dsplit = df[df[args.dataset_col] == args.mimic4_value].copy()
    else:
        dsplit = df.copy()
    dsplit = dsplit[~dsplit[args.target].isna()].copy()

    X = dsplit[feature_cols].copy()
    y = dsplit[args.target].astype(int).values
    groups = dsplit[args.id_cols[0]].values

    train_m, val_m, test_m = groups_split_masks(len(dsplit), y, groups, args.test_size, args.val_size, args.seed)

    # Imputation
    imputer = None
    if args.impute:
        imputer = SimpleImputer(strategy="median")
        X.iloc[train_m, :] = imputer.fit_transform(X.loc[train_m])
        X.iloc[val_m,   :] = imputer.transform(X.loc[val_m])
        X.iloc[test_m,  :] = imputer.transform(X.loc[test_m])

    # --- Train XGB ---
    tuned_params = {
        "n_estimators": args.n_estimators,
        "max_depth":    args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "scale_pos_weight": args.scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": args.seed,
        "n_jobs": -1,
    }

    clf = XGBClassifier(**tuned_params)
    clf.fit(X.loc[train_m], y[train_m])
    val_scores = clf.predict_proba(X.loc[val_m])[:, 1]

    thr = 0.5

    # --- H-stats ---
    try:
        h_df = compute_hstats_with_anchor(clf, X, anchor="dem_sex_F")
        h_path = Path(f"{args.out_prefix}.hstats_gender.csv")
        h_df.to_csv(h_path, index=False)
        print(f"✓ H-statistics saved to {h_path} ({len(h_df)} rows)")
    except Exception as e:
        print(f"⚠️ H-stats computation skipped: {e}")

    print("Wrote XGBoost artifacts to", args.out_prefix)


if __name__ == "__main__":
    main()
