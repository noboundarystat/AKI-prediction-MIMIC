#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Logistic Regression for incident AKI (MIMIC-IV) with tuning and MIMIC-III external validation.

Pipeline (features_all_imputed.parquet expected):
  1) Filter to MIMIC-IV for train/val/test; optionally keep both for feature selection alignment.
  2) Drop non-feature cols, RRT/CRRT, embeddings, *_missing, *_min/_max/_std/_last, and a small correlated set.
  3) Train/Val/Test split (stratified).
  4) Hyperparameter tuning on Val (AUROC/AUPRC).
  5) Final fit on Train (best hyperparams), evaluate on Test.
  6) External validation on full MIMIC-III (same feature set, aligned).
Artifacts:
  - <out>.joblib                  (model + scaler + feature list)
  - <out>.features.txt           (feature names)
  - <out>.coefficients.csv       (LR coefficients)
  - <out>.predictions.csv        (train/val/test + mimic3_full probs)
  - <out>.metrics.json           (all splits)
  - <out>.summary.txt            (human-readable summary)
"""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)

import joblib


# ----------------------- helpers -----------------------

def evaluate_split(y_true, y_score, thr: float = 0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    prec = tp / (tp + fp + 1e-12)

    report_txt = classification_report(y_true, y_pred, digits=4, zero_division=0)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)

    pr_prec, pr_rec, pr_thr = precision_recall_curve(y_true, y_score)
    roc_fpr, roc_tpr, roc_thr = roc_curve(y_true, y_score)

    return {
        "prevalence": float(np.mean(y_true == 1)),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "threshold": float(thr),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": {"labels": [0, 1], "matrix": cm.tolist()},
        "classification_report": report,
        "classification_report_text": report_txt,
        "pr_curve": {
            "precision": pr_prec.tolist(),
            "recall": pr_rec.tolist(),
            "thresholds": pr_thr.tolist() if len(pr_thr) else [],
        },
        "roc_curve": {"fpr": roc_fpr.tolist(), "tpr": roc_tpr.tolist(), "thresholds": roc_thr.tolist()},
    }


def small_param_grid():
    # Keep it tiny/fast; liblinear supports l1/l2.
    Cs = [0.25, 0.5, 1.0, 2.0]
    penalties = ["l1", "l2"]
    return [{"C": c, "penalty": p} for c in Cs for p in penalties]


def align_columns(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    # Add any missing columns as zeros; reorder to feature_cols
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    return df[feature_cols].copy()


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="features/features_all_imputed.parquet")
    ap.add_argument("--target", default="incident_aki_label")
    ap.add_argument("--dataset-col", default="version")
    ap.add_argument("--mimic3-value", default="mimic3")
    ap.add_argument("--mimic4-value", default="mimic4")
    ap.add_argument("--mode", choices=["mimic4_only", "both"], default="mimic4_only")
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--val-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=20250831)

    # tuning
    ap.add_argument("--tune", action="store_true", help="Run small val-based hyperparameter search.")
    ap.add_argument("--tune-metric", choices=["auprc", "auroc"], default="auprc")

    # thresholding (we’ll keep 0.5 default; add options for completeness)
    ap.add_argument("--threshold", type=float, default=0.5)

    # extra column drops on top of baked-in lists
    ap.add_argument("--exclude-cols", nargs="*", default=[], help="Extra columns to drop if present.")

    ap.add_argument("--out-prefix", default="artifacts/logreg_model/logreg")
    args = ap.parse_args()

    p = Path(args.input)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df = df[~((df['cmb_ckd']==1)|(df['renal_impaired_at_adm']==1))]

    # Basic assertions
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.input}")

    # ----- Preprocessing (matches your notebook/script) -----
    rrt_cols = [x for x in df.columns if "rrt" in x]  # history, context, etc.

    non_feature_cols = [
        "version", "subject_id", "hadm_id", "admittime", "target_meta__age",
        "gender", "hospital_expire_flag", "death_flag",
        "time_to_death_from_adm_hr", "time_to_death_from_disch_hr",
        "cmb_ckd", "is_aki", "is_ckd", "ckd_only_flag", "ckd_admission_flag",
        "is_autocar", "kdigo-aki", "renal_impaired_at_adm",
        "aki_history_flag", "onset_aki_flag"
    ]

    null_feature_cols = [
        "vital48h_minute_ventilation_min", "vital48h_minute_ventilation_max",
        "vital48h_minute_ventilation_mean", "vital48h_minute_ventilation_std",
        "vital48h_minute_ventilation_last", "vital48h_minute_ventilation_count"
    ]

    # Work off MIMIC-IV for train/val/test
    if args.dataset_col not in df.columns:
        raise ValueError(f"Column '{args.dataset_col}' not found for dataset split control.")
    df4 = df[df[args.dataset_col] == args.mimic4_value].copy()

    # Drop fixed sets
    drop_now = set(non_feature_cols + null_feature_cols + rrt_cols + args.exclude_cols)
    drop_now = [c for c in drop_now if c in df4.columns]
    df4.drop(columns=drop_now, inplace=True, errors="ignore")

    # embeddings + *_missing/_min/_max/_std/_last
    emb_cols = [x for x in df4.columns if "emb_" in x]
    missing_cols = [x for x in df4.columns if "_missing" in x]
    min_cols = [x for x in df4.columns if x.endswith("_min")]
    max_cols = [x for x in df4.columns if x.endswith("_max")]
    std_cols = [x for x in df4.columns if x.endswith("_std")]
    last_cols = [x for x in df4.columns if x.endswith("_last")]

    # Small correlated set from your example
    drop_more = [
        "labs7d_creatinine_std", "labs7d_bun_std", "labs7d_albumin_std", "labs7d_potassium_std",
        "vital48h_heart_rate_std", "vital48h_tidal_volume_std",
        "vital48h_dbp_value_mean", "fluid6h_net_rate_ml_per_hr",
        "hx31to180d_aminoglycosides_any"
    ]

    to_drop = [c for c in set(emb_cols + missing_cols + min_cols + max_cols + std_cols + last_cols + drop_more) if c in df4.columns]
    df4.drop(columns=to_drop, inplace=True, errors="ignore")

    # Ensure numeric target
    df4[args.target] = (df4[args.target].astype(float) > 0).astype(int)

    # Split: train/val/test (stratified)
    df_trainval, df_test = train_test_split(
        df4.fillna(0.0),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df4[args.target]
    )
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df_trainval[args.target]
    )

    X_train = df_train.drop(columns=[args.target])
    y_train = df_train[args.target].astype(int).values
    X_val = df_val.drop(columns=[args.target])
    y_val = df_val[args.target].astype(int).values
    X_test = df_test.drop(columns=[args.target])
    y_test = df_test[args.target].astype(int).values

    # Scale (liblinear doesn’t require scaling, but helpful for stability; use with_mean=False to avoid densifying sparse)
    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ----- Tuning -----
    best = None
    grid = small_param_grid() if args.tune else [{"C": 1.0, "penalty": "l1"}]

    for cand in grid:
        # liblinear supports l1/l2; class_weight balanced to handle imbalance
        clf = LogisticRegression(
            solver="liblinear",
            penalty=cand["penalty"],
            C=cand["C"],
            max_iter=1000,
            class_weight="balanced"
        )
        clf.fit(X_train_s, y_train)
        val_proba = clf.predict_proba(X_val_s)[:, 1]
        auprc = average_precision_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else -np.inf
        auroc = roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else -np.inf
        crit = auprc if args.tune_metric == "auprc" else auroc
        if (best is None) or (crit > best["crit"]):
            best = {"crit": crit, "clf": clf, "params": cand, "val_proba": val_proba}

    model = best["clf"]
    tuned_params = best["params"]

    # Final evals
    thr = float(args.threshold)

    train_proba = model.predict_proba(X_train_s)[:, 1]
    val_proba = best["val_proba"]
    test_proba = model.predict_proba(X_test_s)[:, 1]

    train_metrics = evaluate_split(y_train, train_proba, thr)
    val_metrics = evaluate_split(y_val, val_proba, thr)
    test_metrics = evaluate_split(y_test, test_proba, thr)

    # ----- External MIMIC-III -----
    results = {
        "mode": args.mode,
        "model": {"type": "LogisticRegression(liblinear)", "params": tuned_params, "class_weight": "balanced"},
        "threshold": thr,
        "tuning": {"enabled": bool(args.tune), "metric": args.tune_metric, "crit": float(best["crit"])},
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }

    preds_frames = []
    for split_name, part_df, probs in [
        ("train", df_train, train_proba),
        ("val", df_val, val_proba),
        ("test", df_test, test_proba),
    ]:
        out = part_df[[c for c in ["subject_id", "hadm_id"] if c in part_df.columns]].copy()
        out[args.target] = part_df[args.target].values
        out["split"] = split_name
        out["prob"] = probs
        preds_frames.append(out)

    if args.mode == "mimic4_only" and args.dataset_col in df.columns:
        df3 = df[df[args.dataset_col] == args.mimic3_value].copy()
        if len(df3):
            # replicate dropping logic for df3, then align to training feature set
            drop_3 = set(non_feature_cols + null_feature_cols + rrt_cols + args.exclude_cols)
            drop_3 = [c for c in drop_3 if c in df3.columns]
            df3.drop(columns=drop_3, inplace=True, errors="ignore")

            emb3 = [x for x in df3.columns if "emb_" in x]
            miss3 = [x for x in df3.columns if "_missing" in x]
            min3 = [x for x in df3.columns if x.endswith("_min")]
            max3 = [x for x in df3.columns if x.endswith("_max")]
            std3 = [x for x in df3.columns if x.endswith("_std")]
            last3 = [x for x in df3.columns if x.endswith("_last")]
            to_drop3 = [c for c in set(emb3 + miss3 + min3 + max3 + std3 + last3 + drop_more) if c in df3.columns]
            df3.drop(columns=to_drop3, inplace=True, errors="ignore")

            # ensure numeric target and no nans in features
            df3[args.target] = (df3[args.target].astype(float) > 0).astype(int)

            X3 = align_columns(df3.drop(columns=[args.target]), list(X_train.columns))
            X3_s = scaler.transform(X3.fillna(0.0))
            y3 = df3[args.target].astype(int).values
            prob3 = model.predict_proba(X3_s)[:, 1]
            m3_metrics = evaluate_split(y3, prob3, thr)
            results["mimic3_full"] = m3_metrics

            out3 = df3[[c for c in ["subject_id", "hadm_id"] if c in df3.columns]].copy()
            out3[args.target] = y3
            out3["split"] = "mimic3_full"
            out3["prob"] = prob3
            preds_frames.append(out3)

    # Coefficients
    coef = pd.Series(model.coef_[0], index=X_train.columns).sort_values(ascending=False)
    coef_df = coef.reset_index(); coef_df.columns = ["feature", "coefficient"]

    # ----- Outputs -----
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "scaler": scaler, "features": list(X_train.columns)}, f"{out_prefix}.joblib")
    Path(f"{out_prefix}.features.txt").write_text("\n".join(X_train.columns))
    coef_df.to_csv(f"{out_prefix}.coefficients.csv", index=False)

    preds_df = pd.concat(preds_frames, axis=0, ignore_index=True)
    preds_df.to_csv(f"{out_prefix}.predictions.csv", index=False)

    Path(f"{out_prefix}.metrics.json").write_text(json.dumps(results, indent=2))

    # summary.txt
    lines = []
    lines.append(f"# Logistic Regression (liblinear) — tuned={bool(args.tune)} metric={args.tune_metric}")
    lines.append(f"- Features: {len(X_train.columns)}")
    lines.append(f"- Tuned params: {tuned_params}")
    lines.append(f"- Threshold: {thr:.4f}")

    def add_block(name, m):
        lines.append(f"\n## {name.upper()}")
        lines.append(f"- AUROC: {m['auroc']:.6f} | AUPRC: {m['auprc']:.6f} | Prev: {m['prevalence']:.6f}")
        lines.append(f"- Sensitivity: {m['sensitivity']:.6f} | Specificity: {m['specificity']:.6f} | Precision: {m['precision']:.6f}")
        lines.append(f"- Confusion (rows=actual 0/1, cols=pred 0/1): {m['confusion_matrix']['matrix']}")
        lines.append("```\n" + m["classification_report_text"] + "\n```")

    for key in ["train", "val", "test"]:
        add_block(key, results[key])
    if "mimic3_full" in results:
        add_block("MIMIC-III (external)", results["mimic3_full"])

    Path(f"{out_prefix}.summary.txt").write_text("\n".join(lines))

    print(f"✓ Done. Artifacts at {out_prefix}.*")
    print(f"Test AUROC={results['test']['auroc']:.4f}, AUPRC={results['test']['auprc']:.4f}")
    if "mimic3_full" in results:
        print(f"External (MIMIC-III) AUROC={results['mimic3_full']['auroc']:.4f}, AUPRC={results['mimic3_full']['auprc']:.4f}")


if __name__ == "__main__":
    main()
