# src/train_rf.py
# Train RandomForestClassifier with group-aware splits and optional val-based hyperparameter tuning.
# Artifacts: .joblib, .features.txt, .splits.csv, .meta.json, .predictions.csv, .metrics.json, .summary.txt, .feature_importance.csv

import argparse, json, itertools
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from pandas.api.types import (
    is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_timedelta64_dtype
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
import joblib


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
    """Tiny grid to keep tuning quick."""
    grid = {
        "n_estimators":      [100, 200],
        "max_depth":         [None, 12],
        "min_samples_leaf":  [1, 5],
        "max_features":      ["sqrt", "log2"],
        "bootstrap":         [True],
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

    # Feature selection & row filtering
    ap.add_argument("--max-null-rate", type=float, default=0.9)
    ap.add_argument("--features-list", default=None)
    ap.add_argument("--exclude-cols", nargs="*", default=['version', 'subject_id', 'hadm_id', 'admittime', 'target_meta__age',
'gender',
'hospital_expire_flag',
'death_flag',
'time_to_death_from_adm_hr',
'time_to_death_from_disch_hr',
'cmb_ckd',
'is_aki',
'is_ckd',
'ckd_only_flag',
'ckd_admission_flag',
'is_autocar',
'kdigo-aki',
'renal_impaired_at_adm',
'aki_history_flag',
'onset_aki_flag',
'hx30d_crrt_context_any',
 'hx31to180d_crrt_context_any',
 'hx30d_rrt_any',
 'hx30d_crrt_any',
 'hx31to180d_rrt_any',
 'hx31to180d_crrt_any'])
    ap.add_argument("--row-filter", default=None)

    # RF hyperparams (defaults used if tuning disabled)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=lambda x: None if str(x).lower()=="none" else int(x), default=None)
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    ap.add_argument("--max-features", default="sqrt")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--class-weight", default="balanced", choices=["balanced","none"])

    # Null handling (RF in sklearn doesn’t accept NaN reliably across versions)
    ap.add_argument("--impute", action="store_true", help="Median-impute features before training (recommended for RF).")

    # Tuning
    ap.add_argument("--tune", action="store_true", help="Run a small val-based grid search.")
    ap.add_argument("--tune-metric", choices=["auprc","auroc"], default="auprc")

    # Thresholding
    ap.add_argument("--threshold-objective", default="specificity",
                    choices=["specificity","sensitivity","youden","f1","precision_at_recall","fixed"])
    ap.add_argument("--threshold-target", type=float, default=0.90)

    ap.add_argument("--out-prefix", default="artifacts/rf_model/rf")
    args = ap.parse_args()

    # --- Load data ---
    p = Path(args.input)
    df = pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)
    df = df[~((df['cmb_ckd']==1)|(df['renal_impaired_at_adm']==1))]
    
    if args.row_filter:
        df = df.query(args.row_filter).copy()

    # Target -> 0/1
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

    # Force-exclude
    exclude = set(args.exclude_cols)
    feature_cols = [c for c in feature_cols if c not in exclude]

    # Split dataset for training
    if args.mode == "mimic4_only":
        if args.dataset_col not in df.columns:
            raise ValueError("mimic4_only mode requires --dataset-col present in the data.")
        dsplit = df[df[args.dataset_col] == args.mimic4_value].copy()
    else:
        dsplit = df.copy()
    dsplit = dsplit[~dsplit[args.target].isna()].copy()

    X = dsplit[feature_cols].copy()
    y = dsplit[args.target].astype(int).values
    groups = dsplit[args.id_cols[0]].values

    train_m, val_m, test_m = groups_split_masks(len(dsplit), y, groups, args.test_size, args.val_size, args.seed)

    # Optional imputation (recommended for RF)
    imputer = None
    if args.impute:
        imputer = SimpleImputer(strategy="median")
        X.iloc[train_m, :] = imputer.fit_transform(X.loc[train_m])
        X.iloc[val_m,   :] = imputer.transform(X.loc[val_m])
        X.iloc[test_m,  :] = imputer.transform(X.loc[test_m])

    # --- Optional val-based tuning ---
    tuned_params = {
        "n_estimators": args.n_estimators,
        "max_depth":    args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "bootstrap":    args.bootstrap or True,   # default True in grid
    }

    if args.tune:
        best = None
        for cand in small_param_grid():
            clf = RandomForestClassifier(
                n_estimators=cand["n_estimators"],
                max_depth=cand["max_depth"],
                min_samples_leaf=cand["min_samples_leaf"],
                max_features=cand["max_features"],
                bootstrap=cand["bootstrap"],
                class_weight=(None if args.class_weight=="none" else args.class_weight),
                random_state=args.seed,
                n_jobs=-1
            )
            clf.fit(X[train_m], y[train_m])
            val_scores = clf.predict_proba(X[val_m])[:, 1]
            auroc = roc_auc_score(y[val_m], val_scores) if len(np.unique(y[val_m]))>1 else np.nan
            auprc = average_precision_score(y[val_m], val_scores) if len(np.unique(y[val_m]))>1 else np.nan
            crit = auprc if args.tune_metric=="auprc" else auroc
            if best is None or (crit > best["crit"]):
                best = {"crit":crit, "params":cand, "val_scores":val_scores, "clf":clf}
        tuned_params.update(best["params"])
        # reuse tuned model for next steps
        clf = best["clf"]
        val_scores = best["val_scores"]
    else:
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            bootstrap=args.bootstrap or True,
            class_weight=(None if args.class_weight=="none" else args.class_weight),
            random_state=args.seed,
            n_jobs=-1
        )
        clf.fit(X[train_m], y[train_m])
        val_scores = clf.predict_proba(X[val_m])[:, 1]

    # Threshold from VAL
    # thr = choose_threshold(y[val_m], val_scores, objective=args.threshold_objective, target=args.threshold_target)
    thr = 0.5
    
    # Eval helper
    def add_eval(mask=None, part_df=None, part_X=None):
        if mask is not None:
            Xp = X[mask]; yp = y[mask]
        else:
            Xp = part_X; yp = part_df[args.target].astype(int).values
        scores = clf.predict_proba(Xp)[:, 1]
        metrics = eval_split(yp, scores, threshold=thr)
        return metrics, scores

    # Collect metrics & predictions
    results: Dict[str, dict] = {
        "mode": args.mode,
        "split_prevalence": {
            "train_prev": prevalence(y[train_m]),
            "val_prev":   prevalence(y[val_m]),
            "test_prev":  prevalence(y[test_m]),
        },
        "thresholding": {"objective": args.threshold_objective, "target": args.threshold_target},
        "model": {
            "type": "RandomForestClassifier",
            "params": tuned_params,
            "class_weight": args.class_weight,
            "seed": args.seed,
            "impute": bool(args.impute),
        },
        "tuning": {
            "enabled": bool(args.tune),
            "metric": args.tune_metric
        }
    }

    preds_frames = []
    for name, mask in [("train", train_m), ("val", val_m), ("test", test_m)]:
        m, scores = add_eval(mask=mask)
        results[name] = m
        out = dsplit.loc[mask, args.id_cols + [args.target]].copy()
        out["split"] = name; out["prob"] = scores
        preds_frames.append(out)

    # External: full MIMIC-III (only for mimic4_only)
    if args.mode == "mimic4_only" and args.dataset_col in df.columns:
        df_m3 = df[df[args.dataset_col] == args.mimic3_value].copy()
        df_m3 = df_m3[~df_m3[args.target].isna()]
        if len(df_m3):
            Xm3 = df_m3[feature_cols].copy()
            if args.impute and imputer is not None:
                Xm3 = pd.DataFrame(imputer.transform(Xm3), columns=feature_cols, index=Xm3.index)
            m3_metrics, scores = add_eval(mask=None, part_df=df_m3, part_X=Xm3)
            results["mimic3_full"] = m3_metrics
            out = df_m3[args.id_cols + [args.target]].copy()
            out["split"] = "mimic3_full"; out["prob"] = scores
            preds_frames.append(out)

    # Feature importance
    fi = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fi_df = fi.reset_index(); fi_df.columns = ["feature", "importance"]

    # Outputs
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Save model + (optional) imputer
    joblib.dump({"model": clf, "imputer": imputer, "features": feature_cols}, f"{args.out_prefix}.joblib")

    # Splits keyed by IDs
    split_arr = np.where(train_m, "train", np.where(val_m, "val", "test"))
    split_df = dsplit.loc[:, args.id_cols].copy()
    split_df["split"] = split_arr
    split_df.to_csv(f"{args.out_prefix}.splits.csv", index=False)

    # Features
    Path(f"{args.out_prefix}.features.txt").write_text("\n".join(feature_cols))

    # Meta
    meta = {
        "input": str(p),
        "id_cols": args.id_cols,
        "target": args.target,
        "dataset_col": args.dataset_col,
        "mimic3_value": args.mimic3_value,
        "mimic4_value": args.mimic4_value,
        "mode": args.mode,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "seed": args.seed,
        "feature_count": len(feature_cols),
        "features_path": f"{args.out_prefix}.features.txt",
        "threshold_objective": args.threshold_objective,
        "threshold_target": args.threshold_target,
        "chosen_threshold": float(results["val"]["threshold"]),
        "model": results["model"],
        "excluded_columns": sorted(list(exclude)),
        "class_weight_impl": args.class_weight,
        "tuning": results["tuning"],
    }
    Path(f"{args.out_prefix}.meta.json").write_text(json.dumps(meta, indent=2))

    # Predictions & metrics
    preds_df = pd.concat(preds_frames, axis=0, ignore_index=True)
    preds_df.to_csv(f"{args.out_prefix}.predictions.csv", index=False)
    Path(f"{args.out_prefix}.metrics.json").write_text(json.dumps(results, indent=2))
    fi_df.to_csv(f"{args.out_prefix}.feature_importance.csv", index=False)

    # Summary
    lines = []
    lines.append(f"# RandomForest Results ({args.mode})")
    lines.append(f"- Features: {len(feature_cols)}")
    lines.append(f"- Train prevalence: {results['split_prevalence']['train_prev']:.6f}")
    lines.append(f"- Val prevalence:   {results['split_prevalence']['val_prev']:.6f}")
    lines.append(f"- Test prevalence:  {results['split_prevalence']['test_prev']:.6f}")
    lines.append(f"- Tuning: {results['tuning']}")
    lines.append(f"- Threshold ({args.threshold_objective} target={args.threshold_target}): {results['val']['threshold']:.4f}")

    def add_block(name, m):
        lines.append(f"\n## {name.upper()}")
        lines.append(f"- AUROC: {m['auroc']:.6f} | AUPRC: {m['auprc']:.6f}")
        lines.append(f"- Sensitivity: {m['sensitivity']:.6f} | Specificity: {m['specificity']:.6f} | Precision: {m['precision']:.6f}")
        lines.append(f"- Confusion matrix (rows=actual 0/1, cols=pred 0/1): {m['confusion_matrix']['matrix']}")
        lines.append("```\n" + m["classification_report_text"] + "\n```")

    for key in ["train", "val", "test"]:
        add_block(key, results[key])
    if "mimic3_full" in results: add_block("MIMIC-III (full external)", results["mimic3_full"])

    Path(f"{args.out_prefix}.summary.txt").write_text("\n".join(lines))

    print("Wrote RF artifacts to", args.out_prefix)


if __name__ == "__main__":
    main()
