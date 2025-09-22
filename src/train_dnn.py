#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
import joblib


# ------------------------
# Dataset
# ------------------------
class akiData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return {
            "X": torch.tensor(self.X[index], dtype=torch.float32),
            "y": torch.tensor(self.y[index], dtype=torch.float32),
        }


# ------------------------
# Model
# ------------------------
class akiClassifier(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 4 * feature_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4 * feature_dim, feature_dim // 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, x):
        return self.classifier(x)


# ------------------------
# Training utils
# ------------------------
def train_one_epoch(model, criterion, optimizer, dataloader, device="cpu"):
    model.train()
    total_loss, total_n = 0.0, 0
    target, prediction = [], []
    for batch in dataloader:
        X, y = batch["X"].to(device), batch["y"].to(device)
        y_pred = model(X).squeeze()
        loss = criterion(y_pred, y)
        target.extend(y.detach().cpu())
        prediction.extend(torch.sigmoid(y_pred).detach().cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch["X"])
        total_n += len(batch["X"])

    return total_loss / total_n, np.array(target), np.array(prediction)


def evaluate(model, criterion, dataloader, device="cpu"):
    model.eval()
    total_loss, total_n = 0.0, 0
    target, prediction = [], []
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch["X"].to(device), batch["y"].to(device)
            y_pred = model(X).squeeze()
            loss = criterion(y_pred, y)
            target.extend(y.detach().cpu())
            prediction.extend(torch.sigmoid(y_pred).detach().cpu())

            total_loss += loss.item() * len(batch["X"])
            total_n += len(batch["X"])

    return total_loss / total_n, np.array(target), np.array(prediction)


def train(model, criterion, optimizer, train_loader, val_loader, device="cpu", n_epochs=20, patience=3):
    best_model_state_dict, best_loss, patience_counter = None, np.inf, 0
    for i in range(n_epochs):
        train_loss, train_target, train_pred = train_one_epoch(model, criterion, optimizer, train_loader, device)
        train_auc = roc_auc_score(train_target, train_pred)

        val_loss, val_target, val_pred = evaluate(model, criterion, val_loader, device)
        val_auc = roc_auc_score(val_target, val_pred)

        print(f"epoch {i+1}, train_loss: {train_loss:.4f}, train_auc: {train_auc:.4f}")
        print(f"           val_loss: {val_loss:.4f}, val_auc: {val_auc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {i+1}")
                break

    return best_model_state_dict


# ------------------------
# Metrics helper
# ------------------------
def eval_split(y_true, y_score, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= thr).astype(int)

    auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    auprc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    cls_report = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "confusion_matrix": cm,
        "classification_report": cls_report,
    }


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="features/features_all_imputed.parquet")
    ap.add_argument("--target", default="incident_aki_label")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-prefix", default="artifacts/dnn_model/dnn")
    args = ap.parse_args()

    # Load data
    df = pd.read_parquet(args.input)
    df = df[~((df['cmb_ckd']==1)|(df['renal_impaired_at_adm']==1))]
    target_col = args.target

    df4 = df[df["version"] == "mimic4"].fillna(0.0).copy()
    df3 = df[df["version"] == "mimic3"].fillna(0.0).copy()

    # Trainval/test split (MIMIC-IV only)
    df_trainval, df_test = train_test_split(
        df4, test_size=0.3, random_state=20250831, stratify=df4[target_col]
    )

    # Non-feature columns to exclude (align with notebook's preprocessing)
    exclude_cols = [
    "version", "subject_id", "hadm_id", "admittime", "target_meta__age",
    "gender", "hospital_expire_flag", "death_flag",
    "time_to_death_from_adm_hr", "time_to_death_from_disch_hr",
    "cmb_ckd", "is_aki", "is_ckd", "ckd_only_flag",
    "ckd_admission_flag", "is_autocar", "kdigo-aki",
    "renal_impaired_at_adm", "aki_history_flag", "onset_aki_flag",
    'hx30d_crrt_context_any', 'hx31to180d_crrt_context_any', 'hx30d_rrt_any', 'hx30d_crrt_any', 'hx31to180d_rrt_any', 'hx31to180d_crrt_any'
    ]

    # Drop non-feature + target
    feature_cols = [c for c in df_trainval.columns if c not in exclude_cols + [target_col]]
    df_trainval = df_trainval[feature_cols + [target_col]]
    df_test = df_test[feature_cols + [target_col]]
    if len(df3) > 0:
        df3 = df3[feature_cols + [target_col]]

    # Fit scaler on trainval
    scaler = StandardScaler().fit(df_trainval.drop(columns=target_col))
    X_trainval = scaler.transform(df_trainval.drop(columns=target_col))
    y_trainval = df_trainval[target_col].values

    X_test = scaler.transform(df_test.drop(columns=target_col))
    y_test = df_test[target_col].values

    # Split train/val inside trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=20250831, stratify=y_trainval
    )

    # External validation (MIMIC-III)
    if len(df3) > 0:
        X_m3 = scaler.transform(df3.drop(columns=target_col))
        y_m3 = df3[target_col].values
    else:
        X_m3, y_m3 = None, None

    # Dataloaders
    train_loader = DataLoader(akiData(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(akiData(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(akiData(X_test, y_test), batch_size=2 * args.batch_size, shuffle=False)
    m3_loader = DataLoader(akiData(X_m3, y_m3), batch_size=2 * args.batch_size, shuffle=False) if y_m3 is not None else None

    # Model
    feature_dim = X_train.shape[1]
    model = akiClassifier(feature_dim).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Train
    best_state_dict = train(model, criterion, optimizer, train_loader, val_loader, device=args.device,
                            n_epochs=args.epochs, patience=args.patience)

    # Save best model
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    best_model = akiClassifier(feature_dim)
    best_model.load_state_dict(best_state_dict)
    best_model.to(args.device)
    best_model.eval()
    torch.save(best_model.state_dict(), f"{out_prefix}.pt")
    joblib.dump(scaler, f"{out_prefix}.scaler.joblib")

    # Evaluate and collect predictions
    results = {}
    preds_frames = []

    split_map = {
        "train": (train_loader, pd.DataFrame(X_train, columns=feature_cols), y_train),
        "val": (val_loader, pd.DataFrame(X_val, columns=feature_cols), y_val),
        "test": (test_loader, df_test, y_test),
        "mimic3_full": (m3_loader, df3 if y_m3 is not None else None, y_m3),
    }

    for split_name, (loader, df_split, y_true) in split_map.items():
        if loader is None or y_true is None:
            continue
        loss, target, pred = evaluate(best_model, criterion, loader, device=args.device)
        results[split_name] = eval_split(target, pred)
        print(f"{split_name} prediction mean={np.mean(pred):.4f}, std={np.std(pred):.4f}")

        # Save row-level predictions (with IDs if available)
        if df_split is not None and {"subject_id", "hadm_id"}.issubset(df_split.columns):
            out_df = df_split[["subject_id", "hadm_id"]].copy()
        else:
            out_df = pd.DataFrame({"row": np.arange(len(y_true))})
        out_df["true"] = target
        out_df["prob"] = pred
        out_df["split"] = split_name
        preds_frames.append(out_df)

    # Concatenate and save predictions
    if preds_frames:
        preds_df = pd.concat(preds_frames, axis=0, ignore_index=True)
        preds_df.to_csv(f"{out_prefix}.predictions.csv", index=False)

    # Save features
    features = list(df_trainval.drop(columns=target_col).columns)
    Path(f"{out_prefix}.features.txt").write_text("\n".join(features))

    # Save metrics
    Path(f"{out_prefix}.metrics.json").write_text(json.dumps(results, indent=2))

    # Save summary
    lines = []
    lines.append(f"# DNN Results")
    lines.append(f"- Features: {len(features)}")
    for name in ["train", "val", "test", "mimic3_full"]:
        if name not in results:
            continue
        m = results[name]
        lines.append(f"\n## {name.upper()}")
        lines.append(f"- AUROC: {m['auroc']:.6f} | AUPRC: {m['auprc']:.6f}")
        lines.append(f"- Confusion matrix: {m['confusion_matrix']}")
    Path(f"{out_prefix}.summary.txt").write_text("\n".join(lines))

    # ------------------------
    # Integrated Gradients feature importance (Captum, DNN version)
    # ------------------------
    try:
        from captum.attr import IntegratedGradients

        print("Computing feature importance with Integrated Gradients (DNN)...")

        # Wrap model to output probability
        class ProbWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return torch.sigmoid(self.model(x))

        model_prob = ProbWrapper(best_model).to(args.device)
        model_prob.eval()

        # Handle X_train / X_val as DataFrame or ndarray
        def to_numpy(x):
            if hasattr(x, "iloc"):  # pandas DataFrame
                return x.values
            return x

        # Baseline: mean of 200 random training samples
        bg_idx = np.random.choice(len(X_train), size=min(200, len(X_train)), replace=False)
        baseline = torch.tensor(to_numpy(X_train)[bg_idx], dtype=torch.float32).to(args.device)

        # Validation subset (200 rows)
        sv = torch.tensor(to_numpy(X_val)[:200], dtype=torch.float32).to(args.device)

        # Integrated Gradients
        ig = IntegratedGradients(model_prob)
        attributions, delta = ig.attribute(
            sv,
            baselines=baseline.mean(dim=0, keepdim=True),
            n_steps=50,
            return_convergence_delta=True
        )

        # Aggregate per feature
        mean_abs_ig = attributions.abs().mean(dim=0).cpu().numpy()

        # Use feature names if available, else f0...fN
        if "features" not in locals() or features is None:
            features = [f"f{i}" for i in range(mean_abs_ig.shape[0])]

        fi_df = pd.DataFrame({
            "feature": features,
            "importance": mean_abs_ig
        }).sort_values("importance", ascending=False)

        # Save
        fi_df.to_csv(f"{out_prefix}.integrated_gradients.csv", index=False)
        print(f"✓ Integrated Gradients importance saved -> {out_prefix}.integrated_gradients.csv")

    except Exception as e:
        print(f"⚠️ Integrated Gradients importance skipped due to error: {e}")

if __name__ == "__main__":
    main()