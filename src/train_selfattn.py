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


def build_feature_group(feature_list):
    groups = {
        "comorbidities": [],
        "demographics": [],
        "insurance": [],
        "labs": [],
        "vitals": [],
        "fluids": [],
        "history": [],
        "embeddings": [],
        "targets_meta": [],
        "other": []
    }

    for feat in feature_list:
        # Comorbidities
        if feat.startswith("cmb_"):
            groups["comorbidities"].append(feat)

        # Demographics
        elif feat.startswith("dem_") or feat == "age":
            groups["demographics"].append(feat)

        # Insurance
        elif feat.startswith("ins_"):
            groups["insurance"].append(feat)

        # Labs (7d labs + *_missing flags)
        elif feat.startswith("labs7d_"):
            groups["labs"].append(feat)

        # Vitals (48h vitals + *_missing flags)
        elif feat.startswith("vital48h_"):
            groups["vitals"].append(feat)

        # Fluids (48h / 6h + *_missing flags)
        elif feat.startswith("fluid"):
            groups["fluids"].append(feat)

        # Recent history (hx30d, hx31to180d, *_missing flags)
        elif feat.startswith("hx"):
            groups["history"].append(feat)

        # Learned embeddings
        elif feat.startswith("emb_") or feat == "emb_missing":
            groups["embeddings"].append(feat)

        # Target / meta columns
        elif feat in ["incident_aki_label", "is_aki", "is_ckd", 
                      "hospital_expire_flag", "death_flag", 
                      "time_to_death_from_adm_hr", "time_to_death_from_disch_hr"]:
            groups["targets_meta"].append(feat)

        else:
            groups["other"].append(feat)

    # Remove empty groups
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    return groups

from sklearn.preprocessing import StandardScaler
def scale_features_train(df, feature_group, scale_groups=['labs', 'vitals', 'fluids']):
    df_scaled = pd.DataFrame()
    scalers = {}
    for (k,v) in feature_group.items():
        if k in scale_groups:
            feature_scaler = StandardScaler()
            feature_scaler.fit(df[v])
            scalers[k] = feature_scaler
            scaled_v = feature_scaler.transform(df[v])
            df_scaled_v = pd.DataFrame(data=scaled_v, columns=v, index=df.index)
            df_scaled = pd.concat([df_scaled, df_scaled_v], axis=1)
        else:
            df_scaled = pd.concat([df_scaled, df[v]], axis=1)
    return df_scaled, scalers

def scale_features_test(df, feature_group, scalers, scale_groups=('labs','vitals','fluids')):
    df_scaled = pd.DataFrame(index=df.index)
    for k, v in feature_group.items():
        if k in scale_groups:
            scaled_v = scalers[k].transform(df[v])
            df_scaled_v = pd.DataFrame(scaled_v, columns=v, index=df.index)
            df_scaled = pd.concat([df_scaled, df_scaled_v], axis=1)
        else:
            df_scaled = pd.concat([df_scaled, df[v]], axis=1)
    return df_scaled

# ------------------------
# Dataset
# ------------------------
import torch
from torch.utils.data import Dataset, DataLoader

class akiData(Dataset):
    def __init__(self, X, y):
        self.X = X.reset_index(drop=True)
        self.y = pd.Series(y).reset_index(drop=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X.iloc[index].values.astype(np.float32)
        y = float(self.y.iloc[index])
        return {'X': torch.tensor(x), 'y': torch.tensor(y)}

# ------------------------
# Model
# ------------------------
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=1, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        # project
        Q = self.q(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)  # [B, h, L, d_k]
        K = self.k(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        V = self.v(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)         # [B,h,L,L]
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ V                                                # [B,h,L,d_k]
        out = out.transpose(1,2).contiguous().view(B, L, self.d_model)   # [B,L,d_model]
        return self.out(out)                                             # [B,L,d_model]


class akiClassifier(nn.Module):
    def __init__(self, feature_group, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.group_slices = []
        self.proj = nn.ModuleList()
        start = 0
        for g in feature_group.values():
            gdim = len(g)
            self.group_slices.append(slice(start, start+gdim))
            start += gdim
            self.proj.append(nn.Linear(gdim, d_model))  # project each group to same dim

        self.attn_block = nn.Sequential(
            SelfAttention(d_model, n_heads, dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        self.classifier = nn.Sequential(
            nn.Linear(len(feature_group)*d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        group_vecs = []
        for sl, proj in zip(self.group_slices, self.proj):
            gx = x[:, sl]                  # [B, group_dim]
            group_vecs.append(proj(gx))    # [B, d_model]

        h = torch.stack(group_vecs, dim=1) # [B, G, d_model]
        h = self.attn_block(h)             # attention across groups
        h = h.flatten(start_dim=1)         # [B, G*d_model]
        return self.classifier(h)

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
    ap.add_argument("--out-prefix", default="artifacts/multiheaded_selfatten_model/selfatten")
    args = ap.parse_args()

    # Load data
    df = pd.read_parquet(args.input)
    df = df[~((df['cmb_ckd']==1)|(df['renal_impaired_at_adm']==1))]
    target_col = args.target

    df4 = df[df["version"] == "mimic4"].fillna(0.0).copy()
    df3 = df[df["version"] == "mimic3"].fillna(0.0).copy()

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
    feature_cols = [c for c in df4.columns if c not in exclude_cols + [target_col]]
    df4 = df4[feature_cols+[target_col]]

    feature_cols = [x for x in df4.columns if x!=target_col]
    feature_group = build_feature_group(feature_cols)

    # Trainval/test split (MIMIC-IV only)
    df_trainval, df_test = train_test_split(
        df4, test_size=0.3, random_state=20250831, stratify=df4[target_col]
    )


    df_train_val, df_test = train_test_split(df4.fillna(0.0), test_size=0.3, random_state=20250831, stratify=df4[target_col])
    df_train_val_scaled, scalers = scale_features_train(df_train_val.drop(columns=target_col), feature_group)

    X_train, X_val, y_train, y_val = train_test_split(df_train_val_scaled, df_train_val[target_col].values, test_size=0.2, stratify=df_train_val[target_col])

    X_test = scale_features_test(df_test.drop(columns=target_col), feature_group, scalers)
    y_test = df_test[target_col].values

    # External validation (MIMIC-III)
    if len(df3) > 0:
        X_m3 =scale_features_test(df3.drop(columns=target_col), feature_group, scalers)
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
    model = akiClassifier(feature_group).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Train
    best_state_dict = train(model, criterion, optimizer, train_loader, val_loader, device=args.device,
                            n_epochs=args.epochs, patience=args.patience)

    # Save best model
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    best_model = akiClassifier(feature_group)
    best_model.load_state_dict(best_state_dict)
    best_model.to(args.device)
    best_model.eval()
    torch.save(best_model.state_dict(), f"{out_prefix}.pt")
    joblib.dump(scalers, f"{out_prefix}.scaler.joblib")

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
    # Integrated Gradients feature importance (via Captum, with grouping)
    # ------------------------
    try:
        from captum.attr import IntegratedGradients

        print("Computing feature importance with Integrated Gradients...")

        # Wrap model to output probability
        class ProbWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return torch.sigmoid(self.model(x))

        model_prob = ProbWrapper(best_model).to(args.device)
        model_prob.eval()

        # Use 200 background rows for baseline
        bg_idx = np.random.choice(len(X_train), size=min(200, len(X_train)), replace=False)
        baseline = torch.tensor(X_train.iloc[bg_idx].values, dtype=torch.float32).to(args.device)

        # Subset of validation data
        sv = torch.tensor(X_val.iloc[:200].values, dtype=torch.float32).to(args.device)

        # Integrated Gradients
        ig = IntegratedGradients(model_prob)
        attributions, delta = ig.attribute(
            sv,
            baselines=baseline.mean(dim=0, keepdim=True),
            n_steps=50,
            return_convergence_delta=True
        )

        # Mean absolute attribution per feature
        mean_abs_ig = attributions.abs().mean(dim=0).cpu().numpy()

        fi_df = pd.DataFrame({
            "feature": features,
            "importance": mean_abs_ig
        }).sort_values("importance", ascending=False)

        # Save per-feature
        fi_df.to_csv(f"{out_prefix}.integrated_gradients_raw.csv", index=False)

        # --- Grouped importance ---
        feat2group = {}
        for gname, flist in feature_group.items():
            for f in flist:
                feat2group[f] = gname
        fi_df["group"] = fi_df["feature"].map(feat2group).fillna("other")

        grouped = (fi_df.groupby("group")["importance"]
                          .sum()
                          .reset_index()
                          .sort_values("importance", ascending=False))

        grouped.to_csv(f"{out_prefix}.integrated_gradients_grouped.csv", index=False)

        print(f"✓ Integrated Gradients importance saved -> {out_prefix}.integrated_gradients_raw.csv (per-feature)")
        print(f"✓ Integrated Gradients importance saved -> {out_prefix}.integrated_gradients_grouped.csv (per-group)")

    except Exception as e:
        print(f"⚠️ Integrated Gradients importance skipped due to error: {e}")


if __name__ == "__main__":
    main()