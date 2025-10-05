#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSurv for CKD with Cox partial likelihood.
- Stable training (full batch, sorted by time)
- Evaluation: C-index, td-AUROC/AUPRC, external MIMIC-III
- Explainability: Integrated Gradients (IG) for dem_sex_M
- Skips scaling for binary features (e.g. dem_sex_M stays 0/1)
"""

import argparse, json
from pathlib import Path
from typing import Dict
import numpy as np, pandas as pd, joblib
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pycox.models.loss import CoxPHLoss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# ---------------- metrics ---------------- #

def concordance_index(event_times, predicted_scores, event_observed):
    n = 0; n_conc = 0.0
    for i in range(len(event_times)):
        for j in range(i + 1, len(event_times)):
            if event_observed[i] == 1 or event_observed[j] == 1:
                ti, tj = event_times[i], event_times[j]
                si, sj = predicted_scores[i], predicted_scores[j]
                if ti != tj:
                    n += 1
                    if (ti < tj and si > sj) or (tj < ti and sj > si):
                        n_conc += 1
                    elif si == sj: n_conc += 0.5
    return n_conc / n if n > 0 else np.nan

def td_binary_metrics_at_time(time, event, score, t_cut: float) -> dict:
    time = np.asarray(time, float); event = np.asarray(event, int); score = np.asarray(score, float)
    pos = (event == 1) & (time <= t_cut); neg = (time > t_cut); use = pos | neg
    if use.sum() == 0: return {"time": t_cut, "auroc": np.nan, "auprc": np.nan,"n_used": 0, "pos_rate": np.nan}
    y_true = pos[use].astype(int); y_score = score[use]
    if len(np.unique(y_true)) < 2:
        return {"time": t_cut, "auroc": np.nan, "auprc": np.nan,
                "n_used": int(use.sum()), "pos_rate": float(y_true.mean())}
    auroc = float(roc_auc_score(y_true, y_score)); auprc = float(average_precision_score(y_true, y_score))
    return {"time": float(t_cut), "auroc": auroc, "auprc": auprc,
            "n_used": int(use.sum()), "pos_rate": float(y_true.mean())}

def td_metrics_over_grid(time, event, score, grid_times: np.ndarray):
    rows = [td_binary_metrics_at_time(time, event, score, t) for t in grid_times]
    df = pd.DataFrame(rows)
    mean_auroc = float(df["auroc"].mean(skipna=True)); mean_auprc = float(df["auprc"].mean(skipna=True))
    def trapz_over(col):
        v = df[["time", col]].dropna()
        if len(v) < 2: return np.nan
        x, y = v["time"].values, v[col].values; order = np.argsort(x)
        return float(np.trapz(y[order], x[order]) / (x[order][-1] - x[order][0]))
    return df, {"mean_td_auroc": mean_auroc, "mean_td_auprc": mean_auprc,
                "iAUC": trapz_over("auroc"), "iAUPRC": trapz_over("auprc")}

# ---------------- data split ---------------- #

def groups_split_masks(n: int, groups: np.ndarray, test_size: float, val_size: float, seed: int):
    idx = np.arange(n)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(idx, groups=groups))
    rel_val = val_size / max(1e-12, (1.0 - test_size))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed+1)
    tr_idx_rel, val_idx_rel = next(gss2.split(trainval_idx, groups=groups[trainval_idx]))
    train_idx = trainval_idx[tr_idx_rel]; val_idx = trainval_idx[val_idx_rel]
    train_m = np.zeros(n, bool); val_m = np.zeros(n, bool); test_m = np.zeros(n, bool)
    train_m[train_idx]=True; val_m[val_idx]=True; test_m[test_idx]=True
    return train_m, val_m, test_m

# ---------------- model ---------------- #

class DeepSurv(nn.Module):
    def __init__(self, in_features, hidden=[512,128,64], dropout=0.1):
        super().__init__()
        layers = []; last=in_features
        for h in hidden:
            layers += [nn.Linear(last,h), nn.ReLU()]
            if dropout>0: layers += [nn.Dropout(dropout)]
            last=h
        layers += [nn.Linear(last,1)]
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)

# ---------------- IG helper ---------------- #

def compute_ig(model, X_tensor, feature_idx, baseline=None, steps=64, device="cpu"):
    ig = IntegratedGradients(model); model.eval()
    if baseline is None: baseline = torch.zeros_like(X_tensor)
    attr, _ = ig.attribute(X_tensor.to(device), baselines=baseline.to(device),
                           n_steps=steps, return_convergence_delta=True)
    return attr[:, feature_idx].detach().cpu().numpy()

# ---------------- main ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="mimic_ckd_e3.txt")
    # ap.add_argument("--input", default="features_ckd_imputed.parquet")
    ap.add_argument("--time-col", default="time_days")
    ap.add_argument("--event-col", default="is_ckd")
    ap.add_argument("--dataset-col", default="version")
    ap.add_argument("--mimic3-value", default="mimic3")
    ap.add_argument("--mimic4-value", default="mimic4")
    ap.add_argument("--mode", choices=["mimic4_only","both"], default="mimic4_only")
    ap.add_argument("--external-test", choices=["auto","none","mimic3"], default="auto",
                    help="Run external test after training. 'auto' runs on MIMIC-III if rows exist.")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--impute", action="store_true", default=True)
    ap.add_argument("--scaler", choices=["standard","robust","none"], default="robust")
    ap.add_argument("--n-grid", type=int, default=50)
    ap.add_argument("--grid-min-quantile", type=float, default=0.05)
    ap.add_argument("--grid-max-quantile", type=float, default=0.95)
    ap.add_argument("--out-prefix", default="artifacts/deepsurv_ckd/deepsurv_ckd")
    ap.add_argument("--ig-steps", type=int, default=64)
    ap.add_argument("--exclude-cols", nargs="*", default=[
        "version", "subject_id", "event_hadm_id", "index_admit",
        "ckd_event_hadm_id", "postckd_event_hadm_id", "death_event_hadm_id",
        "time_to_ckd", "time_to_postckd", "time_to_death", "cmb_ckd", "gender", 
        "is_ckd", "is_postckd", "is_death", "is_censored", "time_days", "event_code", "censor_time"
    ])
    args = ap.parse_args()

    np.random.seed(args.epochs); torch.manual_seed(args.epochs)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load data ---
    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input, sep="\t")
    if "cmb_ckd" in df: df = df[df.cmb_ckd==0].copy()

    exclude = args.exclude_cols + [col for col in df.columns if "_missing" in col]
    feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype,np.number)]

    if "subject_id" not in df: df["subject_id"] = df.index

    # Training dataset selection
    dsplit = df[df[args.dataset_col]==args.mimic4_value] if args.mode=="mimic4_only" else df
    X = dsplit[feature_cols].copy()
    y_time = dsplit[args.time_col].astype(float).values
    y_event = dsplit[args.event_col].astype(int).values
    groups = dsplit["subject_id"].values
    train_m,val_m,test_m = groups_split_masks(len(dsplit),groups,0.2,0.1,42)

    # --- detect binary vs continuous ---
    binary_like = [c for c in feature_cols if set(np.unique(X[c].dropna().values)) <= {0,1}]
    cont_cols   = [c for c in feature_cols if c not in binary_like]

    # --- imputer ---
    imp = SimpleImputer(strategy="median").fit(X.loc[train_m, feature_cols])

    # --- scaler (fit on imputed continuous cols only) ---
    sca = None
    if args.scaler in ("standard","robust") and len(cont_cols) > 0:
        scaler_cls = StandardScaler if args.scaler=="standard" else RobustScaler
        X_train_imp = pd.DataFrame(
            imp.transform(X.loc[train_m, feature_cols]),
            columns=feature_cols, index=X.loc[train_m].index
        )
        sca = scaler_cls().fit(X_train_imp[cont_cols])

    # --- prep function ---
    def prep(dfpart):
        vals = imp.transform(dfpart[feature_cols])
        df_imp = pd.DataFrame(vals, columns=feature_cols, index=dfpart.index)
        if sca is not None and len(cont_cols) > 0:
            df_imp[cont_cols] = sca.transform(df_imp[cont_cols])
        return torch.tensor(df_imp.values, dtype=torch.float32)

    X_train,X_val,X_test = map(lambda m: prep(X.loc[m]),[train_m,val_m,test_m])
    t_train,t_val,t_test = map(lambda a: torch.tensor(a,dtype=torch.float32),[y_time[train_m],y_time[val_m],y_time[test_m]])
    e_train,e_val,e_test = map(lambda a: torch.tensor(a,dtype=torch.float32),[y_event[train_m],y_event[val_m],y_event[test_m]])
    val_loader=DataLoader(TensorDataset(X_val,t_val,e_val),batch_size=2048,shuffle=False)
    test_loader=DataLoader(TensorDataset(X_test,t_test,e_test),batch_size=2048,shuffle=False)

    # --- model ---
    model=DeepSurv(X_train.shape[1]).to(device)
    criterion=CoxPHLoss()
    optimizer=optim.AdamW(model.parameters(),lr=5e-3,weight_decay=1e-3)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=3,min_lr=1e-6)

    # --- training ---
    best_val,best_state=float("inf"),None; patience_left=args.patience
    for ep in range(1,args.epochs+1):
        model.train()
        order=torch.argsort(t_train,descending=True)
        xb,tb,eb=X_train[order].to(device),t_train[order].to(device),e_train[order].to(device)
        optimizer.zero_grad(); risks=model(xb); loss=criterion(risks,tb,eb)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
        tr_loss=float(loss.item())
        # val
        model.eval(); val_losses=[]
        with torch.no_grad():
            for xv,tv,ev in val_loader:
                xv,tv,ev=xv.to(device),tv.to(device),ev.to(device)
                val_losses.append(criterion(model(xv),tv,ev).item())
        va_loss=float(np.mean(val_losses))
        print(f"Epoch {ep}/{args.epochs} train={tr_loss:.4f} val={va_loss:.4f}")
        if va_loss<best_val-1e-6:
            best_val,best_state,patience_left=va_loss,{k:v.cpu().clone() for k,v in model.state_dict().items()},args.patience
        else:
            patience_left-=1
            if patience_left<=0: print("Early stopping."); break
        scheduler.step(va_loss)
    if best_state: model.load_state_dict(best_state)

    # --- predictions ---
    def predict_scores(loader):
        model.eval(); out=[]
        with torch.no_grad():
            for xb,_,_ in loader:
                out.append(model(xb.to(device)).cpu().numpy())
        return np.concatenate(out) if out else np.array([])
    risk_train=model(X_train.to(device)).detach().cpu().numpy()
    risk_val=predict_scores(val_loader); risk_test=predict_scores(test_loader)

    # --- metrics ---
    ci_train=concordance_index(y_time[train_m],risk_train,y_event[train_m])
    ci_val=concordance_index(y_time[val_m],risk_val,y_event[val_m])
    ci_test=concordance_index(y_time[test_m],risk_test,y_event[test_m])

    # grid based on TRAIN event times (used consistently across splits & external)
    train_event_times=y_time[train_m&(y_event==1)]
    lo=np.quantile(train_event_times,args.grid_min_quantile); hi=np.quantile(train_event_times,args.grid_max_quantile)
    grid_times=np.linspace(lo,hi,num=args.n_grid)

    td_train_df,td_train_agg=td_metrics_over_grid(y_time[train_m],y_event[train_m],risk_train,grid_times)
    td_val_df,td_val_agg=td_metrics_over_grid(y_time[val_m],y_event[val_m],risk_val,grid_times)
    td_test_df,td_test_agg=td_metrics_over_grid(y_time[test_m],y_event[test_m],risk_test,grid_times)
    results={"train":{"cindex":float(ci_train),**td_train_agg,"n":int(train_m.sum())},
             "val":{"cindex":float(ci_val),**td_val_agg,"n":int(val_m.sum())},
             "test":{"cindex":float(ci_test),**td_test_agg,"n":int(test_m.sum())}}

    # -------- External test: MIMIC-III -------- #
    run_ext = (args.external_test=="mimic3") or (args.external_test=="auto")
    if run_ext:
        df_ext = df[df[args.dataset_col]==args.mimic3_value].copy()
        if len(df_ext) == 0 and args.external_test == "mimic3":
            print("WARNING: --external-test=mimic3 requested but no MIMIC-III rows found.")
        if len(df_ext) > 0:
            # Use same features / imputer / scaler learned on TRAIN
            X_ext_t = prep(df_ext[feature_cols])
            t_ext = df_ext[args.time_col].astype(float).values
            e_ext = df_ext[args.event_col].astype(int).values
            ext_loader = DataLoader(TensorDataset(X_ext_t,
                                                  torch.tensor(t_ext, dtype=torch.float32),
                                                  torch.tensor(e_ext, dtype=torch.float32)),
                                    batch_size=2048, shuffle=False)
            risk_ext = predict_scores(ext_loader)
            ci_ext = concordance_index(t_ext, risk_ext, e_ext)
            td_ext_df, td_ext_agg = td_metrics_over_grid(t_ext, e_ext, risk_ext, grid_times)
            results["external_mimic3"] = {"cindex": float(ci_ext), **td_ext_agg, "n": int(len(df_ext))}
        else:
            print("No MIMIC-III rows present; skipping external test.")

    # --- explain IG for dem_sex_M ---
    if "dem_sex_M" in feature_cols:
        sex_idx=feature_cols.index("dem_sex_M")
        ig_vals=compute_ig(model,X_val,sex_idx,steps=args.ig_steps,device=device)
        df_val=pd.DataFrame({"sex_M":dsplit.loc[val_m,"dem_sex_M"].values,"ig_sexM":ig_vals})
        print("\nIG attribution for dem_sex_M by sex group:")
        print(df_val.groupby("sex_M")["ig_sexM"].describe())
        plt.figure(figsize=(6,4))
        df_val["sex_label"]=df_val["sex_M"].map({0:"Female",1:"Male"})
        df_val.boxplot(column="ig_sexM",by="sex_label")
        plt.suptitle(""); plt.title("IG attribution for dem_sex_M")
        plt.savefig(f"{args.out_prefix}.ig_dem_sexM.png",dpi=150)

    # --- Integrated Gradients: global feature importance ---
    print("\nComputing IG feature importances on VAL...")

    all_ig = {}
    for j, feat in enumerate(feature_cols):
        ig_vals = compute_ig(model, X_val, j, steps=args.ig_steps, device=device)
        all_ig[feat] = ig_vals

    # DataFrame of raw IG attributions
    df_ig = pd.DataFrame(all_ig)
    df_ig.to_csv(f"{args.out_prefix}.ig_attributions_val.csv", index=False)

    # Aggregate: mean absolute attribution
    ig_importance = df_ig.abs().mean().sort_values(ascending=False)
    ig_importance.to_csv(f"{args.out_prefix}.ig_feature_importance.csv")

    print("\nTop 15 features by mean |IG| on VAL:")
    print(ig_importance.head(15))

    # Plot top 20
    plt.figure(figsize=(8,6))
    ig_importance.head(20).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |IG attribution|")
    plt.title("Top 20 Features by Integrated Gradients (VAL)")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}.ig_feature_importance_top20.png", dpi=150)
    plt.close()

    # --- Counterfactual Δ risk (flip sex) ---
    if "dem_sex_M" in feature_cols:
        sex_idx = feature_cols.index("dem_sex_M")

        Xv_fem = X_val.clone(); Xv_fem[:, sex_idx] = 0.0
        Xv_mal = X_val.clone(); Xv_mal[:, sex_idx] = 1.0

        with torch.no_grad():
            risk_fem = model(Xv_fem.to(device)).cpu().numpy()
            risk_mal = model(Xv_mal.to(device)).cpu().numpy()

        actual_sex = dsplit.loc[val_m, "dem_sex_M"].values.astype(int)
        risk_as_is = np.where(actual_sex==1, risk_mal, risk_fem)
        risk_flipped = np.where(actual_sex==1, risk_fem, risk_mal)
        delta = risk_flipped - risk_as_is

        df_cf = pd.DataFrame({"sex_M": actual_sex, "delta_flip": delta})
        print("\nCounterfactual Δrisk (flip dem_sex_M) on VAL:")
        print(df_cf.groupby("sex_M")["delta_flip"].describe())
        df_cf.to_csv(f"{args.out_prefix}.cf_delta_sexM.csv", index=False)

    # --- Permutation importance for dem_sex_M ---
    if "dem_sex_M" in feature_cols:
        X_val_perm = X_val.clone()
        permuted = X_val_perm[:, sex_idx].numpy().copy()
        np.random.shuffle(permuted)
        X_val_perm[:, sex_idx] = torch.tensor(permuted, dtype=torch.float32)

        with torch.no_grad():
            risk_val_perm = model(X_val_perm.to(device)).cpu().numpy()

        ci_val_perm = concordance_index(y_time[val_m], risk_val_perm, y_event[val_m])
        td_val_df_perm, td_val_agg_perm = td_metrics_over_grid(
            y_time[val_m], y_event[val_m], risk_val_perm, grid_times
        )

        print("\nPermutation importance — dem_sex_M:")
        print(f"Original C-index (val): {results['val']['cindex']:.3f}")
        print(f"Permuted C-index (val): {ci_val_perm:.3f}")
        print(f"Original mean_td_auroc: {results['val']['mean_td_auroc']:.3f}")
        print(f"Permuted mean_td_auroc: {td_val_agg_perm['mean_td_auroc']:.3f}")
        print(f"Original mean_td_auprc: {results['val']['mean_td_auprc']:.3f}")
        print(f"Permuted mean_td_auprc: {td_val_agg_perm['mean_td_auprc']:.3f}")

        td_val_df_perm.to_csv(f"{args.out_prefix}.val_perm_td_metrics.csv", index=False)
        
    # --- save artifacts ---
    out_prefix=Path(args.out_prefix); out_prefix.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),f"{args.out_prefix}.pt")
    joblib.dump({"imputer":imp,"scaler":sca,"features":feature_cols,
                 "binary_like":binary_like,"cont_cols":cont_cols},
                 f"{args.out_prefix}.joblib")
    Path(f"{args.out_prefix}.features.txt").write_text("\n".join(feature_cols))
    td_train_df.to_csv(f"{args.out_prefix}.train_td_metrics.csv",index=False)
    td_val_df.to_csv(f"{args.out_prefix}.val_td_metrics.csv",index=False)
    td_test_df.to_csv(f"{args.out_prefix}.test_td_metrics.csv",index=False)

    # save external MIMIC-III TD metrics if available
    if "external_mimic3" in results:
        # recompute df_ext to avoid carrying a big object; write CSV we already built
        df_ext = df[df[args.dataset_col]==args.mimic3_value].copy()
        X_ext_vals = prep(df_ext[feature_cols])
        with torch.no_grad():
            risk_ext = model(X_ext_vals.to(device)).cpu().numpy()
        t_ext = df_ext[args.time_col].astype(float).values
        e_ext = df_ext[args.event_col].astype(int).values
        td_ext_df, _ = td_metrics_over_grid(t_ext, e_ext, risk_ext, grid_times)
        td_ext_df.to_csv(f"{args.out_prefix}.mimic3_td_metrics.csv", index=False)

    Path(f"{args.out_prefix}.metrics.json").write_text(json.dumps(results,indent=2))
    print("DeepSurv CKD — summary:"); print(json.dumps(results,indent=2))

if __name__=="__main__":
    main()
