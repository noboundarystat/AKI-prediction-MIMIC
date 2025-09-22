# src/utils/io.py
import os
import json
import pandas as pd

def load_pickle(path: str):
    """Load a pickle (e.g., data_dict) with a friendly error."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Empty pickle: {path}")
    with open(path, "rb") as f:
        obj = pd.read_pickle(f)
    return obj

def load_target(path: str) -> pd.DataFrame:
    """Load target parquet and normalize a few types."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Target not found: {path}")
    df = pd.read_parquet(path)
    # normalize dtypes commonly used downstream
    for c in ("admittime", "dischtime", "deathtime"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("hospital_expire_flag", "is_aki", "is_ckd"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)

def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)

def save_json(obj, path: str, indent: int = 2):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
