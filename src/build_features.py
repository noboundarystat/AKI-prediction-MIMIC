# src/build_features.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import List, Dict

import numpy as np
import pandas as pd

KEYS = ["version", "subject_id", "hadm_id"]


# -----------------------
# Small IO helpers
# -----------------------
def _log(msg: str): print(msg, flush=True)

def _exists(path: str) -> bool:
    return path is not None and os.path.exists(path) and os.path.getsize(path) > 0

def _read_parquet_or_empty(path: str, name: str) -> pd.DataFrame:
    if _exists(path):
        df = pd.read_parquet(path)
        # ensure keys exist
        missing = [k for k in KEYS if k not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing key columns {missing}")
        # drop exact dup rows if any
        df = df.drop_duplicates(subset=KEYS)
        _log(f"✓ loaded {os.path.basename(path)} -> rows={len(df):,}, cols={df.shape[1]}")
        return df
    _log(f"• skip {name}: file not found at {path}")
    return pd.DataFrame(columns=KEYS)


# -----------------------
# Merge logic
# -----------------------
def _safe_merge(base: pd.DataFrame, add: pd.DataFrame, block_name: str) -> pd.DataFrame:
    """Left-merge on KEYS. If non-key column name collision, prefix with block name."""
    if add is None or add.empty:
        _log(f"• {block_name}: empty → no merge")
        return base

    add_cols = [c for c in add.columns if c not in KEYS]
    if not add_cols:
        _log(f"• {block_name}: only keys → no merge")
        return base

    # Disambiguate overlapping non-key columns
    overlap = (set(base.columns) & set(add_cols)) - set(KEYS)
    if overlap:
        rename = {c: f"{block_name}__{c}" for c in overlap}
        add = add.rename(columns=rename)

    merged = base.merge(add, on=KEYS, how="left", validate="one_to_one")
    _log(f"✓ merged {block_name:<28s} -> rows={len(merged):,}, cols={merged.shape[1]}")
    return merged


# -----------------------
# Missingness
# -----------------------
def _write_missingness(df: pd.DataFrame, out_csv: str):
    miss = df.isna().mean().rename("missing_frac").reset_index().rename(columns={"index": "column"})
    miss = miss.sort_values("missing_frac", ascending=False)
    miss.to_csv(out_csv, index=False)
    _log(f"✓ wrote missingness -> {out_csv} (columns={len(miss)})")


# -----------------------
# Main build
# -----------------------
def build(features_dir: str, target_path: str, outdir: str,
          drop_thresh: float | None = None) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)

    # Base = target
    if not _exists(target_path):
        raise FileNotFoundError(f"target not found: {target_path}")
    base = pd.read_parquet(target_path)
    for k in KEYS:
        if k not in base.columns:
            raise ValueError(f"target missing key column: {k}")
    base = base.drop_duplicates(subset=KEYS)
    _log(f"✓ loaded target -> rows={len(base):,}, cols={base.shape[1]}")

    # Gather feature files (if a file is missing, it’s skipped gracefully)
    paths = {
        "features_comorbidities": os.path.join(features_dir, "features_comorbidities.parquet"),
        "features_demographics": os.path.join(features_dir, "features_demographics.parquet"),
        "features_insurance": os.path.join(features_dir, "features_insurance.parquet"),
        "features_labs_preicu7d": os.path.join(features_dir, "features_labs_preicu7d.parquet"),
        "features_vitals_preicu48h": os.path.join(features_dir, "features_vitals_preicu48h.parquet"),
        "features_fluids_preicu48h": os.path.join(features_dir, "features_fluids_preicu48h.parquet"),
        "features_meds_procedures_history": os.path.join(features_dir, "features_meds_procedures_history.parquet"),
        "features_text": os.path.join(features_dir, "features_text.parquet"),  
    }

    # Load each block
    blocks: Dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        blocks[name] = _read_parquet_or_empty(path, name)

    # Merge blocks onto base
    merged = base[KEYS].drop_duplicates().copy()
    for name in [
        "features_comorbidities",
        "features_demographics",
        "features_insurance",
        "features_labs_preicu7d",
        "features_vitals_preicu48h",
        "features_fluids_preicu48h",
        "features_meds_procedures_history",
        "features_text"
    ]:
        merged = _safe_merge(merged, blocks[name], name)

    # Final join with target labels/metadata
    merged = _safe_merge(merged, base, "target_meta")

    # Optional column drop by missingness threshold
    if drop_thresh is not None:
        # compute missingness excluding keys and core labels
        protected = set(KEYS) | {"is_aki", "is_ckd", "hospital_expire_flag", "admittime", "age", "gender"}
        miss = merged.isna().mean()
        to_drop = [c for c, frac in miss.items() if (c not in protected) and (frac > drop_thresh)]
        if to_drop:
            merged = merged.drop(columns=to_drop)
            _log(f"• dropped {len(to_drop)} columns with missingness > {drop_thresh:.2f}")

    # Write outputs
    out_all = os.path.join(outdir, "features_all.parquet")
    merged.to_parquet(out_all, index=False)
    _log(f"✓ wrote {out_all}  shape={merged.shape}")

    _write_missingness(merged, os.path.join(outdir, "missingness.csv"))

    # Quick sanity: uniqueness
    dupn = merged.duplicated(subset=KEYS).sum()
    if dupn:
        _log(f"⚠️  WARNING: duplicates on keys after merge: {dupn}")

    return {"features_all": out_all}


def main():
    ap = argparse.ArgumentParser(description="Merge all feature blocks into a single features_all.parquet")
    ap.add_argument("--target", default="target.parquet", help="Path to target.parquet")
    ap.add_argument("--features-dir", default=".", help="Directory containing feature parquet files")
    ap.add_argument("--outdir", default="./features", help="Output directory")
    ap.add_argument("--drop-thresh", type=float, default=None,
                    help="Optional: drop columns with missingness > THRESH (e.g., 0.7)")
    args = ap.parse_args()

    build(features_dir=args.features_dir,
          target_path=args.target,
          outdir=args.outdir,
          drop_thresh=args.drop_thresh)


if __name__ == "__main__":
    main()
