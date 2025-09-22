# src/features/features_text.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
from pathlib import Path
import re
import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False


# ------------ text utilities ------------
_DEID = re.compile(r"\[\*\*.*?\*\*\]")
_WS   = re.compile(r"\s+")

def clean_text(s: str, lower: bool = True, strip_deid: bool = True) -> str:
    if not isinstance(s, str):
        return ""
    if strip_deid:
        s = _DEID.sub(" ", s)
    if lower:
        s = s.lower()
    return _WS.sub(" ", s).strip()


def split_into_chunks(text: str, max_chars: int = 1000) -> list[str]:
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text) and text[end] != " ":
            sp = text.rfind(" ", start, end)
            if sp > start + max_chars // 2:
                end = sp
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


# ------------ notes harvesting (simplified for brevity) ------------
def gather_notes(data_dict: dict) -> pd.DataFrame:
    out = []
    if "noteevents" in data_dict:
        df = data_dict["noteevents"].copy()
        if "charttime" in df.columns:
            df["note_time"] = pd.to_datetime(df["charttime"], errors="coerce")
        df["version"] = "mimic3"
        out.append(df[["version", "subject_id", "hadm_id", "note_time", "text"]])
    if "radiology" in data_dict:
        r = data_dict["radiology"].copy()
        r["note_time"] = pd.to_datetime(r.get("charttime", pd.NaT), errors="coerce")
        r["version"] = "mimic4"
        text_col = "text" if "text" in r.columns else "impression"
        r = r.rename(columns={text_col: "text"})
        out.append(r[["version", "subject_id", "hadm_id", "note_time", "text"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build_text_features(
    data_pkl: str,
    target_parquet: str,
    out_path: str,
    model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2",  # smaller 256-dim model
    batch_size: int = 64,
    max_chars_per_note: int = 1000,
):
    if not ST_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed.")

    with open(data_pkl, "rb") as f:
        dd = pickle.load(f)

    target = pd.read_parquet(target_parquet)[["version", "subject_id", "hadm_id"]].drop_duplicates()
    notes = gather_notes(dd).merge(target, on=["version", "subject_id", "hadm_id"], how="inner")

    notes["text"] = notes["text"].astype(str).map(lambda s: clean_text(s))
    notes = notes[notes["text"].str.len() > 0].copy()

    # expand into chunks
    expanded = []
    for _, row in notes.iterrows():
        for ch in split_into_chunks(row["text"], max_chars=max_chars_per_note):
            expanded.append((row["version"], row["subject_id"], row["hadm_id"], ch))
    exp_df = pd.DataFrame(expanded, columns=["version", "subject_id", "hadm_id", "chunk_text"])

    # pick device → prefer MPS on Apple Silicon
    if TORCH_AVAILABLE and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = SentenceTransformer(model_name, device=device)

    embeddings = model.encode(
        exp_df["chunk_text"].tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )

    exp_df["emb_idx"] = np.arange(len(exp_df))
    grouped = exp_df.groupby(["version", "subject_id", "hadm_id"])["emb_idx"].apply(list).reset_index()

    rows = []
    for _, r in grouped.iterrows():
        v = embeddings[r["emb_idx"]].mean(axis=0)
        row = {"version": r["version"], "subject_id": r["subject_id"], "hadm_id": r["hadm_id"]}
        for i in range(v.shape[0]):
            row[f"emb_{i}"] = float(v[i])
        rows.append(row)

    feats = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    print(f"✅ Text features saved to {out_path}  | shape={feats.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-pkl", default="data.pkl")
    ap.add_argument("--target", default="target.parquet")
    ap.add_argument("--out", default="./features/features_text.parquet")
    ap.add_argument("--model_name", default="sentence-transformers/paraphrase-MiniLM-L3-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    build_text_features(
        data_pkl=args.data_pkl,
        target_parquet=args.target,
        out_path=args.out,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
