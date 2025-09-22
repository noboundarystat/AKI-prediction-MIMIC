# src/utils/time_windows.py
from typing import Tuple
import pandas as pd
import numpy as np

KEYS = ["version", "subject_id", "hadm_id"]

def to_datetime(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def choose_first_icu_after_admit(admissions: pd.DataFrame,
                                 icustays: pd.DataFrame) -> pd.DataFrame:
    """
    Return a frame with one row per (version, subject_id, hadm_id) and the
    FIRST ICU stay that starts AFTER admittime.
    Columns returned: KEYS + ['icu_intime','icu_outtime','icu_stay_id']
    """
    adm = admissions.copy()
    icu = icustays.copy()
    to_datetime(adm, ["admittime", "dischtime"])
    to_datetime(icu, ["intime", "outtime"])
    if "stay_id" not in icu.columns:
        # tolerate icustay_id naming
        if "icustay_id" in icu.columns:
            icu = icu.rename(columns={"icustay_id": "stay_id"})
        else:
            icu["stay_id"] = np.nan

    merged = (icu.merge(adm[KEYS + ["admittime"]], on=KEYS, how="inner", validate="many_to_one")
                .query("intime >= admittime"))
    if merged.empty:
        return adm[KEYS].drop_duplicates().assign(icu_intime=pd.NaT, icu_outtime=pd.NaT, icu_stay_id=np.nan)

    merged = merged.sort_values(["version","subject_id","hadm_id","intime"])
    first = (merged.groupby(KEYS, as_index=False)
                  .first()[KEYS + ["intime","outtime","stay_id"]]
                  .rename(columns={"intime":"icu_intime","outtime":"icu_outtime","stay_id":"icu_stay_id"}))
    return first

def clamp_to_window(df: pd.DataFrame,
                    tcol: str,
                    start_col: str,
                    end_col: str) -> pd.DataFrame:
    """
    Keep rows where start_col <= tcol < end_col (end inclusive can be toggled upstream).
    Assumes datetime cols already parsed.
    """
    m = (df[tcol].notna() &
         df[start_col].notna() &
         df[end_col].notna() &
         (df[tcol] >= df[start_col]) &
         (df[tcol] <  df[end_col]))
    return df.loc[m].copy()

def within_window(ts: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Boolean mask for ts within [start, end)."""
    return (ts.notna()) & (ts >= start) & (ts < end)

def floor_to_hour(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, errors="coerce").dt.floor("h")

def ceil_to_hour(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, errors="coerce").dt.ceil("h")
