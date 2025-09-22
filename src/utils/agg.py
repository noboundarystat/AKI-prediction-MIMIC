# src/utils/agg.py
from typing import Dict, Optional
import numpy as np
import pandas as pd

def safe_min(x: pd.Series) -> Optional[float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.min()) if not x.empty else np.nan

def safe_max(x: pd.Series) -> Optional[float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.max()) if not x.empty else np.nan

def safe_mean(x: pd.Series) -> Optional[float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.mean()) if not x.empty else np.nan

def safe_std(x: pd.Series) -> Optional[float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.std(ddof=1)) if not x.empty and x.size > 1 else np.nan

def safe_count(x: pd.Series) -> int:
    return int(pd.to_numeric(x, errors="coerce").notna().sum())

def safe_last_by_time(values: pd.Series, times: pd.Series) -> Optional[float]:
    """Return value at the latest non-null time."""
    t = pd.to_datetime(times, errors="coerce")
    v = pd.to_numeric(values, errors="coerce")
    m = t.notna() & v.notna()
    if not m.any():
        return np.nan
    idx = t[m].idxmax()
    return float(v.loc[idx])

def winsorize_series(x: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip extremes to given quantiles; ignores NaNs."""
    s = pd.to_numeric(x, errors="coerce")
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)

def aggregate_timeseries(df: pd.DataFrame,
                         value_col: str = "valuenum",
                         time_col: str = "charttime") -> Dict[str, float]:
    """
    Generic set of aggregates for a time series.
    Returns dict with min/max/mean/std/last/count.
    """
    return {
        "min":  safe_min(df[value_col]),
        "max":  safe_max(df[value_col]),
        "mean": safe_mean(df[value_col]),
        "std":  safe_std(df[value_col]),
        "last": safe_last_by_time(df[value_col], df[time_col]),
        "count": safe_count(df[value_col]),
    }
