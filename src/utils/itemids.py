# ----- VITALS (CHARTEVENTS / D_ITEMS) -----
import re
import pandas as pd
from typing import List, Dict, Iterable

VITAL_PATTERNS: Dict[str, Dict[str, Iterable[str]]] = {
    "heart_rate":  {"include": [r"\bheart\s*rate\b", r"\bhr\b"], "exclude": []},
    "sbp":         {"include": [r"\bsystolic\b.*\b(bp|pressure)\b"], "exclude": []},
    "dbp":         {"include": [r"\bdiastolic\b.*\b(bp|pressure)\b"], "exclude": []},
    "mbp":         {"include": [r"\b(mean|map)\b.*\b(bp|pressure)\b|\bmean\s+arterial\s+pressure\b"], "exclude": []},
    "resp_rate":   {"include": [r"\bresp(iratory)?\s*rate\b", r"\brr\b"], "exclude": []},
    "spo2":        {"include": [r"\bspo2\b", r"oxygen\s*saturation"], "exclude": []},
    "temperature": {"include": [r"\btemperature\b", r"\btemp\b"], "exclude": []},
    "cvp":         {"include": [r"\bcvp\b", r"central\s*venous\s*pressure"], "exclude": []},
}

ADVANCED_PATTERNS: Dict[str, Iterable[str]] = {
    # Cardiac output (avoid CO2)
    "cardiac_output": [
        r"cardiac\s*output",
        r"\bco\b(?!\s*2)",
    ],
    # Pulmonary artery pressures
    "pap_sbp": [
        r"(pulmonary|pa)\s*(artery|arterial).*(systolic|sys|\bsbp\b)",
        r"\bpap\s*(systolic|sys|\bsbp\b)",
        r"\bpa\s*(systolic|sys|\bsbp\b)",
    ],
    "pap_dbp": [
        r"(pulmonary|pa)\s*(artery|arterial).*(diastolic|dia|\bdbp\b)",
        r"\bpap\s*(diastolic|dia|\bdbp\b)",
        r"\bpa\s*(diastolic|dia|\bdbp\b)",
    ],
    "pap_mbp": [
        r"(pulmonary|pa)\s*(artery|arterial).*(mean|\bmap\b|\bmbp\b)",
        r"\bpap\s*(mean|\bmap\b|\bmbp\b)",
        r"\bpa\s*(mean|\bmap\b|\bmbp\b)",
    ],
    # Vent settings
    "tidal_volume": [
        r"\btidal\s*volume\b",
        r"\bvt\b(?!\w)",
    ],
    "minute_ventilation": [
        r"\bminute\s*vent(ilation)?\b",
        r"\bve\b(?!\w)",
    ],
    # Neuro
    "gcs": [
        r"\bglasgow\s*coma\s*scale\b.*(total|score)?",
        r"\bgcs\b(?!\w).*?(total|score)?",
        r"\bgcs\s*total\b",
    ],
}

def _regex_or(cols: pd.Series, patterns: Iterable[str]) -> pd.Series:
    """OR across multiple regex patterns against a text Series."""
    if not patterns:
        return pd.Series(False, index=cols.index)
    mask = cols.str.contains(patterns[0], case=False, regex=True, na=False)
    for p in patterns[1:]:
        mask |= cols.str.contains(p, case=False, regex=True, na=False)
    return mask

def chart_itemids_by_pattern(d_items: pd.DataFrame,
                             include: Iterable[str],
                             exclude: Iterable[str] = ()) -> List[int]:
    """
    Match against label and abbreviation (if present).
    """
    di = d_items.copy()
    di.columns = [c.lower() for c in di.columns]
    if "label" not in di.columns or "itemid" not in di.columns:
        raise ValueError("d_items must have columns: itemid, label")

    label = di["label"].astype(str)
    abbr  = di["abbreviation"].astype(str) if "abbreviation" in di.columns else pd.Series("", index=di.index)
    text  = (label + " " + abbr)

    inc = _regex_or(text, include)
    if exclude:
        exc = _regex_or(text, exclude)
        inc &= ~exc

    return sorted(di.loc[inc, "itemid"].dropna().astype(int).unique().tolist())

def get_vital_itemids(d_items: pd.DataFrame, signal: str) -> List[int]:
    """
    Unified resolver: supports base vitals (VITAL_PATTERNS) and advanced signals (ADVANCED_PATTERNS).
    """
    s = signal.lower()
    if s in VITAL_PATTERNS:
        pats = VITAL_PATTERNS[s]
        return chart_itemids_by_pattern(d_items, pats["include"], pats.get("exclude", []))
    if s in ADVANCED_PATTERNS:
        return chart_itemids_by_pattern(d_items, ADVANCED_PATTERNS[s], [])
    raise KeyError(f"Unknown vital signal: {signal}")

# --- BP helpers (discover SBP/DBP/MAP itemids from D_ITEMS labels) ---

import re
import pandas as pd
from typing import List, Dict

def _find_ids(di: pd.DataFrame, include_regex: str, exclude_regex: str = "") -> List[int]:
    s = di["label"].astype(str)
    inc = s.str.contains(include_regex, case=False, regex=True, na=False)
    if exclude_regex:
        exc = s.str.contains(exclude_regex, case=False, regex=True, na=False)
        inc = inc & ~exc
    return di.loc[inc, "itemid"].dropna().astype(int).unique().tolist()

def get_bp_itemids(d_items: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Discover SBP/DBP/MAP itemids from D_ITEMS labels. Excludes pulmonary artery pressures.
    Returns: {"map_ids": [...], "sbp_ids": [...], "dbp_ids": [...]}
    """
    di = d_items.copy()
    di.columns = [c.lower() for c in di.columns]
    if "label" not in di.columns or "itemid" not in di.columns:
        return {"map_ids": [], "sbp_ids": [], "dbp_ids": []}

    # MAP
    map_ids = _find_ids(di, r"\b(mean\s+arterial\s+pressure|map)\b", exclude_regex=r"pulmonary")

    # SBP
    sbp_abp = _find_ids(di, r"\b(arterial|art|abp).*(systolic|sys)\b|\b(systolic|sys).*(arterial|art|abp)\b", exclude_regex=r"pulmonary")
    sbp_nbp = _find_ids(di, r"\b(non[-\s]*invasive|nibp|nbp|cuff).*(systolic|sys)\b|\b(systolic|sys).*(non[-\s]*invasive|nibp|nbp|cuff)\b", exclude_regex=r"pulmonary")
    sbp_generic = _find_ids(di, r"\b(systolic|sys)\b.*\b(blood\s*pressure|bp)\b", exclude_regex=r"pulmonary|pulm\.?|artery")

    # DBP
    dbp_abp = _find_ids(di, r"\b(arterial|art|abp).*(diastolic|dia)\b|\b(diastolic|dia).*(arterial|art|abp)\b", exclude_regex=r"pulmonary")
    dbp_nbp = _find_ids(di, r"\b(non[-\s]*invasive|nibp|nbp|cuff).*(diastolic|dia)\b|\b(diastolic|dia).*(non[-\s]*invasive|nibp|nbp|cuff)\b", exclude_regex=r"pulmonary")
    dbp_generic = _find_ids(di, r"\b(diastolic|dia)\b.*\b(blood\s*pressure|bp)\b", exclude_regex=r"pulmonary|pulm\.?|artery")

    sbp_ids = sorted(set(sbp_abp + sbp_nbp + sbp_generic))
    dbp_ids = sorted(set(dbp_abp + dbp_nbp + dbp_generic))
    return {"map_ids": sorted(set(map_ids)), "sbp_ids": sbp_ids, "dbp_ids": dbp_ids}
