# src/utils/dx_maps.py
from typing import Dict, Iterable, List
import pandas as pd

# Prefix sets (ICD-9 without dot; ICD-10 letters + digits)
COMORBID_PREFIXES: Dict[str, Dict[str, Iterable[str]]] = {
    # Hypertension
    "htn":   {"icd9": ["401","402","403","404","405"], "icd10": ["I10","I11","I12","I13","I15"]},
    # Diabetes (any type)
    "dm":    {"icd9": ["250"], "icd10": ["E08","E09","E10","E11","E13"]},
    # Coronary artery disease / ischemic heart disease
    "cad":   {"icd9": ["410","411","412","413","414"], "icd10": ["I20","I21","I22","I23","I24","I25"]},
    # Heart failure
    "hf":    {"icd9": ["428"], "icd10": ["I50"]},
    # Atrial fibrillation/flutter (optional)
    "afib":  {"icd9": ["42731","42732"], "icd10": ["I48"]},
    # COPD / chronic lower respiratory
    "copd":  {"icd9": ["490","491","492","493","494","495","496"], "icd10": ["J40","J41","J42","J43","J44"]},
    # Cancer (solid + hematologic; broad)
    "cancer":{"icd9": [f"{i:03d}" for i in range(140,240)], "icd10": ["C","D0","D3","D4"]},
    # Infection (very broad; refine as needed)
    "infection": {"icd9": ["001","002","003","004","005","006","007","008","009","038","9959"],
                  "icd10": ["A","B","J1","N39","R65"]},
    # CKD as comorbidity (baseline renal disease)
    "ckd":   {"icd9": ["585"], "icd10": ["N18"]},
    # Liver disease (mild → severe)
    "liver": {"icd9": ["570","571","572","573"], "icd10": ["K70","K71","K72","K73","K74","K75","K76","K77"]},
    # Cerebrovascular disease
    "cerebrovasc": {"icd9": ["430","431","432","433","434","435","436","437","438"], "icd10": ["I60","I61","I62","I63","I64","I65","I66","G45"]},
    # Obesity
    "obesity": {"icd9": ["2780"], "icd10": ["E66"]},
    # (Do NOT include AKI here for modeling AKI)
}

def _starts_with_any(code: str, prefixes: Iterable[str]) -> bool:
    cs = str(code)
    return any(cs.startswith(p) for p in prefixes)

def code_in_category(icd_code: str, icd_version: int, category: str) -> bool:
    """
    True if icd_code belongs to the comorbidity category by prefix sets.
    icd_version: 9 or 10 (others ignored)
    """
    if category not in COMORBID_PREFIXES:
        raise KeyError(f"Unknown comorbidity category: {category}")
    prefs = COMORBID_PREFIXES[category]
    if icd_version == 9:
        return _starts_with_any(icd_code, prefs["icd9"])
    if icd_version == 10:
        return _starts_with_any(icd_code, prefs["icd10"])
    return False

def diagnosis_to_hadm_flags(df_dx: pd.DataFrame,
                            categories: List[str],
                            subject_col: str = "subject_id",
                            hadm_col: str = "hadm_id",
                            icd_col: str = "icd_code",
                            ver_col: str = "icd_version") -> pd.DataFrame:
    """
    Collapse diagnoses to hadm-level 0/1 flags for each requested category.
    Returns columns: subject_id, hadm_id, <cat> flags
    """
    df = df_dx[[subject_col, hadm_col, icd_col, ver_col]].copy()
    df[ver_col] = pd.to_numeric(df[ver_col], errors="coerce").fillna(9).astype(int)
    for cat in categories:
        df[cat] = df.apply(lambda r: int(code_in_category(r[icd_col], r[ver_col], cat)), axis=1)

    agg = (df.groupby([subject_col, hadm_col])[categories]
             .max()
             .reset_index())
    return agg
