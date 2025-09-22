# src/utils/meds_maps.py
import re
from typing import Dict, Iterable, List, Set

# Compile case-insensitive regex patterns for each class
DRUG_CLASS_PATTERNS: Dict[str, Iterable[str]] = {
    # nephro-influential classes
    "nsaids": [
        r"\bibuprofen\b", r"\bnaproxen\b", r"\bdiclofenac\b", r"\bindomethacin\b",
        r"\bketorolac\b", r"\bmeloxicam\b", r"\bcelecoxib\b", r"\betodolac\b",
        r"\bnabumetone\b", r"\bpiroxicam\b", r"\bflurbiprofen\b",
    ],
    "diuretics": [
        r"\bfurosemide\b", r"\bbumetanide\b", r"\btorsemide\b",
        r"\bhydrochlorothiazide\b", r"\bchlorthalidone\b", r"\bindapamide\b",
        r"\bspironolactone\b", r"\beplerenone\b", r"\btriamterene\b", r"\bamiloride\b",
        r"\bacetazolamide\b",  # carbonic anhydrase inhibitor
    ],
    "vasopressors": [
        r"\bnorepinephrine\b|\bnoradrenaline\b",
        r"\bepinephrine\b|\badrenaline\b",
        r"\bphenylephrine\b",
        r"\bvasopressin\b",
        r"\bdopamine\b",
        r"\bdobutamine\b",         # inotrope but often grouped in vaso support
        r"\bangiotensin\s*ii\b|\bgiapreza\b",
    ],
    "aminoglycosides": [
        r"\bgentamicin\b", r"\btobramycin\b", r"\bamikacin\b",
        r"\bstreptomycin\b", r"\bkanamycin\b", r"\bnetilmicin\b",
    ],
    "mannitol": [r"\bmannitol\b"],
    "colloid_bolus": [
        r"\balbumin\b", r"\bhetastarch\b|\bh(es|ydroxyethyl)\s*starch\b",
        r"\bdextran\b", r"\bgelatin\b",
    ],
    # dialysis-related meds can help catch RRT context (rare in PRESCRIPTIONS)
    "crrt_context": [
        r"\bcitrate\b", r"\bheparin\b.*(circuit|dialysis)",
    ],
}

_COMPILED = {k: [re.compile(p, flags=re.I) for p in v] for k, v in DRUG_CLASS_PATTERNS.items()}

def drug_classes(drug_name: str) -> Set[str]:
    """
    Return set of class labels matched by a free-text drug string.
    """
    if not isinstance(drug_name, str) or not drug_name.strip():
        return set()
    s = drug_name.lower()
    hits = set()
    for cls, regs in _COMPILED.items():
        if any(r.search(s) for r in regs):
            hits.add(cls)
    return hits

def flag_row(drug_name: str) -> Dict[str, int]:
    """
    Return dict of {class: 0/1} for a given drug string.
    """
    hits = drug_classes(drug_name)
    return {cls: int(cls in hits) for cls in DRUG_CLASS_PATTERNS.keys()}

def get_drug_class_map() -> Dict[str, Iterable[str]]:
    """
    Return the regex pattern list per drug class.
    These patterns are intended for regex matching (not plain substring).
    """
    return DRUG_CLASS_PATTERNS

def get_drug_class_regex() -> Dict[str, Iterable[re.Pattern]]:
    """
    Return compiled regex patterns per class (case-insensitive).
    """
    return _COMPILED
