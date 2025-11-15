# preparse.py
# -----------
# PURPOSE:
#   Lightweight, deterministic regex parsing of obvious fields from narratives.
#   Gives a working baseline and "hints" you can pass to the LLM later.
#
# WHAT TO ADD NEXT:
#   - Harden regex (handle "BP120/80", "O2 sat 93%", etc.).
#   - Fahrenheit ➜ Celsius conversion if Temp 'F' is detected.
#   - Parse GCS ("GCS 12"), pain score ("8/10"), acronyms (CP, SOB, N/V).
#   - Expand symptom lexicon from configs/ontology later.

import re
from typing import Dict, Tuple, List

# Simple starter regex (keep readable; refine later)
BP   = re.compile(r"\b(?:bp|blood pressure)\s*[:\-]?\s*(\d{2,3})\s*[/ ]\s*(\d{2,3})\b", re.I)
HR   = re.compile(r"\b(?:hr|heart rate|pulse)\s*[:\-]?\s*(\d{2,3})\b", re.I)
RR   = re.compile(r"\b(?:rr|resp(?:iratory)?(?:\s+rate)?)\s*[:\-]?\s*(\d{1,3})\b", re.I)
SPO2 = re.compile(r"\b(?:sp[o0]2|sat(?:uration)?)\s*[:\-]?\s*(\d{2,3})\s*%?\b", re.I)
TEMP = re.compile(r"\b(?:temp|temperature)\s*[:\-]?\s*(\d{2,3}(?:\.\d)?)\s*(?:c|°c|f|°f)?\b", re.I)

AGE_SEX = re.compile(r"\b(\d{1,3})\s*(?:y\/?o|yo|years?)\s*([MFmf])\b")
AGE_ONLY= re.compile(r"\b(\d{1,3})\s*(?:y\/?o|yo|years?)\b")
NEG     = re.compile(r"\b(denies|no|without|negative for)\b", re.I)

# Tiny starter lexicon (expand later or load from configs/ontology)
SYMPTOMS = [
    "chest pain", "shortness of breath", "sob", "dyspnea",
    "headache", "nausea", "vomiting", "diaphoresis",
    "facial droop", "slurred speech", "weakness", "bleeding", "laceration",
]

def parse_vitals(text: str) -> Dict:
    """Extract obvious vitals; safe defaults; clamp SpO2 quickly."""
    out = {}
    if (m := BP.search(text)):   out["sbp_mmHg"], out["dbp_mmHg"] = int(m.group(1)), int(m.group(2))
    if (m := HR.search(text)):   out["hr_bpm"]   = int(m.group(1))
    if (m := RR.search(text)):   out["rr_bpm"]   = int(m.group(1))
    if (m := SPO2.search(text)):
        s = int(m.group(1))
        out["spo2_pct"] = s if 50 <= s <= 100 else None
    if (m := TEMP.search(text)):
        # Baseline: assume Celsius; upgrade later with 'F' detection + conversion.
        out["temp_c"] = float(m.group(1))
    return out

def parse_age_sex(text: str) -> Tuple[int | None, str]:
    """Return (age, sex) with sex in {'male','female','unknown'}."""
    if (m := AGE_SEX.search(text)):
        return int(m.group(1)), ("male" if m.group(2).lower()=="m" else "female")
    if (m := AGE_ONLY.search(text)):
        return int(m.group(1)), "unknown"
    return None, "unknown"

def _negated_window(term: str, text: str, window: int = 30) -> bool:
    """Is a negation token near the term? Crude but effective baseline."""
    low, t = text.lower(), term.lower()
    idx = low.find(t)
    if idx == -1: return False
    span = low[max(0, idx-window): idx+len(t)+window]
    return bool(NEG.search(span))

def parse_symptoms(text: str) -> List[Dict]:
    """Return [{'term':..., 'negated':bool}, ...] for lexicon matches."""
    res, low = [], text.lower()
    for t in SYMPTOMS:
        if t in low:
            res.append({"term": ("shortness of breath" if t=="sob" else t),
                        "negated": _negated_window(t, text)})
    return res

def rule_extract(narrative: str, patient_id: str | None = None) -> Dict:
    """Minimal rules-only extractor producing a dict shaped like your schema."""
    from datetime import datetime, timezone
    age, sex = parse_age_sex(narrative)
    vitals   = parse_vitals(narrative)
    syms     = parse_symptoms(narrative)

    # Heuristic confidence: % vitals present
    keys = ["sbp_mmHg","dbp_mmHg","hr_bpm","rr_bpm","spo2_pct","temp_c"]
    coverage = sum(1 for k in keys if vitals.get(k) is not None)
    conf = round(0.3 + 0.1*coverage, 2)

    return {
        "patient_id": patient_id,
        "encounter_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "demographics": {"age_years": age, "sex": sex, "pregnancy_status": "unknown"},
        "vitals": vitals,
        "key_signs_symptoms": syms,
        "mechanism_of_injury": None,
        "bleeding": {"present":"unknown","estimated_ml":None},
        "airway_breathing_circulation": {"airway":"unknown","breathing":"unknown","circulation":"unknown"},
        "allergies": [],
        "medications": [],
        "comorbidities": [],
        "constraints": {"reliability":"medium","notes":"rule-only baseline"},
        "extraction_confidence": conf
    }


