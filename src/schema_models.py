# schema_models.py
# ----------------
# PURPOSE:
#   Pydantic v2 models that define/validate the canonical triage JSON.
#   Everything should pass through TriageJSON before use downstream.
#
# WHAT TO ADD NEXT:
#   - More validators (e.g., SBP in [50,260], HR in [20,220]).
#   - Enums for categorical fields (ABC statuses).
#   - Optional: add schema_version and validate against configs/schema.json.

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator

class Symptom(BaseModel):
    term: str
    onset_minutes: Optional[int] = None
    severity: Optional[str] = None
    negated: bool = False

class Demographics(BaseModel):
    age_years: Optional[int] = None
    sex: Optional[Literal["male", "female", "unknown"]] = "unknown"
    pregnancy_status: Optional[Literal["yes", "no", "unknown"]] = "unknown"

class Vitals(BaseModel):
    sbp_mmHg: Optional[int] = None
    dbp_mmHg: Optional[int] = None
    hr_bpm: Optional[int] = None
    rr_bpm: Optional[int] = None
    temp_c: Optional[float] = None
    spo2_pct: Optional[int] = None
    gcs_total: Optional[int] = None

    @field_validator("spo2_pct")
    @classmethod
    def clamp_spo2(cls, v):
        # Keep plausible range; null out nonsense values early.
        if v is None: return v
        return v if 50 <= v <= 100 else None

    @field_validator("sbp_mmHg", "dbp_mmHg", "hr_bpm", "rr_bpm", "gcs_total")
    @classmethod
    def nonnegative(cls, v):
        if v is None: return v
        return v if v >= 0 else None

class ABC(BaseModel):
    airway: Optional[Literal["patent", "compromised", "unknown"]] = "unknown"
    breathing: Optional[Literal["adequate", "inadequate", "unknown"]] = "unknown"
    circulation: Optional[Literal["stable", "unstable", "unknown"]] = "unknown"

class TriageJSON(BaseModel):
    patient_id: Optional[str] = None
    encounter_time: Optional[str] = None
    demographics: Demographics = Field(default_factory=Demographics)
    vitals: Vitals = Field(default_factory=Vitals)
    key_signs_symptoms: List[Symptom] = Field(default_factory=list)
    mechanism_of_injury: Optional[str] = None
    bleeding: dict = Field(default_factory=lambda: {"present":"unknown","estimated_ml":None})
    airway_breathing_circulation: ABC = Field(default_factory=ABC)
    allergies: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    comorbidities: List[str] = Field(default_factory=list)
    constraints: dict = Field(default_factory=lambda: {"reliability":"medium","notes":""})
    extraction_confidence: Optional[float] = None
