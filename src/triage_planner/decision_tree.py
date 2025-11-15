"""Skeleton for the rule-based triage decision tree.

The EMT/clinician team will own this module. Right now it only documents the
steps you need to take in order to convert the JSON summaries produced by the
LLM into a priority level (1 = treat now, 2 = urgent, 3 = stable, etc.).

Every function contains TODO markers that point to the logic you need to write.
Feel free to replace the plain Python `if/elif` structure with a real decision
-tree library (e.g., scikit-learn) later on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PatientSummary:
    """Typed view of the JSON summary produced by the LLM.

    Only a handful of fields are represented here to keep the example simple.
    Add more attributes (respiratory rate, mental status, allergies, etc.) as we
    finalize the schema.
    """

    patient_uid: str
    age_years: float | None
    blood_type: str | None
    systolic_bp: float | None
    diastolic_bp: float | None
    heart_rate: float | None
    condition_summary: str


def assign_priority(summary: PatientSummary) -> int:
    """Return a priority bucket for the supplied patient summary.

    The logic below is intentionally incomplete. Follow the comments and fill in
    the missing pieces so that the decision tree captures our clinical rules.
    """

    # 1. High-risk vital overrides -------------------------------------------------
    # If any *immediately life-threatening* condition is detected we should short
    # circuit and return priority 1. Examples: systolic BP < 80, heart rate > 140,
    # or keywords like "unresponsive" in the narrative summary. Extend this block
    # with additional guard rails as you meet with clinicians.
    if summary.systolic_bp is not None and summary.systolic_bp < 80:
        return 1
    if summary.heart_rate is not None and summary.heart_rate > 140:
        return 1
    # TODO(team): parse summary.condition_summary for red-flag phrases ("no pulse",
    # "cardiac arrest", "GCS < 8" etc.) and immediately return 1 when found.

    # 2. Moderate concern ----------------------------------------------------------
    # The placeholder below demonstrates how you might fold multiple vitals into a
    # score. Replace with your actual triage flowchart (e.g., START, SALT, ESI).
    moderate_triggers = 0
    if summary.systolic_bp is not None and summary.systolic_bp < 100:
        moderate_triggers += 1
    if summary.heart_rate is not None and summary.heart_rate > 110:
        moderate_triggers += 1
    # TODO(team): add respiratory rate, oxygen saturation, mental status, mechanism
    # of injury, etc. Each finding should increment `moderate_triggers`.

    if moderate_triggers >= 2:
        return 2

    # 3. Stable / routine ----------------------------------------------------------
    # If the patient didn't trip any high-risk or moderate alarms we default to a
    # lower priority. This is where you could branch into additional buckets (3, 4)
    # if you want to model outpatient follow-ups versus transport now.
    return 3


def rank_patients(summaries: List[Dict]) -> List[Dict]:
    """Helper that sorts raw JSON dicts by the priority returned above.

    This is useful when you want to feed the triage output directly into a UI.
    Replace the `TODO` block to convert the dictionaries into `PatientSummary`
    objects (for validation) before scoring them.
    """

    scored: List[Dict] = []
    for payload in summaries:
        # TODO(team): consider validating the schema (missing vitals, wrong types).
        summary = PatientSummary(
            patient_uid=payload["patient_uid"],
            age_years=payload.get("age_years"),
            blood_type=payload.get("blood_type"),
            systolic_bp=payload.get("systolic_bp"),
            diastolic_bp=payload.get("diastolic_bp"),
            heart_rate=payload.get("heart_rate"),
            condition_summary=payload.get("condition", ""),
        )
        priority = assign_priority(summary)
        scored.append({"priority": priority, **payload})

    return sorted(scored, key=lambda item: item["priority"])


__all__ = ["PatientSummary", "assign_priority", "rank_patients"]
