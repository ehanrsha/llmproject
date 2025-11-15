"""Prompt building utilities.

The existing helpers stay mostly intact so newcomers can see a reference
implementation. Comments highlight where custom logic should be inserted (for
example, adding more structured fields or plugging in a better summarizer).
"""
from __future__ import annotations

import json
import textwrap
from typing import Dict, List, Optional


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a clinical data summarization model. Given a patient narrative, extract the key
    demographic features and produce a concise clinical condition summary. Your output must be a
    valid JSON object with the following keys: patient_uid, age_years, blood_type, condition.
    """
).strip()


def _coerce_age_years(record: Dict) -> Optional[float]:
    """Extract the numeric age value from the structured field when present."""

    age_field = record.get("age") or []
    if isinstance(age_field, (list, tuple)) and age_field:
        value = age_field[0][0]
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - guard against corrupt data
            return None
    return None


def build_input_prompt(record: Dict, few_shot_limit: int = 0) -> str:
    """Create the instruction-style prompt that the seq2seq model will consume.

    Feel free to heavily customize this function. Examples of future tweaks:
    * prepend triage-specific reminders (e.g., "highlight vital instability")
    * add few-shot exemplars that show the expected JSON output style
    * include additional structured vitals from the dataset once available
    """

    parts: List[str] = [SYSTEM_PROMPT, "Patient narrative:"]
    if few_shot_limit > 0:
        parts.append(
            "Example output format: {\"patient_uid\": \"123\", \"age_years\": 45, \"blood_type\": null, "
            "\"condition\": \"Concise summary here.\"}"
        )
    structured = []
    if record.get("gender"):
        structured.append(f"Gender: {record['gender']}")
    age_years = _coerce_age_years(record)
    if age_years is not None:
        structured.append(f"Age: {int(age_years)} years")
    if record.get("pub_date"):
        structured.append(f"Publication year: {record['pub_date']}")
    if structured:
        parts.append("Known structured fields:\n" + "\n".join(structured))
    patient_text = record.get("patient", "").strip()
    parts.append(patient_text)
    return "\n\n".join(parts)


def _summarize_condition_text(text: str, max_sentences: int = 3) -> str:
    """Naive summarization by clipping the narrative.

    TODO(team): replace this logic with whichever approach you prefer:
    - keyword extraction (e.g., YAKE, keyBERT)
    - calling an open LLM zero-shot (if allowed)
    - manual bullet points curated by clinicians
    For now we keep a deterministic implementation so unit tests / dry-runs work.
    """

    # This keeps the framework lightweight while still producing deterministic targets. In a
    # production system this function can be replaced with a keyword extractor or clinician
    # provided notes.
    normalized = text.replace("\n", " ").strip()
    sentences: List[str] = []
    for chunk in normalized.split(". "):
        chunk = chunk.strip()
        if not chunk:
            continue
        sentences.append(chunk.rstrip("."))
        if len(sentences) >= max_sentences:
            break
    summary = ". ".join(sentences)
    if summary and not summary.endswith("."):
        summary += "."
    return summary or normalized[:256]


def build_target_summary(record: Dict) -> str:
    """Create the JSON target string with the extracted key points.

    The payload currently mirrors the simple example shared by the team. When we
    decide to include blood pressure, visual descriptions, etc., expand this
    dictionary and update the README/decision tree to match.
    """

    payload = {
        "patient_uid": record.get("patient_uid") or record.get("patient_id"),
        "age_years": _coerce_age_years(record),
        "blood_type": record.get("blood_type"),
        "condition": _summarize_condition_text(record.get("patient", "")),
    }
    return json.dumps(payload, ensure_ascii=False)


__all__ = [
    "SYSTEM_PROMPT",
    "build_input_prompt",
    "build_target_summary",
]
