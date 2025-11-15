# postprocess.py
# --------------
# PURPOSE:
#   Convert raw text (e.g., from LLM) into validated JSON and/or
#   validate dicts produced by rules. Safe single entrypoints help testing.
#
# WHAT TO ADD NEXT:
#   - Robust JSON block extraction from messy model output.
#   - "Repair loop": if validation fails, re-prompt model with error hints.
#   - Unit normalization (°F ➜ °C), categorical coercion.

import json
from typing import Dict
from .schema_models import TriageJSON

def validate_and_normalize(obj: Dict) -> Dict:
    """Run pydantic validation and return a clean plain dict."""
    return TriageJSON.model_validate(obj).model_dump()

def parse_json_from_text(text: str) -> Dict:
    """
    Grab the last JSON object inside a larger text blob.
    NOTE: keep this simple for now; harden with a proper JSON finder later.
    """
    start = text.rfind("{")
    if start == -1:
        raise ValueError("No JSON object found in text.")
    return json.loads(text[start:])
