# features.py
# -----------
# PURPOSE:
#   Map validated JSON to a fixed-order numeric vector per decision contract.
#   Keep this in sync with configs/decision_contract.yaml.
#
# WHAT TO ADD NEXT:
#   - Derived flags (e.g., hypotension_flag, hypoxia_flag) if your contract includes them.
#   - Output DataFrame/CSV helpers and schema assertions (length/order).

from typing import List, Dict

# Mirror of configs/decision_contract.yaml for quick import in code paths
FEATURE_ORDER = [
    "age_years",
    "sbp_mmHg", "dbp_mmHg",
    "hr_bpm", "rr_bpm",
    "spo2_pct", "temp_c",
]

def _val(x, default=-1):
    """Use a sentinel for missing numeric values so the tree stays deterministic."""
    return x if x is not None else default

def json_to_features(j: Dict) -> List[float]:
    """Return a list[float] matching FEATURE_ORDER exactly."""
    d, v = j.get("demographics", {}), j.get("vitals", {})
    return [
        _val(d.get("age_years")),
        _val(v.get("sbp_mmHg")), _val(v.get("dbp_mmHg")),
        _val(v.get("hr_bpm")),   _val(v.get("rr_bpm")),
        _val(v.get("spo2_pct")), float(_val(v.get("temp_c"), -1.0)),
    ]
