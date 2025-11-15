# Architecture (high-level)
Narrative → (preparse rules) + (LLM via PyTorch/Transformers) → post-validate (pydantic) → JSON
→ features (per decision contract) → decision tree (scikit-learn).

Key contracts:
- `configs/schema.json`
- `configs/decision_contract.yaml`

# COMMENTS / NEXT STEPS
# - Add a block diagram and module responsibilities.
# - Note the retry/repair loop if JSON validation fails.
