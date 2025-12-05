
"""
End-to-End Triage Pipeline

This module orchestrates the complete pipeline:
1. Load narrative report
2. Preprocess text
3. Extract information using LLM → JSON
4. Apply decision tree → triage priority
5. Return results

What it does:

CONNECTS EVERYTHING - This is the main orchestrator!
Takes narrative → preprocesses → LLM extracts → decision tree predicts → returns priority
Handles single narratives or batch processing
Saves results to JSON files
"""
