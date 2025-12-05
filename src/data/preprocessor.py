"""
Text Preprocessor for EMS Narrative Reports

This module handles cleaning, normalization, and preprocessing
of narrative text before it's fed to the LLM.

What it does:

Cleans narrative text before feeding to the LLM
Removes Protected Health Information (PHI) like phone numbers, SSNs, emails, addresses
Normalizes unicode and whitespace
Removes URLs
Filters by length (min/max word counts)
Handles batch processing
"""
