"""
JSON Schema Definition for EMS Narrative Extraction

This module defines the expected structure of the JSON output
that the LLM should extract from EMS narrative reports.

What it does:

Defines the EXACT structure your LLM should extract
Uses Python dataclasses for type safety
Has helper functions to validate, convert to/from JSON
Includes get_schema_template() that you can use in your LLM prompts

"""
