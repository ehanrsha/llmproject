"""
Run Inference Script

This script runs the end-to-end triage pipeline on new narrative reports.
Can process single files, multiple files, or entire directories.

Usage:
    # Single file
    python scripts/run_inference.py --input data/raw/narrative.txt
    
    # Multiple files
    python scripts/run_inference.py --input file1.txt file2.txt file3.txt
    
    # Entire directory
    python scripts/run_inference.py --input-dir data/raw/narratives/
    
    # With custom output directory
    python scripts/run_inference.py --input narrative.txt --output-dir results/


What it does:

Main command-line script to run triage predictions on new narratives
Supports single files, multiple files, or entire directories
Saves results in JSON and/or TXT format
Provides summary statistics and priority distribution
"""
