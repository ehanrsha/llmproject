"""Utilities for running inference with a fine-tuned summarization model.

The actual model loading/generation has intentionally been removed so the team
can practice wiring up the tokenizer/model themselves. Use this CLI to build
muscle memory for evaluating checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .prompts import build_input_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=str, help="Fine-tuned model path or HF hub name")
    parser.add_argument("--input", required=True, type=Path, help="Path to a JSON file with one or more patient objects")
    parser.add_argument("--output", type=Path, help="Optional path to write the predictions as JSON lines")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length cap (pass to HF generate)")
    parser.add_argument("--num-beams", type=int, default=4, help="Beam search width during generation")
    return parser.parse_args()


def _load_records(path: Path) -> List[Dict[str, Any]]:
    """Read JSON objects (single dict or list) from disk."""

    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Input file must contain a JSON array or object")
    return data


def _print_instructions() -> None:
    """Reminder for teammates about what needs to be implemented."""

    print(
        "TODO(team): import AutoTokenizer/AutoModelForSeq2SeqLM, load the model, "
        "tokenize prompts, call model.generate, decode, and dump JSON lines."
    )


def main() -> None:  # pragma: no cover - CLI helper
    args = parse_args()
    records = _load_records(args.input)
    prompts = [build_input_prompt(rec) for rec in records]
    _print_instructions()

    # TODO(team): Replace the pseudo-output below with real model generations.
    # The placeholder keeps the CLI usable for debugging file paths / schemas.
    results = []
    for record, prompt in zip(records, prompts):
        results.append(
            {
                "patient_uid": record.get("patient_uid"),
                "prompt_preview": prompt[:120] + ("..." if len(prompt) > 120 else ""),
                "prediction": "TODO: run the fine-tuned model here",
            }
        )

    serialized = "\n".join(json.dumps(row, ensure_ascii=False) for row in results)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
