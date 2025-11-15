"""Utilities for running inference with a fine-tuned summarization model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .prompts import build_input_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Path to the fine-tuned model or model hub name")
    parser.add_argument("--input", required=True, type=Path, help="Path to a JSON file with one or more patient objects")
    parser.add_argument("--output", type=Path, help="Optional path to write the predictions as JSON lines")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    return parser.parse_args()


def _load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Input file must contain a JSON array or object")
    return data


def main() -> None:  # pragma: no cover - CLI helper
    args = parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.model))

    records = _load_records(args.input)
    prompts = [build_input_prompt(rec) for rec in records]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = []
    for record, text in zip(records, decoded):
        results.append({"patient_uid": record.get("patient_uid"), "prediction": text})
    if args.output:
        args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in results), encoding="utf-8")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
