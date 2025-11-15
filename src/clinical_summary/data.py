"""Data loading and preprocessing utilities for the clinical summarization task.

Important: this module intentionally keeps the "boring" plumbing in place so
new contributors can focus on understanding how raw JSON records turn into
prompt/target pairs. Several extension hooks and TODO notes are sprinkled
throughout the file.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from datasets import Dataset, DatasetDict

from .config import DataConfig
from .prompts import build_input_prompt, build_target_summary


@dataclass
class PreparedExample:
    """Single training example containing an input prompt and target text."""

    uid: str
    input_text: str
    target_text: str


def _coerce_json_array(text: str) -> str:
    """Convert loose JSON objects separated by commas into a valid array string."""

    stripped = text.strip()
    if not stripped.startswith("["):
        stripped = "[" + stripped
    stripped = stripped.rstrip(",\n \t") + "]"
    return stripped


def load_patient_records(dataset_path: Path | str) -> List[Dict]:
    """Load the dataset file and return a list of patient dictionaries.

    TODO(team): if we later ingest the PhysioNet CSVs, add a branch that detects
    `.csv` and uses `pandas.read_csv`. For now we assume newline-delimited JSON.
    """

    raw_text = Path(dataset_path).read_text(encoding="utf-8")
    data = json.loads(_coerce_json_array(raw_text))
    if not isinstance(data, list):  # pragma: no cover - sanity guard
        raise ValueError("Dataset is expected to be a JSON array of patient records")
    return data


def prepare_examples(records: Sequence[Dict], cfg: DataConfig) -> List[PreparedExample]:
    """Transform patient records into prompt/target pairs.

    Notice how the helper defers all formatting details to `prompts.py`. Keeping
    the logic centralized makes it easier to iterate on the schema without
    touching data loading.
    """

    examples: List[PreparedExample] = []
    for row in records:
        uid = row.get("patient_uid") or row.get("patient_id")
        if not uid:
            continue  # skip malformed
        input_text = build_input_prompt(row, few_shot_limit=cfg.few_shot_examples)
        target_text = build_target_summary(row)
        examples.append(PreparedExample(uid=uid, input_text=input_text, target_text=target_text))
    return examples


def build_hf_dataset(records: Sequence[Dict], cfg: DataConfig) -> DatasetDict:
    """Create a :class:`datasets.DatasetDict` with train/test splits.

    The actual splitting logic remains implemented because it is mostly boiler-plate.
    The interesting work—feature engineering and prompt tuning—happens earlier.
    Still, there are clear extension points noted below.
    """

    examples = prepare_examples(records, cfg)
    random.Random(cfg.seed).shuffle(examples)
    if not examples:
        raise ValueError("No training examples were generated. Check the dataset format.")

    split_index = max(1, math.floor((1 - cfg.test_size) * len(examples)))
    train_examples = examples[:split_index]
    eval_examples = examples[split_index:]

    def _to_dicts(items: Iterable[PreparedExample]) -> List[Dict[str, str]]:
        return [{"uid": ex.uid, "input_text": ex.input_text, "target_text": ex.target_text} for ex in items]

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(_to_dicts(train_examples)),
            "validation": Dataset.from_list(_to_dicts(eval_examples)),
        }
    )

    # TODO(team): consider logging a few random samples here to manually sanity
    # check prompts before starting a long training job.
    return dataset_dict


__all__ = [
    "PreparedExample",
    "load_patient_records",
    "prepare_examples",
    "build_hf_dataset",
]
