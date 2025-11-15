"""Command line utility to fine-tune a seq2seq model on the patient dataset.

The previous commit shipped a working Hugging Face training loop. Per the new
requirements we now expose *only the scaffolding* so beginners can practice
implementing the important steps themselves. Every block contains explicit
instructions that describe what needs to happen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import ProjectConfig
from .data import build_hf_dataset, load_patient_records


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Nothing fancy happens here—it's safe for new contributors to modify the CLI
    flags if they want to pass additional knobs to the training loop.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and tokenize the dataset without launching training (useful for smoke tests).",
    )
    return parser.parse_args()


def _print_dataset_stats(dataset_dict) -> None:
    """Utility that prints how many rows live in each split.

    Keeping this helper ensures `--dry-run` still provides fast feedback even
    though the training loop itself is unfinished.
    """

    train_len = len(dataset_dict["train"])
    val_len = len(dataset_dict["validation"])
    print(f"Dataset prepared. train={train_len} validation={val_len}")


def main() -> None:  # pragma: no cover - entry point with I/O
    args = parse_args()
    cfg = ProjectConfig.from_yaml(args.config)

    # STEP 1: Load and split the raw dataset ------------------------------------
    # The helper functions already deal with JSON quirks (dangling commas, etc.).
    # Feel free to extend `load_patient_records` to support CSV or Parquet later.
    records = load_patient_records(cfg.data.dataset_path)
    dataset_dict = build_hf_dataset(records, cfg.data)
    _print_dataset_stats(dataset_dict)

    if args.dry_run:
        # Early exit so contributors can quickly check their data prep changes.
        return

    # STEP 2: Tokenize -----------------------------------------------------------
    # TODO(team):
    #   * Import `AutoTokenizer` from `transformers`.
    #   * Load the tokenizer specified in the config (`cfg.model`).
    #   * Map `dataset_dict` through the tokenizer to produce token IDs.
    #   * Remember to remove the original text columns to keep memory usage low.
    raise NotImplementedError(
        "Tokenization + HF Trainer setup has been intentionally removed. "
        "Follow the comments in training.py to re-implement the steps."
    )

    # STEP 3: Trainer setup ------------------------------------------------------
    # After tokenization is in place, instantiate:
    #   * `AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name_or_path)`
    #   * `TrainingArguments` using the hyper-parameters in `cfg.training`
    #   * `Trainer` with the model, datasets, tokenizer, and data collator
    #   * Call `trainer.train()` and `trainer.save_model(cfg.training.output_dir)`
    # Add custom callbacks / metrics once the basic loop works.


if __name__ == "__main__":  # pragma: no cover
    main()
