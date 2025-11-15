"""Command line utility to fine-tune a seq2seq model on the patient dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .config import ProjectConfig
from .data import build_hf_dataset, load_patient_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and tokenize the dataset without launching training (useful for smoke tests).",
    )
    return parser.parse_args()


def _tokenize_function(tokenizer, max_input: int, max_target: int):
    def _tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_input,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                max_length=max_target,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _tokenize


def main() -> None:  # pragma: no cover - entry point with I/O
    args = parse_args()
    cfg = ProjectConfig.from_yaml(args.config)
    records = load_patient_records(cfg.data.dataset_path)
    dataset_dict = build_hf_dataset(records, cfg.data)

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    tokenizer_name = cfg.model.tokenizer_name or cfg.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized_dataset = dataset_dict.map(
        _tokenize_function(tokenizer, cfg.data.max_input_length, cfg.data.max_target_length),
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
    )

    if args.dry_run:
        print(f"Prepared {len(dataset_dict['train'])} train and {len(dataset_dict['validation'])} validation rows.")
        return

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name_or_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = TrainingArguments(
        output_dir=str(cfg.training.output_dir),
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        weight_decay=cfg.training.weight_decay,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_steps=cfg.training.warmup_steps,
        logging_steps=cfg.training.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.training.save_total_limit,
        predict_with_generate=True,
        fp16=cfg.training.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(cfg.training.output_dir))


if __name__ == "__main__":  # pragma: no cover
    main()
