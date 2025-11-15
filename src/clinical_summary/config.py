"""Configuration helpers for clinical summarization experiments.

Dataclasses keep the YAML structure explicit so beginners can see exactly which
knobs are available. Every field includes a sensible default; edit
`config/training.example.yaml` to customize values per experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    """Settings that describe how the dataset should be loaded and prepared."""

    dataset_path: Path
    test_size: float = 0.1  # 10% of the data becomes validation by default
    seed: int = 7  # used for shuffling so runs are reproducible
    max_input_length: int = 1024  # token limit passed to the tokenizer
    max_target_length: int = 256  # controls JSON target length
    few_shot_examples: int = 0  # when >0 include example outputs in the prompt


@dataclass
class ModelConfig:
    """Model specific settings used for fine-tuning."""

    name_or_path: str = "google/flan-t5-base"  # swap this for any free HF model
    tokenizer_name: Optional[str] = None  # override when tokenizer != model


@dataclass
class TrainingConfig:
    """Hyper-parameters for the trainer."""

    output_dir: Path = Path("artifacts/checkpoints")  # git-ignored directory
    num_train_epochs: float = 1.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    logging_steps: int = 50
    save_total_limit: int = 2
    fp16: bool = False  # set True when training on GPUs that support fp16


@dataclass
class ProjectConfig:
    """Top level configuration that groups the other sections."""

    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ProjectConfig":
        """Load a project configuration from a YAML file."""

        raw = yaml.safe_load(Path(path).read_text())
        data_cfg = DataConfig(dataset_path=Path(raw["data"]["dataset_path"]), **{k: v for k, v in raw["data"].items() if k != "dataset_path"})
        model_cfg = ModelConfig(**raw.get("model", {}))
        training_cfg = TrainingConfig(**raw.get("training", {}))
        return cls(data=data_cfg, model=model_cfg, training=training_cfg)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration into a nested dictionary."""

        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
        }


__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "ProjectConfig",
]
