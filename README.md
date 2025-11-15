# Clinical Patient Key-Point Summarization

This repository contains a lightweight framework for turning the provided clinical narratives
(`dataset`) into instruction pairs that can be used to fine-tune an open Large Language Model (LLM)
(e.g., [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)) to extract structured key
points such as age, blood type, and a concise description of the patient's condition.

## Repository layout

```
.
├── config/
│   └── training.example.yaml   # Starter hyper-parameter file for fine-tuning runs
├── dataset                     # Raw patient JSON objects shipped with the challenge
├── requirements.txt            # Python dependencies (Transformers, Datasets, etc.)
└── src/clinical_summary/
    ├── __init__.py             # Package metadata stub
    ├── config.py               # Dataclasses for experiment configuration
    ├── data.py                 # Dataset loading + HF Dataset conversion
    ├── inference.py            # CLI helper for running inference
    ├── prompts.py              # Prompt/target formatting utilities
    └── training.py             # Fine-tuning entry point (HF Trainer)
```

## Quickstart

1. **Create a virtual environment** (Python 3.10+ is recommended) and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   export PYTHONPATH=src  # needed so `python -m clinical_summary.*` resolves the package
   ```

2. **Run a smoke test** to verify the dataset can be parsed and tokenized without starting a full
   training job:

   ```bash
   python -m clinical_summary.training --config config/training.example.yaml --dry-run
   ```

   The command prints how many training/validation rows were prepared and exits.

3. **Launch fine-tuning** (requires a GPU or a CPU with sufficient memory). This command trains the
   freely available `google/flan-t5-base` model on the generated instruction pairs and stores the
   resulting checkpoint inside `artifacts/flan-t5-base` (configurable in the YAML file):

   ```bash
   python -m clinical_summary.training --config config/training.example.yaml
   ```

   Adjust batch sizes, `gradient_accumulation_steps`, or switch to another open model by editing the
   configuration file.

4. **Run inference** on new patient records by pointing to the fine-tuned model directory (or any
   compatible Hugging Face model) and a JSON file containing one or multiple patient objects:

   ```bash
   python -m clinical_summary.inference \
       --model artifacts/flan-t5-base \
       --input samples/patients.sample.json \
       --output predictions.jsonl
   ```

   When `--output` is omitted, predictions are printed to stdout. Each prediction is a JSON string
   with the fields `patient_uid`, `age_years`, `blood_type`, and `condition`, matching the target
   format described in the prompt instructions.

## How the framework works

1. **Parsing and cleaning** – `clinical_summary.data.load_patient_records` reads the raw JSON file.
   The helper automatically wraps the loose objects inside brackets to convert the document into a
   valid JSON array, so no manual preprocessing is required.
2. **Prompt/target creation** – `clinical_summary.prompts.build_input_prompt` merges the structured
   metadata (age, gender, publication year) with the free-text narrative and emits an instruction
   string understood by instruction-tuned seq2seq models. The paired target is a JSON payload
   created by `build_target_summary`, which currently applies a deterministic (but easily swappable)
   summarizer that clips the narrative to a few sentences.
3. **Dataset preparation** – `clinical_summary.data.build_hf_dataset` converts the prompt/target
   pairs into a Hugging Face `DatasetDict`, performs shuffling, and creates a validation split.
4. **Fine-tuning** – `clinical_summary.training` wires the dataset into the Hugging Face `Trainer`
   using `AutoModelForSeq2SeqLM` and `AutoTokenizer`. Switching to a different free model is as
   simple as changing the `name_or_path` field in the YAML configuration.
5. **Deployment/inference** – `clinical_summary.inference` loads the saved checkpoint, rebuilds the
   prompts for new records, and generates predictions that follow the required JSON schema.

## Customization guide

- **Model choice:** swap `google/flan-t5-base` in `config/training.example.yaml` for any other free
  seq2seq model (e.g., `google/flan-t5-large`, `facebook/bart-base`, or `tiiuae/falcon-7b-instruct`
  when enough hardware is available).
- **Target engineering:** replace `_summarize_condition_text` in `src/clinical_summary/prompts.py`
  with a more advanced algorithm (keyword extraction, clinician-provided bullet points, etc.) to
  produce higher-quality supervision signals.
- **Additional fields:** extend `build_target_summary` to emit more attributes (blood pressure,
  imaging modality, etc.) and update the system prompt accordingly.
- **Evaluation:** plug the validation split into any preferred evaluation metric (BLEU, Rouge, exact
  JSON match) by enhancing `training.py` with a `compute_metrics` function.

This scaffold keeps the heavy lifting (tokenization, training loop, inference) implemented so that
future improvements can focus on better labeling or model experimentation.
