# Triage Planner Framework

We are building **two complementary components** that work together to help EMT crews and hospital
staff triage patients faster:

1. **LLM Summarizer** – ingests narrative reports (for example the EMT notes listed in
   [PREP-5004 examples](https://www.vdh.virginia.gov/content/uploads/sites/23/2016/05/PREP-5004Examples.pdf))
   plus structured vitals (age, blood type, etc.) and turns them into a consistent JSON summary.
2. **Decision Tree Prioritizer** – consumes the JSON summaries and produces a priority score or
   label that indicates who should be treated first.

> 💡 **Goal for this commit:** provide a _beginner-friendly scaffold_ with very explicit comments and
> TODO markers so new teammates can fill in the important pieces themselves.

## Repository layout

```
.
├── config/
│   └── training.example.yaml   # Annotated template for LLM training hyper-parameters
├── dataset/                    # Raw clinical JSON dumps (needs review/cleanup by the team)
├── samples/patients.sample.json# Mini file to test the pipeline manually
├── src/
│   ├── clinical_summary/       # LLM summarizer package (heavily commented scaffold)
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── prompts.py
│   │   ├── training.py
│   │   └── inference.py
│   └── triage_planner/
│       └── decision_tree.py    # Placeholder for the rule-based/decision-tree prioritizer
└── README.md                   # This guide (plan, task breakdown, onboarding steps)
```

Person A: Ehan Shah /n
Person B: Raaghava Deepak /n
Person C: David Castellanos /n
Person D: Dhruv Palli /n

## Beginner-friendly onboarding plan

The work is intentionally split so at least two people can collaborate without stepping on each
other. Each task references the module(s) involved and notes what is already prepared versus what
still needs to be coded.

| Workstream | Sub-task | Owner suggestion | Status | Notes |
|------------|----------|------------------|--------|-------|
| **LLM Summarizer** | Review/clean the raw `dataset/*.json` files | Person A | ⬜️ TODO | Make sure every file is a valid JSON array and redact PHI if necessary. |
| | Implement prompt+target builders inside `src/clinical_summary/prompts.py` | Person A | ⬜️ TODO | Skeleton + docstrings exist; fill in `_summarize_condition_text` with real logic. |
| | Finish `load_patient_records` + `build_hf_dataset` in `data.py` | Person B | ⬜️ TODO | Current file explains each step and where to insert code. |
| | Complete the training CLI in `training.py` | Person B | ⬜️ TODO | Use Hugging Face `Trainer`; the comments outline the flow. |
| | Complete inference CLI in `inference.py` | Person B | ⬜️ TODO | Should mirror training tokenizer/model loading. |
| | Experiment tracking + evaluation metrics | Person C | ⬜️ TODO | See TODO in `training.py` for hooking custom metrics. |
| **Decision Tree Prioritizer** | Define triage criteria with clinical lead | Person D | ⬜️ TODO | E.g., unstable vitals > allergies > transport time. Document in README once finalized. |
| | Implement priority scoring in `triage_planner/decision_tree.py` | Person D | ⬜️ TODO | Function `assign_priority` currently returns `NotImplementedError`. |
| | Connect LLM output JSON to the decision tree | Person A + D | ⬜️ TODO | Determine shared schema (see `samples/patients.sample.json`). |
| **Project Ops** | Document experiments + share checkpoints | Rotating | ⬜️ TODO | Use `/artifacts` folder (not tracked) per config template. |

## How everything fits together

```
Narrative report + vitals
          │
          ▼
  clinical_summary.prompts  (format instruction + desired JSON fields)
          │
          ▼
  clinical_summary.data     (convert dataset to Hugging Face DatasetDict)
          │
          ▼
  clinical_summary.training (fine-tune FLAN-T5 or any free seq2seq model)
          │
          ▼
  clinical_summary.inference (produce JSON summaries for new patients)
          │
          ▼
triage_planner.decision_tree (score priority / produce flowchart result)
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # transformers, datasets, etc.
export PYTHONPATH=src  # so "python -m clinical_summary.training" works
```

The Python dependencies are already listed, but installation may fail in restricted environments.
If that happens, try running on a local machine or Colab where Hugging Face downloads are allowed.

## Implementation roadmap (high level)

1. **Data audit (week 1):** confirm file formats, remove corrupted rows, and list mandatory fields
   (age, blood type, blood pressure, mental status, etc.). Update `samples/patients.sample.json`
   when you find additional attributes so the inference example stays realistic.
2. **Prompt/target iteration (week 1–2):** experiment with how the prompt is phrased and which
   fields we expect in the JSON output. Keep instructions short for small models (FLAN-T5) and add
   more detail for larger models (Llama 3, Mixtral, etc.).
3. **Fine-tuning loop (week 2):** finish `training.py`, run a dry-run (`--dry-run` only loads data),
   then execute a full training job when GPU time is available. Save checkpoints under
   `artifacts/<run-name>` (folder is ignored by git so add a README there for context).
4. **Decision tree prototype (week 2):** codify your triage flowchart in `decision_tree.py`. Start
   simple (e.g., unstable vitals = priority 1) and refine as you gather real EMT feedback.
5. **Integration test (week 3):** feed LLM-generated JSON into the decision tree and compare the
   automated priority order against clinician expectations. Adjust either component as needed.

## Helpful tips embedded in the code

- Every Python file inside `src/` now contains **big block comments** that explain _why_ a function
  exists and mark exactly where to add logic. Search for `TODO(team)` to find your assignments.
- If you are new to Hugging Face, read the inline comments in `training.py` before touching the code.
  They walk you through tokenizer loading, dataset mapping, and the trainer loop.
- The decision tree file gives a plain-English recipe for how to transform vitals (SBP/DBP, heart
  rate, Glasgow Coma Score, etc.) into a simple score. You can start with `if/elif` statements or
  use `sklearn.tree.DecisionTreeClassifier` later.

## Next steps checklist

- [ ] Confirm raw data sources (Virginia EMT narratives + Physionet MIETIC) are downloaded into
      `dataset/`.
- [ ] Decide on the **minimum viable JSON schema** (fields and types) for the LLM output.
- [ ] Finish the `TODO` blocks in `src/clinical_summary/*.py` and `src/triage_planner/decision_tree.py`.
- [ ] Run the dry-run command to ensure data loading works before launching expensive training jobs.
- [ ] Pair the LLM predictions with the decision tree scoring logic to produce a ranked patient list.

Once these boxes are checked we can worry about evaluation metrics, model deployment, and UI/UX.













System Architecture:
EMS Narrative Report (Text)
         ↓
    Preprocessing (Clean text, remove PHI)
         ↓
    LLM Extraction (Extract structured JSON)
         ↓
    Decision Tree (Predict priority 1-5)
         ↓
    Triage Priority Assignment
🏥 Triage Priority Levels

Immediate - Life-threatening, needs immediate intervention
Emergent - Serious but stable, needs prompt care
Urgent - Stable but needs medical care
Less Urgent - Minor injury, can wait
Non-Urgent - Can wait extended period

🚀 Quick Start
Installation
bash# Clone repository
git clone https://github.com/ehanrsha/llmproject.git
cd llmproject

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Basic Usage
bash# Run inference on a single narrative
python scripts/run_inference.py --input data/raw/narrative.txt

# Process all narratives in a directory
python scripts/run_inference.py --input-dir data/raw/narratives/

# Quick test with text
python scripts/run_inference.py --text "45 y/o male with chest pain, BP 160/95, HR 110..."
Run Tests
bash# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest --cov=src tests/
📁 Project Structure
llmproject/
├── config/                          # Configuration files
│   ├── training.example.yaml        # Training configuration
│   ├── llm_config.yaml             # LLM model settings
│   ├── data_config.yaml            # Data paths and preprocessing
│   └── decision_tree_config.yaml   # Decision tree settings
│
├── data/                           # Data at various stages
│   ├── raw/                        # Original narrative reports
│   ├── processed/                  # Cleaned narratives
│   ├── json_outputs/               # LLM-generated JSON
│   └── splits/                     # Train/val/test splits
│
├── models/                         # Trained models
│   ├── checkpoints/                # LLM model weights
│   └── decision_tree/              # Decision tree models
│
├── src/                           # Source code
│   ├── data/                      # Data handling
│   │   ├── data_loader.py         # Load MIMIC-III and other datasets
│   │   ├── preprocessor.py        # Clean and normalize text
│   │   └── json_schema.py         # Define JSON extraction structure
│   │
│   ├── clinical_summary/          # LLM component
│   │   ├── config.py              # LLM configuration
│   │   ├── data.py                # LLM data preparation
│   │   ├── inference.py           # Run LLM extraction
│   │   ├── prompts.py             # Prompt templates
│   │   └── training.py            # LLM training
│   │
│   ├── triage_planner/            # Decision tree component
│   │   └── decision_tree.py       # Decision tree logic
│   │
│   ├── utils/                     # Utilities
│   │   ├── logging.py             # Logging setup
│   │   └── metrics.py             # Evaluation metrics
│   │
│   └── pipeline/                  # End-to-end orchestration
│       └── end_to_end.py          # Connect LLM + decision tree
│
├── scripts/                       # Executable scripts
│   ├── train_llm.py              # Train LLM (TODO)
│   ├── train_tree.py             # Train decision tree (TODO)
│   └── run_inference.py          # Run inference on narratives
│
├── tests/                        # Unit tests
│   └── test_pipeline.py          # Test end-to-end pipeline
│
├── notebooks/                    # Jupyter notebooks (optional)
├── docs/                         # Documentation
├── samples/                      # Sample data
│   └── patients.sample.json     # Example JSON structure
├── .gitignore
├── README.md
├── requirements.txt
└── pyproject.toml
🔧 Components
1. Data Processing (src/data/)
Data Loader (data_loader.py)

Loads narrative reports from MIMIC-III dataset
Supports text files, JSON/JSONL formats
Creates train/validation/test splits

Preprocessor (preprocessor.py)

Cleans narrative text
Removes Protected Health Information (PHI)
Normalizes whitespace and formatting
Filters by length

JSON Schema (json_schema.py)

Defines structured output format for LLM
Patient info, vitals, symptoms, medical history
Validation and serialization utilities

2. LLM Component (src/clinical_summary/)
Extracts structured information from narrative text:

Patient demographics (age, sex, weight)
Vital signs (BP, HR, RR, SpO2, temp)
Symptoms and complaints
Medical history
Incident details
Severity indicators

3. Decision Tree Component (src/triage_planner/)
Uses extracted JSON to predict triage priority:

Takes features from JSON (vitals, symptoms, severity flags)
Applies decision tree classification
Returns priority level (1-5)

4. Pipeline (src/pipeline/)
Orchestrates the complete workflow:

Text preprocessing
LLM extraction
Feature engineering
Priority prediction
Result formatting

5. Utilities (src/utils/)
Logging (logging.py)

Setup logging for training and inference
Track metrics and progress

Metrics (metrics.py)

Accuracy, precision, recall, F1 score
Confusion matrix
Critical error rate
Weighted accuracy
LLM extraction accuracy

📊 Data Sources
MIMIC-III Dataset

Clinical notes from ICU admissions
Used for training LLM and decision tree
Access: https://physionet.org/content/mimiciii/

EMS Narrative Examples

Sample reports: https://www.vdh.virginia.gov/content/uploads/sites/23/2016/05/PREP-5004Examples.pdf
