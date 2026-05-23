# Image Schema LLM Project

Structured image-schema prompting for evaluating LLM interpretation of literal and metaphorical spatial language.

This repository implements a restartable experimental pipeline for comparing **naïve**, **direct-schema**, and **structured role-based** prompts across multiple LLM providers. The system runs a controlled sentence corpus through a full grid of:

```text
model × prompt × condition × sentence × repetition
```

It records raw model responses, tracks runtime cost, supports checkpoint/restart behaviour, parses structured outputs into analysis-ready data, and provides notebooks for model/prompt comparison.

---

## 1. Research purpose

The project investigates whether image-schema prompting can provide a useful **intermediate representational layer** for LLM-based semantic interpretation.

The project is not designed to prove that LLMs possess embodied cognition. Instead, it tests whether theoretically motivated image-schema categories make model interpretations more:

- explicit;
- comparable;
- parseable;
- explainable;
- measurable across literal, metaphorical, and weak-schema control sentences.

The central comparison is between ordinary paraphrase and structured semantic analysis. A naïve prompt may produce a good interpretation of a sentence, but structured prompting asks whether the model can also recover image-schema labels, schematic roles, and source–target domain structure.

---

## 2. Repository structure

```text
image_schema_llm_project/
├── README.md
├── .gitignore
├── pyproject.toml
├── requirements.txt
│
├── data/
│   ├── inputs/
│   │   ├── prompts.jsonl
│   │   ├── conditions.jsonl
│   │   ├── models.jsonl
│   │   └── runtime_config.json
│   │
│   ├── gold/
│   │   ├── sentences_v1.jsonl
│   │   └── sentences_summary.txt
│   │
│   └── outputs/
│       ├── raw_responses.jsonl
│       ├── parsed_responses.jsonl
│       ├── run_log.jsonl
│       ├── cost_log.jsonl
│       ├── cost_summary.json
│       ├── errors.jsonl
│       ├── dry_run_manifest.jsonl
│       ├── dry_run_summary.json
│       └── experiment_manifest.jsonl
│
├── notebooks/
│   ├── 01_parse_quality_and_dataset_coverage.ipynb
│   ├── 02_schema_accuracy_by_sentence_type_and_prompt.ipynb
│   └── 03_tutor_response_evidence_notebook.ipynb
│
├── scripts/
│   ├── validate_inputs.py
│   ├── inspect_inputs.py
│   ├── preview_experiment_grid.py
│   ├── write_experiment_manifest.py
│   ├── dry_run_experiment.py
│   ├── run_provider_manifest.py
│   ├── run_openai_next_job.py
│   ├── run_claude_next_job.py
│   ├── run_gemini_next_job.py
│   ├── parse_responses.py
│   ├── inspect_parsed_responses.py
│   ├── diagnose_parse_errors.py
│   ├── inspect_costs.py
│   ├── rebuild_cost_summary.py
│   └── cleanup_redundant_files.py
│
├── src/
│   └── image_schema_llm/
│       ├── __init__.py
│       ├── config.py
│       ├── schemas.py
│       ├── experiment_grid.py
│       ├── manifest_runner.py
│       ├── checkpoint.py
│       ├── cost_tracker.py
│       ├── runtime_config.py
│       ├── structured_output.py
│       ├── jsonl_utils.py
│       │
│       ├── clients/
│       │   ├── __init__.py
│       │   ├── base_client.py
│       │   ├── openai_client.py
│       │   ├── claude_client.py
│       │   └── gemini_client.py
│       │
│       ├── prompts/
│       │   ├── __init__.py
│       │   └── prompt_builder.py
│       │
│       ├── parsing/
│       │   ├── __init__.py
│       │   └── response_parser.py
│       │
│       └── analysis/
│           ├── __init__.py
│           └── basic_metrics.py
│
└── tests/
    ├── test_basic_imports.py
    ├── test_checkpoint.py
    ├── test_cost_tracker.py
    ├── test_cost_tracker_runtime.py
    ├── test_data_aware_parser.py
    ├── test_dry_run.py
    ├── test_manifest_runner_imports.py
    ├── test_run_keys.py
    ├── test_runtime_config.py
    └── test_structured_output.py
```

---

## 3. Installation

Create or activate your virtual environment, then install the package in editable mode:

```bash
python -m pip install -e '.[dev]'
```

The `-e` flag installs the project in editable mode, so local code changes under `src/image_schema_llm/` are picked up without reinstalling. The `[dev]` extra installs development tools such as `pytest`, if defined in `pyproject.toml`.

Check the installation:

```bash
python -c "import image_schema_llm; print(image_schema_llm.__file__)"
python -m pytest
```

---

## 4. API keys

Set provider API keys before running model calls:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
```

For local development, you may prefer to store these in your shell profile or a local `.env` file. Do not commit API keys to Git.

---

## 5. Input files

The experiment is driven by JSONL input files.

### `data/gold/sentences_v1.jsonl`

Human-validated gold-standard sentence records.

Typical fields include:

```text
sentence_id
text
sentence_type
expected_schema_primary
expected_schema_secondary
expected_literal_or_metaphorical
source_domain
target_domain
annotation_status
annotator_id
validated_at
annotation_guideline_version
```

### `data/inputs/prompts.jsonl`

Prompt definitions for the three prompt families:

```text
naive
direct_schema
structured_role_based
```

Structured prompts may include `recommended_max_output_tokens` to avoid truncated JSON responses.

### `data/inputs/conditions.jsonl`

Inference conditions, including temperature, top-p, maximum output tokens, and repetitions.

### `data/inputs/models.jsonl`

Model/provider configuration, API capability flags, and pricing metadata.

### `data/inputs/runtime_config.json`

Runtime controls and global budget threshold.

Example:

```json
{
  "spend_threshold": 10.0,
  "currency": "USD",
  "stop_on_error": false,
  "dry_run": false,
  "cost_log_filename": "cost_log.jsonl",
  "cost_summary_filename": "cost_summary.json"
}
```

---

## 6. Validate inputs and preview the grid

Run these commands before making API calls:

```bash
python scripts/validate_inputs.py --project-root .
python scripts/inspect_inputs.py --project-root .
python scripts/preview_experiment_grid.py --project-root . --limit 10
```

Write the full experiment manifest:

```bash
python scripts/write_experiment_manifest.py --project-root .
```

The manifest is written to:

```text
data/outputs/experiment_manifest.jsonl
```

---

## 7. Checkpoint and restart behaviour

Successful raw responses are treated as completed jobs. Failed jobs remain pending unless later completed successfully.

Important behaviour:

- successful `run_key`s are skipped on restart;
- failed runs remain pending;
- spend-threshold stops do not mark jobs complete;
- checkpoint state can be regenerated from the current grid and raw responses;
- `raw_responses.jsonl` is the main source of truth for completed runs.

Useful commands:

```bash
python scripts/inspect_checkpoint.py --project-root .
python scripts/inspect_checkpoint.py --project-root . --write-state
python scripts/preview_pending_jobs.py --project-root . --limit 10
python scripts/reset_checkpoint.py --project-root .
python scripts/reset_checkpoint.py --project-root . --execute
```

---

## 8. Dry-run mode

Dry-run mode builds the experiment plan without making API calls.

```bash
python scripts/dry_run_experiment.py --project-root . --limit 10
python scripts/dry_run_experiment.py --project-root . --write-manifest --write-summary
python scripts/dry_run_experiment.py --project-root . --pending-only --limit 10
```

Dry-run outputs:

```text
data/outputs/dry_run_manifest.jsonl
data/outputs/dry_run_summary.json
```

Use dry-run mode to check:

- the number of planned jobs;
- estimated input/output tokens;
- estimated cost;
- pending versus completed jobs;
- prompt rendering;
- whether the manifest matches the intended experimental design.

---

## 9. Run model jobs

### Run one pending job per provider

```bash
python scripts/run_openai_next_job.py --project-root .
python scripts/run_claude_next_job.py --project-root .
python scripts/run_gemini_next_job.py --project-root .
```

### Dry-run one pending job

```bash
python scripts/run_openai_next_job.py --project-root . --dry-run
python scripts/run_claude_next_job.py --project-root . --dry-run
python scripts/run_gemini_next_job.py --project-root . --dry-run
```

### Run a specific job

```bash
python scripts/run_openai_next_job.py --project-root . \
  --run-key "openai_gpt_5_4_mini|p_naive_v1|c_temp_0_v1|s0001|0"
```

Successful responses are written to:

```text
data/outputs/raw_responses.jsonl
```

Cost records are written to:

```text
data/outputs/cost_log.jsonl
```

---

## 10. Run a provider manifest

Use `run_provider_manifest.py` to run all pending jobs for a selected provider.

### Pilot run

```bash
python scripts/run_provider_manifest.py --project-root . --provider openai --execute --max-jobs 20 --stop-on-error
```

### Full OpenAI run

```bash
python scripts/run_provider_manifest.py --project-root . --provider openai --execute --stop-on-error
```

### Full Claude run

```bash
python scripts/run_provider_manifest.py --project-root . --provider anthropic --execute --stop-on-error
```

### Full Gemini run

```bash
python scripts/run_provider_manifest.py --project-root . --provider google --execute --stop-on-error
```

### Rate-limit safety

```bash
python scripts/run_provider_manifest.py --project-root . --provider anthropic --execute --sleep-seconds 1 --stop-on-error
```

Completed `run_key`s are skipped automatically on restart.

---

## 11. Cost tracking

Inspect current estimated spend:

```bash
python scripts/inspect_costs.py --project-root .
```

Write or rebuild the cost summary:

```bash
python scripts/inspect_costs.py --project-root . --write-summary
python scripts/rebuild_cost_summary.py --project-root .
```

Run cost tests:

```bash
python -m pytest tests/test_cost_tracker_runtime.py
```

The default runtime budget is stored in:

```text
data/inputs/runtime_config.json
```

---

## 12. Parsing pipeline

The parser converts raw LLM responses into analysis-ready records.

```bash
python scripts/parse_responses.py --project-root .
python scripts/inspect_parsed_responses.py --project-root .
python scripts/diagnose_parse_errors.py --project-root . --limit 20
```

Input:

```text
data/outputs/raw_responses.jsonl
```

Output:

```text
data/outputs/parsed_responses.jsonl
```

The parser exists to avoid treating raw LLM text as clean data. It:

- preserves raw model evidence;
- extracts structured fields;
- normalises inconsistent model output;
- preserves parse failures;
- supports partial recovery from truncated JSON;
- prepares data for scoring and statistical comparison.

The current parser includes data-aware recovery fields:

```text
parse_status
parse_quality
usable_for_schema_accuracy
usable_for_lm_accuracy
parser_strategy
```

Use:

```text
usable_for_schema_accuracy == true
```

for primary schema metrics, and:

```text
usable_for_lm_accuracy == true
```

for literal/metaphorical metrics.

---

## 13. Analysis notebooks

Run the parser first:

```bash
python scripts/parse_responses.py --project-root .
```

Then open the notebooks in order:

```text
notebooks/01_parse_quality_and_dataset_coverage.ipynb
notebooks/02_schema_accuracy_by_sentence_type_and_prompt.ipynb
notebooks/03_tutor_response_evidence_notebook.ipynb
```

The notebooks support analysis of:

- parse quality;
- primary schema accuracy;
- literal/metaphorical classification accuracy;
- prompt-family differences;
- provider/model differences;
- schema-family difficulty;
- naïve paraphrase versus structured image-schema analysis.

The third notebook supports a tutor-facing interpretation: the project is not claiming that LLMs have embodied cognition, but testing whether image-schema prompting gives a useful intermediate representation beyond ordinary paraphrase.

---

## 14. Raw data quality improvements

Structured prompts should use provider-native JSON controls where possible.

Current design principles:

- OpenAI: use JSON schema structured output where supported;
- Gemini: use `response_mime_type="application/json"` and compatible `response_schema`;
- Claude: use strict prompt-level JSON instructions and avoid sending both `temperature` and `top_p`;
- direct-schema prompts use larger output budgets than naïve prompts;
- structured-role prompts use the largest output budgets;
- raw responses store finish/stop reason metadata where available.

Patch prompt/model configuration if needed:

```bash
python scripts/patch_prompts_for_raw_quality.py --project-root .
python scripts/patch_models_for_raw_quality.py --project-root .
```

Apply patches:

```bash
python scripts/patch_prompts_for_raw_quality.py --project-root . --execute
python scripts/patch_models_for_raw_quality.py --project-root . --execute
```

---

## 15. Reset and rerun

To rerun the entire manifest from scratch, archive or remove the output files that mark jobs complete.

Recommended archive:

```bash
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p data/outputs/archive/run_$timestamp
mv data/outputs/*.jsonl data/outputs/*.json data/outputs/*.csv data/outputs/archive/run_$timestamp/ 2>/dev/null
mkdir -p data/outputs
```

Minimum reset:

```bash
rm -f data/outputs/raw_responses.jsonl
rm -f data/outputs/parsed_responses.jsonl
rm -f data/outputs/run_log.jsonl
rm -f data/outputs/errors.jsonl
rm -f data/outputs/cost_log.jsonl
rm -f data/outputs/cost_summary.json
```

Do not delete:

```text
data/gold/sentences_v1.jsonl
data/inputs/models.jsonl
data/inputs/prompts.jsonl
data/inputs/conditions.jsonl
data/inputs/runtime_config.json
```

unless you are intentionally changing the experiment design.

---

## 16. Testing

Run all tests:

```bash
python -m pytest
```

Run selected tests:

```bash
python -m pytest tests/test_data_aware_parser.py
python -m pytest tests/test_structured_output.py
python -m pytest tests/test_cost_tracker_runtime.py
```

Compile provider clients after edits:

```bash
python -m py_compile src/image_schema_llm/clients/openai_client.py
python -m py_compile src/image_schema_llm/clients/gemini_client.py
python -m py_compile src/image_schema_llm/clients/claude_client.py
```

---

## 17. Recommended workflow

For a clean experimental run:

```bash
python -m pip install -e '.[dev]'
python -m pytest

python scripts/validate_inputs.py --project-root .
python scripts/inspect_runtime_config.py --project-root .
python scripts/dry_run_experiment.py --project-root . --write-manifest --write-summary

python scripts/run_provider_manifest.py --project-root . --provider openai --execute --stop-on-error
python scripts/run_provider_manifest.py --project-root . --provider anthropic --execute --stop-on-error
python scripts/run_provider_manifest.py --project-root . --provider google --execute --stop-on-error

python scripts/parse_responses.py --project-root .
python scripts/diagnose_parse_errors.py --project-root . --limit 20
python scripts/inspect_parsed_responses.py --project-root .
```

Then proceed to notebooks.

---

## 18. Notes on reproducibility

For reproducibility, preserve:

- `data/inputs/*.jsonl`;
- `data/gold/sentences_v1.jsonl`;
- `data/outputs/raw_responses.jsonl`;
- `data/outputs/parsed_responses.jsonl`;
- `data/outputs/cost_log.jsonl`;
- `data/outputs/run_log.jsonl`;
- the Git commit hash for the run;
- provider model names and snapshots;
- prompt versions;
- condition versions;
- parser version or commit.

The raw responses should always be retained because parsed outputs can be regenerated without making further API calls.

## 19. Making PyCharm Jupyter notebooks run in the same .venv as the main repo
cd /Users/Shared/image_schema_llm_project
source .venv/bin/activate

python -m pip install ipykernel
python -m ipykernel install --user \
  --name image-schema-llm-venv \
  --display-name "Python (.venv image-schema-llm)"

Then in PyCharm:
Notebook toolbar → Kernel selector → Python (.venv image-schema-llm)
