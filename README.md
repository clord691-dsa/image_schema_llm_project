### REPO FOLDER STRUCTURE

Below is a project framework designed for: structured image-schema prompting for LLM interpretation of literal and metaphorical spatial language. It implements the experimental logic: per LLM × per prompt × per condition × per sentence, with restartable iteration, raw response capture, and cumulative cost monitoring. This fits the Goldsmith project’s focus on comparing naïve, direct-schema, and structured role-based prompts against a controlled sentence corpus.


image_schema_llm_project/
│
├── README.md
├── .gitignore
├── pyproject.toml
├── requirements.txt
│
├── data/
│   ├── inputs/
│   │   ├── sentences.jsonl
│   │   ├── prompts.jsonl
│   │   ├── conditions.jsonl
│   │   └── models.jsonl
│   │
│   ├── outputs/
│   │   ├── raw_responses.jsonl
│   │   ├── parsed_responses.jsonl
│   │   ├── run_log.jsonl
│   │   ├── cost_log.jsonl
│   │   └── errors.jsonl
│   │
│   └── gold/
│       └── gold_annotations.jsonl
│
├── notebooks/
│   ├── 01_explore_corpus.ipynb
│   ├── 02_run_summary.ipynb
│   └── 03_analysis_metrics.ipynb
│
├── src/
│   └── image_schema_llm/
│       ├── __init__.py
│       │
│       ├── config.py
│       ├── runner.py
│       ├── experiment_grid.py
│       ├── checkpoint.py
│       ├── cost_tracker.py
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
│           ├── metrics.py
│           └── agreement.py
│
├── scripts/
│   ├── run_experiment.py
│   ├── validate_inputs.py
│   └── summarise_costs.py
│
└── tests/
    ├── test_jsonl_utils.py
    ├── test_experiment_grid.py
    └── test_checkpoint.py


### INPUT FILE TESTS
pip install -e .
python scripts/validate_inputs.py --project-root .
python scripts/inspect_inputs.py --project-root .
python scripts/preview_experiment_grid.py --project-root . --limit 10

### WRITE FULL MANIFEST
python scripts/write_experiment_manifest.py --project-root .

### CHECKPOINT MANAGEMENT
python scripts/inspect_checkpoint.py --project-root .
python scripts/inspect_checkpoint.py --project-root . --write-state
python scripts/preview_pending_jobs.py --project-root . --limit 10
python scripts/reset_checkpoint.py --project-root .
python scripts/reset_checkpoint.py --project-root . --execute

Important behaviour:

- successful raw responses are skipped on restart;
- failed runs remain pending unless later completed successfully;
- spend-threshold or manual stop events are recorded but do not mark jobs complete;
- checkpoint state can be regenerated from the current grid and raw responses.

### DRYRUN MODE
python scripts/dry_run_experiment.py --project-root . --limit 10
python scripts/dry_run_experiment.py --project-root . --write-manifest --write-summary
python scripts/dry_run_experiment.py --project-root . --pending-only --limit 10

#### scripts/dry_run_experiment.py

This is the main dry-run script. It:
loads the input files:

data/inputs/models.jsonl
data/inputs/prompts.jsonl
data/inputs/conditions.jsonl
data/gold/sentences_v1.jsonl

builds the full experiment grid:

model × prompt × condition × sentence × repetition
optionally checks data/outputs/raw_responses.jsonl to see which run_keys are already complete;
estimates input tokens from the prompt length;
estimates output tokens from the selected strategy;
estimates API cost using the prices in models.jsonl;
prints a dry-run summary and the first few planned jobs.

Example:

python scripts/dry_run_experiment.py --project-root . --limit 10

This gives a preview such as:

Total dry-run records: 3600
Estimated total cost: ...
First 10 jobs:
openai_gpt_5_4_mini|p_naive_v1|c_temp_0_v1|s0001|0
...

You can also write output files:
python scripts/dry_run_experiment.py --project-root . --write-manifest --write-summary

That produces:
data/outputs/dry_run_manifest.jsonl
data/outputs/dry_run_summary.json

Use this script when you want to ask:
“What exactly will the experiment run, and roughly how much will it cost?”

#### scripts/write_dry_run_manifest.py

This is a convenience wrapper around dry-run mode.
It always writes the dry-run manifest and summary files, rather than just printing a preview.

Example:
python scripts/write_dry_run_manifest.py --project-root .

It creates:
data/outputs/dry_run_manifest.jsonl
data/outputs/dry_run_summary.json

The manifest contains one record per planned job, for example:
{
  "run_key": "openai_gpt_5_4_mini|p_naive_v1|c_temp_0_v1|s0001|0",
  "model_id": "openai_gpt_5_4_mini",
  "prompt_id": "p_naive_v1",
  "condition_id": "c_temp_0_v1",
  "sentence_id": "s0001",
  "estimated_input_tokens": 112,
  "estimated_output_tokens": 800,
  "estimated_cost": 0.00042
}

Use this script when you want a persistent audit file showing:
“These are all the API calls the system is planning to make.”
This is useful before running the real API loop.

#### tests/test_dry_run.py

This is a small test file for pytest. It checks that the token-estimation utility behaves as expected.

For example, it verifies that:
estimate_tokens_from_text("")

returns at least 1, and that a 400-character string with a 4-character-per-token estimate gives roughly 100 tokens.

Run it with:
pytest tests/test_dry_run.py

Use this script when you want to confirm:
“The dry-run utility functions are behaving consistently after code changes.”

#### How they fit together

The relationship is:

dry_run.py
  ↑
  ├── dry_run_experiment.py       # interactive preview / optional output
  ├── write_dry_run_manifest.py   # always writes manifest + summary
  └── test_dry_run.py             # tests utility behaviour

So the core logic lives in:
src/image_schema_llm/dry_run.py
The scripts are just command-line entry points for using that logic.

#### Run Open AI on the manifest
export OPENAI_API_KEY="..."
python scripts/run_openai_next_job.py --project-root . --dry-run

#### Run the next pending OpenAI job
python scripts/run_openai_next_job.py --project-root .

#### Run a specfic job
python scripts/run_openai_next_job.py --project-root . \
  --run-key "openai_gpt_5_4_mini|p_naive_v1|c_temp_0_v1|s0001|0"

#### Successful responses are immediately written to:
data/outputs/raw_responses.jsonl

#### and a cost record is written to:
data/outputs/cost_log.jsonl