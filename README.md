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


