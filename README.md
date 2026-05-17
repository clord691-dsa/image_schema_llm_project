Initial program spec to GPT:
Define a sentences corpus data structure with fields suitable for the project proposal which should be implemented as a jsonl database. Provide a python framework and github folder structure for the project. The design should incorporate a per LLM, per prompt (e.g naive, direct schema and structured role-based), per condition (e.g. model temperature), per sentence, iterative structure. The outputs from each permutation should be written to a jsonl database for subsequent analysis. The raw model responses should be recorded to minimise cost. The loop should monitor the cumulative cost of each model's api charges. The loop should stop if a spend threshold is reached. If the loop stops e.g. because of connection error or spend threshold it must be possible to restart the model from the last successful iteration. Prompts and conditions should be stored in jsonl database files for input into the program. Python function placeholders should be created rather than full code at this stage. e.g. an OpenAI API handler. Three LLMs will be used OpenAI, Claude and Gemini. Each placeholder function or class should contain commenary about inputs, outputs and purpose.

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