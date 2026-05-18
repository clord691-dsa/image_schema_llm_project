#!/usr/bin/env python
from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from image_schema_llm.config import ProjectPaths
from image_schema_llm.experiment_grid import build_experiment_grid
from image_schema_llm.loaders import load_conditions, load_models, load_prompts, load_sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the experiment grid without calling any APIs.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory")
    parser.add_argument("--limit", type=int, default=10, help="Number of jobs to print")
    args = parser.parse_args()

    paths = ProjectPaths(args.project_root)

    models = load_models(paths.models_path)
    prompts = load_prompts(paths.prompts_path)
    conditions = load_conditions(paths.conditions_path)
    sentences = load_sentences(paths.sentences_path)

    jobs = list(build_experiment_grid(models, prompts, conditions, sentences))
    print(f"Total enabled jobs: {len(jobs)}")
    print(f"Showing first {min(args.limit, len(jobs))} jobs:\n")

    for job in islice(jobs, args.limit):
        print(job.run_key)
        print(f"  model: {job.model.model_id}")
        print(f"  prompt: {job.prompt.prompt_id}")
        print(f"  condition: {job.condition.condition_id}")
        print(f"  sentence: {job.sentence.sentence_id} | {job.sentence.text}")
        print(f"  user_prompt_preview: {job.user_prompt[:160].replace(chr(10), ' ')}...")
        print()


if __name__ == "__main__":
    main()
