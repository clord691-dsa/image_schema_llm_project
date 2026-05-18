#!/usr/bin/env python
from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from image_schema_llm.config import ProjectPaths
from image_schema_llm.experiment_grid import (
    build_experiment_grid,
    filter_pending_jobs,
    load_completed_run_keys,
    load_experiment_inputs,
    make_run_hash,
    write_experiment_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the experiment grid without calling APIs.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    paths = ProjectPaths(project_root)
    models, prompts, conditions, sentences = load_experiment_inputs(project_root)

    jobs = list(build_experiment_grid(models, prompts, conditions, sentences))

    if args.pending_only:
        completed = load_completed_run_keys(paths.raw_responses_path)
        jobs = list(filter_pending_jobs(jobs, completed))
        print(f"Completed run keys found: {len(completed)}")

    print(f"Total jobs in selected grid: {len(jobs)}")

    if args.write_manifest:
        manifest_path = write_experiment_manifest(project_root, jobs)
        print(f"Wrote manifest: {manifest_path}")

    print(f"Showing first {min(args.limit, len(jobs))} jobs:\n")

    for job in islice(jobs, args.limit):
        print(job.run_key)
        print(f"  run_index: {job.run_index}")
        print(f"  run_hash: {make_run_hash(job.run_key)}")
        print(f"  model: {job.model.model_id} ({job.model.provider}: {job.model.model_name})")
        print(f"  prompt: {job.prompt.prompt_id} ({job.prompt.prompt_family})")
        print(
            f"  condition: {job.condition.condition_id} "
            f"(temp={job.condition.temperature}, top_p={job.condition.top_p}, "
            f"max_output_tokens={job.condition.max_output_tokens})"
        )
        print(
            f"  sentence: {job.sentence.sentence_id} | "
            f"{job.sentence.expected_schema_primary} | {job.sentence.text}"
        )
        print(f"  user_prompt_preview: {job.user_prompt[:220].replace(chr(10), ' ')}...")
        print()


if __name__ == "__main__":
    main()
