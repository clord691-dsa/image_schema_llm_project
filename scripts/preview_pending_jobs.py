#!/usr/bin/env python
from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from image_schema_llm.checkpoint import CheckpointManager
from image_schema_llm.config import ProjectPaths
from image_schema_llm.experiment_grid import build_grid_from_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview jobs that remain pending after checkpoint filtering.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    paths = ProjectPaths(project_root)
    manager = CheckpointManager(paths.outputs_dir)

    jobs = build_grid_from_project(project_root)
    pending = manager.filter_pending_jobs(jobs)

    print(f"Total planned jobs: {len(jobs)}")
    print(f"Completed jobs: {len(jobs) - len(pending)}")
    print(f"Pending jobs: {len(pending)}")
    print(f"Showing first {min(args.limit, len(pending))} pending jobs:\n")

    for job in islice(pending, args.limit):
        print(job.run_key)
        print(f"  model: {job.model.model_id}")
        print(f"  prompt: {job.prompt.prompt_id}")
        print(f"  condition: {job.condition.condition_id}")
        print(f"  sentence: {job.sentence.sentence_id} | {job.sentence.text}")
        print()


if __name__ == "__main__":
    main()
