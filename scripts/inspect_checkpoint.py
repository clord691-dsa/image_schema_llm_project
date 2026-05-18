#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from image_schema_llm.checkpoint import CheckpointManager
from image_schema_llm.config import ProjectPaths
from image_schema_llm.experiment_grid import build_grid_from_project
from image_schema_llm.restart import build_restart_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect checkpoint/restart state.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--write-state", action="store_true", help="Write data/outputs/checkpoint_state.json")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    paths = ProjectPaths(project_root)
    manager = CheckpointManager(paths.outputs_dir)

    jobs = build_grid_from_project(project_root)
    plan = build_restart_plan(project_root)

    print("Checkpoint / restart summary")
    print("============================")
    print(f"total_jobs: {plan.total_jobs}")
    print(f"completed_jobs: {plan.completed_jobs}")
    print(f"pending_jobs: {plan.pending_jobs}")

    if plan.next_job:
        print("\nNext pending job")
        print("================")
        print(f"run_key: {plan.next_job.run_key}")
        print(f"model: {plan.next_job.model.model_id}")
        print(f"prompt: {plan.next_job.prompt.prompt_id}")
        print(f"condition: {plan.next_job.condition.condition_id}")
        print(f"sentence: {plan.next_job.sentence.sentence_id} | {plan.next_job.sentence.text}")
    else:
        print("\nNo pending jobs. The enabled grid is complete.")

    if args.write_state:
        state_path = manager.write_checkpoint_state(
            planned_run_keys=[job.run_key for job in jobs],
            metadata={"source": "scripts/inspect_checkpoint.py"},
        )
        print(f"\nWrote checkpoint state: {state_path}")


if __name__ == "__main__":
    main()
