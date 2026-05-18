from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from image_schema_llm.checkpoint import CheckpointManager
from image_schema_llm.config import ProjectPaths
from image_schema_llm.experiment_grid import build_experiment_grid, load_experiment_inputs
from image_schema_llm.schemas import ExperimentJob


@dataclass(frozen=True)
class RestartPlan:
    """
    Restart plan for the current experiment grid.

    Fields
    ------
    total_jobs:
        Number of jobs in the enabled experiment grid.
    completed_jobs:
        Number of jobs already successful in raw_responses.jsonl.
    pending_jobs:
        Number of jobs still to run.
    next_job:
        The first pending job, or None if the experiment is complete.
    """

    total_jobs: int
    completed_jobs: int
    pending_jobs: int
    next_job: ExperimentJob | None


def build_restart_plan(project_root: Path) -> RestartPlan:
    """
    Build a restart plan by comparing the current grid with successful runs.

    This function does not call APIs. It is safe to run at any time.
    """

    paths = ProjectPaths(project_root)
    manager = CheckpointManager(paths.outputs_dir)

    models, prompts, conditions, sentences = load_experiment_inputs(project_root)
    jobs = list(build_experiment_grid(models, prompts, conditions, sentences))
    completed = manager.load_completed_run_keys()
    pending = [job for job in jobs if job.run_key not in completed]

    return RestartPlan(
        total_jobs=len(jobs),
        completed_jobs=len(completed & {job.run_key for job in jobs}),
        pending_jobs=len(pending),
        next_job=pending[0] if pending else None,
    )
