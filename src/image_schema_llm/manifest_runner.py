from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from image_schema_llm.checkpoint import CheckpointManager
from image_schema_llm.config import ProjectPaths
from image_schema_llm.experiment_grid import build_grid_from_project
from image_schema_llm.openai_runner import run_openai_job
from image_schema_llm.provider_runner import run_provider_job
from image_schema_llm.runtime_config import load_runtime_config
from image_schema_llm.schemas import ExperimentJob


ProviderName = Literal["openai", "anthropic", "google"]


@dataclass(frozen=True)
class ManifestRunSummary:
    """
    Summary of a provider manifest run.

    Fields
    ------
    provider:
        Provider selected for this run.
    selected_jobs:
        Number of pending jobs selected before execution begins.
    attempted_jobs:
        Number of jobs actually attempted.
    succeeded_jobs:
        Number of jobs that returned status == success.
    failed_jobs:
        Number of jobs that returned status == error.
    skipped_jobs:
        Number of jobs that returned status == skipped.
    stopped:
        Whether the manifest loop stopped early.
    stop_reason:
        Reason for early stop, if applicable.
    """

    provider: ProviderName
    selected_jobs: int
    attempted_jobs: int
    succeeded_jobs: int
    failed_jobs: int
    skipped_jobs: int
    dry_run_jobs: int
    stopped: bool
    stop_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "selected_jobs": self.selected_jobs,
            "attempted_jobs": self.attempted_jobs,
            "succeeded_jobs": self.succeeded_jobs,
            "failed_jobs": self.failed_jobs,
            "skipped_jobs": self.skipped_jobs,
            "dry_run_jobs": self.dry_run_jobs,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
        }


def provider_label(provider: ProviderName) -> str:
    """
    Return a user-facing provider label.
    """

    labels = {
        "openai": "OpenAI",
        "anthropic": "Claude",
        "google": "Gemini",
    }
    return labels[provider]


def pending_jobs_for_provider(
    *,
    project_root: Path,
    provider: ProviderName,
    max_jobs: int | None = None,
) -> list[ExperimentJob]:
    """
    Return pending jobs for a single provider.

    Completion is determined from data/outputs/raw_responses.jsonl. A run is
    complete only if a success record exists for its run_key.
    """

    paths = ProjectPaths(project_root)
    manager = CheckpointManager(paths.outputs_dir)
    completed = manager.load_completed_run_keys()

    jobs = [
        job
        for job in build_grid_from_project(project_root)
        if job.model.provider == provider and job.run_key not in completed
    ]

    if max_jobs is not None:
        jobs = jobs[:max_jobs]

    return jobs


def run_one_job_for_provider(
    *,
    project_root: Path,
    provider: ProviderName,
    job: ExperimentJob,
    dry_run: bool,
):
    """
    Execute one job using the appropriate provider runner.
    """

    if provider == "openai":
        return run_openai_job(
            project_root=project_root,
            job=job,
            dry_run=dry_run,
        )

    if provider in {"anthropic", "google"}:
        return run_provider_job(
            project_root=project_root,
            job=job,
            dry_run=dry_run,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def run_provider_manifest(
    *,
    project_root: Path,
    provider: ProviderName,
    max_jobs: int | None = None,
    sleep_seconds: float = 0.0,
    stop_on_error: bool | None = None,
    dry_run: bool | None = None,
    print_prompts: bool = False,
) -> ManifestRunSummary:
    """
    Run all pending manifest jobs for one provider.

    Parameters
    ----------
    project_root:
        Repository root.
    provider:
        openai, anthropic, or google.
    max_jobs:
        Optional cap for pilot runs. If None, all pending provider jobs are run.
    sleep_seconds:
        Optional delay between calls, useful for rate-limit safety.
    stop_on_error:
        If True, stop the loop after the first error. If None, load from
        runtime_config.json.
    dry_run:
        If True, select jobs but make no API calls. If None, load from
        runtime_config.json.
    print_prompts:
        If True, print a short prompt preview for each selected job.

    Returns
    -------
    ManifestRunSummary
    """

    runtime_config = load_runtime_config(project_root)
    selected_stop_on_error = runtime_config.stop_on_error if stop_on_error is None else stop_on_error
    selected_dry_run = runtime_config.dry_run if dry_run is None else dry_run

    jobs = pending_jobs_for_provider(
        project_root=project_root,
        provider=provider,
        max_jobs=max_jobs,
    )

    selected_jobs = len(jobs)
    attempted = 0
    succeeded = 0
    failed = 0
    skipped = 0
    dry_runs = 0
    stopped = False
    stop_reason: str | None = None

    print(f"{provider_label(provider)} manifest run")
    print("=" * (len(provider_label(provider)) + 13))
    print(f"selected_jobs: {selected_jobs}")
    print(f"dry_run: {selected_dry_run}")
    print(f"stop_on_error: {selected_stop_on_error}")
    print(f"sleep_seconds: {sleep_seconds}")
    print()

    if not jobs:
        print(f"No pending {provider_label(provider)} jobs found.")
        return ManifestRunSummary(
            provider=provider,
            selected_jobs=0,
            attempted_jobs=0,
            succeeded_jobs=0,
            failed_jobs=0,
            skipped_jobs=0,
            dry_run_jobs=0,
            stopped=False,
            stop_reason=None,
        )

    for index, job in enumerate(jobs, start=1):
        attempted += 1

        print(f"[{index}/{selected_jobs}] {job.run_key}")
        print(f"  model: {job.model.model_id} ({job.model.model_name})")
        print(f"  prompt: {job.prompt.prompt_id}")
        print(f"  condition: {job.condition.condition_id}")
        print(f"  sentence: {job.sentence.sentence_id} | {job.sentence.text}")

        if print_prompts:
            preview = job.user_prompt[:500].replace("\n", " ")
            print(f"  prompt_preview: {preview}...")

        result = run_one_job_for_provider(
            project_root=project_root,
            provider=provider,
            job=job,
            dry_run=selected_dry_run,
        )

        print(f"  status: {result.status}")
        print(f"  message: {result.message}")

        if result.raw_record:
            print(f"  input_tokens: {result.raw_record.get('input_tokens')}")
            print(f"  output_tokens: {result.raw_record.get('output_tokens')}")
            print(f"  estimated_cost: {result.raw_record.get('estimated_cost')}")

        if result.status == "success":
            succeeded += 1
        elif result.status == "error":
            failed += 1
            if selected_stop_on_error:
                stopped = True
                stop_reason = "error"
                print("Stopping because stop_on_error=True and a job returned error.")
                break
        elif result.status == "skipped":
            skipped += 1
        elif result.status == "dry_run":
            dry_runs += 1
        elif result.status == "stopped":
            stopped = True
            stop_reason = "budget_or_runner_stop"
            print("Stopping because the provider runner returned stopped.")
            break

        print()

        if sleep_seconds > 0 and index < selected_jobs:
            time.sleep(sleep_seconds)

    summary = ManifestRunSummary(
        provider=provider,
        selected_jobs=selected_jobs,
        attempted_jobs=attempted,
        succeeded_jobs=succeeded,
        failed_jobs=failed,
        skipped_jobs=skipped,
        dry_run_jobs=dry_runs,
        stopped=stopped,
        stop_reason=stop_reason,
    )

    print()
    print("Manifest run summary")
    print("====================")
    for key, value in summary.to_dict().items():
        print(f"{key}: {value}")

    return summary
