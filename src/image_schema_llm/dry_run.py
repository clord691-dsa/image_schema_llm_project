from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from image_schema_llm.checkpoint import CheckpointManager
from image_schema_llm.config import ProjectPaths
from image_schema_llm.experiment_grid import build_grid_from_project
from image_schema_llm.jsonl_utils import write_jsonl
from image_schema_llm.schemas import ExperimentJob


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DryRunSettings:
    """
    Settings for dry-run mode.

    Parameters
    ----------
    pending_only:
        If True, remove run keys already completed in raw_responses.jsonl.
    estimated_output_tokens_strategy:
        How to estimate output tokens before an API call exists.
        Supported values:
        - "condition_max": use condition.max_output_tokens
        - "half_condition_max": use condition.max_output_tokens // 2
        - "zero": estimate output tokens as zero
    token_char_ratio:
        Approximate characters per token for prompt text. A rough value of 4
        is commonly used for English text. This is an estimate only.
    """

    pending_only: bool = False
    estimated_output_tokens_strategy: str = "condition_max"
    token_char_ratio: float = 4.0


@dataclass(frozen=True)
class DryRunSummary:
    """
    Summary of a dry-run plan.

    Fields
    ------
    total_jobs_in_grid:
        Full enabled grid size before pending filtering.
    jobs_in_dry_run:
        Number of jobs included in this dry run.
    completed_jobs_excluded:
        Number of jobs excluded because they were already complete.
    estimated_input_tokens:
        Estimated total input tokens.
    estimated_output_tokens:
        Estimated total output tokens.
    estimated_cost:
        Estimated total cost across the dry-run jobs.
    """

    total_jobs_in_grid: int
    jobs_in_dry_run: int
    completed_jobs_excluded: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "total_jobs_in_grid": self.total_jobs_in_grid,
            "jobs_in_dry_run": self.jobs_in_dry_run,
            "completed_jobs_excluded": self.completed_jobs_excluded,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_cost": self.estimated_cost,
        }


def estimate_tokens_from_text(text: str, *, token_char_ratio: float = 4.0) -> int:
    """
    Estimate tokens from character count.

    This is intentionally simple and provider-agnostic. During real execution,
    provider-reported token counts should replace this estimate.
    """

    if token_char_ratio <= 0:
        raise ValueError("token_char_ratio must be positive")

    return max(1, int(round(len(text) / token_char_ratio)))


def estimate_output_tokens(job: ExperimentJob, strategy: str) -> int:
    """
    Estimate output tokens for a job before an API response exists.

    Supported strategies:
    - condition_max
    - half_condition_max
    - zero
    """

    if strategy == "condition_max":
        return job.condition.max_output_tokens

    if strategy == "half_condition_max":
        return max(1, job.condition.max_output_tokens // 2)

    if strategy == "zero":
        return 0

    raise ValueError(f"Unsupported output-token estimation strategy: {strategy}")


def estimate_job_cost(
    job: ExperimentJob,
    *,
    estimated_input_tokens: int,
    estimated_output_tokens: int,
) -> float:
    """
    Estimate one job's API cost from model pricing.

    Pricing is taken from models.jsonl:
    - input_cost_per_1m_tokens
    - output_cost_per_1m_tokens
    """

    input_cost = (
        estimated_input_tokens / 1_000_000
    ) * job.model.input_cost_per_1m_tokens

    output_cost = (
        estimated_output_tokens / 1_000_000
    ) * job.model.output_cost_per_1m_tokens

    return input_cost + output_cost


def build_dry_run_record(
    job: ExperimentJob,
    *,
    status: str,
    settings: DryRunSettings,
) -> dict[str, Any]:
    """
    Build a JSONL dry-run manifest record for one job.

    Parameters
    ----------
    job:
        ExperimentJob produced by the experiment grid.
    status:
        Dry-run status, usually `pending` or `already_completed`.
    settings:
        DryRunSettings controlling token estimation.

    Returns
    -------
    dict
        JSON-serialisable dry-run record.
    """

    input_text = job.system_message + "\n\n" + job.user_prompt
    estimated_input_tokens = estimate_tokens_from_text(
        input_text,
        token_char_ratio=settings.token_char_ratio,
    )
    estimated_output_tokens = estimate_output_tokens(
        job,
        settings.estimated_output_tokens_strategy,
    )
    estimated_cost = estimate_job_cost(
        job,
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
    )

    return {
        "created_at": utc_now_iso(),
        "status": status,
        "dry_run": True,
        "run_key": job.run_key,
        "run_index": job.run_index,
        "model_id": job.model.model_id,
        "provider": job.model.provider,
        "model_name": job.model.model_name,
        "prompt_id": job.prompt.prompt_id,
        "prompt_family": job.prompt.prompt_family,
        "prompt_version": job.prompt.prompt_version,
        "condition_id": job.condition.condition_id,
        "condition_family": job.condition.condition_family,
        "temperature": job.condition.temperature,
        "top_p": job.condition.top_p,
        "max_output_tokens": job.condition.max_output_tokens,
        "sentence_id": job.sentence.sentence_id,
        "sentence_type": job.sentence.sentence_type,
        "expected_schema_primary": job.sentence.expected_schema_primary,
        "expected_literal_or_metaphorical": job.sentence.expected_literal_or_metaphorical,
        "repetition_index": job.repetition_index,
        "system_message_chars": len(job.system_message),
        "user_prompt_chars": len(job.user_prompt),
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_cost": estimated_cost,
        "currency": job.model.currency,
        "token_estimation_method": f"chars/{settings.token_char_ratio}",
        "output_token_estimation_strategy": settings.estimated_output_tokens_strategy,
    }


def summarise_dry_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarise dry-run manifest records.

    Returns a dictionary suitable for data/outputs/dry_run_summary.json.
    """

    by_model_cost = defaultdict(float)
    by_prompt_count = Counter()
    by_condition_count = Counter()
    by_status = Counter()
    by_sentence_type = Counter()

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    for record in records:
        by_status[record["status"]] += 1
        by_prompt_count[record["prompt_id"]] += 1
        by_condition_count[record["condition_id"]] += 1
        by_sentence_type[record["sentence_type"]] += 1
        by_model_cost[record["model_id"]] += record["estimated_cost"]
        total_input_tokens += record["estimated_input_tokens"]
        total_output_tokens += record["estimated_output_tokens"]
        total_cost += record["estimated_cost"]

    return {
        "created_at": utc_now_iso(),
        "dry_run": True,
        "records": len(records),
        "status_counts": dict(by_status),
        "prompt_counts": dict(by_prompt_count),
        "condition_counts": dict(by_condition_count),
        "sentence_type_counts": dict(by_sentence_type),
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_total_cost": total_cost,
        "estimated_cost_by_model": dict(by_model_cost),
    }


def run_dry_run(
    project_root: Path,
    *,
    settings: DryRunSettings,
    write_manifest: bool = False,
    write_summary: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Execute dry-run planning without calling any provider APIs.

    Steps
    -----
    1. Build the enabled experiment grid.
    2. Load completed run keys from raw_responses.jsonl.
    3. Mark jobs as pending or already_completed.
    4. Optionally filter to pending-only jobs.
    5. Estimate token volume and cost.
    6. Optionally write manifest and summary output files.
    """

    paths = ProjectPaths(project_root)
    manager = CheckpointManager(paths.outputs_dir)

    jobs = build_grid_from_project(project_root)
    completed = manager.load_completed_run_keys()

    records: list[dict[str, Any]] = []
    for job in jobs:
        is_completed = job.run_key in completed
        status = "already_completed" if is_completed else "pending"

        if settings.pending_only and is_completed:
            continue

        records.append(
            build_dry_run_record(
                job,
                status=status,
                settings=settings,
            )
        )

    summary = summarise_dry_run(records)
    summary["total_jobs_in_enabled_grid"] = len(jobs)
    summary["completed_run_keys_found"] = len(completed)
    summary["pending_only"] = settings.pending_only

    if write_manifest:
        manifest_path = paths.outputs_dir / "dry_run_manifest.jsonl"
        write_jsonl(manifest_path, records)
        summary["manifest_path"] = str(manifest_path)

    if write_summary:
        summary_path = paths.outputs_dir / "dry_run_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["summary_path"] = str(summary_path)

    return records, summary
