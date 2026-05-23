from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from image_schema_llm.checkpoint import CheckpointManager
from image_schema_llm.clients.claude_client import ClaudeMessagesClient
from image_schema_llm.clients.gemini_client import GeminiGenerateContentClient
from image_schema_llm.cost_tracker import RuntimeCostTracker
from image_schema_llm.experiment_grid import build_grid_from_project
from image_schema_llm.runtime_config import load_runtime_config
from image_schema_llm.schemas import ExperimentJob
from image_schema_llm.structured_output import response_format_for_job


ProviderName = Literal["anthropic", "google"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ProviderRunResult:
    run_key: str | None
    status: str
    message: str
    raw_record: dict[str, Any] | None = None


def select_next_provider_job(*, project_root: Path, provider: ProviderName, run_key: str | None = None) -> ExperimentJob | None:
    manager = CheckpointManager(project_root / "data" / "outputs")
    completed = manager.load_completed_run_keys()
    jobs = [job for job in build_grid_from_project(project_root) if job.model.provider == provider]
    if run_key is not None:
        matches = [job for job in jobs if job.run_key == run_key]
        if not matches:
            raise ValueError(f"No {provider} job found for run_key: {run_key}")
        return None if matches[0].run_key in completed else matches[0]
    for job in jobs:
        if job.run_key not in completed:
            return job
    return None


def _build_client(job: ExperimentJob):
    if job.model.provider == "anthropic":
        return ClaudeMessagesClient(api_key_env_var=job.model.api_key_env_var or "ANTHROPIC_API_KEY")
    if job.model.provider == "google":
        return GeminiGenerateContentClient(api_key_env_var=job.model.api_key_env_var or "GEMINI_API_KEY")
    raise ValueError(f"Unsupported provider for provider_runner: {job.model.provider}")


def run_provider_job(
    *,
    project_root: Path,
    job: ExperimentJob,
    dry_run: bool = False,
    reasoning_effort: str | None = None,
) -> ProviderRunResult:
    outputs_dir = project_root / "data" / "outputs"
    manager = CheckpointManager(outputs_dir)
    runtime_config = load_runtime_config(project_root)
    tracker = RuntimeCostTracker.from_runtime_config(
        project_root=project_root,
        runtime_config=runtime_config,
    )

    if manager.is_completed(job.run_key):
        return ProviderRunResult(job.run_key, "skipped", "Run key already completed.")

    if tracker.threshold_reached():
        tracker.write_stop_record(reason="spend_threshold_reached_before_call", run_key=job.run_key)
        manager.write_stop_record(
            reason="spend_threshold_reached_before_call",
            run_key=job.run_key,
            details=tracker.summary_dict(),
        )
        return ProviderRunResult(job.run_key, "stopped", "Spend threshold reached before API call.")

    manager.write_run_event(
        event_type="job_started",
        run_key=job.run_key,
        status="started",
        message=f"Starting {job.model.provider} job.",
        details={
            "provider": job.model.provider,
            "model_id": job.model.model_id,
            "prompt_id": job.prompt.prompt_id,
            "condition_id": job.condition.condition_id,
            "sentence_id": job.sentence.sentence_id,
            "effective_max_output_tokens": job.effective_max_output_tokens,
        },
    )

    if dry_run:
        return ProviderRunResult(job.run_key, "dry_run", "Dry run selected job; no API call made.")

    try:
        client = _build_client(job)
        response = client.generate(
            system_message=job.system_message,
            user_prompt=job.user_prompt,
            model_name=job.model.model_name,
            temperature=job.condition.temperature if job.model.supports_temperature else None,
            top_p=job.condition.top_p if job.model.supports_top_p else None,
            max_output_tokens=job.effective_max_output_tokens,
            response_format=response_format_for_job(job),
            reasoning_effort=reasoning_effort,
        )

        cost_record = tracker.record_api_usage(
            run_key=job.run_key,
            model=job.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            provider=job.model.provider,
            metadata={
                "prompt_id": job.prompt.prompt_id,
                "condition_id": job.condition.condition_id,
                "sentence_id": job.sentence.sentence_id,
            },
        )

        raw_record = {
            "created_at": utc_now_iso(),
            "run_key": job.run_key,
            "run_index": job.run_index,
            "status": "success",
            "provider": job.model.provider,
            "model_id": job.model.model_id,
            "model_name": job.model.model_name,
            "model_snapshot": job.model.model_snapshot,
            "prompt_id": job.prompt.prompt_id,
            "prompt_family": job.prompt.prompt_family,
            "prompt_version": job.prompt.prompt_version,
            "expected_output_format": job.prompt.expected_output_format,
            "condition_id": job.condition.condition_id,
            "condition_family": job.condition.condition_family,
            "temperature": job.condition.temperature,
            "top_p": job.condition.top_p,
            "condition_max_output_tokens": job.condition.max_output_tokens,
            "recommended_max_output_tokens": job.prompt.recommended_max_output_tokens,
            "max_output_tokens": job.effective_max_output_tokens,
            "sentence_id": job.sentence.sentence_id,
            "sentence_type": job.sentence.sentence_type,
            "expected_schema_primary": job.sentence.expected_schema_primary,
            "expected_literal_or_metaphorical": job.sentence.expected_literal_or_metaphorical,
            "repetition_index": job.repetition_index,
            "system_message": job.system_message,
            "user_prompt": job.user_prompt,
            "raw_response": response.raw_response,
            "input_tokens": cost_record["input_tokens"],
            "output_tokens": cost_record["output_tokens"],
            "estimated_cost": cost_record["estimated_cost"],
            "currency": cost_record["currency"],
            "provider_response_id": response.provider_response_id,
            "provider_metadata": response.provider_metadata,
            "finish_reason": response.finish_reason,
        }
        manager.write_success_marker_from_raw_response(raw_record)
        return ProviderRunResult(job.run_key, "success", f"{job.model.provider} response persisted successfully.", raw_record)

    except Exception as exc:
        manager.write_error_record(
            run_key=job.run_key,
            error_type=type(exc).__name__,
            error_message=str(exc),
            retryable=True,
            job_metadata={
                "provider": job.model.provider,
                "model_id": job.model.model_id,
                "prompt_id": job.prompt.prompt_id,
                "condition_id": job.condition.condition_id,
                "sentence_id": job.sentence.sentence_id,
            },
        )
        return ProviderRunResult(job.run_key, "error", f"{type(exc).__name__}: {exc}")
