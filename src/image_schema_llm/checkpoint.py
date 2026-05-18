from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from image_schema_llm.jsonl_utils import append_jsonl, read_jsonl, write_jsonl


SUCCESS_STATUS = "success"
ERROR_STATUS = "error"
SKIPPED_STATUS = "skipped"
STOPPED_STATUS = "stopped"


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CheckpointSummary:
    """
    Summary of experiment checkpoint/restart state.

    Fields
    ------
    total_planned_jobs:
        Number of jobs in the enabled experiment grid.
    completed_jobs:
        Number of jobs with successful raw response records.
    pending_jobs:
        Number of jobs still requiring execution.
    error_records:
        Number of error records currently logged.
    stopped_records:
        Number of explicit stopped records logged.
    """

    total_planned_jobs: int
    completed_jobs: int
    pending_jobs: int
    error_records: int
    stopped_records: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serialisable dictionary."""
        return {
            "total_planned_jobs": self.total_planned_jobs,
            "completed_jobs": self.completed_jobs,
            "pending_jobs": self.pending_jobs,
            "error_records": self.error_records,
            "stopped_records": self.stopped_records,
        }


class CheckpointManager:
    """
    Manages checkpoint and restart state for the experiment runner.

    Design principle
    ----------------
    `raw_responses.jsonl` is the authoritative source of truth for successful
    completion. The checkpoint state file is a convenience snapshot, not the
    canonical proof that a job completed.

    Purpose
    -------
    - Detect completed run keys.
    - Filter pending jobs.
    - Write run lifecycle events.
    - Write error records.
    - Write explicit stopped/budget records.
    - Maintain a human-readable checkpoint_state.json snapshot.
    """

    def __init__(
        self,
        outputs_dir: Path,
        *,
        raw_responses_filename: str = "raw_responses.jsonl",
        run_log_filename: str = "run_log.jsonl",
        errors_filename: str = "errors.jsonl",
        checkpoint_state_filename: str = "checkpoint_state.json",
    ) -> None:
        """
        Parameters
        ----------
        outputs_dir:
            Path to `data/outputs`.
        raw_responses_filename:
            Append-only file containing raw model responses.
        run_log_filename:
            Append-only operational run event log.
        errors_filename:
            Append-only error log.
        checkpoint_state_filename:
            JSON snapshot of restart progress.
        """

        self.outputs_dir = outputs_dir
        self.raw_responses_path = outputs_dir / raw_responses_filename
        self.run_log_path = outputs_dir / run_log_filename
        self.errors_path = outputs_dir / errors_filename
        self.checkpoint_state_path = outputs_dir / checkpoint_state_filename
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def load_completed_run_keys(self) -> set[str]:
        """
        Return run keys that have completed successfully.

        A run is treated as complete only when raw_responses.jsonl contains a
        record with `status == "success"` and a non-empty `run_key`.
        """

        if not self.raw_responses_path.exists():
            return set()

        records = read_jsonl(self.raw_responses_path)
        return {
            record["run_key"]
            for record in records
            if record.get("status") == SUCCESS_STATUS and record.get("run_key")
        }

    def load_failed_run_keys(self) -> set[str]:
        """
        Return run keys that appear in errors.jsonl.

        Failed run keys are not automatically skipped; they remain pending
        unless they also appear as successful raw responses. This behaviour
        allows retry after connection errors or rate limits.
        """

        if not self.errors_path.exists():
            return set()

        records = read_jsonl(self.errors_path)
        return {
            record["run_key"]
            for record in records
            if record.get("run_key")
        }

    def is_completed(self, run_key: str) -> bool:
        """Return True if a run key has a successful raw response record."""
        return run_key in self.load_completed_run_keys()

    def filter_pending_jobs(self, jobs: Iterable[Any]) -> list[Any]:
        """
        Return jobs whose run_key is not completed.

        Parameters
        ----------
        jobs:
            Iterable of ExperimentJob-like objects with a `run_key` attribute.

        Returns
        -------
        list
            Jobs that should still be executed.
        """

        completed = self.load_completed_run_keys()
        return [job for job in jobs if job.run_key not in completed]

    def write_run_event(
        self,
        *,
        event_type: str,
        run_key: str | None = None,
        status: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Append one operational event to run_log.jsonl.

        Example event types:
        - experiment_started
        - job_started
        - job_succeeded
        - job_skipped_completed
        - job_failed
        - experiment_stopped
        - experiment_finished
        """

        record = {
            "created_at": utc_now_iso(),
            "event_type": event_type,
            "run_key": run_key,
            "status": status,
            "message": message,
            "details": details or {},
        }
        append_jsonl(self.run_log_path, record)

    def write_error_record(
        self,
        *,
        run_key: str | None,
        error_type: str,
        error_message: str,
        retryable: bool,
        job_metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Append one error record to errors.jsonl.

        Errors do not mark a run as complete. On restart, the corresponding job
        remains pending unless a later success record exists in raw_responses.
        """

        record = {
            "created_at": utc_now_iso(),
            "run_key": run_key,
            "status": ERROR_STATUS,
            "error_type": error_type,
            "error_message": error_message,
            "retryable": retryable,
            "job_metadata": job_metadata or {},
        }
        append_jsonl(self.errors_path, record)
        self.write_run_event(
            event_type="job_failed",
            run_key=run_key,
            status=ERROR_STATUS,
            message=error_message,
            details={"error_type": error_type, "retryable": retryable},
        )

    def write_stop_record(
        self,
        *,
        reason: str,
        run_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Record an intentional stop, such as spend threshold reached.

        Stop records are written to errors.jsonl because they represent an
        incomplete experiment state requiring later restart or review.
        """

        record = {
            "created_at": utc_now_iso(),
            "run_key": run_key,
            "status": STOPPED_STATUS,
            "reason": reason,
            "details": details or {},
        }
        append_jsonl(self.errors_path, record)
        self.write_run_event(
            event_type="experiment_stopped",
            run_key=run_key,
            status=STOPPED_STATUS,
            message=reason,
            details=details or {},
        )

    def write_success_marker_from_raw_response(self, raw_response_record: dict[str, Any]) -> None:
        """
        Persist a successful raw response record.

        This method is provided for the later API runner. It appends the raw
        response immediately and writes a corresponding run event.

        Requirements
        ------------
        raw_response_record must include:
        - run_key
        - status == "success"
        - raw_response or provider-specific response payload
        """

        run_key = raw_response_record.get("run_key")
        status = raw_response_record.get("status")

        if not run_key:
            raise ValueError("raw_response_record must include run_key")

        if status != SUCCESS_STATUS:
            raise ValueError("raw_response_record must have status == 'success'")

        append_jsonl(self.raw_responses_path, raw_response_record)
        self.write_run_event(
            event_type="job_succeeded",
            run_key=run_key,
            status=SUCCESS_STATUS,
            message="Raw response persisted successfully.",
        )

    def summarise(self, planned_run_keys: Iterable[str]) -> CheckpointSummary:
        """
        Build a checkpoint summary against a planned run-key set.
        """

        planned = set(planned_run_keys)
        completed = self.load_completed_run_keys()

        error_records = 0
        stopped_records = 0
        if self.errors_path.exists():
            for record in read_jsonl(self.errors_path):
                if record.get("status") == ERROR_STATUS:
                    error_records += 1
                if record.get("status") == STOPPED_STATUS:
                    stopped_records += 1

        return CheckpointSummary(
            total_planned_jobs=len(planned),
            completed_jobs=len(planned & completed),
            pending_jobs=len(planned - completed),
            error_records=error_records,
            stopped_records=stopped_records,
        )

    def write_checkpoint_state(
        self,
        *,
        planned_run_keys: Iterable[str],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write checkpoint_state.json.

        This is a convenience snapshot for humans and scripts. It should be
        regenerated from raw_responses.jsonl and the current experiment grid.
        """

        planned = set(planned_run_keys)
        completed = self.load_completed_run_keys()
        failed = self.load_failed_run_keys()
        summary = self.summarise(planned)

        state = {
            "created_at": utc_now_iso(),
            "summary": summary.to_dict(),
            "completed_run_keys_count": len(completed),
            "failed_run_keys_count": len(failed),
            "pending_run_keys_count": len(planned - completed),
            "metadata": metadata or {},
        }

        self.checkpoint_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.checkpoint_state_path

    def reset_checkpoint_files(self, *, dry_run: bool = True) -> list[Path]:
        """
        Delete checkpoint-related files.

        This does not delete input data. By default it is a dry run.

        Warning
        -------
        Deleting raw_responses.jsonl removes the authoritative record of
        completed API calls. Use with care.
        """

        files = [
            self.raw_responses_path,
            self.run_log_path,
            self.errors_path,
            self.checkpoint_state_path,
        ]
        existing = [path for path in files if path.exists()]

        if not dry_run:
            for path in existing:
                path.unlink()

        return existing
