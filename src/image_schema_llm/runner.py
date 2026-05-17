from datetime import datetime, timezone
from typing import Dict

from image_schema_llm.checkpoint import load_completed_run_keys, should_skip_run
from image_schema_llm.clients import get_client
from image_schema_llm.config import ProjectPaths, RuntimeConfig
from image_schema_llm.cost_tracker import CostTracker
from image_schema_llm.experiment_grid import build_experiment_grid, make_run_key
from image_schema_llm.jsonl_utils import append_jsonl, read_jsonl
from image_schema_llm.prompts.prompt_builder import build_prompt


class ExperimentRunner:
    """
    Coordinates the full LLM experiment.

    Purpose:
        Runs every model × prompt × condition × sentence × repetition
        permutation, writes raw responses, monitors spend, and supports
        restart from the last successful iteration.

    Restart logic:
        On startup, the runner reads raw_responses.jsonl and builds a set of
        successful run keys. Any completed run is skipped.

    Cost logic:
        After each successful API call, the runner estimates cost, writes a
        cost log entry, and stops if the threshold is reached.
    """

    def __init__(self, paths: ProjectPaths, config: RuntimeConfig):
        """
        Inputs:
            paths:
                ProjectPaths object containing all input/output paths.
            config:
                RuntimeConfig object containing runtime options.

        Outputs:
            ExperimentRunner instance.
        """
        self.paths = paths
        self.config = config
        self.cost_tracker = CostTracker(spend_threshold=config.spend_threshold)

    def run(self) -> None:
        """
        Execute the full experiment loop.

        Inputs:
            None. Reads all input JSONL databases from ProjectPaths.

        Outputs:
            None. Writes output records to JSONL databases.

        Purpose:
            Main orchestration method for the experiment.
        """
        models = read_jsonl(self.paths.models_path)
        prompts = read_jsonl(self.paths.prompts_path)
        conditions = read_jsonl(self.paths.conditions_path)
        sentences = read_jsonl(self.paths.sentences_path)

        completed_run_keys = load_completed_run_keys(self.paths.raw_responses_path)

        for job in build_experiment_grid(models, prompts, conditions, sentences):
            run_key = make_run_key(job)

            if should_skip_run(run_key, completed_run_keys):
                continue

            if self.cost_tracker.threshold_reached():
                self._write_stop_record(job, reason="spend_threshold_reached")
                break

            try:
                self._run_single_job(job, run_key)
            except Exception as exc:
                self._write_error_record(job, exc)

                if self.config.stop_on_error:
                    break

    def _run_single_job(self, job: Dict, run_key: str) -> None:
        """
        Run one model × prompt × condition × sentence × repetition job.

        Inputs:
            job:
                One experiment job dictionary.
            run_key:
                Stable unique identifier for this job.

        Outputs:
            None. Writes raw response and cost log records.

        Purpose:
            Sends a single prompt to a single model and persists the result.
        """
        prompt_text = build_prompt(
            prompt_record=job["prompt_record"],
            sentence_record=job["sentence_record"],
        )

        if self.config.dry_run:
            return

        client = get_client(job["provider"])

        model_response = client.generate(
            prompt_text=prompt_text,
            model_name=job["model_name"],
            temperature=job["temperature"],
            top_p=job["top_p"],
            max_output_tokens=job["max_output_tokens"],
        )

        cost_estimate = self.cost_tracker.estimate_cost(
            model_record=job["model_record"],
            input_tokens=model_response.input_tokens or 0,
            output_tokens=model_response.output_tokens or 0,
        )

        self.cost_tracker.add_cost(
            model_id=job["model_id"],
            cost=cost_estimate.estimated_cost,
        )

        raw_record = self._build_raw_response_record(
            job=job,
            run_key=run_key,
            prompt_text=prompt_text,
            raw_response=model_response.raw_response,
            input_tokens=cost_estimate.input_tokens,
            output_tokens=cost_estimate.output_tokens,
            estimated_cost=cost_estimate.estimated_cost,
            provider_metadata=model_response.provider_metadata,
        )

        append_jsonl(self.paths.raw_responses_path, raw_record)

        cost_record = self._build_cost_record(
            job=job,
            run_key=run_key,
            estimated_cost=cost_estimate.estimated_cost,
        )

        append_jsonl(self.paths.cost_log_path, cost_record)

    def _build_raw_response_record(
        self,
        job: Dict,
        run_key: str,
        prompt_text: str,
        raw_response: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float,
        provider_metadata: Dict,
    ) -> Dict:
        """
        Build the raw response output record.

        Inputs:
            job: Experiment job dictionary.
            run_key: Unique run key.
            prompt_text: Exact prompt sent to the API.
            raw_response: Full raw model response.
            input_tokens: Input token count.
            output_tokens: Output token count.
            estimated_cost: Estimated cost for the call.
            provider_metadata: Provider-specific metadata.

        Outputs:
            Dictionary suitable for raw_responses.jsonl.

        Purpose:
            Centralises the output schema for raw model responses.
        """
        return {
            "run_key": run_key,
            "run_id": self._new_run_id(),
            "model_id": job["model_id"],
            "provider": job["provider"],
            "model_name": job["model_name"],
            "prompt_id": job["prompt_id"],
            "prompt_type": job["prompt_type"],
            "condition_id": job["condition_id"],
            "sentence_id": job["sentence_id"],
            "repetition_index": job["repetition_index"],
            "status": "success",
            "prompt_text": prompt_text,
            "raw_response": raw_response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": estimated_cost,
            "provider_metadata": provider_metadata,
            "created_at": self._utc_now(),
        }

    def _build_cost_record(
        self,
        job: Dict,
        run_key: str,
        estimated_cost: float,
    ) -> Dict:
        """
        Build one cost-log record.

        Inputs:
            job: Experiment job dictionary.
            run_key: Unique run key.
            estimated_cost: Estimated cost for current call.

        Outputs:
            Dictionary suitable for cost_log.jsonl.

        Purpose:
            Tracks incremental and cumulative spend.
        """
        return {
            "created_at": self._utc_now(),
            "run_key": run_key,
            "model_id": job["model_id"],
            "provider": job["provider"],
            "incremental_cost": estimated_cost,
            "cumulative_model_cost": self.cost_tracker.model_totals.get(
                job["model_id"], 0.0
            ),
            "global_cumulative_cost": self.cost_tracker.global_total,
            "spend_threshold": self.config.spend_threshold,
        }

    def _write_error_record(self, job: Dict, exc: Exception) -> None:
        """
        Write an API, connection, parsing, or runtime error to errors.jsonl.

        Inputs:
            job: Experiment job active when error occurred.
            exc: Exception object.

        Outputs:
            None.

        Purpose:
            Makes failures auditable and allows manual restart/debugging.
        """
        error_record = {
            "created_at": self._utc_now(),
            "model_id": job.get("model_id"),
            "provider": job.get("provider"),
            "prompt_id": job.get("prompt_id"),
            "condition_id": job.get("condition_id"),
            "sentence_id": job.get("sentence_id"),
            "repetition_index": job.get("repetition_index"),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "retryable": True,
        }

        append_jsonl(self.paths.errors_path, error_record)

    def _write_stop_record(self, job: Dict, reason: str) -> None:
        """
        Record why the loop stopped before completing all permutations.

        Inputs:
            job: Current job at the point of stopping.
            reason: Human-readable reason for stopping.

        Outputs:
            None.

        Purpose:
            Logs spend-threshold stops or manual stop conditions.
        """
        stop_record = {
            "created_at": self._utc_now(),
            "status": "stopped",
            "reason": reason,
            "model_id": job.get("model_id"),
            "prompt_id": job.get("prompt_id"),
            "condition_id": job.get("condition_id"),
            "sentence_id": job.get("sentence_id"),
            "global_cumulative_cost": self.cost_tracker.global_total,
            "spend_threshold": self.config.spend_threshold,
        }

        append_jsonl(self.paths.errors_path, stop_record)

    def _new_run_id(self) -> str:
        """
        Create a unique run ID.

        Inputs:
            None.

        Outputs:
            String run ID.

        Purpose:
            Human-readable identifier for output records.
        """
        return f"r_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

    def _utc_now(self) -> str:
        """
        Return current UTC timestamp.

        Inputs:
            None.

        Outputs:
            ISO-8601 UTC timestamp string.

        Purpose:
            Standardises timestamps across output databases.
        """
        return datetime.now(timezone.utc).isoformat()
