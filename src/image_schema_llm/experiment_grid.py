from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path

from image_schema_llm.config import ProjectPaths
from image_schema_llm.jsonl_utils import read_jsonl, write_jsonl
from image_schema_llm.loaders import (
    enabled_conditions,
    enabled_models,
    load_conditions,
    load_models,
    load_prompts,
    load_sentences,
)
from image_schema_llm.prompts import build_user_prompt
from image_schema_llm.schemas import ConditionConfig, ExperimentJob, ModelConfig, PromptConfig, SentenceRecord

RUN_KEY_SEPARATOR = "|"


def make_run_key(
    model_id: str,
    prompt_id: str,
    condition_id: str,
    sentence_id: str,
    repetition_index: int,
) -> str:
    """Build the stable human-readable run key for one permutation."""
    return RUN_KEY_SEPARATOR.join(
        [model_id, prompt_id, condition_id, sentence_id, str(repetition_index)]
    )


def parse_run_key(run_key: str) -> dict[str, str | int]:
    """Parse a run key back into component IDs."""
    parts = run_key.split(RUN_KEY_SEPARATOR)
    if len(parts) != 5:
        raise ValueError(f"Invalid run_key {run_key!r}: expected 5 parts")
    model_id, prompt_id, condition_id, sentence_id, repetition_index = parts
    return {
        "model_id": model_id,
        "prompt_id": prompt_id,
        "condition_id": condition_id,
        "sentence_id": sentence_id,
        "repetition_index": int(repetition_index),
    }


def make_run_hash(run_key: str) -> str:
    """Return a compact deterministic hash of a run key."""
    return hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:16]


def build_experiment_grid(
    models: list[ModelConfig],
    prompts: list[PromptConfig],
    conditions: list[ConditionConfig],
    sentences: list[SentenceRecord],
) -> Iterator[ExperimentJob]:
    """
    Yield model × prompt × condition × sentence × repetition jobs.

    Disabled models and disabled conditions are skipped. The iteration order is
    deterministic: model -> prompt -> condition -> sentence -> repetition.
    """
    run_index = 0
    for model in enabled_models(models):
        for prompt in prompts:
            for condition in enabled_conditions(conditions):
                for sentence in sentences:
                    for repetition_index in range(condition.repetitions):
                        run_key = make_run_key(
                            model.model_id,
                            prompt.prompt_id,
                            condition.condition_id,
                            sentence.sentence_id,
                            repetition_index,
                        )
                        yield ExperimentJob(
                            run_key=run_key,
                            run_index=run_index,
                            model=model,
                            prompt=prompt,
                            condition=condition,
                            sentence=sentence,
                            repetition_index=repetition_index,
                            system_message=prompt.system_message,
                            user_prompt=build_user_prompt(prompt, sentence),
                        )
                        run_index += 1


def load_completed_run_keys(raw_responses_path: Path) -> set[str]:
    """
    Load completed run keys from raw_responses.jsonl.

    A job counts as complete only when status == "success".
    """
    if not raw_responses_path.exists():
        return set()
    return {
        r["run_key"]
        for r in read_jsonl(raw_responses_path)
        if r.get("status") == "success" and r.get("run_key")
    }


def filter_pending_jobs(
    jobs: Iterable[ExperimentJob],
    completed_run_keys: set[str],
) -> Iterator[ExperimentJob]:
    """Yield jobs whose run_key is not already completed."""
    for job in jobs:
        if job.run_key not in completed_run_keys:
            yield job


def load_experiment_inputs(project_root: Path) -> tuple[
    list[ModelConfig],
    list[PromptConfig],
    list[ConditionConfig],
    list[SentenceRecord],
]:
    """Load all four canonical input files from the project folder structure."""
    paths = ProjectPaths(project_root)
    return (
        load_models(paths.models_path),
        load_prompts(paths.prompts_path),
        load_conditions(paths.conditions_path),
        load_sentences(paths.sentences_path),
    )


def build_grid_from_project(project_root: Path) -> list[ExperimentJob]:
    """Load canonical inputs and return the full enabled experiment grid."""
    return list(build_experiment_grid(*load_experiment_inputs(project_root)))


def write_experiment_manifest(project_root: Path, jobs: Iterable[ExperimentJob]) -> Path:
    """Write a JSONL manifest of planned jobs to data/outputs."""
    paths = ProjectPaths(project_root)
    write_jsonl(paths.manifest_path, [job.to_manifest_record() for job in jobs])
    return paths.manifest_path
