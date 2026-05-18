from __future__ import annotations

from collections.abc import Iterator

from image_schema_llm.loaders import enabled_conditions, enabled_models
from image_schema_llm.prompts.prompt_builder import build_user_prompt
from image_schema_llm.schemas import ConditionConfig, ExperimentJob, ModelConfig, PromptConfig, SentenceRecord


def make_run_key(
    model_id: str,
    prompt_id: str,
    condition_id: str,
    sentence_id: str,
    repetition_index: int,
) -> str:
    """
    Create the stable run key for one experiment permutation.

    Purpose
    -------
    This key is used to detect completed runs and safely resume the experiment.
    """

    return f"{model_id}|{prompt_id}|{condition_id}|{sentence_id}|{repetition_index}"


def build_experiment_grid(
    models: list[ModelConfig],
    prompts: list[PromptConfig],
    conditions: list[ConditionConfig],
    sentences: list[SentenceRecord],
) -> Iterator[ExperimentJob]:
    """
    Yield model × prompt × condition × sentence × repetition jobs.

    Inputs
    ------
    models:
        Model configurations. Disabled models are skipped.
    prompts:
        Prompt configurations.
    conditions:
        Inference conditions. Disabled conditions are skipped.
    sentences:
        Gold/draft sentence records.

    Outputs
    -------
    Iterator[ExperimentJob]
        One job per experimental API call.

    Purpose
    -------
    Defines the complete controlled experimental grid before API execution.
    """

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
                            model=model,
                            prompt=prompt,
                            condition=condition,
                            sentence=sentence,
                            repetition_index=repetition_index,
                            system_message=prompt.system_message,
                            user_prompt=build_user_prompt(prompt, sentence),
                        )


def completed_run_keys_from_raw_records(raw_records: list[dict]) -> set[str]:
    """
    Extract completed run keys from raw response records.

    Purpose
    -------
    Later used to resume interrupted experiments. A run is complete only when
    raw_responses.jsonl contains a success record for the run key.
    """

    return {
        record["run_key"]
        for record in raw_records
        if record.get("status") == "success" and record.get("run_key")
    }
