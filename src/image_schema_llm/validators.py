from __future__ import annotations

from collections import Counter
from pathlib import Path

from image_schema_llm.loaders import (
    enabled_conditions,
    enabled_models,
    load_conditions,
    load_models,
    load_prompts,
    load_sentences,
)
from image_schema_llm.prompt_builder import validate_prompt_template
from image_schema_llm.schemas import ConditionConfig, ModelConfig, PromptConfig, SentenceRecord


ALLOWED_PROVIDERS = {"openai", "anthropic", "google"}
ALLOWED_SENTENCE_TYPES = {"literal_spatial", "metaphorical_spatial", "control_weak_schema"}
ALLOWED_LM_LABELS = {"literal", "metaphorical", "control", "uncertain"}
PRIMARY_SCHEMAS = {
    "CONTAINER",
    "SOURCE_PATH_GOAL",
    "FORCE",
    "BLOCKAGE",
    "VERTICALITY",
    "SUPPORT_BALANCE",
    "NONE",
}
SECONDARY_SCHEMAS = {
    "CONTAINER",
    "SOURCE_PATH_GOAL",
    "FORCE",
    "BLOCKAGE",
    "VERTICALITY",
    "SUPPORT_BALANCE",
    "SCALE",
    "STRUCTURE_STABILITY",
    "CENTER_PERIPHERY",
    "LINK",
    "PART_WHOLE",
    "NEAR_FAR",
    "CYCLE",
    "CONTACT",
}


def validate_models(models: list[ModelConfig]) -> list[str]:
    """Validate model configuration records."""

    errors: list[str] = []
    seen = set()

    for m in models:
        if m.model_id in seen:
            errors.append(f"Duplicate model_id: {m.model_id}")
        seen.add(m.model_id)

        if m.provider not in ALLOWED_PROVIDERS:
            errors.append(f"{m.model_id}: provider must be one of {sorted(ALLOWED_PROVIDERS)}")

        if m.input_cost_per_1m_tokens < 0 or m.output_cost_per_1m_tokens < 0:
            errors.append(f"{m.model_id}: token costs must be non-negative")

        if not m.model_name:
            errors.append(f"{m.model_id}: model_name is required")

        if not m.api_key_env_var:
            errors.append(f"{m.model_id}: api_key_env_var is recommended")

    return errors


def validate_prompts(prompts: list[PromptConfig]) -> list[str]:
    """Validate prompt records and prompt templates."""

    errors: list[str] = []
    seen = set()

    for p in prompts:
        if p.prompt_id in seen:
            errors.append(f"Duplicate prompt_id: {p.prompt_id}")
        seen.add(p.prompt_id)

        if not p.system_message:
            errors.append(f"{p.prompt_id}: system_message is required")

        if not p.user_prompt_template:
            errors.append(f"{p.prompt_id}: user_prompt_template is required")

        errors.extend(validate_prompt_template(p))

    return errors


def validate_conditions(conditions: list[ConditionConfig]) -> list[str]:
    """Validate condition records."""

    errors: list[str] = []
    seen = set()

    for c in conditions:
        if c.condition_id in seen:
            errors.append(f"Duplicate condition_id: {c.condition_id}")
        seen.add(c.condition_id)

        if c.temperature < 0:
            errors.append(f"{c.condition_id}: temperature must be non-negative")

        if not 0 <= c.top_p <= 1:
            errors.append(f"{c.condition_id}: top_p must be between 0 and 1")

        if c.max_output_tokens <= 0:
            errors.append(f"{c.condition_id}: max_output_tokens must be positive")

        if c.repetitions <= 0:
            errors.append(f"{c.condition_id}: repetitions must be positive")

    return errors


def validate_sentences(sentences: list[SentenceRecord]) -> list[str]:
    """Validate sentence annotation records."""

    errors: list[str] = []
    seen = set()

    for s in sentences:
        if s.sentence_id in seen:
            errors.append(f"Duplicate sentence_id: {s.sentence_id}")
        seen.add(s.sentence_id)

        if not s.text:
            errors.append(f"{s.sentence_id}: text is required")

        if s.sentence_type not in ALLOWED_SENTENCE_TYPES:
            errors.append(f"{s.sentence_id}: invalid sentence_type {s.sentence_type!r}")

        if s.expected_literal_or_metaphorical not in ALLOWED_LM_LABELS:
            errors.append(f"{s.sentence_id}: invalid expected_literal_or_metaphorical")

        if s.expected_schema_primary not in PRIMARY_SCHEMAS:
            errors.append(f"{s.sentence_id}: invalid expected_schema_primary {s.expected_schema_primary!r}")

        unknown_secondary = [x for x in s.expected_schema_secondary if x not in SECONDARY_SCHEMAS]
        if unknown_secondary:
            errors.append(f"{s.sentence_id}: unknown secondary schema labels {unknown_secondary}")

        if not isinstance(s.source_domain, list):
            errors.append(f"{s.sentence_id}: source_domain must be a list")

        if not isinstance(s.target_domain, list):
            errors.append(f"{s.sentence_id}: target_domain must be a list")

        if s.sentence_type in {"literal_spatial", "control_weak_schema"}:
            if s.source_domain:
                errors.append(f"{s.sentence_id}: literal/control source_domain should be empty")
            if s.target_domain:
                errors.append(f"{s.sentence_id}: literal/control target_domain should be empty")

        if s.sentence_type == "metaphorical_spatial":
            if not s.source_domain:
                errors.append(f"{s.sentence_id}: metaphorical sentence should have source_domain")
            if not s.target_domain:
                errors.append(f"{s.sentence_id}: metaphorical sentence should have target_domain")

    return errors


def validate_all(project_root: Path) -> tuple[list[str], dict[str, object]]:
    """
    Validate all canonical input files for the project.

    Returns
    -------
    tuple[list[str], dict[str, object]]
        Validation errors and summary statistics.
    """

    from image_schema_llm.config import ProjectPaths

    paths = ProjectPaths(project_root)

    models = load_models(paths.models_path)
    prompts = load_prompts(paths.prompts_path)
    conditions = load_conditions(paths.conditions_path)
    sentences = load_sentences(paths.sentences_path)

    errors: list[str] = []
    errors.extend(validate_models(models))
    errors.extend(validate_prompts(prompts))
    errors.extend(validate_conditions(conditions))
    errors.extend(validate_sentences(sentences))

    summary = {
        "models_total": len(models),
        "models_enabled": len(enabled_models(models)),
        "prompts_total": len(prompts),
        "conditions_total": len(conditions),
        "conditions_enabled": len(enabled_conditions(conditions)),
        "sentences_total": len(sentences),
        "sentence_types": dict(Counter(s.sentence_type for s in sentences)),
        "primary_schemas": dict(Counter(s.expected_schema_primary for s in sentences)),
        "estimated_enabled_grid_size": sum(
            c.repetitions for c in enabled_conditions(conditions)
        ) * len(enabled_models(models)) * len(prompts) * len(sentences),
    }

    return errors, summary
