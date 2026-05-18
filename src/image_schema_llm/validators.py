from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from image_schema_llm.config import ProjectPaths
from image_schema_llm.loaders import (
    enabled_conditions,
    enabled_models,
    load_conditions,
    load_models,
    load_prompts,
    load_sentences,
)
from image_schema_llm.prompts import validate_prompt_template
from image_schema_llm.schemas import ConditionConfig, ModelConfig, PromptConfig, SentenceRecord

ALLOWED_PROVIDERS = {"openai", "anthropic", "google"}
ALLOWED_SENTENCE_TYPES = {"literal_spatial", "metaphorical_spatial", "control_weak_schema"}
ALLOWED_LM_LABELS = {"literal", "metaphorical", "control", "uncertain"}
PRIMARY_SCHEMAS = {"CONTAINER", "SOURCE_PATH_GOAL", "FORCE", "BLOCKAGE", "VERTICALITY", "SUPPORT_BALANCE", "NONE"}
SECONDARY_SCHEMAS = {
    "CONTAINER", "SOURCE_PATH_GOAL", "FORCE", "BLOCKAGE", "VERTICALITY",
    "SUPPORT_BALANCE", "SCALE", "STRUCTURE_STABILITY", "CENTER_PERIPHERY",
    "CENTRE_PERIPHERY", "LINK", "PART_WHOLE", "NEAR_FAR", "CYCLE", "CONTACT"
}


def _duplicate_errors(records: list[Any], attr: str, label: str) -> list[str]:
    counts = Counter(getattr(r, attr) for r in records)
    return [f"Duplicate {label}: {k}" for k, v in counts.items() if v > 1]


def validate_models(models: list[ModelConfig]) -> list[str]:
    errors = _duplicate_errors(models, "model_id", "model_id")
    for m in models:
        if m.provider not in ALLOWED_PROVIDERS:
            errors.append(f"{m.model_id}: invalid provider {m.provider!r}")
        if not m.model_name:
            errors.append(f"{m.model_id}: model_name is required")
        if m.input_cost_per_1m_tokens < 0 or m.output_cost_per_1m_tokens < 0:
            errors.append(f"{m.model_id}: token costs must be non-negative")
        if m.enabled and not m.api_key_env_var:
            errors.append(f"{m.model_id}: enabled model should define api_key_env_var")
    return errors


def validate_prompts(prompts: list[PromptConfig]) -> list[str]:
    errors = _duplicate_errors(prompts, "prompt_id", "prompt_id")
    for p in prompts:
        if not p.system_message:
            errors.append(f"{p.prompt_id}: system_message is required")
        if not p.user_prompt_template:
            errors.append(f"{p.prompt_id}: user_prompt_template is required")
        errors.extend(validate_prompt_template(p))
    return errors


def validate_conditions(conditions: list[ConditionConfig]) -> list[str]:
    errors = _duplicate_errors(conditions, "condition_id", "condition_id")
    for c in conditions:
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
    errors = _duplicate_errors(sentences, "sentence_id", "sentence_id")
    for s in sentences:
        if not s.text:
            errors.append(f"{s.sentence_id}: text is required")
        if s.sentence_type not in ALLOWED_SENTENCE_TYPES:
            errors.append(f"{s.sentence_id}: invalid sentence_type {s.sentence_type!r}")
        if s.expected_literal_or_metaphorical not in ALLOWED_LM_LABELS:
            errors.append(f"{s.sentence_id}: invalid expected_literal_or_metaphorical")
        if s.expected_schema_primary not in PRIMARY_SCHEMAS:
            errors.append(f"{s.sentence_id}: invalid expected_schema_primary {s.expected_schema_primary!r}")
        unknown = [x for x in s.expected_schema_secondary if x not in SECONDARY_SCHEMAS]
        if unknown:
            errors.append(f"{s.sentence_id}: unknown secondary schemas {unknown}")
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


def validate_all(project_root: Path) -> tuple[list[str], dict[str, Any]]:
    paths = ProjectPaths(project_root)
    models = load_models(paths.models_path)
    prompts = load_prompts(paths.prompts_path)
    conditions = load_conditions(paths.conditions_path)
    sentences = load_sentences(paths.sentences_path)

    errors = []
    errors.extend(validate_models(models))
    errors.extend(validate_prompts(prompts))
    errors.extend(validate_conditions(conditions))
    errors.extend(validate_sentences(sentences))

    enabled_reps = sum(c.repetitions for c in enabled_conditions(conditions))
    summary = {
        "models_total": len(models),
        "models_enabled": len(enabled_models(models)),
        "prompts_total": len(prompts),
        "conditions_total": len(conditions),
        "conditions_enabled": len(enabled_conditions(conditions)),
        "sentences_total": len(sentences),
        "enabled_condition_repetitions": enabled_reps,
        "estimated_enabled_grid_size": len(enabled_models(models)) * len(prompts) * len(sentences) * enabled_reps,
        "sentence_types": dict(Counter(s.sentence_type for s in sentences)),
        "primary_schemas": dict(Counter(s.expected_schema_primary for s in sentences)),
        "enabled_models": [m.model_id for m in enabled_models(models)],
        "enabled_conditions": [c.condition_id for c in enabled_conditions(conditions)],
        "prompts": [p.prompt_id for p in prompts],
    }
    return errors, summary
