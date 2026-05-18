from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """
    One model configuration record loaded from data/inputs/models.jsonl.

    Purpose
    -------
    Stores provider, model name, pricing, endpoint, and capability metadata.
    The API key itself is never stored here; only the environment variable name
    is stored.
    """

    model_id: str
    provider: str
    model_name: str
    enabled: bool
    input_cost_per_1m_tokens: float
    output_cost_per_1m_tokens: float
    currency: str
    api_key_env_var: str | None = None
    default_max_output_tokens: int | None = None
    model_snapshot: str | None = None
    model_family: str | None = None
    primary_recommended: bool = False
    selection_role: str | None = None
    selection_rationale: str | None = None
    cached_input_cost_per_1m_tokens: float | None = None
    pricing_mode: str | None = None
    pricing_checked_at: str | None = None
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    supports_temperature: bool | None = None
    supports_top_p: bool | None = None
    supports_json_mode: bool | None = None
    supports_structured_outputs: bool | None = None
    supports_function_calling: bool | None = None
    supports_reasoning_effort: bool | None = None
    tokenizer_strategy: str | None = None
    recommended_endpoint: str | None = None
    deprecation_status: str | None = None
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptConfig:
    """
    One prompt strategy record loaded from data/inputs/prompts.jsonl.

    Purpose
    -------
    Stores the system message, user prompt template, output format expectation,
    and output schema for a prompt family such as naive, direct_schema, or
    structured_role_based.
    """

    prompt_id: str
    prompt_family: str
    prompt_version: str
    description: str
    intended_use: str
    expected_output_format: str
    system_message: str
    user_prompt_template: str
    required_placeholders: list[str]
    output_schema: dict[str, Any]
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionConfig:
    """
    One inference condition loaded from data/inputs/conditions.jsonl.

    Purpose
    -------
    Stores inference settings such as temperature, top_p, max output tokens,
    and repetitions. Conditions define stochastic and budget-relevant
    experimental variants.
    """

    condition_id: str
    condition_family: str
    condition_version: str
    enabled: bool
    temperature: float
    top_p: float
    max_output_tokens: int
    repetitions: int
    random_seed: int | None = None
    description: str | None = None
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SentenceRecord:
    """
    One gold/draft sentence annotation loaded from data/gold/sentences_v1.jsonl.

    Purpose
    -------
    Provides the sentence text, expected image-schema labels, literal/metaphorical
    status, conceptual domains for metaphorical examples, and annotation metadata.
    """

    sentence_id: str
    text: str
    sentence_type: str
    expected_schema_primary: str
    expected_schema_secondary: list[str]
    expected_literal_or_metaphorical: str
    target_domain: list[str]
    source_domain: list[str]
    notes: str
    annotation_status: str
    annotator_id: str | None
    validated_at: str | None
    annotation_guideline_version: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentJob:
    """
    One model × prompt × condition × sentence × repetition permutation.

    Purpose
    -------
    This object represents one API call to be made later by the experiment
    runner. At this stage it is only used to preview and validate the grid.
    """

    run_key: str
    model: ModelConfig
    prompt: PromptConfig
    condition: ConditionConfig
    sentence: SentenceRecord
    repetition_index: int
    system_message: str
    user_prompt: str
