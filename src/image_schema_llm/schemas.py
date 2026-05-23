from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration loaded from data/inputs/models.jsonl."""

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
    Prompt configuration loaded from data/inputs/prompts.jsonl.

    recommended_max_output_tokens is optional but important for this project:
    structured prompts should not be constrained by a short naïve-prompt output
    limit, otherwise JSON responses may be truncated.
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
    recommended_max_output_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionConfig:
    """Inference condition loaded from data/inputs/conditions.jsonl."""

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
    """Sentence annotation loaded from data/gold/sentences_v1.jsonl."""

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
    """One model × prompt × condition × sentence × repetition permutation."""

    run_key: str
    run_index: int
    model: ModelConfig
    prompt: PromptConfig
    condition: ConditionConfig
    sentence: SentenceRecord
    repetition_index: int
    system_message: str
    user_prompt: str

    @property
    def effective_max_output_tokens(self) -> int:
        """
        Use the larger of condition max_output_tokens and prompt-level
        recommended_max_output_tokens.

        This prevents structured JSON prompts from being truncated by short
        generic condition limits.
        """
        prompt_limit = self.prompt.recommended_max_output_tokens or 0
        return max(self.condition.max_output_tokens, prompt_limit)

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "run_key": self.run_key,
            "run_index": self.run_index,
            "model_id": self.model.model_id,
            "provider": self.model.provider,
            "model_name": self.model.model_name,
            "model_snapshot": self.model.model_snapshot,
            "prompt_id": self.prompt.prompt_id,
            "prompt_family": self.prompt.prompt_family,
            "prompt_version": self.prompt.prompt_version,
            "condition_id": self.condition.condition_id,
            "condition_family": self.condition.condition_family,
            "condition_version": self.condition.condition_version,
            "temperature": self.condition.temperature,
            "top_p": self.condition.top_p,
            "condition_max_output_tokens": self.condition.max_output_tokens,
            "recommended_max_output_tokens": self.prompt.recommended_max_output_tokens,
            "effective_max_output_tokens": self.effective_max_output_tokens,
            "sentence_id": self.sentence.sentence_id,
            "sentence_type": self.sentence.sentence_type,
            "expected_schema_primary": self.sentence.expected_schema_primary,
            "expected_literal_or_metaphorical": self.sentence.expected_literal_or_metaphorical,
            "repetition_index": self.repetition_index,
            "system_message": self.system_message,
            "user_prompt": self.user_prompt,
        }
