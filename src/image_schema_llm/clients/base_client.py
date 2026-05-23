from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelResponse:
    """Normalised response object returned by provider clients."""

    raw_response: str
    input_tokens: int | None
    output_tokens: int | None
    provider_response_id: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None


class BaseLLMClient(Protocol):
    def generate(
        self,
        *,
        system_message: str,
        user_prompt: str,
        model_name: str,
        temperature: float | None,
        top_p: float | None,
        max_output_tokens: int,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> ModelResponse:
        """Generate one model response."""
        ...
