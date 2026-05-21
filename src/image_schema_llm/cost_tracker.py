from __future__ import annotations

from dataclasses import dataclass

from image_schema_llm.schemas import ModelConfig


@dataclass(frozen=True)
class CostEstimate:
    """Estimated token cost for one model call."""

    input_tokens: int
    output_tokens: int
    estimated_cost: float
    currency: str


def estimate_cost_from_usage(
    *,
    model: ModelConfig,
    input_tokens: int | None,
    output_tokens: int | None,
) -> CostEstimate:
    """
    Estimate one call's cost from provider-reported token usage.
    """

    in_tokens = input_tokens or 0
    out_tokens = output_tokens or 0

    estimated_cost = (in_tokens / 1_000_000) * model.input_cost_per_1m_tokens
    estimated_cost += (out_tokens / 1_000_000) * model.output_cost_per_1m_tokens

    return CostEstimate(
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        estimated_cost=estimated_cost,
        currency=model.currency,
    )
