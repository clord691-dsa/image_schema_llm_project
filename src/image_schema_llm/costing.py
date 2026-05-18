from __future__ import annotations

from dataclasses import dataclass

from image_schema_llm.schemas import ModelConfig


@dataclass(frozen=True)
class CostEstimate:
    """
    Estimated cost for one model call.

    Purpose
    -------
    Converts provider-reported token usage into a comparable cost record.
    """

    input_tokens: int
    output_tokens: int
    estimated_cost: float
    currency: str


def estimate_call_cost(
    model: ModelConfig,
    input_tokens: int,
    output_tokens: int,
    use_cached_input_price: bool = False,
) -> CostEstimate:
    """
    Estimate the API cost of one call.

    Inputs
    ------
    model:
        ModelConfig containing per-million-token costs.
    input_tokens:
        Provider-reported input token count.
    output_tokens:
        Provider-reported output token count.
    use_cached_input_price:
        Whether to use cached input pricing when available.

    Outputs
    -------
    CostEstimate

    Notes
    -----
    This is an estimate because provider billing rules can change. The pricing
    date in models.jsonl should be reported in the methodology.
    """

    input_rate = model.input_cost_per_1m_tokens
    if use_cached_input_price and model.cached_input_cost_per_1m_tokens is not None:
        input_rate = model.cached_input_cost_per_1m_tokens

    cost = (input_tokens / 1_000_000) * input_rate
    cost += (output_tokens / 1_000_000) * model.output_cost_per_1m_tokens

    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=cost,
        currency=model.currency,
    )
