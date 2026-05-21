from image_schema_llm.cost_tracker import estimate_cost_from_usage
from image_schema_llm.schemas import ModelConfig


def test_estimate_cost_from_usage():
    model = ModelConfig(
        model_id="m",
        provider="openai",
        model_name="test",
        enabled=True,
        input_cost_per_1m_tokens=1.0,
        output_cost_per_1m_tokens=2.0,
        currency="USD",
    )

    result = estimate_cost_from_usage(model=model, input_tokens=1_000_000, output_tokens=500_000)

    assert result.estimated_cost == 2.0
