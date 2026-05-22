from pathlib import Path

from image_schema_llm.cost_tracker import RuntimeCostTracker, estimate_cost_from_usage
from image_schema_llm.schemas import ModelConfig


def _model() -> ModelConfig:
    return ModelConfig(
        model_id="m1",
        provider="openai",
        model_name="test-model",
        enabled=True,
        input_cost_per_1m_tokens=1.0,
        output_cost_per_1m_tokens=2.0,
        currency="USD",
    )


def test_estimate_cost_from_usage():
    estimate = estimate_cost_from_usage(
        model=_model(),
        input_tokens=1_000_000,
        output_tokens=500_000,
    )
    assert estimate.estimated_cost == 2.0


def test_runtime_tracker_records_and_rebuilds(tmp_path: Path):
    tracker = RuntimeCostTracker(
        cost_log_path=tmp_path / "cost_log.jsonl",
        cost_summary_path=tmp_path / "cost_summary.json",
        spend_threshold=10.0,
    )

    tracker.record_api_usage(
        run_key="m1|p1|c1|s1|0",
        model=_model(),
        input_tokens=1_000,
        output_tokens=500,
        provider="openai",
    )

    rebuilt = RuntimeCostTracker(
        cost_log_path=tmp_path / "cost_log.jsonl",
        cost_summary_path=tmp_path / "cost_summary.json",
        spend_threshold=10.0,
    )

    assert rebuilt.totals.global_total == 0.002
    assert rebuilt.totals.by_model["m1"] == 0.002
    assert rebuilt.totals.by_provider["openai"] == 0.002


def test_threshold_reached(tmp_path: Path):
    tracker = RuntimeCostTracker(
        cost_log_path=tmp_path / "cost_log.jsonl",
        cost_summary_path=tmp_path / "cost_summary.json",
        spend_threshold=0.001,
    )

    tracker.record_api_usage(
        run_key="m1|p1|c1|s1|0",
        model=_model(),
        input_tokens=1_000,
        output_tokens=500,
        provider="openai",
    )

    assert tracker.threshold_reached()
