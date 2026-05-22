from __future__ import annotations

from pathlib import Path

from image_schema_llm.cost_tracker import RuntimeCostTracker
from image_schema_llm.runtime_config import load_runtime_config


def assert_budget_available(
    *,
    project_root: Path,
    spend_threshold: float | None = None,
) -> RuntimeCostTracker:
    """
    Build a RuntimeCostTracker and raise RuntimeError if threshold is reached.

    The threshold is loaded from data/inputs/runtime_config.json unless an
    explicit spend_threshold override is supplied.
    """

    config = load_runtime_config(project_root)

    tracker = RuntimeCostTracker.from_project(
        project_root=project_root,
        spend_threshold=spend_threshold or config.spend_threshold,
        currency=config.currency,
    )

    if tracker.threshold_reached():
        tracker.write_stop_record(
            reason="spend_threshold_reached_before_call",
            metadata={"source": "cost_guard.assert_budget_available"},
        )
        raise RuntimeError(
            f"Spend threshold reached. "
            f"global_total={tracker.totals.global_total:.6f}, "
            f"threshold={tracker.spend_threshold:.6f}"
        )

    return tracker
