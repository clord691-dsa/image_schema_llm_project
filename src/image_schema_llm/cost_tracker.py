from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from image_schema_llm.config import ProjectPaths
from image_schema_llm.jsonl_utils import append_jsonl, read_jsonl
from image_schema_llm.runtime_config import RuntimeConfig, load_runtime_config
from image_schema_llm.schemas import ModelConfig


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CostEstimate:
    """
    Estimated token cost for one model call.
    """

    input_tokens: int
    output_tokens: int
    estimated_cost: float
    currency: str


@dataclass
class CostTotals:
    """
    Runtime cumulative cost totals.
    """

    global_total: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)
    by_provider: dict[str, float] = field(default_factory=dict)
    calls_by_model: dict[str, int] = field(default_factory=dict)
    calls_by_provider: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_total": self.global_total,
            "by_model": self.by_model,
            "by_provider": self.by_provider,
            "calls_by_model": self.calls_by_model,
            "calls_by_provider": self.calls_by_provider,
        }


def estimate_cost_from_usage(
    *,
    model: ModelConfig,
    input_tokens: int | None,
    output_tokens: int | None,
    use_cached_input_price: bool = False,
) -> CostEstimate:
    """
    Estimate one API call's cost from provider-reported token usage.

    Pricing is read from models.jsonl via ModelConfig.
    """

    in_tokens = int(input_tokens or 0)
    out_tokens = int(output_tokens or 0)

    input_rate = model.input_cost_per_1m_tokens
    if use_cached_input_price and model.cached_input_cost_per_1m_tokens is not None:
        input_rate = model.cached_input_cost_per_1m_tokens

    estimated_cost = (in_tokens / 1_000_000) * input_rate
    estimated_cost += (out_tokens / 1_000_000) * model.output_cost_per_1m_tokens

    return CostEstimate(
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        estimated_cost=estimated_cost,
        currency=model.currency,
    )


class RuntimeCostTracker:
    """
    Runtime cost tracker with spend-threshold protection.

    Design
    ------
    - Reads runtime budget from data/inputs/runtime_config.json.
    - Rebuilds cumulative totals from data/outputs/cost_log.jsonl on startup.
    - Checks whether the spend threshold has been reached before an API call.
    - Appends a cost record after each successful API response.
    - Writes data/outputs/cost_summary.json after each cost update.
    """

    def __init__(
        self,
        *,
        cost_log_path: Path,
        cost_summary_path: Path,
        spend_threshold: float,
        currency: str = "USD",
    ) -> None:
        self.cost_log_path = cost_log_path
        self.cost_summary_path = cost_summary_path
        self.spend_threshold = float(spend_threshold)
        self.currency = currency
        self.totals = self.rebuild_totals_from_log()

    @classmethod
    def from_project(
        cls,
        *,
        project_root: Path,
        spend_threshold: float | None = None,
        currency: str | None = None,
    ) -> "RuntimeCostTracker":
        """
        Construct a tracker from canonical project paths.

        If spend_threshold or currency are omitted, they are loaded from:

            data/inputs/runtime_config.json
        """

        paths = ProjectPaths(project_root)
        runtime_config = load_runtime_config(project_root)

        threshold = (
            float(spend_threshold)
            if spend_threshold is not None
            else runtime_config.spend_threshold
        )
        selected_currency = currency or runtime_config.currency

        return cls(
            cost_log_path=paths.outputs_dir / runtime_config.cost_log_filename,
            cost_summary_path=paths.outputs_dir / runtime_config.cost_summary_filename,
            spend_threshold=threshold,
            currency=selected_currency,
        )

    @classmethod
    def from_runtime_config(
        cls,
        *,
        project_root: Path,
        runtime_config: RuntimeConfig,
    ) -> "RuntimeCostTracker":
        """
        Construct a tracker from a RuntimeConfig object.
        """

        paths = ProjectPaths(project_root)
        return cls(
            cost_log_path=paths.outputs_dir / runtime_config.cost_log_filename,
            cost_summary_path=paths.outputs_dir / runtime_config.cost_summary_filename,
            spend_threshold=runtime_config.spend_threshold,
            currency=runtime_config.currency,
        )

    def rebuild_totals_from_log(self) -> CostTotals:
        totals = CostTotals(
            global_total=0.0,
            by_model=defaultdict(float),
            by_provider=defaultdict(float),
            calls_by_model=defaultdict(int),
            calls_by_provider=defaultdict(int),
        )

        if not self.cost_log_path.exists():
            return self._freeze_totals(totals)

        for record in read_jsonl(self.cost_log_path):
            if record.get("status", "success") not in {"success", "cost_recorded"}:
                continue

            model_id = record.get("model_id", "unknown_model")
            provider = record.get("provider", "unknown_provider")
            cost = float(record.get("estimated_cost") or 0.0)

            totals.global_total += cost
            totals.by_model[model_id] += cost
            totals.by_provider[provider] += cost
            totals.calls_by_model[model_id] += 1
            totals.calls_by_provider[provider] += 1

        return self._freeze_totals(totals)

    @staticmethod
    def _freeze_totals(totals: CostTotals) -> CostTotals:
        totals.by_model = dict(totals.by_model)
        totals.by_provider = dict(totals.by_provider)
        totals.calls_by_model = dict(totals.calls_by_model)
        totals.calls_by_provider = dict(totals.calls_by_provider)
        return totals

    def threshold_reached(self) -> bool:
        return self.totals.global_total >= self.spend_threshold

    def remaining_budget(self) -> float:
        return max(0.0, self.spend_threshold - self.totals.global_total)

    def can_start_next_call(self) -> bool:
        return not self.threshold_reached()

    def record_api_usage(
        self,
        *,
        run_key: str,
        model: ModelConfig,
        input_tokens: int | None,
        output_tokens: int | None,
        provider: str,
        use_cached_input_price: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        estimate = estimate_cost_from_usage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            use_cached_input_price=use_cached_input_price,
        )

        projected_global_total = self.totals.global_total + estimate.estimated_cost

        record = {
            "created_at": utc_now_iso(),
            "status": "cost_recorded",
            "run_key": run_key,
            "provider": provider,
            "model_id": model.model_id,
            "model_name": model.model_name,
            "input_tokens": estimate.input_tokens,
            "output_tokens": estimate.output_tokens,
            "estimated_cost": estimate.estimated_cost,
            "currency": estimate.currency,
            "input_cost_per_1m_tokens": (
                model.cached_input_cost_per_1m_tokens
                if use_cached_input_price and model.cached_input_cost_per_1m_tokens is not None
                else model.input_cost_per_1m_tokens
            ),
            "output_cost_per_1m_tokens": model.output_cost_per_1m_tokens,
            "pricing_checked_at": model.pricing_checked_at,
            "cumulative_global_cost_before": self.totals.global_total,
            "cumulative_global_cost_after": projected_global_total,
            "spend_threshold": self.spend_threshold,
            "remaining_budget_after": max(0.0, self.spend_threshold - projected_global_total),
            "threshold_reached_after": projected_global_total >= self.spend_threshold,
            "metadata": metadata or {},
        }

        append_jsonl(self.cost_log_path, record)
        self.totals = self.rebuild_totals_from_log()
        self.write_summary()
        return record

    def write_stop_record(
        self,
        *,
        reason: str,
        run_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "created_at": utc_now_iso(),
            "status": "stopped",
            "reason": reason,
            "run_key": run_key,
            "cumulative_global_cost": self.totals.global_total,
            "spend_threshold": self.spend_threshold,
            "remaining_budget": self.remaining_budget(),
            "metadata": metadata or {},
        }
        append_jsonl(self.cost_log_path, record)
        self.write_summary()
        return record

    def write_summary(self) -> Path:
        summary = self.summary_dict()
        summary["created_at"] = utc_now_iso()
        summary["cost_log_path"] = str(self.cost_log_path)

        self.cost_summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.cost_summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.cost_summary_path

    def summary_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "spend_threshold": self.spend_threshold,
            "remaining_budget": self.remaining_budget(),
            "threshold_reached": self.threshold_reached(),
            "totals": self.totals.to_dict(),
        }
