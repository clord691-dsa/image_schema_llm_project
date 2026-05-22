#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_schema_llm.cost_tracker import RuntimeCostTracker
from image_schema_llm.runtime_config import load_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect runtime cost tracking state.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--spend-threshold",
        type=float,
        default=None,
        help="Optional override. Default is read from data/inputs/runtime_config.json.",
    )
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    runtime_config = load_runtime_config(project_root)

    tracker = RuntimeCostTracker.from_project(
        project_root=project_root,
        spend_threshold=args.spend_threshold or runtime_config.spend_threshold,
        currency=runtime_config.currency,
    )

    if args.write_summary:
        tracker.write_summary()

    summary = tracker.summary_dict()

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    print("Runtime cost summary")
    print("====================")
    print(f"spend_threshold: {summary['spend_threshold']}")
    print(f"global_total: {summary['totals']['global_total']:.8f}")
    print(f"remaining_budget: {summary['remaining_budget']:.8f}")
    print(f"threshold_reached: {summary['threshold_reached']}")

    print("\nCost by model")
    print("=============")
    for model_id, cost in summary["totals"]["by_model"].items():
        calls = summary["totals"]["calls_by_model"].get(model_id, 0)
        print(f"{model_id}: {cost:.8f} ({calls} calls)")

    print("\nCost by provider")
    print("================")
    for provider, cost in summary["totals"]["by_provider"].items():
        calls = summary["totals"]["calls_by_provider"].get(provider, 0)
        print(f"{provider}: {cost:.8f} ({calls} calls)")


if __name__ == "__main__":
    main()
