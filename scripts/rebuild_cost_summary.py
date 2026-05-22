#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from image_schema_llm.cost_tracker import RuntimeCostTracker
from image_schema_llm.runtime_config import load_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild cost_summary.json from cost_log.jsonl.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--spend-threshold",
        type=float,
        default=None,
        help="Optional override. Default is read from data/inputs/runtime_config.json.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    runtime_config = load_runtime_config(project_root)

    tracker = RuntimeCostTracker.from_project(
        project_root=project_root,
        spend_threshold=args.spend_threshold or runtime_config.spend_threshold,
        currency=runtime_config.currency,
    )
    path = tracker.write_summary()

    print(f"Wrote cost summary: {path}")
    print(f"global_total: {tracker.totals.global_total:.8f}")
    print(f"remaining_budget: {tracker.remaining_budget():.8f}")


if __name__ == "__main__":
    main()
