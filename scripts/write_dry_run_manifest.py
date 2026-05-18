#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from image_schema_llm.dry_run import DryRunSettings, run_dry_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Write dry-run manifest and summary.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--pending-only", action="store_true")
    parser.add_argument(
        "--output-token-strategy",
        choices=["condition_max", "half_condition_max", "zero"],
        default="condition_max",
    )
    args = parser.parse_args()

    settings = DryRunSettings(
        pending_only=args.pending_only,
        estimated_output_tokens_strategy=args.output_token_strategy,
    )

    _, summary = run_dry_run(
        args.project_root.resolve(),
        settings=settings,
        write_manifest=True,
        write_summary=True,
    )

    print(f"Wrote dry-run manifest: {summary.get('manifest_path')}")
    print(f"Wrote dry-run summary: {summary.get('summary_path')}")
    print(f"Records: {summary['records']}")
    print(f"Estimated total cost: {summary['estimated_total_cost']:.6f}")


if __name__ == "__main__":
    main()
