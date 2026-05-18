#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

from image_schema_llm.dry_run import DryRunSettings, run_dry_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dry-run experiment plan without API calls.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory")
    parser.add_argument("--limit", type=int, default=10, help="Number of dry-run records to print")
    parser.add_argument("--pending-only", action="store_true", help="Exclude jobs already completed in raw_responses.jsonl")
    parser.add_argument("--write-manifest", action="store_true", help="Write data/outputs/dry_run_manifest.jsonl")
    parser.add_argument("--write-summary", action="store_true", help="Write data/outputs/dry_run_summary.json")
    parser.add_argument(
        "--output-token-strategy",
        choices=["condition_max", "half_condition_max", "zero"],
        default="condition_max",
        help="How to estimate output tokens before API responses exist",
    )
    parser.add_argument(
        "--token-char-ratio",
        type=float,
        default=4.0,
        help="Approximate characters per token for prompt text",
    )
    parser.add_argument(
        "--print-prompts",
        action="store_true",
        help="Print system/user prompt previews for the displayed records",
    )
    parser.add_argument("--json", action="store_true", help="Print summary as JSON")
    args = parser.parse_args()

    settings = DryRunSettings(
        pending_only=args.pending_only,
        estimated_output_tokens_strategy=args.output_token_strategy,
        token_char_ratio=args.token_char_ratio,
    )

    records, summary = run_dry_run(
        args.project_root.resolve(),
        settings=settings,
        write_manifest=args.write_manifest,
        write_summary=args.write_summary,
    )

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    print("Dry-run summary")
    print("===============")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nShowing first {min(args.limit, len(records))} dry-run records:\n")

    for record in islice(records, args.limit):
        print(record["run_key"])
        print(f"  status: {record['status']}")
        print(f"  model: {record['model_id']} ({record['provider']}: {record['model_name']})")
        print(f"  prompt: {record['prompt_id']} ({record['prompt_family']})")
        print(f"  condition: {record['condition_id']} temp={record['temperature']} top_p={record['top_p']}")
        print(f"  sentence: {record['sentence_id']} | {record['expected_schema_primary']}")
        print(f"  estimated_input_tokens: {record['estimated_input_tokens']}")
        print(f"  estimated_output_tokens: {record['estimated_output_tokens']}")
        print(f"  estimated_cost: {record['estimated_cost']:.8f} {record['currency']}")

        if args.print_prompts:
            print("  prompt previews are available in the generated grid/manifests, not in dry-run summary records.")
            print("  use preview_experiment_grid.py for full prompt previews.")

        print()


if __name__ == "__main__":
    main()
