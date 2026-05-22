#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_schema_llm.manifest_runner import run_provider_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all pending manifest jobs for one provider."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root directory.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "google"],
        required=True,
        help="Provider to run.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Optional cap for pilot runs. Omit to run all pending provider jobs.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between jobs for rate-limit safety.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first job that returns status=error.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after errors. Overrides runtime_config stop_on_error.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select jobs and print them without making API calls.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Force actual execution even if runtime_config has dry_run=true.",
    )
    parser.add_argument(
        "--print-prompts",
        action="store_true",
        help="Print a short prompt preview for each job.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final summary as JSON.",
    )
    args = parser.parse_args()

    if args.stop_on_error and args.continue_on_error:
        raise SystemExit("Use either --stop-on-error or --continue-on-error, not both.")

    if args.dry_run and args.execute:
        raise SystemExit("Use either --dry-run or --execute, not both.")

    stop_on_error = None
    if args.stop_on_error:
        stop_on_error = True
    elif args.continue_on_error:
        stop_on_error = False

    dry_run = None
    if args.dry_run:
        dry_run = True
    elif args.execute:
        dry_run = False

    summary = run_provider_manifest(
        project_root=args.project_root.resolve(),
        provider=args.provider,
        max_jobs=args.max_jobs,
        sleep_seconds=args.sleep_seconds,
        stop_on_error=stop_on_error,
        dry_run=dry_run,
        print_prompts=args.print_prompts,
    )

    if args.json:
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
