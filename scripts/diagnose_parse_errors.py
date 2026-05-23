#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from image_schema_llm.config import ProjectPaths
from image_schema_llm.jsonl_utils import read_jsonl
from image_schema_llm.parsing.response_parser import parse_raw_response_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose parse statuses and recovery strategies.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    paths = ProjectPaths(args.project_root.resolve())
    raw_records = read_jsonl(paths.raw_responses_path)

    results = [parse_raw_response_record(record) for record in raw_records]

    print("Parser diagnostic")
    print("=================")
    print(f"raw_records: {len(raw_records)}")
    print(f"status_counts: {dict(Counter(r.get('parse_status') for r in results))}")
    print(f"parser_strategy_counts: {dict(Counter(r.get('parser_strategy') for r in results))}")
    print(f"parse_quality_counts: {dict(Counter(r.get('parse_quality') for r in results))}")
    print(f"usable_for_schema_accuracy: {sum(bool(r.get('usable_for_schema_accuracy')) for r in results)}")
    print(f"usable_for_lm_accuracy: {sum(bool(r.get('usable_for_lm_accuracy')) for r in results)}")

    errors = [
        (raw, result)
        for raw, result in zip(raw_records, results)
        if result.get("parse_status") == "parse_error"
    ]

    print(f"\nFirst {min(args.limit, len(errors))} remaining parse errors")
    print("=======================================")
    for raw, result in errors[:args.limit]:
        print(raw.get("run_key"))
        print(f"  prompt_family: {raw.get('prompt_family')}")
        print(f"  provider: {raw.get('provider')}")
        print(f"  parse_error: {result.get('parse_error')}")
        response = str(raw.get("raw_response", ""))
        print(f"  raw_preview: {response[:500].replace(chr(10), ' ')}")
        print()


if __name__ == "__main__":
    main()
