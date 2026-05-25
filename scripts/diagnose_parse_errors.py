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

    finish_reasons = Counter()
    for raw in raw_records:
        reason = raw.get("finish_reason")
        if not reason:
            meta = raw.get("provider_metadata") or {}
            reason = meta.get("finish_reason")
        finish_reasons[reason] += 1
    print(f"finish_reason_counts: {dict(finish_reasons)}")

    errors = [
        (raw, result)
        for raw, result in zip(raw_records, results)
        if result.get("parse_status") == "parse_error"
    ]
    partials = [
        (raw, result)
        for raw, result in zip(raw_records, results)
        if result.get("parse_status") == "partial"
    ]

    print(f"\nFirst {min(args.limit, len(partials))} partial recoveries")
    print("======================================")
    for raw, result in partials[:args.limit]:
        print(raw.get("run_key"))
        print(f"  provider: {raw.get('provider')}")
        print(f"  prompt_id: {raw.get('prompt_id')}")
        print(f"  parse_quality: {result.get('parse_quality')}")
        print(f"  literality: {result.get('literal_or_metaphorical')}")
        print(f"  schema: {result.get('main_image_schema')}")
        response = str(raw.get("raw_response", ""))
        print(f"  raw_preview: {response[:300].replace(chr(10), ' ')}")
        print()

    print(f"\nFirst {min(args.limit, len(errors))} remaining parse errors")
    print("=======================================")
    for raw, result in errors[:args.limit]:
        print(raw.get("run_key"))
        print(f"  prompt_family: {raw.get('prompt_family')}")
        print(f"  provider: {raw.get('provider')}")
        print(f"  finish_reason: {raw.get('finish_reason') or (raw.get('provider_metadata') or {}).get('finish_reason')}")
        print(f"  parse_error: {result.get('parse_error')}")
        response = str(raw.get("raw_response", ""))
        print(f"  raw_preview: {response[:500].replace(chr(10), ' ')}")
        print()


if __name__ == "__main__":
    main()
