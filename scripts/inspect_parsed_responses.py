#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_schema_llm.analysis.basic_metrics import compute_basic_metrics
from image_schema_llm.config import ProjectPaths
from image_schema_llm.jsonl_utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect parsed response records.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--parsed-responses-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths(args.project_root.resolve())
    parsed_path = args.parsed_responses_path or (paths.outputs_dir / "parsed_responses.jsonl")

    records = read_jsonl(parsed_path)
    summary = compute_basic_metrics(records).to_dict()

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print("Parsed response summary")
    print("=======================")
    print(f"records: {summary['total_records']}")
    print(f"parse_status_counts: {summary['parse_status_counts']}")
    print(f"parser_strategy_counts: {summary['parser_strategy_counts']}")
    print(f"parse_quality_counts: {summary['parse_quality_counts']}")
    print(f"usable_for_schema_accuracy: {summary['usable_for_schema_accuracy']}")
    print(f"usable_for_lm_accuracy: {summary['usable_for_lm_accuracy']}")
    print(f"primary_schema_accuracy: {summary['primary_schema_accuracy']}")
    print(f"literal_metaphorical_accuracy: {summary['literal_metaphorical_accuracy']}")

    print(f"\nFirst {min(args.limit, len(records))} records")
    print("================")
    for record in records[:args.limit]:
        print(record.get("run_key"))
        print(f"  parse_status: {record.get('parse_status')}")
        print(f"  parser_strategy: {record.get('parser_strategy')}")
        print(f"  parse_quality: {record.get('parse_quality')}")
        print(f"  usable_schema: {record.get('usable_for_schema_accuracy')}")
        print(f"  usable_lm: {record.get('usable_for_lm_accuracy')}")
        print(f"  expected_schema: {record.get('expected_schema_primary')}")
        print(f"  predicted_schema: {record.get('main_image_schema')}")
        print(f"  expected_lm: {record.get('expected_literal_or_metaphorical')}")
        print(f"  predicted_lm: {record.get('literal_or_metaphorical')}")
        print(f"  parse_error: {record.get('parse_error')}")
        print()


if __name__ == "__main__":
    main()
