#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_schema_llm.parsing.pipeline import run_parsing_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse raw LLM responses into parsed_responses.jsonl.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--raw-responses-path", type=Path, default=None)
    parser.add_argument("--parsed-responses-path", type=Path, default=None)
    parser.add_argument("--include-non-success", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_parsing_pipeline(
        project_root=args.project_root.resolve(),
        raw_responses_path=args.raw_responses_path,
        parsed_responses_path=args.parsed_responses_path,
        only_success=not args.include_non_success,
    )

    payload = result.to_dict()

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Parsing pipeline complete")
    print("=========================")
    print(f"input_records: {payload['input_records']}")
    print(f"parsed_records: {payload['parsed_records']}")
    print(f"output_path: {payload['output_path']}")
    print(f"status_counts: {payload['status_counts']}")


if __name__ == "__main__":
    main()
