#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_schema_llm.validators import validate_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate image-schema project input files.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    errors, summary = validate_all(args.project_root)

    if args.json:
        print(json.dumps({"errors": errors, "summary": summary}, indent=2, ensure_ascii=False))
        raise SystemExit(1 if errors else 0)

    print("Input validation summary")
    print("========================")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if errors:
        print("\nValidation errors")
        print("=================")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nNo validation errors found.")


if __name__ == "__main__":
    main()
