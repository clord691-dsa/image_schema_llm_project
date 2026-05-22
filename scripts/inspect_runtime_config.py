#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_schema_llm.runtime_config import load_runtime_config, runtime_config_path, validate_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect runtime configuration.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    config = load_runtime_config(project_root)
    errors = validate_runtime_config(config)

    payload = {
        "path": str(runtime_config_path(project_root)),
        "config": {
            "spend_threshold": config.spend_threshold,
            "currency": config.currency,
            "stop_on_error": config.stop_on_error,
            "dry_run": config.dry_run,
            "cost_log_filename": config.cost_log_filename,
            "cost_summary_filename": config.cost_summary_filename,
            "notes": config.notes,
        },
        "errors": errors,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Runtime configuration")
    print("=====================")
    print(f"path: {payload['path']}")
    for key, value in payload["config"].items():
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
