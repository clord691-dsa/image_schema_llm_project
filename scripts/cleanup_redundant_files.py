#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REDUNDANT_PATHS = [
    # Top-level duplicate package outside src/; imports should use src/image_schema_llm/clients.
    "clients",
    # Build/edit caches that should not live in the repository.
    "src/image_schema_llm.egg-info",
    ".pytest_cache",
    "scripts/__pycache__",
    "tests/__pycache__",
    # Obsolete/placeholder modules superseded by manifest_runner/provider runners.
    "src/image_schema_llm/runner.py",
    "src/image_schema_llm/openai_runner_budget_patch.py",
    "src/image_schema_llm/costing.py",
    "src/image_schema_llm/analysis/metrics.py",
    # Superseded parser tests from earlier development phases.
    "tests/test_response_parser.py",
    "tests/test_tolerant_response_parser.py",
    "tests/test_partial_recovery_parser.py",
    # Local patch backups; keep only if you specifically need audit history.
    "data/inputs/models.jsonl.bak_raw_quality",
    "data/inputs/prompts.jsonl.bak_raw_quality",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove redundant development files from the repo.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true", help="Actually delete files. Default is dry-run.")
    args = parser.parse_args()

    root = args.project_root.resolve()
    print("Repository cleanup")
    print("==================")
    print(f"project_root: {root}")
    print(f"mode: {'execute' if args.execute else 'dry-run'}")
    print()

    for rel in REDUNDANT_PATHS:
        path = root / rel
        if not path.exists():
            print(f"missing: {rel}")
            continue
        print(f"remove:  {rel}")
        if args.execute:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    if not args.execute:
        print("\nDry run only. Re-run with --execute to delete these paths.")


if __name__ == "__main__":
    main()
