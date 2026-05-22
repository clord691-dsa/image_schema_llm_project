#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

from image_schema_llm.manifest_runner import run_provider_manifest


def main() -> None:
    project_root = Path(".").resolve()
    max_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_provider_manifest(
        project_root=project_root,
        provider="openai",
        max_jobs=max_jobs,
        stop_on_error=True,
        dry_run=False,
    )


if __name__ == "__main__":
    main()
