#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from image_schema_llm.checkpoint import CheckpointManager
from image_schema_llm.config import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset checkpoint/restart output files.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files. Without this flag, the script performs a dry run.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    paths = ProjectPaths(project_root)
    manager = CheckpointManager(paths.outputs_dir)

    dry_run = not args.execute
    affected = manager.reset_checkpoint_files(dry_run=dry_run)

    if dry_run:
        print("Dry run: the following files would be deleted:")
    else:
        print("Deleted the following files:")

    if not affected:
        print("- No checkpoint files found.")
    else:
        for path in affected:
            print(f"- {path}")


if __name__ == "__main__":
    main()
