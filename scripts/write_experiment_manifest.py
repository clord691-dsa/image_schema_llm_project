#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from image_schema_llm.experiment_grid import build_grid_from_project, write_experiment_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Write full experiment grid manifest.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    jobs = build_grid_from_project(project_root)
    manifest_path = write_experiment_manifest(project_root, jobs)
    print(f"Wrote {len(jobs)} jobs to {manifest_path}")


if __name__ == "__main__":
    main()
