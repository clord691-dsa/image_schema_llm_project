#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from image_schema_llm.runtime_config import write_default_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Create data/inputs/runtime_config.json.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    path = write_default_runtime_config(
        args.project_root.resolve(),
        overwrite=args.overwrite,
    )
    print(f"Runtime config path: {path}")


if __name__ == "__main__":
    main()
