#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rerun key list from parsed responses.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--include-partial", action="store_true", help="Also rerun parse_status=partial records.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.project_root.resolve()
    parsed_path = root / "data" / "outputs" / "parsed_responses.jsonl"
    out_path = args.output or (root / "data" / "outputs" / "rerun_keys.txt")

    records = read_jsonl(parsed_path)

    statuses = {"parse_error"}
    if args.include_partial:
        statuses.add("partial")

    keys = [
        r["run_key"]
        for r in records
        if r.get("run_key") and r.get("parse_status") in statuses and r.get("prompt_family") != "naive"
    ]

    out_path.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")
    print(f"Wrote {len(keys)} keys to {out_path}")


if __name__ == "__main__":
    main()
