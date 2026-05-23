#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch models.jsonl provider capabilities for better raw quality.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    path = args.project_root.resolve() / "data" / "inputs" / "models.jsonl"
    backup = path.with_suffix(".jsonl.bak_raw_quality")
    records = read_jsonl(path)

    patched = []
    for record in records:
        r = dict(record)
        provider = r.get("provider")

        if provider == "anthropic":
            # Avoid Anthropic API error: temperature and top_p cannot both be supplied.
            r["supports_top_p"] = False
            r["supports_temperature"] = True

        if provider == "google":
            r["supports_json_mode"] = True
            r["supports_structured_outputs"] = True

        if provider == "openai":
            r["supports_json_mode"] = True
            r["supports_structured_outputs"] = True

        patched.append(r)

    print(f"Would patch: {path}")
    for r in patched:
        print(
            f"{r.get('model_id')}: provider={r.get('provider')} "
            f"supports_json_mode={r.get('supports_json_mode')} "
            f"supports_structured_outputs={r.get('supports_structured_outputs')} "
            f"supports_top_p={r.get('supports_top_p')}"
        )

    if args.execute:
        if not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"Backup written: {backup}")
        write_jsonl(path, patched)
        print(f"Patched: {path}")
    else:
        print("Dry run only. Re-run with --execute to write changes.")


if __name__ == "__main__":
    main()
