#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


STRICT_JSON_INSTRUCTIONS = """
Return only one valid JSON object.
Do not include markdown.
Do not wrap the answer in code fences.
Do not include commentary before or after the JSON.
Use double quotes for all JSON keys and string values.
Use empty strings "" or empty arrays [] when a field is not applicable.
Keep interpretation under 25 words.
Keep schema_explanation under 25 words.
"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch prompts.jsonl for better structured raw output.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true", help="Write changes. Without this, dry-run only.")
    args = parser.parse_args()

    path = args.project_root.resolve() / "data" / "inputs" / "prompts.jsonl"
    backup = path.with_suffix(".jsonl.bak_raw_quality")

    records = read_jsonl(path)
    patched = []

    for record in records:
        r = dict(record)
        family = r.get("prompt_family")

        if family == "naive":
            r["recommended_max_output_tokens"] = r.get("recommended_max_output_tokens") or 250

        elif family == "direct_schema":
            r["recommended_max_output_tokens"] = max(int(r.get("recommended_max_output_tokens") or 0), 700)
            if STRICT_JSON_INSTRUCTIONS.strip() not in r.get("system_message", ""):
                r["system_message"] = r.get("system_message", "").rstrip() + "\n\n" + STRICT_JSON_INSTRUCTIONS.strip()

        elif family == "structured_role_based":
            r["recommended_max_output_tokens"] = max(int(r.get("recommended_max_output_tokens") or 0), 1200)
            if STRICT_JSON_INSTRUCTIONS.strip() not in r.get("system_message", ""):
                r["system_message"] = r.get("system_message", "").rstrip() + "\n\n" + STRICT_JSON_INSTRUCTIONS.strip()

        patched.append(r)

    print(f"Would patch: {path}")
    for r in patched:
        print(f"{r.get('prompt_id')}: family={r.get('prompt_family')} recommended_max_output_tokens={r.get('recommended_max_output_tokens')}")

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
