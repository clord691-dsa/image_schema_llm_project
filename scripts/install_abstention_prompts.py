
#!/usr/bin/env python
from __future__ import annotations
import argparse, json, shutil
from datetime import datetime
from pathlib import Path

def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Install v2 abstention prompts into prompts.jsonl.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    if args.replace and args.append:
        raise SystemExit("Use either --replace or --append, not both.")
    if not args.replace and not args.append:
        raise SystemExit("Specify either --replace or --append.")

    project_root = args.project_root.resolve()
    target = project_root / "data" / "inputs" / "prompts.jsonl"
    source = args.source or (project_root / "data" / "inputs" / "prompts_v2_abstention.jsonl")

    if not source.exists():
        raise FileNotFoundError(source)
    if not target.exists():
        raise FileNotFoundError(target)

    backup = target.with_name(f"prompts.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    shutil.copy2(target, backup)

    new_records = read_jsonl(source)
    if args.replace:
        final_records = new_records
    else:
        by_id = {r["prompt_id"]: r for r in read_jsonl(target)}
        for r in new_records:
            by_id[r["prompt_id"]] = r
        final_records = list(by_id.values())

    write_jsonl(target, final_records)
    print(f"Backed up existing prompts to: {backup}")
    print(f"Wrote updated prompts to: {target}")
    print(f"Total prompt records: {len(final_records)}")

if __name__ == "__main__":
    main()
