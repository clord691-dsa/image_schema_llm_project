
#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from image_schema_llm.analysis.abstention_metrics import compute_abstention_metrics
from image_schema_llm.config import ProjectPaths
from image_schema_llm.jsonl_utils import read_jsonl

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect NONE / weak-schema abstention-gate metrics.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--parsed-responses-path", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths(args.project_root.resolve())
    parsed_path = args.parsed_responses_path or (paths.outputs_dir / "parsed_responses.jsonl")
    parsed = pd.DataFrame(read_jsonl(parsed_path))
    metrics = compute_abstention_metrics(parsed).to_dict()

    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    print("NONE / weak-schema abstention metrics")
    print("=====================================")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    if "prompt_id" in parsed.columns:
        print("\\nBy prompt_id")
        print("============")
        for prompt_id, group in parsed.groupby("prompt_id"):
            m = compute_abstention_metrics(group).to_dict()
            print(prompt_id)
            for key, value in m.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
