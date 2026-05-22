#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from image_schema_llm.provider_runner import run_provider_job, select_next_provider_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the next pending Gemini experiment job.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run-key", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    job = select_next_provider_job(project_root=project_root, provider="google", run_key=args.run_key)
    if job is None:
        print("No pending Gemini job found, or requested run_key is already complete.")
        return

    print("Selected Gemini job")
    print("===================")
    print(f"run_key: {job.run_key}")
    print(f"model: {job.model.model_id} ({job.model.model_name})")
    print(f"prompt: {job.prompt.prompt_id}")
    print(f"condition: {job.condition.condition_id}")
    print(f"sentence: {job.sentence.sentence_id} | {job.sentence.text}")

    result = run_provider_job(project_root=project_root, job=job, dry_run=args.dry_run)
    print("\nResult")
    print("======")
    print(f"status: {result.status}")
    print(f"message: {result.message}")
    if result.raw_record:
        print(f"input_tokens: {result.raw_record.get('input_tokens')}")
        print(f"output_tokens: {result.raw_record.get('output_tokens')}")
        print(f"estimated_cost: {result.raw_record.get('estimated_cost')}")


if __name__ == "__main__":
    main()
