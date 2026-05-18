#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from image_schema_llm.config import ProjectPaths
from image_schema_llm.loaders import load_conditions, load_models, load_prompts, load_sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect image-schema input files.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory")
    args = parser.parse_args()

    paths = ProjectPaths(args.project_root)

    models = load_models(paths.models_path)
    prompts = load_prompts(paths.prompts_path)
    conditions = load_conditions(paths.conditions_path)
    sentences = load_sentences(paths.sentences_path)

    print("Files")
    print("=====")
    print(paths.models_path)
    print(paths.prompts_path)
    print(paths.conditions_path)
    print(paths.sentences_path)

    print("\nCounts")
    print("======")
    print(f"models: {len(models)}")
    print(f"prompts: {len(prompts)}")
    print(f"conditions: {len(conditions)}")
    print(f"sentences: {len(sentences)}")

    print("\nEnabled models")
    print("==============")
    for m in models:
        if m.enabled:
            print(f"- {m.model_id} ({m.provider}: {m.model_name})")

    print("\nPrompt families")
    print("===============")
    for p in prompts:
        print(f"- {p.prompt_id}: {p.prompt_family}")

    print("\nEnabled conditions")
    print("==================")
    for c in conditions:
        if c.enabled:
            print(f"- {c.condition_id}: temp={c.temperature}, top_p={c.top_p}, reps={c.repetitions}")

    print("\nSentence types")
    print("==============")
    print(dict(Counter(s.sentence_type for s in sentences)))

    print("\nPrimary schemas")
    print("===============")
    print(dict(Counter(s.expected_schema_primary for s in sentences)))


if __name__ == "__main__":
    main()
