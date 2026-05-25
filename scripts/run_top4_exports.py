
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".").resolve()
OUT = PROJECT_ROOT / "data" / "outputs" / "top4_investigations"
OUT.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])

def prompt_generation(prompt_id):
    prompt_id = str(prompt_id)
    return "v2_abstention" if "v2" in prompt_id or "abstention" in prompt_id else "v1"

def prompt_base(prompt_id):
    prompt_id = str(prompt_id)
    if "direct_schema" in prompt_id:
        return "direct_schema"
    if "structured_roles" in prompt_id:
        return "structured_roles"
    return "other"

def add_derived(df):
    out = df.copy()
    out["prompt_generation"] = out["prompt_id"].map(prompt_generation)
    out["prompt_base"] = out["prompt_id"].map(prompt_base)
    out["is_control"] = out["sentence_type"].eq("control_weak_schema")
    out["is_non_control"] = ~out["is_control"]
    out["gold_schema_present"] = np.where(out["is_control"], "no", "yes")
    if "schema_present" not in out.columns:
        out["schema_present"] = np.where(out["main_image_schema"].eq("NONE"), "no", "yes")
    out["schema_present_correct"] = out["schema_present"].eq(out["gold_schema_present"])
    out["primary_schema_correct"] = out["main_image_schema"].eq(out["expected_schema_primary"])
    out["lm_correct"] = out["literal_or_metaphorical"].eq(out["expected_literal_or_metaphorical"])
    out["control_correct"] = out["is_control"] & out["literal_or_metaphorical"].eq("control") & out["main_image_schema"].eq("NONE")
    out["control_false_positive_schema"] = out["is_control"] & out["main_image_schema"].notna() & ~out["main_image_schema"].eq("NONE")
    out["predicted_none_control"] = out["schema_present"].eq("no") | out["main_image_schema"].eq("NONE") | out["literal_or_metaphorical"].eq("control")
    out["over_abstained_on_non_control"] = out["is_non_control"] & out["predicted_none_control"]
    return out

def save(df, name):
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"Wrote {path}")

def main():
    parsed = add_derived(read_jsonl(PROJECT_ROOT / "data" / "outputs" / "parsed_responses.jsonl"))
    structured = parsed[parsed["parse_status"].eq("parsed")].copy()

    # Prompt summary
    prompt_summary = (
        structured.groupby(["prompt_id", "prompt_generation", "prompt_base"])
        .agg(
            n=("run_key", "count"),
            schema_present_accuracy=("schema_present_correct", "mean"),
            primary_schema_accuracy=("primary_schema_correct", "mean"),
            lm_accuracy=("lm_correct", "mean"),
            control_accuracy=("control_correct", "mean"),
            control_false_positive_rate=("control_false_positive_schema", "mean"),
            non_control_over_abstention_rate=("over_abstained_on_non_control", "mean"),
        )
        .reset_index()
    )
    save(prompt_summary, "summary_prompt_metrics_top4.csv")

    # Provider summary
    provider_summary = (
        structured.groupby(["provider", "model_id", "prompt_id"])
        .agg(
            n=("run_key", "count"),
            schema_present_accuracy=("schema_present_correct", "mean"),
            primary_schema_accuracy=("primary_schema_correct", "mean"),
            lm_accuracy=("lm_correct", "mean"),
            control_accuracy=("control_correct", "mean"),
            control_false_positive_rate=("control_false_positive_schema", "mean"),
            non_control_over_abstention_rate=("over_abstained_on_non_control", "mean"),
        )
        .reset_index()
    )
    save(provider_summary, "summary_provider_prompt_metrics_top4.csv")

    # Schema family summary
    schema_summary = (
        structured.groupby(["expected_schema_primary", "prompt_generation"])
        .agg(
            n=("run_key", "count"),
            primary_schema_accuracy=("primary_schema_correct", "mean"),
            lm_accuracy=("lm_correct", "mean"),
            schema_present_accuracy=("schema_present_correct", "mean"),
            control_false_positive_rate=("control_false_positive_schema", "mean"),
        )
        .reset_index()
    )
    save(schema_summary, "summary_schema_family_metrics_top4.csv")

    # Residual v2 control false positives
    v2_fp = structured[
        structured["prompt_generation"].eq("v2_abstention")
        & structured["is_control"]
        & structured["control_false_positive_schema"]
    ].copy()
    save(v2_fp, "residual_v2_control_false_positives_top4.csv")

if __name__ == "__main__":
    main()
