
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])

def safe_rate(series):
    if len(series) == 0:
        return None
    return float(pd.Series(series).mean())

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prompt_generation"] = out["prompt_id"].map(lambda x: "v2_abstention" if "_v2_" in str(x) else "v1")
    out["uses_abstention_gate"] = out["prompt_id"].astype(str).str.contains("abstention", case=False, na=False)
    out["gold_schema_present"] = np.where(out["sentence_type"].eq("control_weak_schema"), "no", "yes")
    if "schema_present" not in out.columns:
        out["schema_present"] = np.where(out["main_image_schema"].eq("NONE"), "no", "yes")
    out["is_control"] = out["sentence_type"].eq("control_weak_schema")
    out["control_correct"] = out["is_control"] & out["literal_or_metaphorical"].eq("control") & out["main_image_schema"].eq("NONE")
    out["control_false_positive_schema"] = out["is_control"] & out["main_image_schema"].notna() & ~out["main_image_schema"].eq("NONE")
    out["schema_present_correct"] = out["schema_present"].eq(out["gold_schema_present"])
    out["primary_schema_correct"] = out["main_image_schema"].eq(out["expected_schema_primary"])
    out["lm_correct"] = out["literal_or_metaphorical"].eq(out["expected_literal_or_metaphorical"])
    return out

def summarize(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        controls = g[g["is_control"]]
        non_controls = g[~g["is_control"]]
        rows.append({
            **dict(zip(group_cols, keys)),
            "n": len(g),
            "schema_present_accuracy": safe_rate(g["schema_present_correct"]),
            "primary_schema_accuracy": safe_rate(g["primary_schema_correct"]),
            "literal_metaphorical_accuracy": safe_rate(g["lm_correct"]),
            "control_accuracy": safe_rate(controls["control_correct"]) if len(controls) else None,
            "control_false_positive_schema_rate": safe_rate(controls["control_false_positive_schema"]) if len(controls) else None,
            "non_control_lm_accuracy": safe_rate(non_controls["lm_correct"]) if len(non_controls) else None,
        })
    return pd.DataFrame(rows)

def main():
    root = Path(".").resolve()
    parsed = read_jsonl(root / "data" / "outputs" / "parsed_responses.jsonl")
    structured = add_derived_columns(parsed[parsed["parse_status"].eq("parsed")].copy())
    out_dir = root / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    summarize(structured, ["prompt_id", "prompt_family", "prompt_generation"]).to_csv(out_dir / "analysis_prompt_metrics_v2.csv", index=False)
    summarize(structured, ["provider", "model_id", "prompt_id"]).to_csv(out_dir / "analysis_model_prompt_metrics_v2.csv", index=False)
    summarize(structured, ["expected_schema_primary", "prompt_generation"]).to_csv(out_dir / "analysis_schema_family_metrics_v2.csv", index=False)
    summarize(structured, ["prompt_id", "sentence_type"]).to_csv(out_dir / "analysis_sentence_type_metrics_v2.csv", index=False)

    print("Exported analysis tables to data/outputs")

if __name__ == "__main__":
    main()
