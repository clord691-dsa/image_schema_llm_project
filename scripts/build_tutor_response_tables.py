
#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])

def safe_accuracy(df, pred_col, gold_col):
    sub = df[df[pred_col].notna() & df[gold_col].notna()]
    return None if sub.empty else float((sub[pred_col].astype(str) == sub[gold_col].astype(str)).mean())

def main():
    root = Path(".").resolve()
    parsed = read_jsonl(root / "data" / "outputs" / "parsed_responses.jsonl")
    structured = parsed[parsed["parse_status"].eq("parsed")].copy()
    if structured.empty:
        print("No structured parsed records available yet.")
        return
    metrics = (
        structured.groupby(["prompt_family", "sentence_type"])
        .apply(lambda g: pd.Series({
            "n": len(g),
            "schema_accuracy": safe_accuracy(g, "main_image_schema", "expected_schema_primary"),
            "literal_metaphorical_accuracy": safe_accuracy(g, "literal_or_metaphorical", "expected_literal_or_metaphorical"),
        }))
        .reset_index()
    )
    out = root / "data" / "outputs" / "tutor_response_metrics_by_prompt_and_sentence_type.csv"
    metrics.to_csv(out, index=False)
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
