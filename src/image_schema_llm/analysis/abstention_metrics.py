from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AbstentionMetrics:
    control_accuracy: float | None
    control_false_positive_schema_rate: float | None
    schema_present_accuracy: float | None
    literal_metaphorical_non_control_accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_accuracy": self.control_accuracy,
            "control_false_positive_schema_rate": self.control_false_positive_schema_rate,
            "schema_present_accuracy": self.schema_present_accuracy,
            "literal_metaphorical_non_control_accuracy": self.literal_metaphorical_non_control_accuracy,
        }


def _safe_mean(series: pd.Series) -> float | None:
    return None if series.empty else float(series.mean())


def _usable(df: pd.DataFrame, column: str) -> pd.Series:
    flag = f"usable_for_{column}_accuracy"
    if flag in df.columns:
        return df[flag].fillna(False).astype(bool)
    if column == "schema":
        return df["parse_status"].isin(["parsed", "partial"]) & df["main_image_schema"].notna()
    if column == "lm":
        return df["parse_status"].isin(["parsed", "partial"]) & df["literal_or_metaphorical"].notna()
    return pd.Series(False, index=df.index)


def compute_abstention_metrics(parsed: pd.DataFrame) -> AbstentionMetrics:
    """
    Compute metrics for the schema_present / NONE abstention gate.

    Partially recovered records can contribute only when the required field is
    usable. For example, a truncated record with only literality recovered can
    contribute to literal/metaphorical accuracy but not schema accuracy.
    """

    if parsed.empty:
        return AbstentionMetrics(None, None, None, None)

    working = parsed.copy()
    if "schema_present" not in working.columns:
        working["schema_present"] = None

    usable_schema = _usable(working, "schema")
    usable_lm = _usable(working, "lm")
    structured = working[usable_schema | usable_lm].copy()

    if structured.empty:
        return AbstentionMetrics(None, None, None, None)

    controls = structured[structured["sentence_type"].eq("control_weak_schema")]
    non_controls = structured[~structured["sentence_type"].eq("control_weak_schema")]

    control_accuracy = None
    control_false_positive_schema_rate = None
    if not controls.empty:
        control_rows = controls[usable_schema.loc[controls.index] & usable_lm.loc[controls.index]]
        if not control_rows.empty:
            control_accuracy = _safe_mean(
                control_rows["literal_or_metaphorical"].eq("control")
                & control_rows["main_image_schema"].eq("NONE")
            )
        schema_control_rows = controls[usable_schema.loc[controls.index]]
        if not schema_control_rows.empty:
            control_false_positive_schema_rate = _safe_mean(
                schema_control_rows["main_image_schema"].notna()
                & ~schema_control_rows["main_image_schema"].eq("NONE")
            )

    schema_present_rows = structured[working["schema_present"].notna().loc[structured.index]]
    schema_present_accuracy = None
    if not schema_present_rows.empty:
        schema_present_gold = schema_present_rows["sentence_type"].map(
            lambda x: "no" if x == "control_weak_schema" else "yes"
        )
        schema_present_accuracy = _safe_mean(schema_present_rows["schema_present"].eq(schema_present_gold))

    lm_non_control_accuracy = None
    if not non_controls.empty:
        lm_rows = non_controls[usable_lm.loc[non_controls.index]]
        if not lm_rows.empty:
            lm_non_control_accuracy = _safe_mean(
                lm_rows["literal_or_metaphorical"].eq(lm_rows["expected_literal_or_metaphorical"])
            )

    return AbstentionMetrics(
        control_accuracy=control_accuracy,
        control_false_positive_schema_rate=control_false_positive_schema_rate,
        schema_present_accuracy=schema_present_accuracy,
        literal_metaphorical_non_control_accuracy=lm_non_control_accuracy,
    )
