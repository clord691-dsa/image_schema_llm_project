from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BasicMetricSummary:
    """
    Lightweight metrics computed from parsed response records.

    These are preliminary convenience metrics, not the final analysis layer.
    """

    total_records: int
    parse_status_counts: dict[str, int]
    primary_schema_accuracy: float | None
    literal_metaphorical_accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "parse_status_counts": self.parse_status_counts,
            "primary_schema_accuracy": self.primary_schema_accuracy,
            "literal_metaphorical_accuracy": self.literal_metaphorical_accuracy,
        }


def compute_basic_metrics(parsed_records: list[dict[str, Any]]) -> BasicMetricSummary:
    """
    Compute lightweight accuracy metrics from parsed responses.

    A record contributes to accuracy only if the relevant parsed field is
    present and the parse_status is `parsed`.
    """

    parse_counts = Counter(record.get("parse_status", "unknown") for record in parsed_records)

    schema_total = 0
    schema_correct = 0
    lm_total = 0
    lm_correct = 0

    for record in parsed_records:
        if record.get("parse_status") != "parsed":
            continue

        expected_schema = record.get("expected_schema_primary")
        predicted_schema = record.get("main_image_schema")
        if expected_schema and predicted_schema:
            schema_total += 1
            if expected_schema == predicted_schema:
                schema_correct += 1

        expected_lm = record.get("expected_literal_or_metaphorical")
        predicted_lm = record.get("literal_or_metaphorical")
        if expected_lm and predicted_lm:
            lm_total += 1
            if expected_lm == predicted_lm:
                lm_correct += 1

    return BasicMetricSummary(
        total_records=len(parsed_records),
        parse_status_counts=dict(parse_counts),
        primary_schema_accuracy=(schema_correct / schema_total if schema_total else None),
        literal_metaphorical_accuracy=(lm_correct / lm_total if lm_total else None),
    )
