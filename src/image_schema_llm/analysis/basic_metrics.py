from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BasicMetricSummary:
    """
    Lightweight metrics computed from parsed response records.

    Records contribute to a metric when their parser usability flag for that
    metric is true. This allows carefully recovered partial JSON to contribute
    to the fields it actually contains without pretending that every field was
    fully parsed.
    """

    total_records: int
    parse_status_counts: dict[str, int]
    parser_strategy_counts: dict[str, int]
    parse_quality_counts: dict[str, int]
    usable_for_schema_accuracy: int
    usable_for_lm_accuracy: int
    primary_schema_accuracy: float | None
    literal_metaphorical_accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "parse_status_counts": self.parse_status_counts,
            "parser_strategy_counts": self.parser_strategy_counts,
            "parse_quality_counts": self.parse_quality_counts,
            "usable_for_schema_accuracy": self.usable_for_schema_accuracy,
            "usable_for_lm_accuracy": self.usable_for_lm_accuracy,
            "primary_schema_accuracy": self.primary_schema_accuracy,
            "literal_metaphorical_accuracy": self.literal_metaphorical_accuracy,
        }


def _usable_for_schema(record: dict[str, Any]) -> bool:
    if "usable_for_schema_accuracy" in record:
        return bool(record.get("usable_for_schema_accuracy"))
    return record.get("parse_status") == "parsed" and bool(record.get("main_image_schema"))


def _usable_for_lm(record: dict[str, Any]) -> bool:
    if "usable_for_lm_accuracy" in record:
        return bool(record.get("usable_for_lm_accuracy"))
    return record.get("parse_status") == "parsed" and bool(record.get("literal_or_metaphorical"))


def compute_basic_metrics(parsed_records: list[dict[str, Any]]) -> BasicMetricSummary:
    parse_counts = Counter(record.get("parse_status", "unknown") for record in parsed_records)
    strategy_counts = Counter(record.get("parser_strategy", "unknown") for record in parsed_records)
    quality_counts = Counter(record.get("parse_quality", "unknown") for record in parsed_records)

    schema_total = 0
    schema_correct = 0
    lm_total = 0
    lm_correct = 0

    for record in parsed_records:
        expected_schema = record.get("expected_schema_primary")
        predicted_schema = record.get("main_image_schema")
        if _usable_for_schema(record) and expected_schema and predicted_schema:
            schema_total += 1
            if expected_schema == predicted_schema:
                schema_correct += 1

        expected_lm = record.get("expected_literal_or_metaphorical")
        predicted_lm = record.get("literal_or_metaphorical")
        if _usable_for_lm(record) and expected_lm and predicted_lm:
            lm_total += 1
            if expected_lm == predicted_lm:
                lm_correct += 1

    return BasicMetricSummary(
        total_records=len(parsed_records),
        parse_status_counts=dict(parse_counts),
        parser_strategy_counts=dict(strategy_counts),
        parse_quality_counts=dict(quality_counts),
        usable_for_schema_accuracy=schema_total,
        usable_for_lm_accuracy=lm_total,
        primary_schema_accuracy=(schema_correct / schema_total if schema_total else None),
        literal_metaphorical_accuracy=(lm_correct / lm_total if lm_total else None),
    )
