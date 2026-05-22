from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_schema_llm.config import ProjectPaths
from image_schema_llm.jsonl_utils import read_jsonl, write_jsonl
from image_schema_llm.parsing.response_parser import parse_raw_response_record


@dataclass(frozen=True)
class ParsePipelineResult:
    """
    Summary of one parsing pipeline run.
    """

    input_records: int
    parsed_records: int
    output_path: Path
    status_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "input_records": self.input_records,
            "parsed_records": self.parsed_records,
            "output_path": str(self.output_path),
            "status_counts": self.status_counts,
        }


def run_parsing_pipeline(
    *,
    project_root: Path,
    raw_responses_path: Path | None = None,
    parsed_responses_path: Path | None = None,
    only_success: bool = True,
) -> ParsePipelineResult:
    """
    Parse raw model responses into analysis-ready JSONL records.

    Parameters
    ----------
    project_root:
        Repository root.
    raw_responses_path:
        Optional override for input raw response file.
    parsed_responses_path:
        Optional override for output parsed response file.
    only_success:
        If True, parse only raw response records with status == "success".

    Returns
    -------
    ParsePipelineResult
    """

    paths = ProjectPaths(project_root)
    raw_path = raw_responses_path or paths.raw_responses_path
    parsed_path = parsed_responses_path or (paths.outputs_dir / "parsed_responses.jsonl")

    raw_records = read_jsonl(raw_path)

    parsed_records: list[dict[str, Any]] = []
    for record in raw_records:
        if only_success and record.get("status") != "success":
            continue
        parsed_records.append(parse_raw_response_record(record))

    write_jsonl(parsed_path, parsed_records)

    status_counts = Counter(record.get("parse_status", "unknown") for record in parsed_records)

    return ParsePipelineResult(
        input_records=len(raw_records),
        parsed_records=len(parsed_records),
        output_path=parsed_path,
        status_counts=dict(status_counts),
    )
