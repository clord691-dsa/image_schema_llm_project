from typing import Any, Dict, Optional


def parse_structured_response(raw_response: str) -> Optional[Dict[str, Any]]:
    """
    Parse a raw model response into structured fields.

    Inputs:
        raw_response:
            Full text from raw_responses.jsonl.

    Outputs:
        Dictionary containing parsed fields, or None if parsing fails.

    Purpose:
        Converts raw model responses into an analysis-ready structure.

    Notes:
        Parsing should be done after collection to avoid expensive API reruns.
        If parsing fails, the raw response remains available for manual review.
    """
    raise NotImplementedError


def normalise_schema_label(label: str) -> str:
    """
    Normalise model-produced schema labels.

    Inputs:
        label:
            Raw schema label produced by a model, such as:
            - "Source-Path-Goal"
            - "SOURCE_PATH_GOAL"
            - "source/path/goal"

    Outputs:
        Canonical label, such as:
            - "SOURCE_PATH_GOAL"

    Purpose:
        Supports fair comparison across models and prompt conditions.
    """
    raise NotImplementedError
