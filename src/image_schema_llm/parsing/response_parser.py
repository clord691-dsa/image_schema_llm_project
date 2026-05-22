from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


ALLOWED_LITERALITY = {"literal", "metaphorical", "control", "uncertain"}
ALLOWED_SCHEMAS = {
    "CONTAINER",
    "SOURCE_PATH_GOAL",
    "FORCE",
    "BLOCKAGE",
    "VERTICALITY",
    "SUPPORT_BALANCE",
    "NONE",
    "uncertain",
}


@dataclass(frozen=True)
class ParsedResponse:
    """
    Normalised parsed response produced from one raw model response.

    Purpose
    -------
    Converts provider-specific raw text into a stable analysis-ready record.
    The parser is intentionally conservative: parse errors are preserved as
    records rather than causing API reruns.
    """

    parse_status: str
    parse_error: str | None
    literal_or_metaphorical: str | None
    main_image_schema: str | None
    secondary_image_schemas: list[str]
    trajector: str
    landmark_or_container: str
    source: str
    path: str
    goal: str
    obstacle: str
    force: str
    source_domain: list[str]
    target_domain: list[str]
    interpretation: str
    schema_explanation: str
    confidence: str | None
    parsed_json: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "parse_status": self.parse_status,
            "parse_error": self.parse_error,
            "literal_or_metaphorical": self.literal_or_metaphorical,
            "main_image_schema": self.main_image_schema,
            "secondary_image_schemas": self.secondary_image_schemas,
            "trajector": self.trajector,
            "landmark_or_container": self.landmark_or_container,
            "source": self.source,
            "path": self.path,
            "goal": self.goal,
            "obstacle": self.obstacle,
            "force": self.force,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "interpretation": self.interpretation,
            "schema_explanation": self.schema_explanation,
            "confidence": self.confidence,
            "parsed_json": self.parsed_json,
        }


def _empty_parse(parse_status: str, parse_error: str | None) -> ParsedResponse:
    """
    Create an empty parse record for free-text or failed parses.
    """

    return ParsedResponse(
        parse_status=parse_status,
        parse_error=parse_error,
        literal_or_metaphorical=None,
        main_image_schema=None,
        secondary_image_schemas=[],
        trajector="",
        landmark_or_container="",
        source="",
        path="",
        goal="",
        obstacle="",
        force="",
        source_domain=[],
        target_domain=[],
        interpretation="",
        schema_explanation="",
        confidence=None,
        parsed_json={},
    )


def _coerce_list(value: Any) -> list[str]:
    """
    Coerce parser values into a list of strings.

    LLMs sometimes return a string where the schema asks for a list. This
    function makes downstream analysis more robust while retaining the raw
    parsed JSON in the output record.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [text]

    return [str(value)]


def _coerce_str(value: Any) -> str:
    """Coerce a value into a stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def _extract_json_candidate(text: str) -> str:
    """
    Extract the most likely JSON object from a model response.

    Supports:
    - plain JSON object;
    - fenced ```json blocks;
    - responses with prose before/after a JSON object.

    Raises
    ------
    ValueError
        If no JSON object candidate is found.
    """

    stripped = text.strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in raw response text.")

    return stripped[start:end + 1].strip()


def parse_response_text(raw_response: str, *, expected_output_format: str) -> ParsedResponse:
    """
    Parse one raw response text into a normalised ParsedResponse.

    Parameters
    ----------
    raw_response:
        Text returned by the model.
    expected_output_format:
        Prompt-level output expectation, usually `free_text` or
        `json_object`.

    Returns
    -------
    ParsedResponse

    Notes
    -----
    Naive prompts may produce free text. Those records are marked
    `free_text_unparsed` and preserve the interpretation field.
    """

    if expected_output_format == "free_text":
        parsed = _empty_parse("free_text_unparsed", None)
        return ParsedResponse(
            parse_status=parsed.parse_status,
            parse_error=parsed.parse_error,
            literal_or_metaphorical=None,
            main_image_schema=None,
            secondary_image_schemas=[],
            trajector="",
            landmark_or_container="",
            source="",
            path="",
            goal="",
            obstacle="",
            force="",
            source_domain=[],
            target_domain=[],
            interpretation=raw_response.strip(),
            schema_explanation="",
            confidence=None,
            parsed_json={},
        )

    try:
        candidate = _extract_json_candidate(raw_response)
        data = json.loads(candidate)
    except Exception as exc:
        return _empty_parse("parse_error", f"{type(exc).__name__}: {exc}")

    if not isinstance(data, dict):
        return _empty_parse("parse_error", "Parsed JSON is not an object.")

    literal_or_metaphorical = data.get("literal_or_metaphorical")
    if literal_or_metaphorical is not None:
        literal_or_metaphorical = str(literal_or_metaphorical).strip()
        if literal_or_metaphorical not in ALLOWED_LITERALITY:
            literal_or_metaphorical = "uncertain"

    main_image_schema = data.get("main_image_schema")
    if main_image_schema is not None:
        main_image_schema = str(main_image_schema).strip()
        if main_image_schema not in ALLOWED_SCHEMAS:
            main_image_schema = "uncertain"

    confidence = data.get("confidence")
    if confidence is not None:
        confidence = str(confidence).strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = None

    return ParsedResponse(
        parse_status="parsed",
        parse_error=None,
        literal_or_metaphorical=literal_or_metaphorical,
        main_image_schema=main_image_schema,
        secondary_image_schemas=_coerce_list(data.get("secondary_image_schemas")),
        trajector=_coerce_str(data.get("trajector")),
        landmark_or_container=_coerce_str(data.get("landmark_or_container")),
        source=_coerce_str(data.get("source")),
        path=_coerce_str(data.get("path")),
        goal=_coerce_str(data.get("goal")),
        obstacle=_coerce_str(data.get("obstacle")),
        force=_coerce_str(data.get("force")),
        source_domain=_coerce_list(data.get("source_domain")),
        target_domain=_coerce_list(data.get("target_domain")),
        interpretation=_coerce_str(data.get("interpretation")),
        schema_explanation=_coerce_str(data.get("schema_explanation")),
        confidence=confidence,
        parsed_json=data,
    )


def parse_raw_response_record(raw_record: dict[str, Any]) -> dict[str, Any]:
    """
    Parse one raw_responses.jsonl record.

    The returned record preserves key run metadata and appends parsed fields.
    It is suitable for writing to parsed_responses.jsonl.
    """

    raw_response = raw_record.get("raw_response", "")
    expected_output_format = raw_record.get("expected_output_format")

    # Earlier raw records may not contain expected_output_format. Infer it
    # from prompt_family as a robust fallback.
    if not expected_output_format:
        prompt_family = raw_record.get("prompt_family")
        expected_output_format = "free_text" if prompt_family == "naive" else "json_object"

    parsed = parse_response_text(
        str(raw_response),
        expected_output_format=str(expected_output_format),
    )

    result = {
        "run_key": raw_record.get("run_key"),
        "run_index": raw_record.get("run_index"),
        "provider": raw_record.get("provider"),
        "model_id": raw_record.get("model_id"),
        "model_name": raw_record.get("model_name"),
        "prompt_id": raw_record.get("prompt_id"),
        "prompt_family": raw_record.get("prompt_family"),
        "prompt_version": raw_record.get("prompt_version"),
        "condition_id": raw_record.get("condition_id"),
        "condition_family": raw_record.get("condition_family"),
        "temperature": raw_record.get("temperature"),
        "top_p": raw_record.get("top_p"),
        "sentence_id": raw_record.get("sentence_id"),
        "sentence_type": raw_record.get("sentence_type"),
        "expected_schema_primary": raw_record.get("expected_schema_primary"),
        "expected_literal_or_metaphorical": raw_record.get("expected_literal_or_metaphorical"),
        "repetition_index": raw_record.get("repetition_index"),
        "raw_response_status": raw_record.get("status"),
        "provider_response_id": raw_record.get("provider_response_id"),
    }
    result.update(parsed.to_dict())
    return result
