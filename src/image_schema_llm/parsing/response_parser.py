from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


ALLOWED_SCHEMA_PRESENT = {"yes", "no", "uncertain"}
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

SCHEMA_ALIASES = {
    "PATH": "SOURCE_PATH_GOAL",
    "SOURCE-PATH-GOAL": "SOURCE_PATH_GOAL",
    "SOURCE_PATH_GOAL": "SOURCE_PATH_GOAL",
    "SOURCE PATH GOAL": "SOURCE_PATH_GOAL",
    "CONTAINMENT": "CONTAINER",
    "CONTAINER": "CONTAINER",
    "BLOCKAGE": "BLOCKAGE",
    "OBSTACLE": "BLOCKAGE",
    "FORCE": "FORCE",
    "FORCE_DYNAMICS": "FORCE",
    "FORCE-DYNAMICS": "FORCE",
    "UP_DOWN": "VERTICALITY",
    "UP-DOWN": "VERTICALITY",
    "VERTICALITY": "VERTICALITY",
    "SCALE": "VERTICALITY",
    "SUPPORT": "SUPPORT_BALANCE",
    "BALANCE": "SUPPORT_BALANCE",
    "SUPPORT_BALANCE": "SUPPORT_BALANCE",
    "SUPPORT-BALANCE": "SUPPORT_BALANCE",
    "NONE": "NONE",
    "NO_SCHEMA": "NONE",
    "NO SCHEMA": "NONE",
    "WEAK_SCHEMA": "NONE",
    "WEAK-SCHEMA": "NONE",
    "UNCERTAIN": "uncertain",
}


@dataclass(frozen=True)
class ParsedResponse:
    """
    Normalised parsed response produced from one raw model response.

    The parser supports both complete JSON and data-aware partial recovery for
    provider outputs that begin a valid JSON object but are truncated before the
    closing brace. Partial recovery is deliberately conservative: it only marks
    a field usable when the exact relevant key/value pair can be recovered.
    """

    parse_status: str
    parse_error: str | None
    parser_strategy: str | None
    parse_quality: str | None
    usable_for_schema_accuracy: bool
    usable_for_lm_accuracy: bool
    schema_present: str | None
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
        return {
            "parse_status": self.parse_status,
            "parse_error": self.parse_error,
            "parser_strategy": self.parser_strategy,
            "parse_quality": self.parse_quality,
            "usable_for_schema_accuracy": self.usable_for_schema_accuracy,
            "usable_for_lm_accuracy": self.usable_for_lm_accuracy,
            "schema_present": self.schema_present,
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


def _make_response(
    *,
    parse_status: str,
    parse_error: str | None = None,
    parser_strategy: str | None = None,
    parse_quality: str | None = None,
    schema_present: str | None = None,
    literal_or_metaphorical: str | None = None,
    main_image_schema: str | None = None,
    secondary_image_schemas: list[str] | None = None,
    trajector: str = "",
    landmark_or_container: str = "",
    source: str = "",
    path: str = "",
    goal: str = "",
    obstacle: str = "",
    force: str = "",
    source_domain: list[str] | None = None,
    target_domain: list[str] | None = None,
    interpretation: str = "",
    schema_explanation: str = "",
    confidence: str | None = None,
    parsed_json: dict[str, Any] | None = None,
) -> ParsedResponse:
    usable_for_schema = main_image_schema not in {None, ""}
    usable_for_lm = literal_or_metaphorical not in {None, ""}

    return ParsedResponse(
        parse_status=parse_status,
        parse_error=parse_error,
        parser_strategy=parser_strategy,
        parse_quality=parse_quality,
        usable_for_schema_accuracy=usable_for_schema,
        usable_for_lm_accuracy=usable_for_lm,
        schema_present=schema_present,
        literal_or_metaphorical=literal_or_metaphorical,
        main_image_schema=main_image_schema,
        secondary_image_schemas=secondary_image_schemas or [],
        trajector=trajector,
        landmark_or_container=landmark_or_container,
        source=source,
        path=path,
        goal=goal,
        obstacle=obstacle,
        force=force,
        source_domain=source_domain or [],
        target_domain=target_domain or [],
        interpretation=interpretation,
        schema_explanation=schema_explanation,
        confidence=confidence,
        parsed_json=parsed_json or {},
    )


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [str(value).strip()]


def _coerce_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalise_schema(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    key = text.upper().replace(" ", "_")
    return SCHEMA_ALIASES.get(key, "uncertain")


def _normalise_literality(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in ALLOWED_LITERALITY else "uncertain"


def _normalise_schema_present(
    value: Any,
    main_schema: str | None,
    literality: str | None,
) -> str | None:
    if value is not None:
        text = str(value).strip().lower()
        if text in ALLOWED_SCHEMA_PRESENT:
            return text
        if text in {"true", "present"}:
            return "yes"
        if text in {"false", "absent", "none", "no_schema", "no schema"}:
            return "no"

    if main_schema == "NONE" or literality == "control":
        return "no"
    if main_schema in ALLOWED_SCHEMAS and main_schema not in {"NONE", "uncertain", None}:
        return "yes"
    return None


def _normalise_confidence(value: Any) -> str | None:
    if value is None:
        return None
    confidence = str(value).strip().lower()
    return confidence if confidence in {"high", "medium", "low"} else None


def _extract_json_candidate(text: str) -> tuple[str, str]:
    """
    Extract a likely JSON object and the strategy used.

    The function now returns a partial candidate if the response starts with
    ``{`` but lacks a closing brace. That lets the partial-field parser recover
    core annotation fields from truncated provider outputs.
    """

    stripped = text.strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped, "strict_json"

    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return fenced.group(1).strip(), "fenced_json"

    start = stripped.find("{")
    end = stripped.rfind("}")

    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1].strip(), "embedded_json"

    if start != -1:
        return stripped[start:].strip(), "partial_json_prefix"

    raise ValueError("No JSON object found in raw response text.")


def _repair_common_json_issues(candidate: str) -> str:
    repaired = candidate.strip()
    repaired = repaired.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    repaired = re.sub(r"//.*?$", "", repaired, flags=re.MULTILINE)
    repaired = re.sub(r"/\*.*?\*/", "", repaired, flags=re.DOTALL)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _load_json_candidate(candidate: str) -> dict[str, Any]:
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        data = json.loads(_repair_common_json_issues(candidate))

    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not an object.")
    return data


def _extract_string_field(text: str, key: str) -> str | None:
    """
    Extract a completed JSON string field from complete or truncated JSON text.
    """
    pattern = re.compile(
        rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return match.group(1)


def _extract_array_field(text: str, key: str) -> list[str]:
    pattern = re.compile(
        rf'"{re.escape(key)}"\s*:\s*(\[(?:[^\[\]]|"(?:\\.|[^"\\])*")*\])',
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return []
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return _coerce_list(value)


def _apply_abstention_consistency(
    *,
    schema_present: str | None,
    literality: str | None,
    main_schema: str | None,
    secondary: list[str],
    source_domain: list[str],
    target_domain: list[str],
) -> tuple[str | None, str | None, str | None, list[str], list[str], list[str]]:
    """
    Apply the pre-declared NONE / weak-schema consistency rule.

    This only forces NONE when the model itself selected schema_present=no,
    literal_or_metaphorical=control, or main_schema=NONE.
    """

    if schema_present == "no" or literality == "control" or main_schema == "NONE":
        return "no", "control", "NONE", [], [], []

    return schema_present, literality, main_schema, secondary, source_domain, target_domain


def _normalised_response_from_data(data: dict[str, Any], *, parser_strategy: str, parse_quality: str) -> ParsedResponse:
    literality = _normalise_literality(data.get("literal_or_metaphorical"))
    main_schema = _normalise_schema(data.get("main_image_schema"))
    schema_present = _normalise_schema_present(data.get("schema_present"), main_schema, literality)

    secondary = [_normalise_schema(item) or "uncertain" for item in _coerce_list(data.get("secondary_image_schemas"))]
    secondary = [item for item in secondary if item not in {"NONE", None}]

    source_domain = _coerce_list(data.get("source_domain"))
    target_domain = _coerce_list(data.get("target_domain"))

    schema_present, literality, main_schema, secondary, source_domain, target_domain = _apply_abstention_consistency(
        schema_present=schema_present,
        literality=literality,
        main_schema=main_schema,
        secondary=secondary,
        source_domain=source_domain,
        target_domain=target_domain,
    )

    return _make_response(
        parse_status="parsed",
        parse_error=None,
        parser_strategy=parser_strategy,
        parse_quality=parse_quality,
        schema_present=schema_present,
        literal_or_metaphorical=literality,
        main_image_schema=main_schema,
        secondary_image_schemas=secondary,
        trajector=_coerce_str(data.get("trajector")),
        landmark_or_container=_coerce_str(data.get("landmark_or_container")),
        source=_coerce_str(data.get("source")),
        path=_coerce_str(data.get("path")),
        goal=_coerce_str(data.get("goal")),
        obstacle=_coerce_str(data.get("obstacle")),
        force=_coerce_str(data.get("force")),
        source_domain=source_domain,
        target_domain=target_domain,
        interpretation=_coerce_str(data.get("interpretation")),
        schema_explanation=_coerce_str(data.get("schema_explanation")),
        confidence=_normalise_confidence(data.get("confidence")),
        parsed_json=data,
    )


def _partial_parse(candidate: str, *, parse_error: str | None = None) -> ParsedResponse:
    """
    Recover core fields from incomplete JSON.

    This is mainly intended for Gemini outputs that are valid JSON prefixes but
    are truncated before the final brace. The recovered record is flagged with
    parser_strategy='partial_field_recovery' so analyses can include or exclude
    it explicitly.
    """

    literality = _normalise_literality(_extract_string_field(candidate, "literal_or_metaphorical"))
    main_schema = _normalise_schema(_extract_string_field(candidate, "main_image_schema"))
    schema_present = _normalise_schema_present(
        _extract_string_field(candidate, "schema_present"),
        main_schema,
        literality,
    )

    secondary = [_normalise_schema(item) or "uncertain" for item in _extract_array_field(candidate, "secondary_image_schemas")]
    secondary = [item for item in secondary if item not in {"NONE", None}]
    source_domain = _extract_array_field(candidate, "source_domain")
    target_domain = _extract_array_field(candidate, "target_domain")

    schema_present, literality, main_schema, secondary, source_domain, target_domain = _apply_abstention_consistency(
        schema_present=schema_present,
        literality=literality,
        main_schema=main_schema,
        secondary=secondary,
        source_domain=source_domain,
        target_domain=target_domain,
    )

    if literality and main_schema:
        parse_status = "parsed"
        parse_quality = "partial_core_fields"
    elif literality:
        parse_status = "partial"
        parse_quality = "partial_literality_only"
    elif main_schema:
        parse_status = "partial"
        parse_quality = "partial_schema_only"
    else:
        return _make_response(
            parse_status="parse_error",
            parse_error=parse_error or "Could not recover core fields from partial JSON.",
            parser_strategy="partial_field_recovery",
            parse_quality="failed",
        )

    return _make_response(
        parse_status=parse_status,
        parse_error=parse_error,
        parser_strategy="partial_field_recovery",
        parse_quality=parse_quality,
        schema_present=schema_present,
        literal_or_metaphorical=literality,
        main_image_schema=main_schema,
        secondary_image_schemas=secondary,
        source_domain=source_domain,
        target_domain=target_domain,
        interpretation=_coerce_str(_extract_string_field(candidate, "interpretation")),
        schema_explanation=_coerce_str(_extract_string_field(candidate, "schema_explanation")),
        confidence=_normalise_confidence(_extract_string_field(candidate, "confidence")),
        parsed_json={},
    )


def parse_response_text(raw_response: str, *, expected_output_format: str) -> ParsedResponse:
    if expected_output_format == "free_text":
        return _make_response(
            parse_status="free_text_unparsed",
            parse_error=None,
            parser_strategy="free_text",
            parse_quality="free_text",
            interpretation=raw_response.strip(),
        )

    try:
        candidate, strategy = _extract_json_candidate(raw_response)
    except Exception as exc:
        return _make_response(
            parse_status="parse_error",
            parse_error=f"{type(exc).__name__}: {exc}",
            parser_strategy="no_json_found",
            parse_quality="failed",
        )

    try:
        data = _load_json_candidate(candidate)
    except Exception as exc:
        return _partial_parse(candidate, parse_error=f"{type(exc).__name__}: {exc}")

    return _normalised_response_from_data(
        data,
        parser_strategy=strategy,
        parse_quality="complete",
    )


def parse_raw_response_record(raw_record: dict[str, Any]) -> dict[str, Any]:
    raw_response = raw_record.get("raw_response", "")
    expected_output_format = raw_record.get("expected_output_format")

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
        "condition_max_output_tokens": raw_record.get("condition_max_output_tokens"),
        "recommended_max_output_tokens": raw_record.get("recommended_max_output_tokens"),
        "max_output_tokens": raw_record.get("max_output_tokens"),
        "sentence_id": raw_record.get("sentence_id"),
        "sentence_type": raw_record.get("sentence_type"),
        "expected_schema_primary": raw_record.get("expected_schema_primary"),
        "expected_literal_or_metaphorical": raw_record.get("expected_literal_or_metaphorical"),
        "repetition_index": raw_record.get("repetition_index"),
        "raw_response_status": raw_record.get("status"),
        "provider_response_id": raw_record.get("provider_response_id"),
        "finish_reason": raw_record.get("finish_reason"),
    }
    result.update(parsed.to_dict())
    return result
