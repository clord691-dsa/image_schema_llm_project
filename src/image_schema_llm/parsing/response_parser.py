from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from json import JSONDecoder
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
ALLOWED_CONFIDENCE = {"high", "medium", "low"}


@dataclass(frozen=True)
class ParsedResponse:
    """
    Normalised parsed response produced from one raw model response.

    parse_status values:
    - parsed: complete JSON or at least the core fields were recovered.
    - partial: some useful fields were recovered, but not enough for schema scoring.
    - free_text_unparsed: expected free-text baseline output.
    - parse_error: no useful structured evidence was recovered.

    The parser is intentionally transparent. It does not silently pretend that
    truncated JSON is complete. Instead it adds parse_quality and usability flags.
    """

    parse_status: str
    parse_error: str | None
    parser_strategy: str | None
    parse_quality: str | None
    usable_for_schema_accuracy: bool
    usable_for_lm_accuracy: bool
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


def _empty_parse(
    parse_status: str,
    parse_error: str | None,
    *,
    parser_strategy: str | None = None,
    parse_quality: str | None = None,
) -> ParsedResponse:
    return ParsedResponse(
        parse_status=parse_status,
        parse_error=parse_error,
        parser_strategy=parser_strategy,
        parse_quality=parse_quality,
        usable_for_schema_accuracy=False,
        usable_for_lm_accuracy=False,
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
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text and len(text) < 120:
            parts = [part.strip() for part in text.split(",")]
            if all(parts):
                return parts
        return [text]
    return [str(value).strip()]


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _normalise_label(value: Any, allowed: set[str], *, default: str | None = None) -> str | None:
    if value is None:
        return default

    if allowed in (ALLOWED_LITERALITY, ALLOWED_CONFIDENCE):
        label = str(value).strip().lower()
        return label if label in allowed else default

    label = str(value).strip().replace("-", "_").replace(" ", "_").upper()
    aliases = {
        "PATH": "SOURCE_PATH_GOAL",
        "PATH_SCHEMA": "SOURCE_PATH_GOAL",
        "SOURCE_PATH_GOAL_SCHEMA": "SOURCE_PATH_GOAL",
        "SOURCE_PATH": "SOURCE_PATH_GOAL",
        "SPG": "SOURCE_PATH_GOAL",
        "SUPPORT": "SUPPORT_BALANCE",
        "BALANCE": "SUPPORT_BALANCE",
        "SUPPORT_BALANCE_SCHEMA": "SUPPORT_BALANCE",
        "BLOCKAGE_SCHEMA": "BLOCKAGE",
        "FORCE_SCHEMA": "FORCE",
        "CONTAINER_SCHEMA": "CONTAINER",
        "VERTICALITY_SCHEMA": "VERTICALITY",
        "NO_CLEAR_IMAGE_SCHEMA": "NONE",
        "NO_CLEAR_SCHEMA": "NONE",
        "NO_SCHEMA": "NONE",
        "N/A": "NONE",
        "NA": "NONE",
    }
    label = aliases.get(label, label)
    return label if label in allowed else default


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    full = re.fullmatch(
        r"```(?:json|javascript|js|python)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if full:
        return full.group(1).strip()

    inner = re.search(
        r"```(?:json|javascript|js|python)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if inner:
        return inner.group(1).strip()

    return stripped


def _replace_smart_quotes(text: str) -> str:
    for src, dst in {"“": '"', "”": '"', "‘": "'", "’": "'"}.items():
        text = text.replace(src, dst)
    return text


def _remove_json_comments(text: str) -> str:
    text = re.sub(r"^\s*//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _quote_unquoted_keys(text: str) -> str:
    return re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)', r'\1"\2"\3', text)


def _extract_balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    stack = 0
    start: int | None = None
    in_string = False
    escape = False
    quote_char = ""

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False
            continue

        if ch in {'"', "'"}:
            in_string = True
            quote_char = ch
            continue

        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    objects.append(text[start:i + 1])
                    start = None

    return objects


def _attempt_json_loads(candidate: str) -> tuple[dict[str, Any], str]:
    candidate = candidate.strip()

    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data, "strict_json"
    except Exception:
        pass

    cleaned = _replace_smart_quotes(candidate)
    cleaned = _remove_json_comments(cleaned)
    cleaned = _remove_trailing_commas(cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data, "strict_json_after_basic_cleanup"
    except Exception:
        pass

    keyed = _remove_trailing_commas(_quote_unquoted_keys(cleaned))
    try:
        data = json.loads(keyed)
        if isinstance(data, dict):
            return data, "json_after_key_quoting"
    except Exception:
        pass

    try:
        data = ast.literal_eval(cleaned)
        if isinstance(data, dict):
            return data, "python_literal_eval"
    except Exception:
        pass

    raise ValueError("Unable to parse candidate after repair attempts.")


def extract_json_object(raw_response: str) -> tuple[dict[str, Any], str]:
    """
    Extract and parse the best complete JSON object from an LLM response.
    """
    text = _replace_smart_quotes(strip_code_fences(raw_response)).strip()
    candidates = [text]

    decoder = JSONDecoder()
    for idx, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[idx:])
                if isinstance(obj, dict):
                    return obj, "json_decoder_raw_decode"
            except Exception:
                pass

    candidates.extend(_extract_balanced_json_objects(text))
    candidates = list(dict.fromkeys([c for c in candidates if c.strip()]))

    expected_keys = {
        "literal_or_metaphorical",
        "main_image_schema",
        "secondary_image_schemas",
        "interpretation",
        "schema_explanation",
        "confidence",
        "trajector",
        "landmark_or_container",
        "source_domain",
        "target_domain",
    }

    parsed_candidates: list[tuple[int, dict[str, Any], str]] = []
    errors: list[str] = []

    for candidate in candidates:
        try:
            data, strategy = _attempt_json_loads(candidate)
            score = len(expected_keys & set(data.keys()))
            parsed_candidates.append((score, data, strategy))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    if parsed_candidates:
        parsed_candidates.sort(key=lambda item: item[0], reverse=True)
        _, data, strategy = parsed_candidates[0]
        return data, strategy

    raise ValueError("; ".join(errors[-3:]) if errors else "No parseable JSON object found.")


def _extract_string_field(text: str, key: str, *, allow_unclosed: bool = False) -> str | None:
    """
    Extract a JSON-style string field. When allow_unclosed=True, a truncated
    final string can also be recovered.
    """
    complete = re.search(
        rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        flags=re.DOTALL,
    )
    if complete:
        value = complete.group(1)
        try:
            return json.loads(f'"{value}"')
        except Exception:
            return value.replace('\\"', '"')

    if allow_unclosed:
        partial = re.search(
            rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)$',
            text,
            flags=re.DOTALL,
        )
        if partial:
            return partial.group(1).replace('\\"', '"')

    return None


def _extract_array_field(text: str, key: str) -> list[str] | None:
    m = re.search(rf'"{re.escape(key)}"\s*:\s*(\[[^\]]*\])', text, flags=re.DOTALL)
    if not m:
        return None
    candidate = _remove_trailing_commas(m.group(1))
    try:
        return _coerce_list(json.loads(candidate))
    except Exception:
        return None


def _extract_empty_array_field(text: str, key: str) -> list[str] | None:
    return [] if re.search(rf'"{re.escape(key)}"\s*:\s*\[\s*\]', text, flags=re.DOTALL) else None


def partial_field_recovery(raw_response: str) -> dict[str, Any]:
    """
    Recover fields from incomplete/truncated JSON-like model output.

    This is data-aware for the current raw_responses.jsonl pattern: many Gemini
    records start with `literal_or_metaphorical` but are cut off before the
    full JSON object closes. The parser therefore preserves the usable fields
    rather than treating all truncated objects as complete failure.
    """
    text = _replace_smart_quotes(strip_code_fences(raw_response))

    data: dict[str, Any] = {}
    string_fields = [
        "literal_or_metaphorical",
        "main_image_schema",
        "trajector",
        "landmark_or_container",
        "source",
        "path",
        "goal",
        "obstacle",
        "force",
        "interpretation",
        "schema_explanation",
        "confidence",
    ]

    for key in string_fields:
        value = _extract_string_field(
            text,
            key,
            allow_unclosed=(key in {"literal_or_metaphorical", "main_image_schema"}),
        )
        if value is not None:
            data[key] = value

    for key in ["secondary_image_schemas", "source_domain", "target_domain"]:
        value = _extract_array_field(text, key)
        if value is None:
            value = _extract_empty_array_field(text, key)
        if value is not None:
            data[key] = value

    return data


def _classify_partial_data(data: dict[str, Any]) -> tuple[str, str, bool, bool]:
    """
    Return parse_status, parse_quality, usable_schema, usable_lm.
    """
    lm = _normalise_label(data.get("literal_or_metaphorical"), ALLOWED_LITERALITY, default=None)
    schema = _normalise_label(data.get("main_image_schema"), ALLOWED_SCHEMAS, default=None)

    usable_lm = lm is not None
    usable_schema = lm is not None and schema is not None

    if usable_schema:
        return "parsed", "partial_core_fields", True, True
    if usable_lm:
        return "partial", "partial_literality_only", False, True
    if data:
        return "partial", "partial_insufficient_fields", False, False
    return "parse_error", "failed", False, False


def _build_parsed_response(
    data: dict[str, Any],
    *,
    parse_status: str,
    parse_error: str | None,
    parser_strategy: str,
    parse_quality: str,
    usable_for_schema_accuracy: bool,
    usable_for_lm_accuracy: bool,
) -> ParsedResponse:
    literal_or_metaphorical = _normalise_label(
        data.get("literal_or_metaphorical"),
        ALLOWED_LITERALITY,
        default=None,
    )

    main_image_schema = _normalise_label(
        data.get("main_image_schema"),
        ALLOWED_SCHEMAS,
        default=None,
    )

    confidence = _normalise_label(
        data.get("confidence"),
        ALLOWED_CONFIDENCE,
        default=None,
    )

    secondary = []
    for item in _coerce_list(data.get("secondary_image_schemas")):
        normalised = _normalise_label(item, ALLOWED_SCHEMAS, default=None)
        if normalised and normalised != "NONE":
            secondary.append(normalised)

    return ParsedResponse(
        parse_status=parse_status,
        parse_error=parse_error,
        parser_strategy=parser_strategy,
        parse_quality=parse_quality,
        usable_for_schema_accuracy=usable_for_schema_accuracy,
        usable_for_lm_accuracy=usable_for_lm_accuracy,
        literal_or_metaphorical=literal_or_metaphorical,
        main_image_schema=main_image_schema,
        secondary_image_schemas=secondary,
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


def parse_response_text(raw_response: str, *, expected_output_format: str) -> ParsedResponse:
    raw_response = raw_response or ""

    if expected_output_format == "free_text":
        return ParsedResponse(
            parse_status="free_text_unparsed",
            parse_error=None,
            parser_strategy="free_text_passthrough",
            parse_quality="free_text",
            usable_for_schema_accuracy=False,
            usable_for_lm_accuracy=False,
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
        data, strategy = extract_json_object(raw_response)
        complete_quality = "strict_complete" if strategy in {"strict_json", "json_decoder_raw_decode"} else "repaired_complete"
        return _build_parsed_response(
            data,
            parse_status="parsed",
            parse_error=None,
            parser_strategy=strategy,
            parse_quality=complete_quality,
            usable_for_schema_accuracy=True,
            usable_for_lm_accuracy=True,
        )
    except Exception as complete_exc:
        partial = partial_field_recovery(raw_response)
        parse_status, parse_quality, usable_schema, usable_lm = _classify_partial_data(partial)

        if partial:
            return _build_parsed_response(
                partial,
                parse_status=parse_status,
                parse_error=f"complete_json_failed_partial_recovery: {type(complete_exc).__name__}: {complete_exc}",
                parser_strategy="partial_key_value_recovery",
                parse_quality=parse_quality,
                usable_for_schema_accuracy=usable_schema,
                usable_for_lm_accuracy=usable_lm,
            )

        return _empty_parse(
            "parse_error",
            f"{type(complete_exc).__name__}: {complete_exc}",
            parser_strategy="tolerant_json_failed",
            parse_quality="failed",
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
        "sentence_id": raw_record.get("sentence_id"),
        "sentence_type": raw_record.get("sentence_type"),
        "expected_schema_primary": raw_record.get("expected_schema_primary"),
        "expected_literal_or_metaphorical": raw_record.get("expected_literal_or_metaphorical"),
        "repetition_index": raw_record.get("repetition_index"),
        "raw_response_status": raw_record.get("status"),
        "provider_response_id": raw_record.get("provider_response_id"),
        "input_tokens": raw_record.get("input_tokens"),
        "output_tokens": raw_record.get("output_tokens"),
    }
    result.update(parsed.to_dict())
    return result
