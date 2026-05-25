from __future__ import annotations

from copy import deepcopy
from typing import Any

from image_schema_llm.schemas import ExperimentJob


SCHEMA_ENUM = [
    "CONTAINER",
    "SOURCE_PATH_GOAL",
    "FORCE",
    "BLOCKAGE",
    "VERTICALITY",
    "SUPPORT_BALANCE",
    "NONE",
    "uncertain",
]

LITERALITY_ENUM = ["literal", "metaphorical", "control", "uncertain"]
SCHEMA_PRESENT_ENUM = ["yes", "no", "uncertain"]
CONFIDENCE_ENUM = ["high", "medium", "low"]


def _string_enum(values: list[str]) -> dict[str, Any]:
    return {"type": "string", "enum": values}


def _string_field() -> dict[str, Any]:
    return {"type": "string"}


def _string_array() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def _base_properties(*, include_schema_present: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    if include_schema_present:
        properties["schema_present"] = _string_enum(SCHEMA_PRESENT_ENUM)

    properties.update(
        {
            "literal_or_metaphorical": _string_enum(LITERALITY_ENUM),
            "main_image_schema": _string_enum(SCHEMA_ENUM),
            "secondary_image_schemas": _string_array(),
            "source_domain": _string_array(),
            "target_domain": _string_array(),
            "interpretation": _string_field(),
            "schema_explanation": _string_field(),
            "confidence": _string_enum(CONFIDENCE_ENUM),
        }
    )
    return properties


def _schema_from_properties(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


DIRECT_SCHEMA_JSON_SCHEMA: dict[str, Any] = _schema_from_properties(
    _base_properties(include_schema_present=False)
)

DIRECT_SCHEMA_V2_JSON_SCHEMA: dict[str, Any] = _schema_from_properties(
    _base_properties(include_schema_present=True)
)

STRUCTURED_ROLE_JSON_SCHEMA: dict[str, Any] = _schema_from_properties(
    {
        **_base_properties(include_schema_present=False),
        "trajector": _string_field(),
        "landmark_or_container": _string_field(),
        "source": _string_field(),
        "path": _string_field(),
        "goal": _string_field(),
        "obstacle": _string_field(),
        "force": _string_field(),
    }
)

# Keep role fields in a more readable order for the structured prompt.
STRUCTURED_ROLE_JSON_SCHEMA["required"] = [
    "literal_or_metaphorical",
    "main_image_schema",
    "secondary_image_schemas",
    "trajector",
    "landmark_or_container",
    "source",
    "path",
    "goal",
    "obstacle",
    "force",
    "source_domain",
    "target_domain",
    "interpretation",
    "schema_explanation",
    "confidence",
]

STRUCTURED_ROLE_V2_JSON_SCHEMA: dict[str, Any] = _schema_from_properties(
    {
        **_base_properties(include_schema_present=True),
        "trajector": _string_field(),
        "landmark_or_container": _string_field(),
        "source": _string_field(),
        "path": _string_field(),
        "goal": _string_field(),
        "obstacle": _string_field(),
        "force": _string_field(),
    }
)

STRUCTURED_ROLE_V2_JSON_SCHEMA["required"] = [
    "schema_present",
    "literal_or_metaphorical",
    "main_image_schema",
    "secondary_image_schemas",
    "trajector",
    "landmark_or_container",
    "source",
    "path",
    "goal",
    "obstacle",
    "force",
    "source_domain",
    "target_domain",
    "interpretation",
    "schema_explanation",
    "confidence",
]


def schema_for_prompt(prompt_family: str, prompt_id: str | None = None) -> dict[str, Any] | None:
    """
    Return the project JSON schema for a specific prompt.

    v2 abstention prompts require the extra `schema_present` field. This cannot
    be selected by prompt_family alone because v1 and v2 direct-schema prompts
    share the same family.
    """

    is_v2 = bool(prompt_id and ("_v2" in prompt_id or "abstention" in prompt_id))

    if prompt_family == "direct_schema":
        return deepcopy(DIRECT_SCHEMA_V2_JSON_SCHEMA if is_v2 else DIRECT_SCHEMA_JSON_SCHEMA)

    if prompt_family == "structured_role_based":
        return deepcopy(STRUCTURED_ROLE_V2_JSON_SCHEMA if is_v2 else STRUCTURED_ROLE_JSON_SCHEMA)

    return None


def schema_for_prompt_family(prompt_family: str) -> dict[str, Any] | None:
    """
    Backwards-compatible family-level schema lookup.

    This returns the v1 schema. New runtime code should call schema_for_prompt()
    or response_format_for_job() so v2 prompts receive the schema_present field.
    """

    return schema_for_prompt(prompt_family, prompt_id=None)


def response_format_for_job(job: ExperimentJob) -> dict[str, Any] | None:
    """
    Return provider-agnostic response-format hints.

    OpenAI and Gemini clients adapt this dictionary to provider-specific API
    parameters. Claude currently treats JSON compliance as prompt-controlled.
    """

    if job.prompt.expected_output_format != "json_object":
        return None

    schema = schema_for_prompt(job.prompt.prompt_family, job.prompt.prompt_id)
    if not schema:
        return {"response_mime_type": "application/json"}

    return {
        "response_mime_type": "application/json",
        "response_schema": schema,
        "json_schema_name": f"{job.prompt.prompt_id}_response",
        "strict": True,
    }
