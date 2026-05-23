from __future__ import annotations

from typing import Any

from image_schema_llm.schemas import ExperimentJob


DIRECT_SCHEMA_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "literal_or_metaphorical": {
            "type": "string",
            "enum": ["literal", "metaphorical", "control", "uncertain"],
        },
        "main_image_schema": {
            "type": "string",
            "enum": [
                "CONTAINER",
                "SOURCE_PATH_GOAL",
                "FORCE",
                "BLOCKAGE",
                "VERTICALITY",
                "SUPPORT_BALANCE",
                "NONE",
                "uncertain",
            ],
        },
        "secondary_image_schemas": {
            "type": "array",
            "items": {"type": "string"},
        },
        "source_domain": {
            "type": "array",
            "items": {"type": "string"},
        },
        "target_domain": {
            "type": "array",
            "items": {"type": "string"},
        },
        "interpretation": {"type": "string"},
        "schema_explanation": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "literal_or_metaphorical",
        "main_image_schema",
        "secondary_image_schemas",
        "source_domain",
        "target_domain",
        "interpretation",
        "schema_explanation",
        "confidence",
    ],
    "additionalProperties": False,
}


STRUCTURED_ROLE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "literal_or_metaphorical": {
            "type": "string",
            "enum": ["literal", "metaphorical", "control", "uncertain"],
        },
        "main_image_schema": {
            "type": "string",
            "enum": [
                "CONTAINER",
                "SOURCE_PATH_GOAL",
                "FORCE",
                "BLOCKAGE",
                "VERTICALITY",
                "SUPPORT_BALANCE",
                "NONE",
                "uncertain",
            ],
        },
        "secondary_image_schemas": {
            "type": "array",
            "items": {"type": "string"},
        },
        "trajector": {"type": "string"},
        "landmark_or_container": {"type": "string"},
        "source": {"type": "string"},
        "path": {"type": "string"},
        "goal": {"type": "string"},
        "obstacle": {"type": "string"},
        "force": {"type": "string"},
        "source_domain": {
            "type": "array",
            "items": {"type": "string"},
        },
        "target_domain": {
            "type": "array",
            "items": {"type": "string"},
        },
        "interpretation": {"type": "string"},
        "schema_explanation": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [
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
    ],
    "additionalProperties": False,
}


def schema_for_prompt_family(prompt_family: str) -> dict[str, Any] | None:
    """Return project JSON schema for a prompt family."""
    if prompt_family == "direct_schema":
        return DIRECT_SCHEMA_JSON_SCHEMA
    if prompt_family == "structured_role_based":
        return STRUCTURED_ROLE_JSON_SCHEMA
    return None


def response_format_for_job(job: ExperimentJob) -> dict[str, Any] | None:
    """
    Return provider-agnostic response-format hints.

    OpenAI and Gemini clients adapt this dictionary to their provider-specific
    API parameters. Claude ignores it in the basic Messages integration.
    """
    if job.prompt.expected_output_format != "json_object":
        return None

    schema = schema_for_prompt_family(job.prompt.prompt_family)
    if not schema:
        return {"response_mime_type": "application/json"}

    return {
        "response_mime_type": "application/json",
        "response_schema": schema,
        "json_schema_name": f"{job.prompt.prompt_family}_response",
        "strict": True,
    }
