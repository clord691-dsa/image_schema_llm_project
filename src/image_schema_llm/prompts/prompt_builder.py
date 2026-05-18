from __future__ import annotations

from string import Formatter

from image_schema_llm.schemas import PromptConfig, SentenceRecord


def template_fields(template: str) -> set[str]:
    """
    Return placeholder names used by a format template.

    Example
    -------
    "Sentence: {sentence_text}" -> {"sentence_text"}
    """

    return {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }


def validate_prompt_template(prompt: PromptConfig) -> list[str]:
    """
    Validate that required placeholders are present in the prompt template.

    This avoids Python Formatter parsing because the templates contain literal
    JSON braces.
    """
    errors: list[str] = []

    for required in prompt.required_placeholders:
        placeholder = "{" + required + "}"
        if placeholder not in prompt.user_prompt_template:
            errors.append(
                f"{prompt.prompt_id}: required placeholder {placeholder} "
                "is missing from user_prompt_template"
            )

    return errors


def build_user_prompt(prompt: PromptConfig, sentence: SentenceRecord) -> str:
    """
    Build the user prompt for one sentence.

    Inputs
    ------
    prompt:
        PromptConfig loaded from prompts.jsonl.
    sentence:
        SentenceRecord loaded from sentences_v1.jsonl.

    Outputs
    -------
    str
        Fully formatted user prompt ready to send to an LLM provider.

    Purpose
    -------
    Ensures all providers receive the same user-facing prompt text for a given
    prompt family and sentence.

    Uses direct replacement rather than str.format(), because prompt templates
    contain JSON examples with literal braces. Python str.format() would treat
    those JSON braces as placeholders and raise KeyError.
    """

    return prompt.user_prompt_template.replace("{sentence_text}", sentence.text)
