from __future__ import annotations

from image_schema_llm.schemas import PromptConfig, SentenceRecord


def validate_prompt_template(prompt: PromptConfig) -> list[str]:
    """
    Validate required placeholders without parsing JSON braces as format fields.
    """
    errors: list[str] = []
    for placeholder_name in prompt.required_placeholders:
        placeholder = "{" + placeholder_name + "}"
        if placeholder not in prompt.user_prompt_template:
            errors.append(
                f"{prompt.prompt_id}: required placeholder {placeholder} is missing"
            )
    return errors


def build_user_prompt(prompt: PromptConfig, sentence: SentenceRecord) -> str:
    """
    Render one user prompt for one sentence.

    Uses direct replacement rather than str.format(), because prompt templates
    contain literal JSON examples with braces.
    """
    rendered = prompt.user_prompt_template
    for placeholder_name in prompt.required_placeholders:
        placeholder = "{" + placeholder_name + "}"
        if placeholder_name == "sentence_text":
            rendered = rendered.replace(placeholder, sentence.text)
        else:
            raise ValueError(
                f"Unsupported placeholder {placeholder!r} in {prompt.prompt_id}; "
                "only {sentence_text} is supported."
            )
    return rendered
