from __future__ import annotations

import os
from typing import Any

from image_schema_llm.clients.base_client import ModelResponse


def _sanitize_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Remove JSON Schema keywords that Gemini response_schema may reject.

    Gemini supports a useful subset of JSON schema for structured output. This
    keeps the project schema strict enough for shape guidance while avoiding
    provider-specific incompatibilities.
    """
    unsupported_keys = {"additionalProperties", "additional_properties", "$schema", "$id"}

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: clean(child)
                for key, child in value.items()
                if key not in unsupported_keys
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(schema)


class GeminiGenerateContentClient:
    """Google Gemini client using the Google Gen AI Python SDK.

    The Google SDK is imported lazily in __init__ so importing the package does
    not require every optional provider dependency to be installed.
    """

    def __init__(self, *, api_key_env_var: str = "GEMINI_API_KEY") -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "The Google Gen AI SDK is not installed. Run: python -m pip install google-genai"
            ) from exc

        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Missing Gemini API key. Set environment variable {api_key_env_var}."
            )

        self.api_key_env_var = api_key_env_var
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        *,
        system_message: str,
        user_prompt: str,
        model_name: str,
        temperature: float | None,
        top_p: float | None,
        max_output_tokens: int,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> ModelResponse:
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "The Google Gen AI SDK is not installed. Run: python -m pip install google-genai"
            ) from exc

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_message,
            "max_output_tokens": max_output_tokens,
        }

        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if top_p is not None:
            config_kwargs["top_p"] = top_p

        if response_format:
            if response_format.get("response_mime_type"):
                config_kwargs["response_mime_type"] = response_format["response_mime_type"]
            if response_format.get("response_schema"):
                config_kwargs["response_schema"] = _sanitize_schema_for_gemini(
                    response_format["response_schema"]
                )

        response = self.client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        input_tokens, output_tokens = self._extract_usage(response)
        metadata = self._safe_metadata(response)
        metadata["request_config"] = {
            "response_mime_type": config_kwargs.get("response_mime_type"),
            "has_response_schema": "response_schema" in config_kwargs,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": reasoning_effort,
        }

        return ModelResponse(
            raw_response=self._extract_output_text(response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_response_id=getattr(response, "response_id", None),
            provider_metadata=metadata,
            finish_reason=self._extract_finish_reason(response),
        )

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        """
        Extract text from Gemini candidates directly before falling back to the
        convenience response.text accessor.

        Candidate walking is more transparent for debugging blocked or truncated
        outputs because the raw text parts are read explicitly.
        """

        chunks: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content is not None else []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(part_text)

        if chunks:
            return "\n".join(chunks)

        text = getattr(response, "text", None)
        return text if text else ""

    @staticmethod
    def _extract_usage(response: Any) -> tuple[int | None, int | None]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return None, None
        if isinstance(usage, dict):
            return usage.get("prompt_token_count"), usage.get("candidates_token_count")
        return (
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )

    @staticmethod
    def _extract_finish_reason(response: Any) -> str | None:
        reasons: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason:
                reasons.append(str(finish_reason))
        return "|".join(dict.fromkeys(reasons)) if reasons else None

    @staticmethod
    def _safe_metadata(response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict):
            usage_metadata = usage
        elif usage is not None:
            usage_metadata = {
                "prompt_token_count": getattr(usage, "prompt_token_count", None),
                "candidates_token_count": getattr(usage, "candidates_token_count", None),
                "cached_content_token_count": getattr(usage, "cached_content_token_count", None),
                "total_token_count": getattr(usage, "total_token_count", None),
            }
        else:
            usage_metadata = None

        return {
            "provider": "google",
            "response_id": getattr(response, "response_id", None),
            "model_version": getattr(response, "model_version", None),
            "finish_reason": GeminiGenerateContentClient._extract_finish_reason(response),
            "usage": usage_metadata,
        }
