from __future__ import annotations

import os
from typing import Any

from google import genai
from google.genai import types

from image_schema_llm.clients.base_client import ModelResponse


class GeminiGenerateContentClient:
    """
    Google Gemini client using the Google Gen AI Python SDK.

    Uses a Gemini Developer API key from GEMINI_API_KEY by default, or from the
    environment variable named in models.jsonl.
    """

    def __init__(self, *, api_key_env_var: str = "GEMINI_API_KEY") -> None:
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
        """
        Generate one Gemini response.

        `system_message` is passed as system_instruction. If response_format
        includes {"response_mime_type": "application/json"}, Gemini JSON mode is
        requested.
        """

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
                config_kwargs["response_schema"] = response_format["response_schema"]

        response = self.client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        return ModelResponse(
            raw_response=self._extract_output_text(response),
            input_tokens=self._extract_usage(response)[0],
            output_tokens=self._extract_usage(response)[1],
            provider_response_id=getattr(response, "response_id", None),
            provider_metadata=self._safe_metadata(response),
        )

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text

        chunks: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content is not None else []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(part_text)
        return "\n".join(chunks)

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
            "usage": usage_metadata,
        }
