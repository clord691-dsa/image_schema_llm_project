from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from image_schema_llm.clients.base_client import ModelResponse


class OpenAIResponsesClient:
    """
    OpenAI client using the Responses API.

    The SDK reads OPENAI_API_KEY by default, but this client supports any
    environment variable name supplied by the model config.
    """

    def __init__(self, *, api_key_env_var: str = "OPENAI_API_KEY") -> None:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Missing OpenAI API key. Set environment variable {api_key_env_var}."
            )

        self.api_key_env_var = api_key_env_var
        self.client = OpenAI(api_key=api_key)

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
        Generate one response using OpenAI Responses API.
        """

        request: dict[str, Any] = {
            "model": model_name,
            "input": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            "max_output_tokens": max_output_tokens,
        }

        if temperature is not None:
            request["temperature"] = temperature

        if top_p is not None:
            request["top_p"] = top_p

        if response_format is not None:
            request["text"] = {"format": response_format}

        if reasoning_effort is not None:
            request["reasoning"] = {"effort": reasoning_effort}

        response = self.client.responses.create(**request)

        raw_text = self._extract_output_text(response)
        input_tokens, output_tokens = self._extract_usage(response)

        return ModelResponse(
            raw_response=raw_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_response_id=getattr(response, "id", None),
            provider_metadata=self._safe_metadata(response),
        )

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        """
        Extract text from an OpenAI Responses API response.
        """

        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)

        return "\n".join(chunks)

    @staticmethod
    def _extract_usage(response: Any) -> tuple[int | None, int | None]:
        """
        Extract provider-reported token usage.
        """

        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None

        def get_usage_value(*names: str) -> int | None:
            for name in names:
                if isinstance(usage, dict) and name in usage:
                    return usage[name]
                value = getattr(usage, name, None)
                if value is not None:
                    return value
            return None

        input_tokens = get_usage_value("input_tokens", "prompt_tokens")
        output_tokens = get_usage_value("output_tokens", "completion_tokens")

        return input_tokens, output_tokens

    @staticmethod
    def _safe_metadata(response: Any) -> dict[str, Any]:
        """
        Return minimal provider metadata suitable for JSON serialisation.
        """

        usage = getattr(response, "usage", None)
        usage_metadata: dict[str, Any] | None = None

        if usage is not None:
            if isinstance(usage, dict):
                usage_metadata = usage
            else:
                usage_metadata = {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }

        return {
            "provider": "openai",
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "usage": usage_metadata,
        }
