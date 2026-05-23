from __future__ import annotations

import os
from typing import Any

from image_schema_llm.clients.base_client import ModelResponse


class OpenAIResponsesClient:
    """OpenAI client using the Responses API with optional structured output.

    The OpenAI SDK is imported lazily in __init__ so that local unit tests can
    import the package without requiring all provider SDKs to be installed.
    """

    def __init__(self, *, api_key_env_var: str = "OPENAI_API_KEY") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI SDK is not installed. Run: python -m pip install openai"
            ) from exc

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

        if response_format and response_format.get("response_schema"):
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": response_format.get("json_schema_name", "image_schema_response"),
                    "schema": response_format["response_schema"],
                    "strict": bool(response_format.get("strict", True)),
                }
            }
        elif response_format and response_format.get("response_mime_type") == "application/json":
            request["text"] = {"format": {"type": "json_object"}}

        if reasoning_effort is not None:
            request["reasoning"] = {"effort": reasoning_effort}

        response = self.client.responses.create(**request)
        input_tokens, output_tokens = self._extract_usage(response)

        return ModelResponse(
            raw_response=self._extract_output_text(response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_response_id=getattr(response, "id", None),
            provider_metadata=self._safe_metadata(response),
            finish_reason=self._extract_finish_reason(response),
        )

    @staticmethod
    def _extract_output_text(response: Any) -> str:
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
        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None
        def get_value(*names: str) -> int | None:
            for name in names:
                if isinstance(usage, dict) and name in usage:
                    return usage[name]
                value = getattr(usage, name, None)
                if value is not None:
                    return value
            return None
        return get_value("input_tokens", "prompt_tokens"), get_value("output_tokens", "completion_tokens")

    @staticmethod
    def _extract_finish_reason(response: Any) -> str | None:
        reasons: list[str] = []
        for item in getattr(response, "output", []) or []:
            for name in ("status", "finish_reason"):
                value = getattr(item, name, None)
                if value:
                    reasons.append(str(value))
        if reasons:
            return "|".join(dict.fromkeys(reasons))
        status = getattr(response, "status", None)
        return str(status) if status else None

    @staticmethod
    def _safe_metadata(response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if isinstance(usage, dict):
            usage_metadata = usage
        elif usage is not None:
            usage_metadata = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        else:
            usage_metadata = None
        return {
            "provider": "openai",
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "status": getattr(response, "status", None),
            "usage": usage_metadata,
        }
