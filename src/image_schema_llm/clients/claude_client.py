from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic

from image_schema_llm.clients.base_client import ModelResponse


class ClaudeMessagesClient:
    """
    Anthropic Claude client using the Messages API.

    The Anthropic SDK reads ANTHROPIC_API_KEY by default, but this client
    supports any environment variable name supplied by models.jsonl.
    """

    def __init__(self, *, api_key_env_var: str = "ANTHROPIC_API_KEY") -> None:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Missing Anthropic API key. Set environment variable {api_key_env_var}."
            )
        self.api_key_env_var = api_key_env_var
        self.client = Anthropic(api_key=api_key)

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
        Generate one Claude response.

        Claude uses `max_tokens`, not `max_output_tokens`. JSON output is
        controlled by the prompt in this simple integration.
        """

        request: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_output_tokens,
            "system": system_message,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if temperature is not None:
            request["temperature"] = temperature
        if top_p is not None:
            request["top_p"] = top_p

        response = self.client.messages.create(**request)

        return ModelResponse(
            raw_response=self._extract_output_text(response),
            input_tokens=self._extract_usage(response)[0],
            output_tokens=self._extract_usage(response)[1],
            provider_response_id=getattr(response, "id", None),
            provider_metadata=self._safe_metadata(response),
        )

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        chunks: list[str] = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        return "\n".join(chunks)

    @staticmethod
    def _extract_usage(response: Any) -> tuple[int | None, int | None]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None
        if isinstance(usage, dict):
            return usage.get("input_tokens"), usage.get("output_tokens")
        return getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)

    @staticmethod
    def _safe_metadata(response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if isinstance(usage, dict):
            usage_metadata = usage
        elif usage is not None:
            usage_metadata = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            }
        else:
            usage_metadata = None
        return {
            "provider": "anthropic",
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "usage": usage_metadata,
        }
