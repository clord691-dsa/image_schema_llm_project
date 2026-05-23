from __future__ import annotations

import os
import warnings
from typing import Any

from image_schema_llm.clients.base_client import ModelResponse


class ClaudeMessagesClient:
    """Anthropic Claude client using the Messages API.

    The Anthropic SDK is imported lazily in __init__ so importing the package
    does not require every optional provider dependency to be installed.
    """

    def __init__(self, *, api_key_env_var: str = "ANTHROPIC_API_KEY") -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The Anthropic SDK is not installed. Run: python -m pip install anthropic"
            ) from exc

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
        request: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_output_tokens,
            "system": system_message,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if temperature is not None and top_p is not None:
            warnings.warn(
                "Claude request received both temperature and top_p. Suppressing top_p.",
                RuntimeWarning,
                stacklevel=2,
            )
            request["temperature"] = temperature
        elif temperature is not None:
            request["temperature"] = temperature
        elif top_p is not None:
            request["top_p"] = top_p

        response = self.client.messages.create(**request)
        input_tokens, output_tokens = self._extract_usage(response)
        return ModelResponse(
            raw_response=self._extract_output_text(response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_response_id=getattr(response, "id", None),
            provider_metadata=self._safe_metadata(response),
            finish_reason=getattr(response, "stop_reason", None),
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
