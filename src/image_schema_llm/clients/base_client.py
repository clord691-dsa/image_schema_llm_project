from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelResponse:
    """
    Standard internal representation of a model response.

    Fields:
        raw_response:
            Full text response from the model.
        input_tokens:
            Number of input tokens reported by the API if available.
        output_tokens:
            Number of output tokens reported by the API if available.
        provider_metadata:
            Additional provider-specific metadata.
    """

    raw_response: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    provider_metadata: Dict[str, Any]


class BaseLLMClient:
    """
    Abstract base class for LLM providers.

    Purpose:
        Provide a common interface for OpenAI, Claude and Gemini clients.

    Expected subclasses:
        OpenAIClient
        ClaudeClient
        GeminiClient
    """

    def generate(
        self,
        prompt_text: str,
        model_name: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
    ) -> ModelResponse:
        """
        Send one prompt to an LLM provider and return a standard response.

        Inputs:
            prompt_text:
                Final prompt to send to the model.
            model_name:
                Provider-specific model name.
            temperature:
                Sampling temperature for this condition.
            top_p:
                Top-p sampling value for this condition.
            max_output_tokens:
                Maximum number of output tokens.

        Outputs:
            ModelResponse object containing raw text and token usage.

        Purpose:
            Standardises API behaviour across providers.
        """
        raise NotImplementedError
    