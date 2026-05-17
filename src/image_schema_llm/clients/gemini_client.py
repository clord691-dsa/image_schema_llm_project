from image_schema_llm.clients.base_client import BaseLLMClient, ModelResponse


class GeminiClient(BaseLLMClient):
    """
    Placeholder client for Google Gemini models.

    Purpose:
        Encapsulate all Gemini-specific API logic.

    Inputs:
        API key should be read from environment variables.

    Outputs:
        ModelResponse containing raw response text and token usage.

    Notes:
        Gemini usage metadata should be normalised into the shared
        ModelResponse format.
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
        Send a prompt to a Gemini model.

        Inputs:
            prompt_text: Full prompt text.
            model_name: Gemini model name.
            temperature: Sampling temperature.
            top_p: Top-p value.
            max_output_tokens: Maximum completion length.

        Outputs:
            ModelResponse.

        Purpose:
            Allows Gemini to participate in the same experiment loop as
            OpenAI and Claude.
        """
        raise NotImplementedError
    