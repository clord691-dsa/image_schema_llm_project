from image_schema_llm.clients.base_client import BaseLLMClient, ModelResponse


class OpenAIClient(BaseLLMClient):
    """
    Placeholder client for OpenAI models.

    Purpose:
        Encapsulate all OpenAI-specific API logic.

    Inputs:
        API key should be read from environment variables, not stored in code.

    Outputs:
        ModelResponse with:
            - raw model text
            - input token count
            - output token count
            - provider metadata

    Notes:
        This class should eventually call the OpenAI API.
        At this design stage, implementation is intentionally omitted.
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
        Send a prompt to an OpenAI model.

        Inputs:
            prompt_text: Full prompt text.
            model_name: OpenAI model name.
            temperature: Sampling temperature.
            top_p: Top-p value.
            max_output_tokens: Maximum completion length.

        Outputs:
            ModelResponse.

        Purpose:
            Allows the experiment runner to call OpenAI models through the
            same interface used for Claude and Gemini.
        """
        raise NotImplementedError

