
from image_schema_llm.clients.base_client import BaseLLMClient
from image_schema_llm.clients.openai_client import OpenAIClient
from image_schema_llm.clients.claude_client import ClaudeClient
from image_schema_llm.clients.gemini_client import GeminiClient


def get_client(provider: str) -> BaseLLMClient:
    """
    Return the correct LLM client for a provider.

    Inputs:
        provider:
            Provider name from models.jsonl.
            Expected values:
                - openai
                - anthropic
                - google

    Outputs:
        Instance of a BaseLLMClient subclass.

    Purpose:
        Keeps provider selection out of the main experiment loop.
    """
    if provider == "openai":
        return OpenAIClient()
    if provider == "anthropic":
        return ClaudeClient()
    if provider == "google":
        return GeminiClient()

    raise ValueError(f"Unsupported provider: {provider}")
