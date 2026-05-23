def test_client_modules_import_without_instantiating_sdks():
    from image_schema_llm.clients.base_client import ModelResponse
    from image_schema_llm.clients.openai_client import OpenAIResponsesClient
    from image_schema_llm.clients.claude_client import ClaudeMessagesClient
    from image_schema_llm.clients.gemini_client import GeminiGenerateContentClient

    assert ModelResponse is not None
    assert OpenAIResponsesClient is not None
    assert ClaudeMessagesClient is not None
    assert GeminiGenerateContentClient is not None
