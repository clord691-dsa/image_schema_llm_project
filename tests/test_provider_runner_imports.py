def test_provider_imports():
    from image_schema_llm.clients.claude_client import ClaudeMessagesClient
    from image_schema_llm.clients.gemini_client import GeminiGenerateContentClient
    from image_schema_llm.provider_runner import select_next_provider_job, run_provider_job

    assert ClaudeMessagesClient is not None
    assert GeminiGenerateContentClient is not None
    assert select_next_provider_job is not None
    assert run_provider_job is not None
