def test_manifest_runner_imports():
    from image_schema_llm.manifest_runner import (
        pending_jobs_for_provider,
        provider_label,
        run_provider_manifest,
    )

    assert provider_label("openai") == "OpenAI"
    assert provider_label("anthropic") == "Claude"
    assert provider_label("google") == "Gemini"
    assert pending_jobs_for_provider is not None
    assert run_provider_manifest is not None
