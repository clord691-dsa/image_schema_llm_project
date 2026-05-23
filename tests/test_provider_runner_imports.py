def test_provider_runner_imports_without_provider_sdks():
    from image_schema_llm.manifest_runner import provider_label
    from image_schema_llm.provider_runner import select_next_provider_job, run_provider_job
    from image_schema_llm.openai_runner import select_next_openai_job, run_openai_job

    assert provider_label("openai") == "OpenAI"
    assert select_next_provider_job is not None
    assert run_provider_job is not None
    assert select_next_openai_job is not None
    assert run_openai_job is not None
