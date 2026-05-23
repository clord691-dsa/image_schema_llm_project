from image_schema_llm.structured_output import schema_for_prompt_family


def test_direct_schema_available():
    schema = schema_for_prompt_family("direct_schema")
    assert schema is not None
    assert "literal_or_metaphorical" in schema["properties"]


def test_structured_role_schema_available():
    schema = schema_for_prompt_family("structured_role_based")
    assert schema is not None
    assert "trajector" in schema["properties"]


def test_naive_has_no_schema():
    assert schema_for_prompt_family("naive") is None
