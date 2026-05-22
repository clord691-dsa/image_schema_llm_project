from image_schema_llm.parsing.response_parser import parse_response_text


def test_parse_json_response():
    raw = """
    {
      "literal_or_metaphorical": "literal",
      "main_image_schema": "CONTAINER",
      "secondary_image_schemas": [],
      "interpretation": "The keys are located inside the box.",
      "schema_explanation": "The sentence uses containment.",
      "confidence": "high"
    }
    """

    parsed = parse_response_text(raw, expected_output_format="json_object")

    assert parsed.parse_status == "parsed"
    assert parsed.literal_or_metaphorical == "literal"
    assert parsed.main_image_schema == "CONTAINER"


def test_parse_fenced_json_response():
    raw = """```json
    {
      "literal_or_metaphorical": "metaphorical",
      "main_image_schema": "BLOCKAGE",
      "secondary_image_schemas": ["FORCE"],
      "source_domain": ["physical obstruction"],
      "target_domain": ["progress"],
      "confidence": "medium"
    }
    ```"""

    parsed = parse_response_text(raw, expected_output_format="json_object")

    assert parsed.parse_status == "parsed"
    assert parsed.main_image_schema == "BLOCKAGE"
    assert parsed.source_domain == ["physical obstruction"]
    assert parsed.target_domain == ["progress"]


def test_parse_free_text_response():
    parsed = parse_response_text(
        "The sentence means the keys are inside the box.",
        expected_output_format="free_text",
    )

    assert parsed.parse_status == "free_text_unparsed"
    assert parsed.interpretation == "The sentence means the keys are inside the box."


def test_parse_error_preserved():
    parsed = parse_response_text(
        "This is not JSON",
        expected_output_format="json_object",
    )

    assert parsed.parse_status == "parse_error"
    assert parsed.parse_error is not None
