from image_schema_llm.parsing.response_parser import parse_response_text


def test_complete_json_parses():
    raw = '{"literal_or_metaphorical":"literal","main_image_schema":"CONTAINER"}'
    parsed = parse_response_text(raw, expected_output_format="json_object")
    assert parsed.parse_status == "parsed"
    assert parsed.usable_for_schema_accuracy
    assert parsed.usable_for_lm_accuracy


def test_fenced_json_parses():
    raw = """```json
    {"literal_or_metaphorical":"literal","main_image_schema":"CONTAINER"}
    ```"""
    parsed = parse_response_text(raw, expected_output_format="json_object")
    assert parsed.parse_status == "parsed"


def test_truncated_core_fields_are_schema_usable():
    raw = '{"literal_or_metaphorical": "metaphorical", "main_image_schema": "SOURCE_PATH_GOAL", "interpretation": "The'
    parsed = parse_response_text(raw, expected_output_format="json_object")
    assert parsed.parse_status == "parsed"
    assert parsed.parse_quality == "partial_core_fields"
    assert parsed.usable_for_schema_accuracy
    assert parsed.usable_for_lm_accuracy
    assert parsed.main_image_schema == "SOURCE_PATH_GOAL"


def test_truncated_literal_only_is_partial_lm_usable():
    raw = '{"literal_or_metaphorical": "metaphorical", "main'
    parsed = parse_response_text(raw, expected_output_format="json_object")
    assert parsed.parse_status == "partial"
    assert parsed.parse_quality == "partial_literality_only"
    assert not parsed.usable_for_schema_accuracy
    assert parsed.usable_for_lm_accuracy
    assert parsed.literal_or_metaphorical == "metaphorical"


def test_path_alias_normalised():
    raw = '{"literal_or_metaphorical":"metaphorical","main_image_schema":"PATH"}'
    parsed = parse_response_text(raw, expected_output_format="json_object")
    assert parsed.main_image_schema == "SOURCE_PATH_GOAL"
