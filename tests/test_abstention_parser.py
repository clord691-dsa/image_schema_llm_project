
from image_schema_llm.parsing.response_parser import parse_response_text

def test_schema_present_no_forces_control_none():
    raw = '{"schema_present":"no","literal_or_metaphorical":"literal","main_image_schema":"CONTAINER","secondary_image_schemas":["FORCE"],"source_domain":["x"],"target_domain":["y"],"confidence":"medium"}'
    parsed = parse_response_text(raw, expected_output_format="json_object")
    assert parsed.schema_present == "no"
    assert parsed.literal_or_metaphorical == "control"
    assert parsed.main_image_schema == "NONE"
    assert parsed.secondary_image_schemas == []
    assert parsed.source_domain == []
    assert parsed.target_domain == []

def test_control_forces_none():
    raw = '{"schema_present":"yes","literal_or_metaphorical":"control","main_image_schema":"FORCE","confidence":"high"}'
    parsed = parse_response_text(raw, expected_output_format="json_object")
    assert parsed.schema_present == "no"
    assert parsed.literal_or_metaphorical == "control"
    assert parsed.main_image_schema == "NONE"

def test_path_alias_normalised():
    raw = '{"schema_present":"yes","literal_or_metaphorical":"metaphorical","main_image_schema":"PATH","source_domain":["movement"],"target_domain":["time"],"confidence":"high"}'
    parsed = parse_response_text(raw, expected_output_format="json_object")
    assert parsed.main_image_schema == "SOURCE_PATH_GOAL"
    assert parsed.schema_present == "yes"
