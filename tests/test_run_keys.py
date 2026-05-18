from image_schema_llm.experiment_grid import make_run_key, parse_run_key


def test_make_and_parse_run_key():
    key = make_run_key("m1", "p1", "c1", "s0001", 0)
    assert key == "m1|p1|c1|s0001|0"
    assert parse_run_key(key) == {
        "model_id": "m1",
        "prompt_id": "p1",
        "condition_id": "c1",
        "sentence_id": "s0001",
        "repetition_index": 0,
    }
