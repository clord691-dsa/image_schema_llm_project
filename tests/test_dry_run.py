from image_schema_llm.dry_run import estimate_tokens_from_text


def test_estimate_tokens_from_text_minimum_one():
    assert estimate_tokens_from_text("") == 1


def test_estimate_tokens_from_text_ratio():
    assert estimate_tokens_from_text("a" * 400, token_char_ratio=4.0) == 100
