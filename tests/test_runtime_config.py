from pathlib import Path

from image_schema_llm.runtime_config import (
    load_runtime_config,
    validate_runtime_config,
    write_default_runtime_config,
)


def test_write_and_load_runtime_config(tmp_path: Path):
    path = write_default_runtime_config(tmp_path)
    assert path.exists()

    config = load_runtime_config(tmp_path)
    assert config.spend_threshold == 10.0
    assert config.currency == "USD"
    assert validate_runtime_config(config) == []
