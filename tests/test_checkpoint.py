from pathlib import Path

from image_schema_llm.checkpoint import CheckpointManager


def test_completed_run_keys(tmp_path: Path):
    manager = CheckpointManager(tmp_path)
    manager.write_success_marker_from_raw_response(
        {
            "run_key": "m|p|c|s|0",
            "status": "success",
            "raw_response": "example",
        }
    )

    assert manager.load_completed_run_keys() == {"m|p|c|s|0"}


def test_error_does_not_mark_completed(tmp_path: Path):
    manager = CheckpointManager(tmp_path)
    manager.write_error_record(
        run_key="m|p|c|s|0",
        error_type="ConnectionError",
        error_message="network failed",
        retryable=True,
    )

    assert manager.load_completed_run_keys() == set()
    assert manager.load_failed_run_keys() == {"m|p|c|s|0"}
