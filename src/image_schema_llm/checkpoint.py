from pathlib import Path
from typing import Set

from image_schema_llm.jsonl_utils import read_jsonl


def load_completed_run_keys(raw_responses_path: Path) -> Set[str]:
    """
    Load completed runs from raw_responses.jsonl.

    Inputs:
        raw_responses_path:
            Path to the raw response JSONL database.

    Outputs:
        A set of completed run keys.

    Purpose:
        Allows the experiment to resume after interruption.
        If a run already has status == "success", it should be skipped
        rather than calling the API again.
    """
    raise NotImplementedError


def should_skip_run(run_key: str, completed_run_keys: Set[str]) -> bool:
    """
    Decide whether a job should be skipped.

    Inputs:
        run_key:
            Stable unique key for the current job.
        completed_run_keys:
            Set of successfully completed run keys.

    Outputs:
        Boolean.

    Purpose:
        Prevents duplicate API calls and unnecessary spending after restart.
    """
    raise NotImplementedError
