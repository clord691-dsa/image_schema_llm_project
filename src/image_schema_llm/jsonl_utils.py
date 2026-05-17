import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file into a list of dictionaries.

    Inputs:
        path: Path to a JSONL file.

    Outputs:
        A list of dictionaries, one dictionary per JSONL line.

    Purpose:
        Used for loading sentences, prompts, models, conditions,
        previous raw responses, cost logs, and errors.
    """
    raise NotImplementedError


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """
    Append a single dictionary to a JSONL file.

    Inputs:
        path: Output JSONL path.
        record: Dictionary to write as one JSONL line.

    Outputs:
        None.

    Purpose:
        Used to persist each successful model response immediately.
        Immediate writing makes the experiment restartable after errors.
    """
    raise NotImplementedError


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """
    Write multiple dictionaries to a JSONL file.

    Inputs:
        path: Output JSONL path.
        records: Iterable of dictionaries.

    Outputs:
        None.

    Purpose:
        Useful for creating fixture files, validation outputs, or
        derived parsed-response databases.
    """
    raise NotImplementedError
