from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Read a JSONL file into a list of dictionaries.

    Parameters
    ----------
    path:
        Path to a JSONL file.

    Returns
    -------
    list[dict[str, Any]]
        One dictionary per non-empty line.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If a line is not valid JSON or does not contain a JSON object.
    """

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc

            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}, got {type(obj).__name__}")

            records.append(obj)

    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """
    Write dictionaries to a JSONL file.

    Purpose
    -------
    Useful for generated files such as validation reports or fixture datasets.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """
    Append one dictionary to a JSONL file.

    Purpose
    -------
    Later used by the experiment runner to persist raw model outputs immediately
    after each successful API response.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def read_pretty_json(path: Path) -> Any:
    """
    Read a standard pretty JSON file.

    Purpose
    -------
    Provided for inspection files such as models_v1_pretty.json, although the
    canonical machine-readable inputs should remain JSONL.
    """

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
