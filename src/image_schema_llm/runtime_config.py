from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_schema_llm.config import ProjectPaths


@dataclass(frozen=True)
class RuntimeConfig:
    """
    Runtime execution settings for the image-schema LLM experiment.

    Purpose
    -------
    Stores budget and execution controls outside Python code so that
    spend thresholds and runtime behaviour are auditable and consistent
    across scripts and provider runners.

    The canonical file is:

        data/inputs/runtime_config.json
    """

    spend_threshold: float
    currency: str = "USD"
    stop_on_error: bool = False
    dry_run: bool = False
    cost_log_filename: str = "cost_log.jsonl"
    cost_summary_filename: str = "cost_summary.json"
    notes: str | None = None
    raw: dict[str, Any] | None = None


DEFAULT_RUNTIME_CONFIG = RuntimeConfig(
    spend_threshold=10.0,
    currency="USD",
    stop_on_error=False,
    dry_run=False,
    cost_log_filename="cost_log.jsonl",
    cost_summary_filename="cost_summary.json",
    notes="Default runtime configuration used when data/inputs/runtime_config.json is absent.",
    raw=None,
)


def runtime_config_path(project_root: Path) -> Path:
    """
    Return the canonical runtime config path for a project.
    """

    return ProjectPaths(project_root).inputs_dir / "runtime_config.json"


def load_runtime_config(project_root: Path) -> RuntimeConfig:
    """
    Load runtime configuration from data/inputs/runtime_config.json.

    If the file is absent, a safe default configuration is returned. This
    keeps early development scripts working while still supporting a fully
    auditable config file once the project matures.
    """

    path = runtime_config_path(project_root)

    if not path.exists():
        return DEFAULT_RUNTIME_CONFIG

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in runtime config file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Runtime config must be a JSON object: {path}")

    return RuntimeConfig(
        spend_threshold=float(data.get("spend_threshold", DEFAULT_RUNTIME_CONFIG.spend_threshold)),
        currency=str(data.get("currency", DEFAULT_RUNTIME_CONFIG.currency)),
        stop_on_error=bool(data.get("stop_on_error", DEFAULT_RUNTIME_CONFIG.stop_on_error)),
        dry_run=bool(data.get("dry_run", DEFAULT_RUNTIME_CONFIG.dry_run)),
        cost_log_filename=str(data.get("cost_log_filename", DEFAULT_RUNTIME_CONFIG.cost_log_filename)),
        cost_summary_filename=str(data.get("cost_summary_filename", DEFAULT_RUNTIME_CONFIG.cost_summary_filename)),
        notes=data.get("notes"),
        raw=data,
    )


def write_default_runtime_config(project_root: Path, *, overwrite: bool = False) -> Path:
    """
    Write a default runtime_config.json file.

    Parameters
    ----------
    project_root:
        Repository root.
    overwrite:
        If False, do not overwrite an existing runtime_config.json.

    Returns
    -------
    Path
        Path to the runtime config file.
    """

    path = runtime_config_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        return path

    data = {
        "spend_threshold": DEFAULT_RUNTIME_CONFIG.spend_threshold,
        "currency": DEFAULT_RUNTIME_CONFIG.currency,
        "stop_on_error": DEFAULT_RUNTIME_CONFIG.stop_on_error,
        "dry_run": DEFAULT_RUNTIME_CONFIG.dry_run,
        "cost_log_filename": DEFAULT_RUNTIME_CONFIG.cost_log_filename,
        "cost_summary_filename": DEFAULT_RUNTIME_CONFIG.cost_summary_filename,
        "notes": DEFAULT_RUNTIME_CONFIG.notes,
    }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def validate_runtime_config(config: RuntimeConfig) -> list[str]:
    """
    Validate runtime configuration values.

    Returns
    -------
    list[str]
        Empty list if valid; otherwise human-readable error messages.
    """

    errors: list[str] = []

    if config.spend_threshold <= 0:
        errors.append("spend_threshold must be greater than zero")

    if not config.currency:
        errors.append("currency must not be empty")

    if not config.cost_log_filename.endswith(".jsonl"):
        errors.append("cost_log_filename should end with .jsonl")

    if not config.cost_summary_filename.endswith(".json"):
        errors.append("cost_summary_filename should end with .json")

    return errors
