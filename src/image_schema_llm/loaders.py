from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from image_schema_llm.jsonl_utils import read_jsonl
from image_schema_llm.schemas import ConditionConfig, ModelConfig, PromptConfig, SentenceRecord

T = TypeVar("T")


def _construct_dataclass(cls: type[T], record: dict[str, Any]) -> T:
    """
    Construct a dataclass from a dictionary while preserving the original record.

    Purpose
    -------
    JSONL records may contain optional fields that evolve over time. This helper
    loads known dataclass fields and stores the full original object in `raw`.
    """

    field_names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in record.items() if k in field_names and k != "raw"}
    kwargs["raw"] = record
    return cls(**kwargs)


def load_models(path: Path) -> list[ModelConfig]:
    """Load model configurations from models.jsonl."""
    return [_construct_dataclass(ModelConfig, r) for r in read_jsonl(path)]


def load_prompts(path: Path) -> list[PromptConfig]:
    """Load prompt configurations from prompts.jsonl."""
    return [_construct_dataclass(PromptConfig, r) for r in read_jsonl(path)]


def load_conditions(path: Path) -> list[ConditionConfig]:
    """Load condition configurations from conditions.jsonl."""
    return [_construct_dataclass(ConditionConfig, r) for r in read_jsonl(path)]


def load_sentences(path: Path) -> list[SentenceRecord]:
    """Load sentence annotation records from sentences_v1.jsonl."""
    return [_construct_dataclass(SentenceRecord, r) for r in read_jsonl(path)]


def enabled_models(models: list[ModelConfig]) -> list[ModelConfig]:
    """Return only enabled model configurations."""
    return [m for m in models if m.enabled]


def enabled_conditions(conditions: list[ConditionConfig]) -> list[ConditionConfig]:
    """Return only enabled condition configurations."""
    return [c for c in conditions if c.enabled]
