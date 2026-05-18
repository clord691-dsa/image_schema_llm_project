from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from image_schema_llm.jsonl_utils import read_jsonl
from image_schema_llm.schemas import ConditionConfig, ModelConfig, PromptConfig, SentenceRecord

T = TypeVar("T")


def _construct_dataclass(cls: type[T], record: dict[str, Any], *, source_path: Path, index: int) -> T:
    """Construct a dataclass while preserving the original record in raw."""
    field_names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in record.items() if k in field_names and k != "raw"}
    kwargs["raw"] = record
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Could not construct {cls.__name__} from {source_path} record {index}: {exc}"
        ) from exc


def load_models(path: Path) -> list[ModelConfig]:
    return [_construct_dataclass(ModelConfig, r, source_path=path, index=i)
            for i, r in enumerate(read_jsonl(path), start=1)]


def load_prompts(path: Path) -> list[PromptConfig]:
    return [_construct_dataclass(PromptConfig, r, source_path=path, index=i)
            for i, r in enumerate(read_jsonl(path), start=1)]


def load_conditions(path: Path) -> list[ConditionConfig]:
    return [_construct_dataclass(ConditionConfig, r, source_path=path, index=i)
            for i, r in enumerate(read_jsonl(path), start=1)]


def load_sentences(path: Path) -> list[SentenceRecord]:
    return [_construct_dataclass(SentenceRecord, r, source_path=path, index=i)
            for i, r in enumerate(read_jsonl(path), start=1)]


def enabled_models(models: list[ModelConfig]) -> list[ModelConfig]:
    return [m for m in models if m.enabled]


def enabled_conditions(conditions: list[ConditionConfig]) -> list[ConditionConfig]:
    return [c for c in conditions if c.enabled]
