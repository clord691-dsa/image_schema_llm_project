from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical project paths for the image-schema LLM project."""

    project_root: Path

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def inputs_dir(self) -> Path:
        return self.data_dir / "inputs"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def models_path(self) -> Path:
        return self.inputs_dir / "models.jsonl"

    @property
    def prompts_path(self) -> Path:
        return self.inputs_dir / "prompts.jsonl"

    @property
    def conditions_path(self) -> Path:
        return self.inputs_dir / "conditions.jsonl"

    @property
    def sentences_path(self) -> Path:
        return self.gold_dir / "sentences_v1.jsonl"

    @property
    def raw_responses_path(self) -> Path:
        return self.outputs_dir / "raw_responses.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.outputs_dir / "experiment_manifest.jsonl"
