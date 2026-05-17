from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    """
    Stores central project paths.

    Purpose:
        Avoid hard-coding paths throughout the project.

    Inputs:
        project_root: Root directory of the GitHub project.

    Outputs:
        Path objects pointing to input and output JSONL databases.
    """

    project_root: Path

    @property
    def input_dir(self) -> Path:
        return self.project_root / "data" / "inputs"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "data" / "outputs"

    @property
    def sentences_path(self) -> Path:
        return self.input_dir / "sentences.jsonl"

    @property
    def prompts_path(self) -> Path:
        return self.input_dir / "prompts.jsonl"

    @property
    def conditions_path(self) -> Path:
        return self.input_dir / "conditions.jsonl"

    @property
    def models_path(self) -> Path:
        return self.input_dir / "models.jsonl"

    @property
    def raw_responses_path(self) -> Path:
        return self.output_dir / "raw_responses.jsonl"

    @property
    def cost_log_path(self) -> Path:
        return self.output_dir / "cost_log.jsonl"

    @property
    def errors_path(self) -> Path:
        return self.output_dir / "errors.jsonl"


@dataclass
class RuntimeConfig:
    """
    Stores runtime settings for an experiment run.

    Inputs:
        spend_threshold: Maximum permitted total estimated spend.
        stop_on_error: Whether the loop should stop after a connection/API error.
        dry_run: If True, build prompts and estimate flow without calling APIs.

    Outputs:
        Configuration used by the experiment runner.
    """

    spend_threshold: float
    stop_on_error: bool = False
    dry_run: bool = False