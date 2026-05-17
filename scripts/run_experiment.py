from pathlib import Path

from image_schema_llm.config import ProjectPaths, RuntimeConfig
from image_schema_llm.runner import ExperimentRunner


def main() -> None:
    """
    Command-line entry point for running the experiment.

    Purpose:
        Creates project paths and runtime config, then starts the experiment.

    Notes:
        Later versions can replace hard-coded values with argparse arguments:
            --project-root
            --spend-threshold
            --dry-run
            --stop-on-error
    """
    project_root = Path(__file__).resolve().parents[1]

    paths = ProjectPaths(project_root=project_root)

    config = RuntimeConfig(
        spend_threshold=10.00,
        stop_on_error=False,
        dry_run=False,
    )

    runner = ExperimentRunner(paths=paths, config=config)
    runner.run()


if __name__ == "__main__":
    main()
