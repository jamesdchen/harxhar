"""ML-specific config parsing + re-exports of core aggregation utilities."""

from __future__ import annotations

from pathlib import Path

from core.core.log import get_logger

# Re-export core aggregation functions so existing ML imports continue to work.
from core.evaluation.aggregation import (  # noqa: F401
    filter_by_time,
    load_all_chunks,
    process_single_experiment,
)

logger = get_logger(__name__)


def parse_config(exp_dir: str | Path) -> tuple[int, str, str]:
    """Parses the config.txt file to extract the experiment name, ID, and model type."""
    config_path = Path(exp_dir) / "config.txt"
    exp_name = "Unknown"
    exp_id = -1
    model_type = "Unknown"

    if config_path.exists():
        try:
            with open(config_path) as f:
                for line in f:
                    if line.startswith("Experiment Name:"):
                        exp_name = line.split(":", 1)[1].strip()
                    elif line.startswith("Experiment ID:"):
                        exp_id = int(line.split(":", 1)[1].strip())
                    elif line.startswith("Model Type:"):
                        model_type = line.split(":", 1)[1].strip()
        except (OSError, ValueError) as e:
            logger.warning("Could not parse config %s: %s", config_path, e)

    if exp_id == -1:
        try:
            exp_id = int(str(exp_dir).split("_")[-1])
        except (ValueError, IndexError) as e:
            logger.warning("Could not infer exp_id from path %s: %s", exp_dir, e)

    return exp_id, exp_name, model_type
