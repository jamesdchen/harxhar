"""ML-specific result aggregation utilities.

Re-exports core evaluation functions (load_all_chunks, process_single_experiment,
filter_by_time) and adds ``parse_config`` for reading ML experiment config.txt
files into structured metadata.
"""

from core.evaluation.aggregation import (
    filter_by_time,
    load_all_chunks,
    process_single_experiment,
)
from projects.ml.evaluation.aggregation import parse_config

__all__ = [
    "filter_by_time",
    "load_all_chunks",
    "parse_config",
    "process_single_experiment",
]
