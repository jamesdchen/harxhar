"""Evaluation metrics and result aggregation."""

from src.evaluation.metrics import calculate_global_metrics, calculate_baseline_deltas
from src.evaluation.aggregation import (
    load_all_chunks,
    parse_config,
    filter_by_time,
    process_single_experiment,
)
