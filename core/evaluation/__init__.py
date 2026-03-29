"""Evaluation metrics and result aggregation.

Metrics (calculate_global_metrics): MSE, MAE (adjusted scale); QLIKE (raw
scale); winsorized variants of each.

Aggregation (load_all_chunks → process_single_experiment): stitches per-chunk
CSVs, computes per-horizon metrics, adds cross-horizon aggregates.  Supports
global, pre-segmented, and time-of-day filtering evaluation modes.

Baseline comparison (calculate_baseline_deltas): delta metrics and OOS R².
"""

__all__ = [
    "calculate_baseline_deltas",
    "calculate_global_metrics",
    "filter_by_time",
    "load_all_chunks",
    "process_single_experiment",
]

from core.evaluation.aggregation import (
    filter_by_time as filter_by_time,
)
from core.evaluation.aggregation import (
    load_all_chunks as load_all_chunks,
)
from core.evaluation.aggregation import (
    process_single_experiment as process_single_experiment,
)
from core.evaluation.metrics import (
    calculate_baseline_deltas as calculate_baseline_deltas,
)
from core.evaluation.metrics import (
    calculate_global_metrics as calculate_global_metrics,
)
