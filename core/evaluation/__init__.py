"""Evaluation metrics and result aggregation.

Metrics (calculate_global_metrics): MSE, MAE (adjusted scale); QLIKE (raw
scale); winsorized variants of each.

Aggregation (load_all_chunks → process_single_experiment): stitches per-chunk
CSVs, computes per-horizon metrics, adds cross-horizon aggregates.  Supports
global, pre-segmented, and time-of-day filtering evaluation modes.

Segment helpers (build_segment_configs, TARGET_SEGMENTS, TOD_BOUNDS): shared
constants and config builders for time-of-day and pre-segmented evaluation.

Summary formatting (format_summary, print_and_save_summary, SUMMARY_COLUMNS,
SUMMARY_FORMATTERS): tabular display and CSV persistence of experiment results.

Metadata loading (load_experiment_metadata): reads config.txt / metadata.json
from experiment directories.

Baseline comparison (calculate_baseline_deltas): delta metrics and OOS R².
"""

__all__ = [
    "SUMMARY_COLUMNS",
    "SUMMARY_FORMATTERS",
    "TARGET_SEGMENTS",
    "TOD_BOUNDS",
    "build_segment_configs",
    "calculate_baseline_deltas",
    "calculate_global_metrics",
    "filter_by_time",
    "format_summary",
    "load_all_chunks",
    "load_experiment_metadata",
    "print_and_save_summary",
    "process_single_experiment",
]

from core.evaluation.aggregation import (
    SUMMARY_COLUMNS as SUMMARY_COLUMNS,
)
from core.evaluation.aggregation import (
    SUMMARY_FORMATTERS as SUMMARY_FORMATTERS,
)
from core.evaluation.aggregation import (
    TARGET_SEGMENTS as TARGET_SEGMENTS,
)
from core.evaluation.aggregation import (
    TOD_BOUNDS as TOD_BOUNDS,
)
from core.evaluation.aggregation import (
    build_segment_configs as build_segment_configs,
)
from core.evaluation.aggregation import (
    filter_by_time as filter_by_time,
)
from core.evaluation.aggregation import (
    format_summary as format_summary,
)
from core.evaluation.aggregation import (
    load_all_chunks as load_all_chunks,
)
from core.evaluation.aggregation import (
    load_experiment_metadata as load_experiment_metadata,
)
from core.evaluation.aggregation import (
    print_and_save_summary as print_and_save_summary,
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
