"""Result aggregation: chunk stitching, time filtering, and experiment processing.

These utilities are project-agnostic — they work on any directory containing
``results_chunk_*.csv`` files with the standard schema
(date, horizon, true_adj, pred_adj, true_raw, pred_raw).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.core.log import get_logger
from core.evaluation.metrics import calculate_global_metrics

logger = get_logger(__name__)


def load_all_chunks(
    exp_dir: str | Path,
    ignore_suffixes: list[str] | None = None,
    require_suffixes: list[str] | None = None,
) -> pd.DataFrame:
    """Stitches chunk CSVs into a DataFrame with optional suffix filtering.

    Delegates to :func:`hpc.chunking.collect_chunks` for the common case
    (no filtering).  Suffix filtering is domain-specific (market segments)
    and handled here.
    """
    from hpc.chunking import collect_chunks

    # No filtering — delegate entirely to protocol
    if not ignore_suffixes and not require_suffixes:
        return collect_chunks(exp_dir)

    # With suffix filtering — filter files, then stitch manually
    all_files = sorted(Path(exp_dir).glob("results_chunk_*.csv"))
    filtered = []
    for f in all_files:
        base = f.stem
        if ignore_suffixes and any(base.endswith(f"_{s}") for s in ignore_suffixes):
            continue
        if require_suffixes and not any(base.endswith(f"_{s}") for s in require_suffixes):
            continue
        filtered.append(f)

    if not filtered:
        return pd.DataFrame()

    dfs = []
    for f in filtered:
        try:
            dfs.append(pd.read_csv(f))
        except (OSError, pd.errors.ParserError) as e:
            logger.warning("Could not read %s: %s", f.stem, e)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.set_index("date").sort_index()
    else:
        combined = combined.sort_index()

    return combined


def filter_by_time(df: pd.DataFrame, start_time: str | None = None, end_time: str | None = None) -> pd.DataFrame:
    """Slices the DataFrame to the specified time-of-day window."""
    if df.empty or (start_time is None and end_time is None):
        return df

    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        start = start_time if start_time else "00:00:00"
        end = end_time if end_time else "23:59:59"

        # inclusive='left' prevents double-counting the exact overlapping minute (e.g., 11:30)
        return df.between_time(start, end, inclusive="left")
    except (TypeError, ValueError) as e:
        logger.warning("Time filtering failed: %s", e)
        return df


def process_single_experiment(exp_dir: str | Path, metadata: dict[str, Any], segment_configs: list[dict]) -> list[dict]:
    """Agnostically loads data, applies optional time boundaries, and calculates metrics.

    Supports multi-horizon results: if loaded data contains a 'horizon' column,
    metrics are computed per-horizon and a cross-horizon aggregate row is added.
    """
    exp_results = []

    # Cache: when multiple segments share the same load_kwargs (e.g. filter_by_tod
    # mode), load the data once and reuse it instead of re-reading CSVs each time.
    _chunk_cache: dict[tuple, pd.DataFrame] = {}

    for seg_conf in segment_configs:
        seg_name = seg_conf["name"]
        load_kwargs = seg_conf["load_kwargs"]
        time_bounds = seg_conf.get("time_bounds", None)

        logger.info(
            "Processing Exp %s | %s | %s | %s...",
            str(metadata.get("exp_id", "?")).ljust(3),
            metadata.get("model", "?").upper().ljust(8),
            metadata.get("experiment_name", "?")[:16].ljust(16),
            seg_name.ljust(12),
        )

        # 1. Load Data — cache by (require_suffixes, ignore_suffixes) to avoid
        #    redundant disk I/O when only the time filter differs between segments.
        cache_key = (
            tuple(load_kwargs.get("require_suffixes") or []),
            tuple(load_kwargs.get("ignore_suffixes") or []),
        )
        if cache_key not in _chunk_cache:
            _chunk_cache[cache_key] = load_all_chunks(exp_dir, **load_kwargs)
        base_df = _chunk_cache[cache_key]
        df = base_df

        if df.empty:
            logger.info("[EMPTY]")
            continue

        # 2. Apply Time-of-Day Filter in Memory
        if time_bounds:
            df = filter_by_time(df, time_bounds["start"], time_bounds["end"])
            if df.empty:
                logger.info("[EMPTY AFTER TOD FILTER]")
                continue

        # 3. Calculate Metrics — horizon-aware
        if "horizon" in df.columns:
            horizons = sorted(df["horizon"].unique())
            horizon_metrics = []

            for h in horizons:
                df_h = df[df["horizon"] == h]
                m = calculate_global_metrics(df_h)
                m.update(metadata)
                m["segment"] = seg_name
                m["horizon"] = int(h)
                horizon_metrics.append(m)
                exp_results.append(m)

            # Cross-horizon aggregate
            if len(horizon_metrics) > 1:
                agg = dict(metadata)
                agg["segment"] = seg_name
                agg["horizon"] = "mean"
                agg["n_samples"] = sum(m["n_samples"] for m in horizon_metrics)
                for metric_key in ("mse", "mae", "qlike", "w_mse", "w_mae", "w_qlike"):
                    vals = [m[metric_key] for m in horizon_metrics if not np.isnan(m.get(metric_key, np.nan))]
                    agg[metric_key] = np.mean(vals) if vals else np.nan
                exp_results.append(agg)

            logger.info(
                "[OK] %d horizons | n=%d",
                len(horizons),
                sum(m["n_samples"] for m in horizon_metrics),
            )
        else:
            m = calculate_global_metrics(df)
            m.update(metadata)
            m["segment"] = seg_name
            m["horizon"] = 1

            logger.info(
                "[OK] n=%-6d | QLIKE: %.6f | MSE: %.4e | MAE: %.4e",
                m.get("n_samples", 0),
                m.get("qlike", np.nan),
                m.get("mse", np.nan),
                m.get("mae", np.nan),
            )

            exp_results.append(m)

    return exp_results
