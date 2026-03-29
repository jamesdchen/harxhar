"""Result aggregation: chunk stitching, time filtering, and experiment processing.

These utilities are project-agnostic — they work on any directory containing
``results_chunk_*.csv`` files with the standard schema
(date, horizon, true_adj, pred_adj, true_raw, pred_raw).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.core.log import get_logger
from core.evaluation.metrics import calculate_global_metrics

logger = get_logger(__name__)

# ── Segment constants (shared by ML & DL aggregation scripts) ──────────────

TARGET_SEGMENTS = ["morning", "midday", "closing", "overnight"]

TOD_BOUNDS: dict[str, dict[str, str]] = {
    "morning": {"start": "09:30", "end": "11:30"},
    "midday": {"start": "11:30", "end": "14:00"},
    "closing": {"start": "14:00", "end": "16:00"},
    "overnight": {"start": "16:00", "end": "09:30"},
}


def build_segment_configs(eval_mode: str) -> tuple[str, str, list[dict]]:
    """Build (title, output_filename, segment_configs) for a given evaluation mode.

    Supported modes: ``"global"``, ``"segments"``, ``"filter_by_tod"``.
    """
    if eval_mode == "segments":
        title = f"PRE-SEGMENTED FILES SUMMARY: {TARGET_SEGMENTS}"
        out_file = "segment_results_summary.csv"
        configs = [
            {"name": seg.upper(), "load_kwargs": {"require_suffixes": [seg], "ignore_suffixes": None}}
            for seg in TARGET_SEGMENTS
        ]
    elif eval_mode == "filter_by_tod":
        title = "GLOBAL DATA (Filtered into TOD Segments in Memory)"
        out_file = "global_results_tod_filtered.csv"
        configs = [
            {
                "name": f"GLOBAL_{seg.upper()}",
                "load_kwargs": {"require_suffixes": None, "ignore_suffixes": TARGET_SEGMENTS},
                "time_bounds": bounds,
            }
            for seg, bounds in TOD_BOUNDS.items()
        ]
    else:
        title = "GLOBAL SUMMARY (All Hours)"
        out_file = "global_results_summary.csv"
        configs = [{"name": "GLOBAL", "load_kwargs": {"require_suffixes": None, "ignore_suffixes": TARGET_SEGMENTS}}]
    return title, out_file, configs


# ── Summary formatting (shared by ML & DL aggregation scripts) ─────────────

SUMMARY_COLUMNS = [
    "exp_id",
    "model",
    "experiment_name",
    "segment",
    "horizon",
    "mse",
    "delta_mse",
    "oos_r2",
    "mae",
    "delta_mae",
    "qlike",
    "delta_qlike",
    "n_samples",
]

SUMMARY_FORMATTERS = {
    "mse": "{:.4e}".format,
    "delta_mse": "{:.4e}".format,
    "mae": "{:.4e}".format,
    "delta_mae": "{:.4e}".format,
    "qlike": "{:.6f}".format,
    "delta_qlike": "{:.6f}".format,
    "oos_r2": "{:.4%}".format,
}


def format_summary(summary_df: pd.DataFrame) -> str:
    """Format a summary DataFrame for console output using standard columns and formatters."""
    final_cols = [c for c in SUMMARY_COLUMNS if c in summary_df.columns]
    active_formatters = {k: v for k, v in SUMMARY_FORMATTERS.items() if k in final_cols}
    pd.set_option("display.width", 1000)
    return summary_df[final_cols].to_string(index=False, formatters=active_formatters)


def print_and_save_summary(
    summary_df: pd.DataFrame,
    title: str,
    output_path: str | Path,
) -> None:
    """Print formatted summary to stdout and save CSV to *output_path*."""
    print(f"\n{'=' * 175}")
    print(title)
    print("=" * 165)
    print(format_summary(summary_df))

    output_path = Path(output_path)
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved summary to: {output_path}")


# ── Unified experiment metadata loading ────────────────────────────────────


def _parse_config_txt(config_path: Path) -> dict[str, Any]:
    """Parse a config.txt file (ML-style) into a metadata dict."""
    meta: dict[str, Any] = {}
    key_map = {
        "Experiment Name:": "experiment_name",
        "Experiment ID:": "exp_id",
        "Model Type:": "model",
    }
    with open(config_path) as f:
        for line in f:
            for prefix, key in key_map.items():
                if line.startswith(prefix):
                    val = line.split(":", 1)[1].strip()
                    meta[key] = int(val) if key == "exp_id" else val
    return meta


def _infer_metadata_from_dirname(dir_name: str) -> dict[str, Any]:
    """Infer experiment metadata from a directory name (DL-style fallback)."""
    parts = dir_name.split("_")
    model = parts[1] if len(parts) >= 2 else dir_name
    return {"experiment_name": dir_name, "model": model}


def load_experiment_metadata(
    exp_dir: str | Path,
    fallback_exp_id: int = -1,
) -> dict[str, Any]:
    """Load experiment metadata with fallback chain.

    Priority: ``metadata.json`` → ``config.txt`` → directory name inference.
    Always returns a dict with keys ``exp_id``, ``experiment_name``, ``model``.
    """
    exp_dir = Path(exp_dir)

    # 1. Try metadata.json (written by core.cli.metadata.save_metadata)
    meta_json = exp_dir / "metadata.json"
    if meta_json.exists():
        try:
            with open(meta_json) as f:
                raw = json.load(f)
            # Extract the standard keys from experiment_config if present
            cfg = raw.get("experiment_config", {})
            return {
                "exp_id": cfg.get("exp_id", raw.get("exp_id", fallback_exp_id)),
                "experiment_name": cfg.get("experiment_name", raw.get("experiment_name", exp_dir.name)),
                "model": cfg.get("model", raw.get("model", "unknown")),
            }
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read %s: %s", meta_json, e)

    # 2. Try config.txt (ML-style)
    config_txt = exp_dir / "config.txt"
    if config_txt.exists():
        try:
            meta = _parse_config_txt(config_txt)
            meta.setdefault("exp_id", fallback_exp_id)
            meta.setdefault("experiment_name", "Unknown")
            meta.setdefault("model", "Unknown")
            return meta
        except (OSError, ValueError) as e:
            logger.warning("Could not parse %s: %s", config_txt, e)

    # 3. Fallback: infer from directory name
    meta = _infer_metadata_from_dirname(exp_dir.name)
    if fallback_exp_id == -1:
        # Try to extract numeric ID from dir name (e.g. "exp_3_ridge_har" → 3)
        try:
            for part in exp_dir.name.split("_"):
                if part.isdigit():
                    fallback_exp_id = int(part)
                    break
        except (ValueError, IndexError):
            pass
    meta["exp_id"] = fallback_exp_id
    return meta


def load_all_chunks(
    exp_dir: str | Path,
    ignore_suffixes: list[str] | None = None,
    require_suffixes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Stitches chunk CSVs into a DataFrame with flexible filtering.

    Returns
    -------
    pd.DataFrame
    """
    all_files = sorted(Path(exp_dir).glob("results_chunk_*.csv"))

    if not all_files:
        return pd.DataFrame()

    dfs = []
    for filename in all_files:
        base_name = filename.stem

        # 1. Check if we should ignore this file
        if ignore_suffixes and any(base_name.endswith(f"_{seg}") for seg in ignore_suffixes):
            continue

        # 2. Check if we strictly require a specific suffix
        if require_suffixes and not any(base_name.endswith(f"_{seg}") for seg in require_suffixes):
            continue

        try:
            dfs.append(pd.read_csv(filename))
        except (OSError, pd.errors.ParserError) as e:
            logger.warning("Could not read %s: %s", base_name, e)

    if not dfs:
        return pd.DataFrame()

    # Concat first, then parse dates once on the combined frame
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
