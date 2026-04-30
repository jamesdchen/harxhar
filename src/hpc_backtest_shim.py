"""HPC chunking shim — translates chunk_id/total_chunks to --start/--end.

claude-hpc fans out tasks using (chunk_id, total_chunks). Backtest executors
in this repo expect absolute (--start, --end) indices. This shim bridges
the two interfaces:

    1. Determines total post-transform rows (cached after first call)
    2. Splits [0, total_rows) evenly across total_chunks
    3. Forwards --start/--end to the downstream executor

Usage (called by _hpc_dispatch.py via hpc.yaml):

    python3 src/hpc_backtest_shim.py --chunk-id 3 --total-chunks 100 \\
        -- python3 src/ml_xgboost.py --output-file results.csv

Inspect / modify:
    - get_total_rows()       — change if the data pipeline changes
    - range_split_overlap()  — change if you need non-uniform chunks
    - _CACHE_FILE            — set to None to disable caching
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_CACHE_FILE = "_backtest_total_rows.json"


# ── Data length probe ────────────────────────────────────────────────────────


def get_total_rows(data_path: str = "all30min", horizon: int = 1) -> int:
    """Compute post-transform row count for the harxhar backtest pipeline.

    This runs the same transforms as the executors (load → robust_transform →
    HAR features → calendar features → lag trim → horizon shift) and returns
    len(X) — the number of rows an executor would iterate over.
    """
    import numpy as np

    from src.loading import load_raw_data
    from src.transforms import (
        add_calendar_features,
        apply_horizon_shift,
        generate_har_features,
        resolve_har_lags,
        robust_transform,
    )

    df = load_raw_data(data_path, allow_missing=True)
    # Match executor.load_and_transform: drop RV NaN. The diurnal-baseline
    # zero case is now handled inside diurnal_adjust (smallest-nonzero
    # fallback), so post-transform rows stay finite for the new vintage.
    df = df.dropna(subset=["RV"]).reset_index(drop=True)
    adj_rv, baseline = robust_transform(
        df,
        "RV",
        is_target=True,
        use_diurnal=True,
        winsor_window=240,
    )
    df["adj_RV"] = adj_rv

    df, har_names = generate_har_features(df, target_col="adj_RV")
    cal_names = add_calendar_features(df)

    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)

    X = df[har_names + cal_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64) if "baseline" in df else np.zeros(len(df))

    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, horizon)
    return len(X)


def _cached_total_rows(data_path: str = "all30min", horizon: int = 1) -> int:
    """Return total_rows, reading from cache or computing and caching."""
    if _CACHE_FILE and os.path.isfile(_CACHE_FILE):
        with open(_CACHE_FILE) as f:
            cache = json.load(f)
        key = f"{data_path}__h{horizon}"
        if key in cache:
            return cache[key]

    total = get_total_rows(data_path, horizon)

    if _CACHE_FILE:
        cache = {}
        if os.path.isfile(_CACHE_FILE):
            with open(_CACHE_FILE) as f:
                cache = json.load(f)
        key = f"{data_path}__h{horizon}"
        cache[key] = total
        tmp = _CACHE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, _CACHE_FILE)

    return total


# ── Range arithmetic ─────────────────────────────────────────────────────────


def range_split_overlap(
    total_rows: int,
    total_chunks: int,
    chunk_id: int,
    overlap: int,
) -> tuple[int, int]:
    """Split the OOS range [overlap:total_rows) into chunks, prepend overlap.

    For walk-forward backtests: each chunk gets ``overlap`` rows of training
    prefix plus its share of the OOS predictions.  The executor trains on
    the first ``overlap`` rows and predicts on the rest.

    Returns (start, end) — a half-open range suitable for array slicing.
    """
    oos_rows = total_rows - overlap
    base = oos_rows // total_chunks
    remainder = oos_rows % total_chunks
    oos_start = overlap + base * chunk_id + min(chunk_id, remainder)
    oos_end = oos_start + base + (1 if chunk_id < remainder else 0)
    # Prepend training window so the executor can train
    start = oos_start - overlap
    end = oos_end
    return start, end


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate chunk_id/total_chunks to --start/--end",
    )
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--total-chunks", type=int, required=True)
    parser.add_argument(
        "--overlap",
        type=int,
        required=True,
        help="Training window rows to prepend to each chunk (for walk-forward backtests). "
        "Must match the executor's train_window × PERIODS_PER_DAY (e.g. 500×48=24000).",
    )
    parser.add_argument("--data-path", default="all30min")
    parser.add_argument("--horizon", type=int, default=1)
    args, downstream = parser.parse_known_args()

    # Strip leading "--" separator if present
    if downstream and downstream[0] == "--":
        downstream = downstream[1:]

    if not downstream:
        parser.error("no downstream command provided after --")

    if args.overlap <= 0:
        parser.error("--overlap must be positive")

    total_rows = _cached_total_rows(args.data_path, args.horizon)
    start, end = range_split_overlap(
        total_rows,
        args.total_chunks,
        args.chunk_id,
        args.overlap,
    )

    cmd = downstream + ["--start", str(start), "--end", str(end)]
    print(
        f"[shim] chunk {args.chunk_id}/{args.total_chunks} -> "
        f"rows [{start}:{end}) of {total_rows} (overlap={args.overlap})"
    )
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
