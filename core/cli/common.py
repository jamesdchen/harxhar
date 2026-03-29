"""Shared CLI argument helpers for backtest executors."""

from __future__ import annotations

import argparse


def add_chunking_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard chunking and backtest args shared by ML and DL executors.

    Adds: ``--input-path``, ``--chunk-id``, ``--total-chunks``, ``--horizon``,
    ``--train-window``.
    """
    parser.add_argument("--input-path", type=str, default="all30min", help="Directory containing .parquet data files.")
    parser.add_argument("--chunk-id", type=int, default=None, help="Zero-based chunk index for this worker.")
    parser.add_argument("--total-chunks", type=int, default=None, help="Total number of chunks.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon.")
    parser.add_argument("--train-window", type=int, default=None, help="Training window size (periods).")
