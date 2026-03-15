"""Backtesting engines (CPU and GPU)."""

from src.backtest.engine import (
    run_backtest_agnostic,
    get_chunk_indices_strided,
    apply_duan_smearing,
    save_chunk_results,
)
