"""Backtesting engines (CPU and GPU)."""

__all__ = [
    "apply_duan_smearing",
    "build_results_dataframe",
    "extract_subset",
    "get_chunk_indices_strided",
    "run_backtest_agnostic",
    "save_chunk_results",
    "run_multigpu_backtest",
    "run_ae_multigpu_backtest",
]

from src.backtest.engine import (
    apply_duan_smearing as apply_duan_smearing,
)
from src.backtest.engine import (
    build_results_dataframe as build_results_dataframe,
)
from src.backtest.engine import (
    extract_subset as extract_subset,
)
from src.backtest.engine import (
    get_chunk_indices_strided as get_chunk_indices_strided,
)
from src.backtest.engine import (
    run_backtest_agnostic as run_backtest_agnostic,
)
from src.backtest.engine import (
    save_chunk_results as save_chunk_results,
)

# Lazy imports for torch-dependent GPU engines
_GPU_ATTRS = {
    "run_multigpu_backtest",
    "run_ae_multigpu_backtest",
}


def __getattr__(name: str):
    if name in _GPU_ATTRS:
        from src.backtest import gpu_engine

        return getattr(gpu_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
