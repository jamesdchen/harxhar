"""CPU backtesting engine for walk-forward evaluation.

- run_backtest_agnostic — model-agnostic walk-forward loop: burn-in
  initialization, then predict → update for each test step.
- apply_duan_smearing — converts adjusted-space forecasts to raw space:
  pred_raw = (forecast² + smear) × baseline.
- get_chunk_indices_strided — splits test indices into N chunks for HPC.
- save_chunk_results / build_results_dataframe — persist per-chunk CSVs.
"""

__all__ = [
    "apply_duan_smearing",
    "build_results_dataframe",
    "extract_subset",
    "get_chunk_indices_strided",
    "run_backtest_agnostic",
    "save_chunk_results",
]

from core.backtest.engine import apply_duan_smearing as apply_duan_smearing
from core.backtest.engine import build_results_dataframe as build_results_dataframe
from core.backtest.engine import extract_subset as extract_subset
from core.backtest.engine import get_chunk_indices_strided as get_chunk_indices_strided
from core.backtest.engine import run_backtest_agnostic as run_backtest_agnostic
from core.backtest.engine import save_chunk_results as save_chunk_results
