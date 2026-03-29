"""Core package for harxhar volatility forecasting framework.

Shared foundation used by both ``projects.ml`` and ``projects.dl`` (which are
independent of each other).  Re-exports the three most common entry points:

- ``load_and_prep_data_strided`` — full data pipeline (load → grid → transform → lags → horizon shift)
- ``BaseModel`` — walk-forward model ABC (initialize → predict → update)
- ``run_backtest_agnostic`` — CPU walk-forward backtester

Subpackages: data, features, models, backtest, evaluation, backends.
Each subpackage's ``__init__.py`` lists its public API via ``__all__``.
"""

__all__ = [
    "load_and_prep_data_strided",
    "BaseModel",
    "run_backtest_agnostic",
]


def __getattr__(name: str) -> object:
    if name == "load_and_prep_data_strided":
        from core.data import load_and_prep_data_strided

        return load_and_prep_data_strided
    if name == "BaseModel":
        from core.models import BaseModel

        return BaseModel
    if name == "run_backtest_agnostic":
        from core.backtest import run_backtest_agnostic

        return run_backtest_agnostic
    raise AttributeError(f"module 'core' has no attribute {name!r}")
