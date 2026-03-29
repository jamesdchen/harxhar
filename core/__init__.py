"""Core package for harxhar volatility forecasting framework."""

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
