"""Top-level package for harxhar volatility forecasting framework."""

__all__ = [
    "load_and_prep_data_strided",
    "BaseModel",
    "create_model",
    "run_backtest_agnostic",
]


def __getattr__(name):
    if name == "load_and_prep_data_strided":
        from src.data import load_and_prep_data_strided

        return load_and_prep_data_strided
    if name == "BaseModel":
        from src.models import BaseModel

        return BaseModel
    if name == "create_model":
        from src.models import create_model

        return create_model
    if name == "run_backtest_agnostic":
        from src.backtest import run_backtest_agnostic

        return run_backtest_agnostic
    raise AttributeError(f"module 'src' has no attribute {name!r}")
