"""GPU backtesting engines."""

__all__ = [
    "run_multigpu_backtest",
    "run_ae_multigpu_backtest",
]


def __getattr__(name):
    if name in __all__:
        from projects.dl.backtest import gpu_engine

        return getattr(gpu_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
