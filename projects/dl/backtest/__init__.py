"""GPU backtesting engines for deep learning models.

- run_multigpu_backtest — PatchTST: 3D strided windows via torch.as_strided,
  per-GPU instance normalization → compiled training kernel → predict.
- run_ae_multigpu_backtest — AE+Ridge: 2D strided windows, train AE → encode
  → closed-form Ridge solve (X'X + αI)⁻¹X'y → predict.

Both distribute chunks across GPUs via torch.multiprocessing.Pool.
"""

__all__ = [
    "run_multigpu_backtest",
    "run_ae_multigpu_backtest",
]


def __getattr__(name):
    if name in __all__:
        from projects.dl.backtest import gpu_engine

        return getattr(gpu_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
