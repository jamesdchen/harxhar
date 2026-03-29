"""Deep learning project — depends on ``core`` only, independent of ``projects.ml``.

Models: PatchTST (transformer-based patch time series) and LagAutoEncoder
(hybrid supervised/unsupervised, encoder feeds into Ridge regression).
GPU backtest engines use torch.func.vmap for vectorized batch training.

CLI: ``python -m projects.dl.cli.gpu_executor --help`` for single runs;
``python -m projects.dl.cli.lifecycle --help`` for submit + status tracking.
Imports are lazy to avoid pulling torch at import time.
"""

__all__ = [
    "PatchTSTForecaster",
    "LagAutoEncoder",
    "get_model",
    "get_ae_model",
    "train_autoencoder",
    "functional_qlike_loss",
    "run_multigpu_backtest",
    "run_ae_multigpu_backtest",
]


def __getattr__(name: str) -> object:
    _MODEL_ATTRS = {
        "PatchTSTForecaster",
        "LagAutoEncoder",
        "get_model",
        "get_ae_model",
        "train_autoencoder",
    }
    _LOSS_ATTRS = {"functional_qlike_loss"}
    _BACKTEST_ATTRS = {"run_multigpu_backtest", "run_ae_multigpu_backtest"}

    if name in _MODEL_ATTRS:
        from projects.dl.models import deep_learning

        return getattr(deep_learning, name)
    if name in _LOSS_ATTRS:
        from projects.dl.models import losses

        return getattr(losses, name)
    if name in _BACKTEST_ATTRS:
        from projects.dl.backtest import gpu_engine

        return getattr(gpu_engine, name)
    raise AttributeError(f"module 'projects.dl' has no attribute {name!r}")
