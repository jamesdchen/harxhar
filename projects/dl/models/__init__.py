"""Deep learning model implementations."""

__all__ = [
    "PatchTSTForecaster",
    "LagAutoEncoder",
    "get_model",
    "get_ae_model",
    "train_autoencoder",
    "functional_qlike_loss",
]


def __getattr__(name):
    _DL_ATTRS = {
        "PatchTSTForecaster",
        "LagAutoEncoder",
        "get_model",
        "get_ae_model",
        "train_autoencoder",
    }
    _LOSS_ATTRS = {"functional_qlike_loss"}

    if name in _DL_ATTRS:
        from projects.dl.models import deep_learning

        return getattr(deep_learning, name)
    if name in _LOSS_ATTRS:
        from projects.dl.models import losses

        return getattr(losses, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
