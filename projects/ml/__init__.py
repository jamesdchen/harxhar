"""Traditional ML models for harxhar volatility forecasting."""

__all__ = [
    "create_model",
    "RidgeModel",
    "XGBoostModel",
    "LightGBMModel",
    "RandomForestModel",
    "SARIMAXModel",
]


def __getattr__(name: str):
    if name == "create_model":
        from projects.ml.models.registry import create_model

        return create_model
    if name == "SARIMAXModel":
        from projects.ml.models.sarimax import SARIMAXModel

        return SARIMAXModel
    _sklearn = {
        "RidgeModel",
        "XGBoostModel",
        "LightGBMModel",
        "RandomForestModel",
    }
    if name in _sklearn:
        from projects.ml.models import sklearn_models

        return getattr(sklearn_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
