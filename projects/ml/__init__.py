"""Traditional ML project — depends on ``core`` only, independent of ``projects.dl``.

Models: Ridge, XGBoost, LightGBM, RandomForest, SARIMAX, NaiveBaseline.
All registered in ``projects.ml.models.MODEL_REGISTRY``; use ``create_model()``
factory.  Feature groups (~50 exogenous features in subgroups) defined in
``projects.ml.features.feature_groups``.

CLI: ``python -m projects.ml.cli.executor --help`` for single-chunk backtests;
``python projects/ml/scripts/submit.py --help`` for HPC batch submission.
Imports are lazy to keep submit-only paths lightweight.
"""

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
