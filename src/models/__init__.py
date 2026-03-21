"""Model implementations and factory."""

__all__ = [
    "BaseModel",
    "NaiveBaseline",
    "RollingRegressionModel",
    "MODEL_REGISTRY",
    "create_model",
    "SARIMAXModel",
    "LightGBMModel",
    "RandomForestModel",
    "RidgeModel",
    "XGBoostModel",
    "LagAutoEncoder",
    "PatchTSMixerForecaster",
    "get_ae_model",
    "get_model",
    "train_autoencoder",
    "functional_qlike_loss",
]

from src.models.base import (
    BaseModel as BaseModel,
)
from src.models.base import (
    NaiveBaseline as NaiveBaseline,
)
from src.models.base import (
    RollingRegressionModel as RollingRegressionModel,
)
from src.models.registry import (
    MODEL_REGISTRY as MODEL_REGISTRY,
)
from src.models.registry import (
    create_model as create_model,
)
from src.models.sarimax import SARIMAXModel as SARIMAXModel
from src.models.sklearn_models import (
    LightGBMModel as LightGBMModel,
)
from src.models.sklearn_models import (
    RandomForestModel as RandomForestModel,
)
from src.models.sklearn_models import (
    RidgeModel as RidgeModel,
)
from src.models.sklearn_models import (
    XGBoostModel as XGBoostModel,
)

# Lazy imports for torch-dependent modules
_DEEP_LEARNING_ATTRS = {
    "LagAutoEncoder",
    "PatchTSMixerForecaster",
    "get_ae_model",
    "get_model",
    "train_autoencoder",
}

_LOSSES_ATTRS = {
    "functional_qlike_loss",
}


def __getattr__(name: str):
    if name in _DEEP_LEARNING_ATTRS:
        try:
            from src.models import deep_learning
        except ImportError as e:
            raise ImportError(
                f"'{name}' requires PyTorch and transformers. Install them with: pip install torch transformers"
            ) from e
        return getattr(deep_learning, name)
    if name in _LOSSES_ATTRS:
        try:
            from src.models import losses
        except ImportError as e:
            raise ImportError(f"'{name}' requires PyTorch. Install it with: pip install torch") from e
        return getattr(losses, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
