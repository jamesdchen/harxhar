"""Model implementations and factory."""

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

# Lazy imports for torch-dependent deep learning models
_DEEP_LEARNING_ATTRS = {
    "LagAutoEncoder",
    "PatchTSMixerForecaster",
    "get_ae_model",
    "get_model",
    "train_autoencoder",
}


def __getattr__(name: str):
    if name in _DEEP_LEARNING_ATTRS:
        from src.models import deep_learning

        return getattr(deep_learning, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
