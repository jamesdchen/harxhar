"""Feature engineering transforms and lag feature generation pipeline."""

__all__ = [
    "BaseFeatureTransform",
    "HARFeatures",
    "LagFeatureBase",
    "PCATransform",
    "RawLagFeatures",
    "AETransform",
    "generate_lag_features",
    "generate_lag_features_segmented",
    "resolve_lags",
]

from src.features.pipeline import (
    generate_lag_features as generate_lag_features,
)
from src.features.pipeline import (
    generate_lag_features_segmented as generate_lag_features_segmented,
)
from src.features.pipeline import (
    resolve_lags as resolve_lags,
)
from src.features.transforms import (
    BaseFeatureTransform as BaseFeatureTransform,
)
from src.features.transforms import (
    HARFeatures as HARFeatures,
)
from src.features.transforms import (
    LagFeatureBase as LagFeatureBase,
)
from src.features.transforms import (
    PCATransform as PCATransform,
)
from src.features.transforms import (
    RawLagFeatures as RawLagFeatures,
)


# Lazy import for torch-dependent AETransform
def __getattr__(name: str):
    if name == "AETransform":
        try:
            from src.features.transforms import AETransform
        except ImportError as e:
            raise ImportError("AETransform requires PyTorch. Install it with: pip install torch") from e
        return AETransform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
