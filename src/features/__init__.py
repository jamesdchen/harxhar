"""Feature engineering transforms."""

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
        from src.features.transforms import AETransform

        return AETransform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
