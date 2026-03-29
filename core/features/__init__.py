"""Feature engineering transforms and lag feature generation pipeline.

Class hierarchy rooted in BaseFeatureTransform (sklearn-style fit/transform):
- HARFeatures — rolling-mean lags at geometric base-5 scales (1,5,25,125,625,3125)
- RawLagFeatures — simple point-shift lags
- PCATransform — sklearn PCA wrapper for dimensionality reduction

Factory: create_feature_transform(kind, ...) builds the right transform.
Pipeline: generate_lag_features() / generate_lag_features_segmented() are the
public entry points; resolve_lags() returns the geometric lag sequence.
"""

__all__ = [
    "BaseFeatureTransform",
    "HARFeatures",
    "LagFeatureBase",
    "PCATransform",
    "RawLagFeatures",
    "create_feature_transform",
    "generate_lag_features",
    "generate_lag_features_segmented",
    "resolve_lags",
]

from core.features.factory import create_feature_transform as create_feature_transform
from core.features.pipeline import (
    generate_lag_features as generate_lag_features,
)
from core.features.pipeline import (
    generate_lag_features_segmented as generate_lag_features_segmented,
)
from core.features.pipeline import (
    resolve_lags as resolve_lags,
)
from core.features.transforms import (
    BaseFeatureTransform as BaseFeatureTransform,
)
from core.features.transforms import (
    HARFeatures as HARFeatures,
)
from core.features.transforms import (
    LagFeatureBase as LagFeatureBase,
)
from core.features.transforms import (
    PCATransform as PCATransform,
)
from core.features.transforms import (
    RawLagFeatures as RawLagFeatures,
)
