"""Base model classes for walk-forward volatility forecasting.

All models implement the BaseModel ABC: initialize(X_init, y_init) →
predict(x_t) → update(x_t, y_t).

- RollingRegressionModel — wraps any sklearn estimator with RollingBuffer,
  RollingRobustScaler, and optional feature_transform (PCA/AE).  Configurable
  refit_frequency (every step for Ridge, every 5 for trees).
- NaiveBaseline — returns a specific lag value as the forecast.
"""

__all__ = [
    "BaseModel",
    "NaiveBaseline",
    "RollingRegressionModel",
]

from core.models.base import BaseModel as BaseModel
from core.models.base import NaiveBaseline as NaiveBaseline
from core.models.base import RollingRegressionModel as RollingRegressionModel
