"""Feature transform classes: HAR lags, raw lags, and PCA transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# --- Base Class ---
class BaseFeatureTransform:
    """
    Base class for all feature transforms.

    Two interfaces are supported:

    1. **sklearn-style** (`fit` / `transform`): operates on numpy arrays.
       Used by PCATransform and plugged into
       RollingRegressionModel for online dimensionality reduction.

    2. **pandas-level** (`generate_pandas`): operates on a DataFrame and
       column list, returning ``(feature_dict, feature_names)`` for
       concatenation in `data.py`.  Only implemented by LagFeatureBase
       subclasses (HARFeatures, RawLagFeatures).
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BaseFeatureTransform:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


# --- Lag Feature Base (shared iteration logic) ---
class LagFeatureBase(BaseFeatureTransform):
    """Shared iteration logic for lag-based feature generators."""

    def __init__(self, lags: list[int], target_col: str = "adj_RV") -> None:
        self.lags = lags
        self.target_col = target_col

    def _compute_lag(self, col_series: pd.Series, lag: int) -> pd.Series:
        """Override: apply per-column, per-lag transform on a pandas Series."""
        raise NotImplementedError

    def _compute_lag_np(self, col_array: np.ndarray, lag: int) -> np.ndarray:
        """Override: apply per-column, per-lag transform on a numpy array."""
        raise NotImplementedError

    def _feature_name(self, col: str, lag: int) -> str:
        """Override: return the feature name string for this col/lag pair."""
        raise NotImplementedError

    def generate_pandas(self, df: pd.DataFrame, cols: list[str]) -> tuple[dict[str, pd.Series], list[str]]:
        """Pandas-level feature builder: iterate cols x lags, return (feature_dict, feature_names)."""
        feature_dict: dict[str, pd.Series] = {}
        feature_names: list[str] = []
        for col in cols:
            for lag in self.lags:
                name = self._feature_name(col, lag)
                feature_dict[name] = self._compute_lag(df[col], lag)
                feature_names.append(name)
        return feature_dict, feature_names

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Numpy-level transform: iterate columns x lags, stack results."""
        n_samples, n_cols = X.shape
        result_cols: list[np.ndarray] = []
        for col_idx in range(n_cols):
            for lag in self.lags:
                result_cols.append(self._compute_lag_np(X[:, col_idx], lag))
        return np.column_stack(result_cols)


# --- HAR Features (rolling mean aggregates) ---
class HARFeatures(LagFeatureBase):
    """Rolling-mean HAR lag features."""

    def _compute_lag(self, col_series: pd.Series, lag: int) -> pd.Series:
        return col_series.rolling(window=lag, min_periods=1).mean().shift(1)

    def _compute_lag_np(self, col_array: np.ndarray, lag: int) -> np.ndarray:
        n = len(col_array)
        cumsum = np.cumsum(col_array, dtype=np.float64)
        rolling_mean = np.empty(n)
        for i in range(n):
            window_start = max(0, i - lag + 1)
            rolling_mean[i] = cumsum[i]
            if window_start > 0:
                rolling_mean[i] -= cumsum[window_start - 1]
            rolling_mean[i] /= i - window_start + 1
        # shift by 1
        result = np.empty(n)
        result[0] = np.nan
        result[1:] = rolling_mean[:-1]
        return result

    def _feature_name(self, col: str, lag: int) -> str:
        if col == self.target_col:
            return f"har_ma_{lag}"
        return f"{col}_ma_{lag}"


# --- Raw Lag Features (individual shifted lags) ---
class RawLagFeatures(LagFeatureBase):
    """Individual point-shift lag features."""

    def _compute_lag(self, col_series: pd.Series, lag: int) -> pd.Series:
        return col_series.shift(lag)

    def _compute_lag_np(self, col_array: np.ndarray, lag: int) -> np.ndarray:
        result = np.empty_like(col_array, dtype=float)
        result[:lag] = np.nan
        result[lag:] = col_array[:-lag] if lag > 0 else col_array
        return result

    def _feature_name(self, col: str, lag: int) -> str:
        return f"{col}_lag_{lag}"


# --- PCA Transform ---
class PCATransform(BaseFeatureTransform):
    """Sklearn-like PCA wrapper for use as a rolling feature transform."""

    def __init__(self, n_components: int) -> None:
        self.pca = PCA(n_components=n_components)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> PCATransform:
        self.pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)
