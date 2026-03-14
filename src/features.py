import os
import numpy as np
import torch
from sklearn.decomposition import PCA
from src.dl_models import LagAutoEncoder, train_autoencoder


# --- Base Class ---
class BaseFeatureTransform:
    """
    Base class for all feature transforms.

    Two interfaces are supported:

    1. **sklearn-style** (`fit` / `transform`): operates on numpy arrays.
       Used by PCATransform, AETransform, and plugged into
       RollingRegressionModel for online dimensionality reduction.

    2. **pandas-level** (`generate_pandas`): operates on a DataFrame and
       column list, returning ``(feature_dict, feature_names)`` for
       concatenation in `data.py`.  Only implemented by LagFeatureBase
       subclasses (HARFeatures, RawLagFeatures).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


# --- Lag Feature Base (shared iteration logic) ---
class LagFeatureBase(BaseFeatureTransform):
    """Shared iteration logic for lag-based feature generators."""

    def __init__(self, lags, target_col='adj_RV'):
        self.lags = lags
        self.target_col = target_col

    def _compute_lag(self, col_series, lag):
        """Override: apply per-column, per-lag transform on a pandas Series."""
        raise NotImplementedError

    def _compute_lag_np(self, col_array, lag):
        """Override: apply per-column, per-lag transform on a numpy array."""
        raise NotImplementedError

    def _feature_name(self, col, lag):
        """Override: return the feature name string for this col/lag pair."""
        raise NotImplementedError

    def generate_pandas(self, df, cols):
        """Pandas-level feature builder: iterate cols x lags, return (feature_dict, feature_names)."""
        feature_dict = {}
        feature_names = []
        for col in cols:
            for lag in self.lags:
                name = self._feature_name(col, lag)
                feature_dict[name] = self._compute_lag(df[col], lag)
                feature_names.append(name)
        return feature_dict, feature_names

    def transform(self, X):
        """Numpy-level transform: iterate columns x lags, stack results."""
        n_samples, n_cols = X.shape
        result_cols = []
        for col_idx in range(n_cols):
            for lag in self.lags:
                result_cols.append(self._compute_lag_np(X[:, col_idx], lag))
        return np.column_stack(result_cols)


# --- HAR Features (rolling mean aggregates) ---
class HARFeatures(LagFeatureBase):
    """Rolling-mean HAR lag features."""

    def _compute_lag(self, col_series, lag):
        return col_series.rolling(window=lag, min_periods=1).mean().shift(1)

    def _compute_lag_np(self, col_array, lag):
        n = len(col_array)
        cumsum = np.cumsum(col_array)
        rolling_mean = np.empty(n)
        for i in range(n):
            window_start = max(0, i - lag + 1)
            rolling_mean[i] = cumsum[i]
            if window_start > 0:
                rolling_mean[i] -= cumsum[window_start - 1]
            rolling_mean[i] /= (i - window_start + 1)
        # shift by 1
        result = np.empty(n)
        result[0] = np.nan
        result[1:] = rolling_mean[:-1]
        return result

    def _feature_name(self, col, lag):
        if col == self.target_col:
            return f"har_ma_{lag}"
        return f"{col}_ma_{lag}"


# --- Raw Lag Features (individual shifted lags) ---
class RawLagFeatures(LagFeatureBase):
    """Individual point-shift lag features."""

    def _compute_lag(self, col_series, lag):
        return col_series.shift(lag)

    def _compute_lag_np(self, col_array, lag):
        result = np.empty_like(col_array, dtype=float)
        result[:lag] = np.nan
        result[lag:] = col_array[:-lag] if lag > 0 else col_array
        return result

    def _feature_name(self, col, lag):
        return f"{col}_lag_{lag}"


# --- PCA Transform ---
class PCATransform(BaseFeatureTransform):
    """Sklearn-like PCA wrapper for use as a rolling feature transform."""

    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)


# --- Autoencoder Transform ---
class AETransform(BaseFeatureTransform):
    """Autoencoder feature transform with reconstruction + prediction loss."""

    def __init__(self, n_features, n_components, alpha=0.5, hidden_dim=None,
                 epochs=50, lr=1e-3, ae_loss_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ae = LagAutoEncoder(n_features, n_components, hidden_dim).to(self.device)
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.ae_loss_path = ae_loss_path
        self._loss_log = []

    def fit(self, X, y=None):
        y_flat = y.ravel() if y is not None else np.zeros(X.shape[0])
        train_autoencoder(
            self.ae, X, y_flat,
            alpha=self.alpha, epochs=self.epochs, lr=self.lr,
            device=self.device, loss_log=self._loss_log,
        )
        if self.ae_loss_path is not None:
            self._flush_loss_log()
        return self

    def transform(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        return self.ae.encode(X_t).cpu().numpy()

    def _flush_loss_log(self):
        import csv

        if not self._loss_log:
            return
        write_header = not os.path.exists(self.ae_loss_path)
        with open(self.ae_loss_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["recon", "pred", "total"])
            if write_header:
                writer.writeheader()
            writer.writerows(self._loss_log)
        self._loss_log.clear()
