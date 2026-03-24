"""Base model classes: BaseModel, RollingRegressionModel, NaiveBaseline."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.core import config as cfg
from core.data.rolling import RollingBuffer, RollingRobustScaler


class BaseModel(ABC):
    """Abstract base for all forecasting models.

    Subclasses must implement ``initialize``, ``predict``, and ``update``
    to participate in the walk-forward backtest loop.
    """

    @abstractmethod
    def initialize(self, X_init: np.ndarray, y_init: np.ndarray) -> None:
        """Receive initial burn-in history and fit the model."""

    @abstractmethod
    def predict(self, x_t: np.ndarray) -> float:
        """Return a scalar forecast given feature vector *x_t*."""

    @abstractmethod
    def update(self, x_t: np.ndarray, y_t: float) -> None:
        """Ingest newly observed data point and optionally refit."""

    def get_coefs(self) -> np.ndarray | None:
        """Return model coefficients, or ``None`` if not applicable."""
        return None


class RollingRegressionModel(BaseModel):
    def __init__(
        self, model, train_win_periods, n_features, use_scaling=True, refit_frequency=1, feature_transform=None
    ):
        cfg.check_positive(train_win_periods, "train_win_periods")
        cfg.check_positive(n_features, "n_features")
        self.model = model
        self.train_win_periods = train_win_periods
        self.n_features = n_features
        self.use_scaling = use_scaling
        self.refit_frequency = refit_frequency
        self.feature_transform = feature_transform
        self.steps_since_refit = 0

        self.buffer = RollingBuffer(train_win_periods, n_features, 1)

        if self.use_scaling:
            self.scaler = RollingRobustScaler(train_win_periods, n_features)

        self.mean_x = np.zeros(n_features)
        self.std_x = np.ones(n_features)

    def _fit_model(self, X, y):
        """Fit feature transform (if any) and model on the buffer data."""
        if self.feature_transform is not None:
            self.feature_transform.fit(X, y)
            X = self.feature_transform.transform(X)
        self.model.fit(X, y)

    def _transform_input(self, X):
        """Apply feature transform (if any) to input data."""
        if self.feature_transform is not None:
            return self.feature_transform.transform(X)
        return X

    def initialize(self, X_init, y_init):
        if y_init.ndim == 1:
            y_init = y_init.reshape(-1, 1)

        if self.use_scaling:
            self.scaler.initialize(X_init)
            self.mean_x, self.std_x = self.scaler.get_scaler()
            X_buffered = (X_init - self.mean_x) / self.std_x
        else:
            X_buffered = X_init

        self.buffer.X_buffer[:] = X_buffered
        self.buffer.y_buffer[:] = y_init
        self.buffer.count = self.buffer.window_size

        X_tr, y_tr = self.buffer.get_view()
        self._fit_model(X_tr, y_tr)

    def predict(self, x_t):
        if self.use_scaling:
            x_input = (x_t - self.mean_x) / self.std_x
        else:
            x_input = x_t

        x_input = self._transform_input(x_input.reshape(1, -1))
        return self.model.predict(x_input).item()

    def get_coefs(self):
        if hasattr(self.model, "coef_"):
            return self.model.coef_.ravel()
        return None

    def update(self, x_t, y_t):
        # Update Scaler
        if self.use_scaling:
            self.scaler.update(x_t)
            self.mean_x, self.std_x = self.scaler.get_scaler()
            x_new = (x_t - self.mean_x) / self.std_x
        else:
            x_new = x_t

        # Add new to buffer
        self.buffer.add(x_new, y_t)

        # Conditionally Refit
        self.steps_since_refit += 1
        if self.steps_since_refit >= self.refit_frequency:
            X_tr, y_tr = self.buffer.get_view()
            self._fit_model(X_tr, y_tr)
            self.steps_since_refit = 0


class NaiveBaseline(BaseModel):
    """Returns a lagged feature value as prediction — no buffers or scaling needed."""

    def __init__(self, lag_index=0):
        self.lag_index = lag_index

    def initialize(self, X_init, y_init):
        pass

    def predict(self, x_t):
        return x_t[self.lag_index]

    def update(self, x_t, y_t):
        pass
