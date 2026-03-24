"""SARIMAX model wrapper with rolling window support."""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX

from core.core.log import get_logger
from core.models.base import RollingRegressionModel
from projects.ml import config as cfg

logger = get_logger(__name__)


class _SARIMAXEstimator:
    """
    Thin sklearn-style wrapper around statsmodels SARIMAX.

    Accepts fit(X, y) where X/y are in chronological order, and
    predict(X) for h-step-ahead forecasting with exogenous data.
    Retains the previous fitted result on transient failures. After
    MAX_CONSECUTIVE_FAILURES consecutive fit failures, resets to naive
    fallback (last observed value) to avoid stale predictions.
    """

    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self, order, seasonal_order, horizon=1):
        self.order = order
        self.seasonal_order = seasonal_order
        self.horizon = horizon
        self._result = None
        self._fit_count = 0
        self._fail_count = 0
        self._consecutive_failures = 0
        self._last_y_val = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64).ravel()
        self._last_y_val = float(y[-1])
        exog = np.asarray(X, dtype=np.float64) if X.shape[1] > 0 else None
        try:
            m = _SARIMAX(
                endog=y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._result = m.fit(
                disp=False,
                method=cfg.SARIMAX_FIT_METHOD,
                maxiter=cfg.SARIMAX_FIT_MAXITER,
            )
            self._fit_count += 1
            self._consecutive_failures = 0
        except (np.linalg.LinAlgError, ValueError) as e:
            self._fail_count += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "SARIMAX fit failed %d times consecutively (%d total); falling back to naive prediction: %s",
                    self._consecutive_failures,
                    self._fail_count,
                    e,
                )
                self._result = None
            else:
                logger.warning(
                    "SARIMAX fit failed (%d total, %d consecutive), retaining previous fit: %s",
                    self._fail_count,
                    self._consecutive_failures,
                    e,
                )
        return self

    def predict(self, X):
        """Return np.array([scalar]) so .item() in RollingRegressionModel works.

        Uses native multi-step forecasting: forecast(steps=horizon) and
        return the h-th step value. Falls back to naive (last observed value)
        when no successful fit is available.
        """
        if self._result is None:
            if self._last_y_val is not None:
                return np.array([self._last_y_val])
            raise RuntimeError(
                "SARIMAX predict called with no successful fit and no observed data — "
                f"all {self._fail_count} fit attempt(s) failed"
            )
        exog = np.asarray(X, dtype=np.float64) if X.shape[1] > 0 else None
        # For multi-step, exog must have `horizon` rows (one per step)
        if exog is not None and self.horizon > 1:
            exog = np.tile(exog, (self.horizon, 1))
        try:
            fc = self._result.forecast(steps=self.horizon, exog=exog)
            # Take the last value (the h-step-ahead forecast)
            val = fc.iloc[-1] if hasattr(fc, "iloc") else fc[-1]
            return np.array([float(val)])
        except (ValueError, IndexError) as e:
            logger.warning("SARIMAX predict failed, returning last observed value: %s", e)
            return np.array([self._last_y_val if self._last_y_val is not None else 0.0])


class SARIMAXModel(RollingRegressionModel):
    """
    SARIMAX baseline that inherits RollingRegressionModel for buffer/scaler reuse.

    Uses raw-lag exogenous features (same lag indices as config.LAG but as individual
    point values) together with ARMA(p,q) and seasonal AR components. The rolling
    window is fit_window periods (default 480 = 10 trading days), much shorter than
    the outer train_win_periods, since SARIMAX is a parametric model.

    Parameters
    ----------
    train_win_periods : int
        Burn-in window passed in by the backtester (only used to slice X_init/y_init).
    n_features : int
        Number of exogenous features (raw-lag columns).
    fit_window : int
        Internal buffer / fitting window. Defaults to 480 (10 days of 30-min bars).
    refit_frequency : int
        Steps between refits. Defaults to 48 (once per simulated day).
    order : tuple
        ARIMA (p, d, q) non-seasonal order.
    seasonal_order : tuple
        (P, D, Q, s) seasonal order. s=48 for daily seasonality on 30-min bars.
    """

    def __init__(
        self,
        train_win_periods,
        n_features,
        fit_window=480,
        refit_frequency=48,
        order=(2, 0, 1),
        seasonal_order=(1, 0, 0, 48),
        horizon=1,
    ):
        estimator = _SARIMAXEstimator(order, seasonal_order, horizon=horizon)
        # Pass fit_window as train_win_periods so buffer/scaler size = fit_window
        super().__init__(
            model=estimator,
            train_win_periods=fit_window,
            n_features=n_features,
            use_scaling=True,
            refit_frequency=refit_frequency,
        )

    def initialize(self, X_init, y_init):
        """Override to slice X_init/y_init to fit_window and use ordered fit."""
        if y_init.ndim == 1:
            y_init = y_init.reshape(-1, 1)

        # Slice to the buffer size (fit_window rows)
        X_init = X_init[-self.train_win_periods :]
        y_init = y_init[-self.train_win_periods :]

        if self.use_scaling:
            self.scaler.initialize(X_init)
            self.mean_x, self.std_x = self.scaler.get_scaler()
            X_buffered = (X_init - self.mean_x) / self.std_x
        else:
            X_buffered = X_init

        # Fill buffer directly; ptr stays 0 → data is already chronological
        self.buffer.X_buffer[:] = X_buffered
        self.buffer.y_buffer[:] = y_init
        self.buffer.count = self.buffer.window_size

        # Fit SARIMAX with chronologically ordered data
        X_tr, y_tr = self.buffer.get_ordered_view()
        self.model.fit(X_tr, y_tr)

    def update(self, x_t, y_t):
        """Override to use get_ordered_view() when refitting."""
        if self.use_scaling:
            self.scaler.update(x_t)
            self.mean_x, self.std_x = self.scaler.get_scaler()
            x_new = (x_t - self.mean_x) / self.std_x
        else:
            x_new = x_t

        self.buffer.add(x_new, y_t)

        self.steps_since_refit += 1
        if self.steps_since_refit >= self.refit_frequency:
            X_tr, y_tr = self.buffer.get_ordered_view()
            self.model.fit(X_tr, y_tr)
            self.steps_since_refit = 0
