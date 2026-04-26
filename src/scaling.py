# Auto-generated from notebooks/04_scaling.ipynb. Do not edit by hand.

"""Rolling robust scaler and walk-forward backtest infrastructure.

Numba-accelerated sorted-matrix quantile tracking for O(W) median/IQR
scaling, ring buffer for training data, and generic walk-forward loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numba import njit
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _update_sorted_matrix(sorted_mat: np.ndarray, x_old: np.ndarray, x_new: np.ndarray) -> None:
    """Replace *x_old* with *x_new* in each feature's sorted window."""
    n_features, w = sorted_mat.shape
    for i in range(n_features):
        v_old = x_old[i]
        v_new = x_new[i]
        idx_old = np.searchsorted(sorted_mat[i], v_old)
        idx_new = np.searchsorted(sorted_mat[i], v_new)
        if idx_old < idx_new:
            idx_new -= 1
            for j in range(idx_old, idx_new):
                sorted_mat[i, j] = sorted_mat[i, j + 1]
        elif idx_old > idx_new:
            for j in range(idx_old, idx_new, -1):
                sorted_mat[i, j] = sorted_mat[i, j - 1]
        sorted_mat[i, idx_new] = v_new


@njit(cache=True)
def _get_robust_stats(
    sorted_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute median and IQR from pre-sorted rolling window."""
    n_features, w = sorted_mat.shape
    median = np.empty(n_features, dtype=np.float64)
    iqr = np.empty(n_features, dtype=np.float64)
    idx_25 = (w - 1) * 0.25
    idx_50 = (w - 1) * 0.50
    idx_75 = (w - 1) * 0.75
    i25_floor, rem_25 = int(idx_25), idx_25 - int(idx_25)
    i50_floor, rem_50 = int(idx_50), idx_50 - int(idx_50)
    i75_floor, rem_75 = int(idx_75), idx_75 - int(idx_75)
    for i in range(n_features):
        q25 = sorted_mat[i, i25_floor] * (1.0 - rem_25) + sorted_mat[i, min(i25_floor + 1, w - 1)] * rem_25
        med = sorted_mat[i, i50_floor] * (1.0 - rem_50) + sorted_mat[i, min(i50_floor + 1, w - 1)] * rem_50
        q75 = sorted_mat[i, i75_floor] * (1.0 - rem_75) + sorted_mat[i, min(i75_floor + 1, w - 1)] * rem_75
        median[i] = med
        iq = q75 - q25
        iqr[i] = iq if iq >= 1e-12 else 1.0
    return median, iqr


# ---------------------------------------------------------------------------
# Rolling robust scaler
# ---------------------------------------------------------------------------


class RollingRobustScaler:
    """Online robust scaler backed by sorted-matrix quantile tracking.

    Maintains a rolling window of observations and a parallel sorted matrix
    per feature for O(W) median/IQR computation.  Supports two usage styles:

    1. **get_scaler** -- return (median, iqr) for manual scaling (ridge-style).
    2. **transform_single / transform_buffer** -- scale directly (PCR-style).

    Parameters
    ----------
    window_size : int
        Rolling window length.
    n_features : int, optional
        Number of features.  If given, buffers are pre-allocated;
        otherwise allocation is deferred to :meth:`initialize`.
    """

    buffer: np.ndarray | None
    sorted_mat: np.ndarray | None

    def __init__(self, window_size: int, n_features: int | None = None) -> None:
        self.window_size = window_size
        if n_features is not None:
            self.buffer = np.zeros((window_size, n_features), dtype=np.float64)
            self.sorted_mat = np.zeros((n_features, window_size), dtype=np.float64)
        else:
            self.buffer = None
            self.sorted_mat = None
        self.pos: int = 0

    def initialize(self, data_block: np.ndarray) -> None:
        """Fill buffers from *data_block* ``(window_size, n_features)``."""
        w = self.window_size
        n_features = data_block.shape[1]
        if self.buffer is None:
            self.buffer = np.empty((w, n_features), dtype=np.float64)
            self.sorted_mat = np.empty((n_features, w), dtype=np.float64)
        self.buffer[:] = data_block[:w]
        for i in range(n_features):
            self.sorted_mat[i] = np.sort(data_block[:w, i])  # type: ignore[index]
        self.pos = 0

    def update(self, x_new: np.ndarray) -> None:
        """Slide window: replace oldest row with *x_new*."""
        assert self.buffer is not None and self.sorted_mat is not None
        x_old = self.buffer[self.pos].copy()
        self.buffer[self.pos] = x_new
        _update_sorted_matrix(self.sorted_mat, x_old, x_new)
        self.pos = (self.pos + 1) % self.window_size

    def get_scaler(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(median, iqr)`` arrays from the current sorted buffer."""
        assert self.sorted_mat is not None
        return _get_robust_stats(self.sorted_mat)

    def transform_single(self, x: np.ndarray) -> np.ndarray:
        """Scale a single observation using current median/IQR."""
        median, iqr = self.get_scaler()
        return (x - median) / iqr

    def transform_buffer(self) -> np.ndarray:
        """Scale the entire current buffer using current median/IQR."""
        assert self.buffer is not None
        median, iqr = self.get_scaler()
        return (self.buffer - median) / iqr


# ---------------------------------------------------------------------------
# Rolling buffer
# ---------------------------------------------------------------------------


class RollingBuffer:
    """Ring buffer for (X, y) pairs used in walk-forward backtests."""

    def __init__(self, window_size: int, n_features: int, n_targets: int = 1) -> None:
        self.window_size = window_size
        self.X = np.zeros((window_size, n_features), dtype=np.float64)
        self.y = np.zeros((window_size, n_targets), dtype=np.float64)
        self.pos = 0
        self.count = 0

    def add(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        self.X[self.pos] = x_new
        self.y[self.pos] = y_new
        self.pos = (self.pos + 1) % self.window_size
        self.count = min(self.count + 1, self.window_size)

    def get_view(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < self.window_size:
            return self.X[: self.count], self.y[: self.count]
        idx = np.roll(np.arange(self.window_size), -self.pos)
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------


def run_backtest(
    model_fn: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    train_win: int,
    refit_frequency: int = 1,
    use_scaling: bool = True,
) -> np.ndarray:
    """Walk-forward backtest with periodic refit.

    Parameters
    ----------
    model_fn : Callable[[], Any]
        Zero-argument factory returning a fresh model exposing ``.fit(X, y)``
        and ``.predict(X)``. Called once for the initial fit and again at
        every refit step.
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Target vector of shape ``(n_samples,)``.
    train_win : int
        Rolling training window size, in samples.
    refit_frequency : int, default=1
        Cadence (in steps) at which the model is refit on the latest window.
        ``1`` refits every step; larger values amortize fit cost.
    use_scaling : bool, default=True
        If True, apply ``RollingRobustScaler`` (median / IQR) to features
        using statistics computed only over the trailing ``train_win`` window.

    Returns
    -------
    np.ndarray
        Predictions of shape ``(n_samples - train_win,)``. Entry ``k``
        corresponds to the prediction for sample ``t = train_win + k``.

    Notes
    -----
    **Refit cadence invariant.** A model is refit when
    ``(t - train_win + 1) % refit_frequency == 0``, where ``t`` is the
    current step index in ``[train_win, n_samples)``. This is an amortized
    schedule: cheap models like Ridge typically use ``refit_frequency=1``
    (refit every step), while heavier tree models (XGBoost, LightGBM) use
    larger values to amortize fitting cost across multiple predictions.

    **Strict causality.** At step ``t`` the model is trained on the closed-
    open window ``[t - train_win : t]`` and predicts ``y[t]``. The feature
    row ``X[t]`` is scaled using statistics from ``[t - train_win : t]``
    (the scaler is updated *after* the prediction at step ``t`` is made),
    and ``y[t]`` is appended to the training buffer *after* prediction.
    No look-ahead is possible: target and same-step feature information
    never enter the model used to produce the prediction at ``t``.

    Scaling, when enabled, uses ``RollingRobustScaler`` -- a rolling median
    and inter-quartile range computed over the trailing ``train_win``
    window via numba-accelerated sorted-matrix tracking.
    """
    n_samples, n_features = X.shape
    predictions = np.full(n_samples - train_win, np.nan)

    # Initialize scaler + buffer
    scaler_obj = RollingRobustScaler(train_win, n_features) if use_scaling else None
    buf = RollingBuffer(train_win, n_features, n_targets=1)

    X_init = X[:train_win].copy()
    y_init = y[:train_win].copy()

    if use_scaling:
        assert scaler_obj is not None  # mypy: scaler_obj is non-None when use_scaling=True
        scaler_obj.initialize(X_init)
        med, iqr = scaler_obj.get_scaler()
        X_scaled_init = (X_init - med) / iqr
    else:
        X_scaled_init = X_init

    for i in range(train_win):
        buf.add(X_scaled_init[i], y_init[i : i + 1])

    # Initial fit
    X_buf, y_buf = buf.get_view()
    model = model_fn()
    model.fit(X_buf, y_buf.ravel())

    # Walk forward
    for t in tqdm(range(train_win, n_samples), desc="backtest"):
        x_t_raw = X[t]

        # Scale
        if use_scaling:
            assert scaler_obj is not None  # mypy: scaler_obj is non-None when use_scaling=True
            med, iqr = scaler_obj.get_scaler()
            x_t_scaled = (x_t_raw - med) / iqr
        else:
            x_t_scaled = x_t_raw

        # Predict
        predictions[t - train_win] = model.predict(x_t_scaled.reshape(1, -1))[0]

        # Update scaler
        if use_scaling:
            assert scaler_obj is not None  # mypy: scaler_obj is non-None when use_scaling=True
            scaler_obj.update(x_t_raw)

        # Add to buffer
        buf.add(x_t_scaled, y[t : t + 1])

        # Refit
        if (t - train_win + 1) % refit_frequency == 0:
            X_buf, y_buf = buf.get_view()
            model = model_fn()
            model.fit(X_buf, y_buf.ravel())

    return predictions
