from __future__ import annotations

import numpy as np
from numba import njit

# ------------------------------------------------------------------------------
# Numba Kernels for O(W) Sorted Array Maintenance
# ------------------------------------------------------------------------------


@njit(cache=True)
def _update_sorted_matrix(sorted_mat, x_old, x_new):
    """
    Maintains a perfectly sorted rolling window by shifting elements.
    sorted_mat shape: (n_features, window_size)
    """
    n_features, w = sorted_mat.shape

    for i in range(n_features):
        v_old = x_old[i]
        v_new = x_new[i]

        # 1. Find the old value (Binary Search)
        idx_old = np.searchsorted(sorted_mat[i], v_old)

        # 2. Find where the new value belongs (Binary Search)
        idx_new = np.searchsorted(sorted_mat[i], v_new)

        # 3. Shift the elements to overwrite old and make room for new
        if idx_old < idx_new:
            # Shift left
            idx_new -= 1
            for j in range(idx_old, idx_new):
                sorted_mat[i, j] = sorted_mat[i, j + 1]
        elif idx_old > idx_new:
            # Shift right
            for j in range(idx_old, idx_new, -1):
                sorted_mat[i, j] = sorted_mat[i, j - 1]

        # 4. Insert the new value
        sorted_mat[i, idx_new] = v_new


@njit(cache=True)
def _get_robust_stats(sorted_mat):
    """
    Extracts Median and IQR instantly from the pre-sorted array using
    standard linear interpolation for percentiles.
    """
    n_features, w = sorted_mat.shape
    median = np.empty(n_features, dtype=np.float64)
    iqr = np.empty(n_features, dtype=np.float64)

    # Calculate exact float indices for 25th, 50th, and 75th percentiles
    idx_25 = (w - 1) * 0.25
    idx_50 = (w - 1) * 0.50
    idx_75 = (w - 1) * 0.75

    i25_floor, rem_25 = int(idx_25), idx_25 - int(idx_25)
    i50_floor, rem_50 = int(idx_50), idx_50 - int(idx_50)
    i75_floor, rem_75 = int(idx_75), idx_75 - int(idx_75)

    for i in range(n_features):
        # Linear interpolation
        q25 = sorted_mat[i, i25_floor] * (1.0 - rem_25) + sorted_mat[i, min(i25_floor + 1, w - 1)] * rem_25
        med = sorted_mat[i, i50_floor] * (1.0 - rem_50) + sorted_mat[i, min(i50_floor + 1, w - 1)] * rem_50
        q75 = sorted_mat[i, i75_floor] * (1.0 - rem_75) + sorted_mat[i, min(i75_floor + 1, w - 1)] * rem_75

        median[i] = med

        iq = q75 - q25
        iqr[i] = iq if iq >= 1e-12 else 1.0

    return median, iqr


# ------------------------------------------------------------------------------
# Main Class
# ------------------------------------------------------------------------------


class RollingRobustScaler:
    """
    Rolling Robust Scaler (JIT Compiled).
    Maintains a physical ring buffer and a synchronized sorted buffer.
    """

    def __init__(self, window_size: int, n_features: int):
        self.window_size = window_size
        self.n_features = n_features

        # Chronological buffer to know exactly which value is dropping out
        self.chrono_buffer = np.zeros((window_size, n_features), dtype=np.float64)

        # Sorted buffer (transposed for contiguous memory access per feature)
        self.sorted_buffer = np.zeros((n_features, window_size), dtype=np.float64)

        self.ptr = 0
        self.is_full = False

    def initialize(self, data_block: np.ndarray) -> None:
        if data_block.ndim == 1:
            data_block = data_block.reshape(-1, 1)

        n = data_block.shape[0]
        start = max(0, n - self.window_size)
        relevant_data = data_block[start:]

        # Fill chronological buffer
        for row in relevant_data:
            self.chrono_buffer[self.ptr] = row
            self.ptr = (self.ptr + 1) % self.window_size
            if self.ptr == 0:
                self.is_full = True

        # Initialize the sorted buffer
        self.sorted_buffer[:, :] = self.chrono_buffer.T
        self.sorted_buffer.sort(axis=1)

    def update(self, x_new: np.ndarray) -> None:
        # Even though you pass x_old from the model, we use the exact float
        # from our chrono_buffer to guarantee a perfect match in the sorted array.
        x_new_arr = np.atleast_1d(x_new)
        actual_old_arr = self.chrono_buffer[self.ptr]

        # 1. Update the sorted array in O(W) via Numba
        _update_sorted_matrix(self.sorted_buffer, actual_old_arr, x_new_arr)

        # 2. Update the chronological array
        self.chrono_buffer[self.ptr] = x_new_arr
        self.ptr = (self.ptr + 1) % self.window_size

    def get_scaler(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns tuple: (median, iqr) instantly in O(1) time"""
        return _get_robust_stats(self.sorted_buffer)


# ------------------------------------------------------------------------------
# Helper: Vector Rolling Median (Replacing the Naive Rolling Mean)
# ------------------------------------------------------------------------------
class RollingMedian:
    def __init__(self, window_size, n_features):
        self.window_size = window_size
        self.n_features = n_features
        self.buffer = np.zeros((window_size, n_features), dtype=np.float64)
        self.ptr = 0
        self.is_full = False

    def initialize(self, data_block):
        if data_block.ndim == 1:
            data_block = data_block.reshape(-1, 1)

        n = data_block.shape[0]
        start = max(0, n - self.window_size)
        relevant_data = data_block[start:]
        for row in relevant_data:
            self.add(row)

    def add(self, x_new):
        self.buffer[self.ptr] = x_new
        self.ptr = (self.ptr + 1) % self.window_size
        if self.ptr == 0 and not self.is_full:
            self.is_full = True

    def get_median(self):
        count = self.window_size if self.is_full else self.ptr
        if count == 0:
            return np.zeros(self.n_features)

        valid_buffer = self.buffer if self.is_full else self.buffer[: self.ptr]
        return np.median(valid_buffer, axis=0)


# ------------------------------------------------------------------------------
# Helper: Rolling Buffer (For Regression Training)
# ------------------------------------------------------------------------------
class RollingBuffer:
    def __init__(self, window_size: int, n_features: int, n_targets: int):
        self.window_size = window_size
        self.ptr = 0
        self.count = 0
        self.X_buffer = np.zeros((window_size, n_features), dtype=np.float64)
        self.y_buffer = np.zeros((window_size, n_targets), dtype=np.float64)

    def add(self, x_new: np.ndarray, y_new: float | np.ndarray) -> None:
        self.X_buffer[self.ptr] = x_new
        self.y_buffer[self.ptr] = y_new
        self.ptr = (self.ptr + 1) % self.window_size
        if self.count < self.window_size:
            self.count += 1

    def get_view(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X_buffer, self.y_buffer

    def get_ordered_view(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) in oldest-to-newest chronological order.

        After a direct buffer fill (ptr=0) this is identical to get_view().
        After k updates (ptr=k) it correctly reorders the ring buffer.
        When the buffer is not yet full, returns only the filled portion.
        """
        if self.count < self.window_size:
            return self.X_buffer[: self.count], self.y_buffer[: self.count]
        p = self.ptr
        X = np.concatenate([self.X_buffer[p:], self.X_buffer[:p]], axis=0)
        y = np.concatenate([self.y_buffer[p:], self.y_buffer[:p]], axis=0)
        return X, y
