import numpy as np
from numba import njit

# --- Numba Kernels for Speed ---

@njit(cache=True)
def _stats_update_core(sum_x, sum_sq, x_new, x_old):
    """
    Updates sums and sum-of-squares in-place.
    """
    diff_sum = x_new - x_old
    diff_sq = x_new**2 - x_old**2
    
    sum_x += diff_sum
    sum_sq += diff_sq
    
    # Numerical stability clip for sum_sq
    for i in range(len(sum_sq)):
        if sum_sq[i] < 0.0:
            sum_sq[i] = 0.0

@njit(cache=True)
def _get_zscore_params(sum_x, sum_sq, n):
    """
    Calculates Mean and Std from sums.
    """
    if n <= 1:
        return np.zeros_like(sum_x), np.ones_like(sum_x)

    # 1. Calculate Mean
    mean = sum_x / n
    
    # 2. Calculate Variance: E[X^2] - (E[X])^2
    # var = (sum_sq / n) - (mean ** 2)
    # Using max(0) to prevent negative variance due to float precision
    var = np.maximum((sum_sq / n) - (mean * mean), 0.0)
    
    # 3. Calculate Std
    std = np.sqrt(var)
    
    # Thresholding: If std is 0 (constant value), prevent divide by zero
    for i in range(len(std)):
        if std[i] < 1e-12:
            std[i] = 1.0
            
    return mean, std

# --- Main Class ---

class RollingStandardScaler:
    """
    Rolling Z-Score Scaler (StandardScaler).
    Maintains running Sum and SumSq to calculate Mean and Std in O(1).
    """
    def __init__(self, n_features):
        self.n_features = n_features
        self.sum_x = np.zeros(n_features, dtype=np.float64)
        self.sum_sq = np.zeros(n_features, dtype=np.float64)
        self.n = 0

    def initialize(self, data_block):
        # Handle 1D case
        if data_block.ndim == 1:
            data_block = data_block.reshape(-1, 1)
            
        self.sum_x = np.sum(data_block, axis=0).astype(np.float64)
        self.sum_sq = np.sum(data_block**2, axis=0).astype(np.float64)
        self.n = data_block.shape[0]

    def update(self, x_new, x_old):
        # Ensure array inputs
        x_new_arr = np.atleast_1d(x_new)
        x_old_arr = np.atleast_1d(x_old)
        
        _stats_update_core(self.sum_x, self.sum_sq, x_new_arr, x_old_arr)

    def get_scaler(self):
        # Returns tuple: (mean, std)
        return _get_zscore_params(self.sum_x, self.sum_sq, self.n)

# ------------------------------------------------------------------------------
# Helper: Vector Rolling Mean (For Naive Baseline)
# ------------------------------------------------------------------------------
class RollingMean:
    def __init__(self, window_size, n_features):
        self.window_size = window_size
        self.n_features = n_features
        self.buffer = np.zeros((window_size, n_features), dtype=np.float64)
        self.ptr = 0
        self.is_full = False
        self.current_sum = np.zeros(n_features, dtype=np.float64)

    def initialize(self, data_block):
        n = data_block.shape[0]
        start = max(0, n - self.window_size)
        relevant_data = data_block[start:]
        for row in relevant_data:
            self.add(row)

    def add(self, x_new):
        if self.is_full:
            self.current_sum -= self.buffer[self.ptr]
        
        self.buffer[self.ptr] = x_new
        self.current_sum += x_new
        
        self.ptr = (self.ptr + 1) % self.window_size
        if self.ptr == 0 and not self.is_full:
            self.is_full = True

    def get_mean(self):
        count = self.window_size if self.is_full else self.ptr
        if count == 0: return np.zeros(self.n_features)
        return self.current_sum / count

# ------------------------------------------------------------------------------
# Helper: Rolling Buffer (For Regression Training)
# ------------------------------------------------------------------------------
class RollingBuffer:
    def __init__(self, window_size, n_features, n_targets):
        self.window_size = window_size
        self.ptr = 0
        self.X_buffer = np.zeros((window_size, n_features), dtype=np.float32)
        self.y_buffer = np.zeros((window_size, n_targets), dtype=np.float32)
        
    def add(self, x_new, y_new):
        self.X_buffer[self.ptr] = x_new
        self.y_buffer[self.ptr] = y_new
        self.ptr = (self.ptr + 1) % self.window_size
            
    def get_view(self):
        return self.X_buffer, self.y_buffer