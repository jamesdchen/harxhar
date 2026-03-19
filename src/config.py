from __future__ import annotations

# --- PIPELINE CONFIGURATION ---
DIURNAL_WINDOW = 20
DIURNAL_MIN_PERIODS = 5

# Maximum lag (used to derive geometric or consecutive lag sequences)
LAG = 3125

# Naive baseline uses a fixed 144 raw lags (3 days × 48 periods)
NAIVE_LAG = 144

DEFAULT_RESULTS_DIR = "results"

START_DATE = "2005-01-01"

# Temporal constants
PERIODS_PER_DAY = 48  # 30-min bars per trading day

# Forecast horizon
DEFAULT_HORIZON = 1  # 1-step ahead; max is PERIODS_PER_DAY (48)

# Normalization
NORM_EPS = 1e-8

# Gradient clipping (GPU training)
GRAD_CLIP_BOUND = 5.0

# QLIKE clamping bounds (GPU training, log-space)
QLIKE_CLAMP_MIN = -30.0
QLIKE_CLAMP_MAX = 30.0

# Circuit breaker dates (market-wide trading halts)
CIRCUIT_BREAKER_DATES = ["2020-03-09", "2020-03-12", "2020-03-16", "2020-03-18"]

# Winsorization quantiles
WINSOR_LOWER_Q = 0.05
WINSOR_UPPER_Q = 0.95

# SARIMAX defaults
SARIMAX_ORDER = (2, 0, 1)
SARIMAX_SEASONAL_ORDER = (1, 0, 0, PERIODS_PER_DAY)
SARIMAX_FIT_WINDOW = 480  # 10 trading days
SARIMAX_REFIT_FREQUENCY = 48  # once per simulated day

# AE refit frequency (steps between autoencoder refits)
AE_REFIT_FREQUENCY = 240

# SARIMAX fitting defaults
SARIMAX_FIT_METHOD = "lbfgs"
SARIMAX_FIT_MAXITER = 100

# AdamW optimizer betas (GPU training)
ADAMW_BETA1 = 0.9
ADAMW_BETA2 = 0.999
ADAMW_WEIGHT_DECAY = 0.01

# GPU logging
GPU_WORKER_LOG = "worker_log.txt"


# --- Validation helpers ---
def check_positive(val: int | float, name: str) -> None:
    """Raise ValueError if val is not positive."""
    if val <= 0:
        raise ValueError(f"{name} must be positive, got {val}")


def check_backtest_inputs(X, y, indices) -> None:
    """Validate backtest array shapes and index bounds."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y row mismatch: {X.shape[0]} vs {y.shape[0]}")
    if len(indices) == 0:
        raise ValueError("indices array is empty")
    if indices[-1] >= X.shape[0]:
        raise ValueError(f"indices out of bounds: max index {indices[-1]} >= X length {X.shape[0]}")


def check_sorted_index(index) -> None:
    """Raise ValueError if a pandas Index is not monotonically increasing."""
    if not index.is_monotonic_increasing:
        offender = (index.to_series().diff() < 0).argmax()
        raise ValueError(f"Index must be sorted — first offender at position {offender}")


def check_finite(arr, name: str) -> None:
    """Raise ValueError if *arr* contains NaN or Inf."""
    import numpy as _np

    if not _np.all(_np.isfinite(arr)):
        n_bad = int(_np.sum(~_np.isfinite(arr)))
        raise ValueError(f"{name} contains {n_bad} non-finite values (NaN/Inf)")


def find_naive_lag(feature_names: list[str]) -> int:
    """Return the index of the naive-baseline lag feature.

    Searches for a feature containing 'lag_125' or exactly equal to
    'har_ma_125'.  Raises ValueError with a clear message if not found.
    """
    for i, f in enumerate(feature_names):
        if "lag_125" in f or f == "har_ma_125":
            return i
    raise ValueError(
        "Naive model requires a feature matching 'lag_125' or 'har_ma_125', "
        f"but none found in {feature_names[:10]}{'...' if len(feature_names) > 10 else ''}"
    )


# 1. Define Segments with Overlaps
SEGMENT_DEFINITIONS = {
    "morning": {"start": 510, "end": 660},  # 08:30 - 11:00
    "midday": {"start": 630, "end": 870},  # 10:30 - 14:30
    "closing": {"start": 840, "end": 960},  # 14:00 - 16:00
    "overnight": {"start": 990, "end": 510},  # 16:30 - 08:30 (Wraps)
}

# --- DL (PatchTSMixer) Configuration ---
DL_CONFIG = {
    "output_path": "results.csv",
    "train_window": 50000,
    "gpu_count": 2,
    "model": {
        "context_len": 241,
        "num_input_channels": 1,
        "hidden_dim": 4,
        "num_layers": 4,
        "dropout": 0.25,
        "patch_len": 47,
        "stride": 31,
        "prediction_length": 1,
    },
    "train": {
        "num_epochs": 150,
        "learning_rate": 1e-4,
        "batch_size": 50,
        "optimizer": "ADAMW",
        "loss_fn": "QLIKE",
    },
}

# --- AE+Ridge GPU Configuration ---
AE_RIDGE_GPU_CONFIG = {
    "output_path": "ae_ridge_results.csv",
    "train_window": 24000,  # 500 days * 48 periods
    "gpu_count": 2,
    "model": {
        "n_features": 0,  # set at runtime from X.shape[1]
        "n_components": 5,
        "hidden_dim": 0,  # 0 = auto (n_features // 2)
        "alpha_recon": 0.5,  # weight: alpha*recon + (1-alpha)*pred
        "alpha_ridge": 1.0,  # Ridge regularization strength
    },
    "train": {
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "batch_size": 10,  # windows per batch (each ~10MB)
    },
}
