"""Pipeline constants, validation helpers, and model hyperparameter defaults."""

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

# Winsorization quantiles
WINSOR_LOWER_Q = 0.05
WINSOR_UPPER_Q = 0.95


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
