"""Data loading, cleaning, and rolling window utilities.

Pipeline: load_and_clean_base_data (parquet → grid → market-hours filter →
robust_transform) → generate_lag_features → apply_horizon_shift →
load_and_prep_data_strided (combines all steps).

Key classes for online walk-forward evaluation:
- RollingRobustScaler — O(W) online (median, IQR) normalization via Numba JIT
- RollingBuffer — ring buffer storing (X, y) pairs with chronological views
- RollingMedian — simple rolling median over a ring buffer
"""

__all__ = [
    "load_and_clean_base_data",
    "apply_horizon_shift",
    "load_and_prep_data_strided",
    "RollingBuffer",
    "RollingMedian",
    "RollingRobustScaler",
    "apply_data_transform",
    "diurnal_adjust",
    "robust_transform",
    "rolling_winsorize",
]

from core.data.loading import load_and_clean_base_data as load_and_clean_base_data
from core.data.pipeline import apply_horizon_shift as apply_horizon_shift
from core.data.pipeline import load_and_prep_data_strided as load_and_prep_data_strided
from core.data.rolling import RollingBuffer as RollingBuffer
from core.data.rolling import RollingMedian as RollingMedian
from core.data.rolling import RollingRobustScaler as RollingRobustScaler
from core.data.transforms import (
    apply_data_transform as apply_data_transform,
)
from core.data.transforms import (
    diurnal_adjust as diurnal_adjust,
)
from core.data.transforms import (
    robust_transform as robust_transform,
)
from core.data.transforms import (
    rolling_winsorize as rolling_winsorize,
)
