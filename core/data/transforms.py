"""Data transforms: diurnal adjustment, column transforms, winsorization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.core import config
from core.core.config import check_sorted_index

SKIP_VARS = {"hour", "DOW", "t", "date"}
DEFAULT_DIURNAL_EXCLUDED = SKIP_VARS | {"vix", "sentiment"}


def diurnal_adjust(
    series: pd.Series,
    time_of_day_series: pd.Series,
    has_negatives: bool,
    window: int,
    min_periods: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Divide series by a rolling per-time-slot baseline.

    - Non-negative vars: baseline = rolling mean per slot.
    - Signed vars:       baseline = rolling std per slot.

    Returns (adjusted_series, baseline).
    """
    baseline = pd.Series(index=series.index, dtype=float)
    for _slot, idx in time_of_day_series.groupby(time_of_day_series).groups.items():
        slot_series = series.loc[idx].sort_index()
        if not has_negatives:
            rolled = slot_series.rolling(window=window, min_periods=min_periods).mean().shift(1)
        else:
            rolled = slot_series.rolling(window=window, min_periods=min_periods).std().shift(1)
        baseline.loc[idx] = rolled
    baseline = baseline.fillna(1.0)
    return series / baseline, baseline


def apply_data_transform(
    series: pd.Series,
    col_name: str,
    has_negatives: bool,
    allow_missing: bool,
) -> pd.Series:
    """
    Apply a data-driven transform based on column semantics.

    - ret2/RV/turnover/bipow/effspread → sqrt
    - autocov                          → signed sqrt
    - ret3                             → cube root
    - ret4                             → fourth root
    - signed/sumabsret                 → identity (fill NaN if needed)
    - default                          → log
    """

    def _col_matches(*keywords):
        return any(kw in col_name for kw in keywords)

    if _col_matches("ret2", "RV", "turnover", "bipow", "effspread"):
        return pd.Series(np.sqrt(series), index=series.index)

    elif _col_matches("autocov"):
        return pd.Series(np.sign(series) * np.sqrt(np.abs(series)), index=series.index)

    elif _col_matches("ret3"):
        return pd.Series(np.cbrt(series), index=series.index)

    elif _col_matches("ret4"):
        return pd.Series(np.power(series, 0.25), index=series.index)

    elif has_negatives or _col_matches("sumabsret"):
        if not allow_missing:
            return series.fillna(0.0)
        return series

    else:
        return pd.Series(np.log(series), index=series.index)


def rolling_winsorize(
    series: pd.Series,
    window: int,
    allow_missing: bool,
    is_target: bool,
) -> pd.Series:
    """
    Clip series to rolling 5th/95th quantile bounds.

    Uses nanquantile for allow_missing mode (except targets).
    """
    if allow_missing and not is_target:
        lower = (
            series.rolling(window=window, min_periods=1)
            .apply(lambda x: np.nanquantile(x, config.WINSOR_LOWER_Q), raw=True)
            .shift(1)
        )
        upper = (
            series.rolling(window=window, min_periods=1)
            .apply(lambda x: np.nanquantile(x, config.WINSOR_UPPER_Q), raw=True)
            .shift(1)
        )
    else:
        lower = series.rolling(window=window, min_periods=1).quantile(config.WINSOR_LOWER_Q).shift(1)
        upper = series.rolling(window=window, min_periods=1).quantile(config.WINSOR_UPPER_Q).shift(1)
    return series.clip(lower=lower, upper=upper)


def robust_transform(
    df: pd.DataFrame,
    col_name: str,
    time_col: str = "time_of_day",
    diurnal_window: int = config.DIURNAL_WINDOW,
    min_periods: int = config.DIURNAL_MIN_PERIODS,
    use_transform: bool = True,
    allow_missing: bool = False,
    use_diurnal: bool = True,
    winsor_window: int | None = None,
    is_target: bool = False,
    diurnal_excluded_cols: set[str] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Applies diurnal adjustment, data-driven transform, then winsorization.

    Pipeline: diurnal_adjust → apply_data_transform → rolling_winsorize.
    """
    if diurnal_excluded_cols is None:
        diurnal_excluded_cols = DEFAULT_DIURNAL_EXCLUDED

    if col_name in SKIP_VARS:
        return df[col_name], pd.Series(0, index=df.index)

    series = df[col_name]
    has_negatives = bool((series.dropna() < 0).any())

    check_sorted_index(df.index)

    # 1. Diurnal adjustment
    do_diurnal = use_diurnal and (col_name not in diurnal_excluded_cols)
    if do_diurnal:
        series, baseline = diurnal_adjust(series, df[time_col], has_negatives, diurnal_window, min_periods)
    else:
        baseline = pd.Series(1.0, index=df.index)

    # 2. Data-driven transform
    if use_transform:
        series = apply_data_transform(series, col_name, has_negatives, allow_missing)

    # 3. Winsorization
    if winsor_window is not None:
        series = rolling_winsorize(series, winsor_window, allow_missing, is_target)

    return series, baseline
