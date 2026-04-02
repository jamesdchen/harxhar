"""Standalone data transforms for volatility forecasting.

Diurnal adjustment, semantic transforms, rolling winsorization, and a
combined robust_transform pipeline.  No imports from core/ or projects/.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIURNAL_WINDOW: int = 20
DIURNAL_MIN_PERIODS: int = 5
WINSOR_LOWER_Q: float = 0.05
WINSOR_UPPER_Q: float = 0.95
SKIP_VARS: set[str] = {"hour", "DOW", "t", "date"}
DIURNAL_EXCLUDED: set[str] = SKIP_VARS | {"vix", "sentiment"}

# ---------------------------------------------------------------------------
# Diurnal adjustment
# ---------------------------------------------------------------------------


def diurnal_adjust(
    series: pd.Series,
    time_of_day_series: pd.Series,
    has_negatives: bool,
    window: int = DIURNAL_WINDOW,
    min_periods: int = DIURNAL_MIN_PERIODS,
) -> tuple[pd.Series, pd.Series]:
    """Remove intraday seasonality via rolling per-slot baseline.

    Parameters
    ----------
    series : pd.Series
        Raw values to adjust.
    time_of_day_series : pd.Series
        Aligned series of time-of-day slot labels (same index as *series*).
    has_negatives : bool
        If True the variable can be negative and the baseline is rolling std;
        otherwise the baseline is rolling mean.
    window, min_periods : int
        Rolling window parameters applied *within* each slot.

    Returns
    -------
    (adjusted, baseline) where adjusted = series / baseline.
    """
    df = pd.DataFrame({"val": series, "slot": time_of_day_series})

    if has_negatives:
        baseline = (
            df.groupby("slot")["val"]
            .transform(
                lambda g: g.rolling(window, min_periods=min_periods).std().shift(1)
            )
        )
    else:
        baseline = (
            df.groupby("slot")["val"]
            .transform(
                lambda g: g.rolling(window, min_periods=min_periods).mean().shift(1)
            )
        )

    baseline = baseline.fillna(1.0)
    adjusted = series / baseline
    return adjusted, baseline


# ---------------------------------------------------------------------------
# Semantic (column-name-based) transforms
# ---------------------------------------------------------------------------


def apply_semantic_transform(
    series: pd.Series,
    col_name: str,
    has_negatives: bool,
    allow_missing: bool = False,
) -> pd.Series:
    """Apply a variance-stabilising transform chosen by column name.

    Rules (checked in order):
    1. name contains ret2 / RV / turnover / bipow / effspread → sqrt
    2. name contains autocov → sign(x) * sqrt(|x|)
    3. name contains ret3 → cbrt
    4. name contains ret4 → fourth root (x ** 0.25)
    5. has_negatives or name contains sumabsret → identity (NaN → 0)
    6. default → log
    """
    name = col_name.lower()

    if any(tok in name for tok in ("ret2", "rv", "turnover", "bipow", "effspread")):
        return np.sqrt(series)

    if "autocov" in name:
        return np.sign(series) * np.sqrt(np.abs(series))

    if "ret3" in name:
        return np.cbrt(series)

    if "ret4" in name:
        return np.power(np.abs(series), 0.25) * np.sign(series)

    if has_negatives or "sumabsret" in name:
        out = series.copy()
        if not allow_missing:
            out = out.fillna(0.0)
        return out

    # default: log (guard against non-positive values)
    return np.log(series.clip(lower=1e-12))


# ---------------------------------------------------------------------------
# Rolling winsorization
# ---------------------------------------------------------------------------


def rolling_winsorize(
    series: pd.Series,
    window: int = 240,
    allow_missing: bool = False,
    is_target: bool = False,
) -> pd.Series:
    """Clip values to rolling 5th / 95th quantile bounds (shifted by 1).

    Parameters
    ----------
    series : pd.Series
    window : int
        Lookback window for quantile estimation.
    allow_missing : bool
        If True and not is_target, use nanquantile-style (min_periods=1).
    is_target : bool
        Targets never use nanquantile even when allow_missing is True.
    """
    use_nan = allow_missing and not is_target
    min_per = 1 if use_nan else window

    lower = series.rolling(window, min_periods=min_per).quantile(WINSOR_LOWER_Q).shift(1)
    upper = series.rolling(window, min_periods=min_per).quantile(WINSOR_UPPER_Q).shift(1)
    return series.clip(lower=lower, upper=upper)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def robust_transform(
    df: pd.DataFrame,
    col_name: str,
    time_col: str = "time_of_day",
    use_transform: bool = True,
    use_diurnal: bool = True,
    allow_missing: bool = False,
    winsor_window: int | None = None,
    is_target: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Chain diurnal_adjust → apply_semantic_transform → rolling_winsorize.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *col_name* and (if diurnal is used) *time_col*.
    col_name : str
        Column to transform.
    time_col : str
        Column holding the time-of-day slot labels.
    use_transform, use_diurnal : bool
        Toggle individual stages.
    allow_missing : bool
        Forwarded to downstream helpers.
    winsor_window : int | None
        Override default winsorization window (240).
    is_target : bool
        Forwarded to rolling_winsorize.

    Returns
    -------
    (adjusted_series, baseline)
    """
    if col_name in SKIP_VARS:
        return df[col_name].copy(), pd.Series(1.0, index=df.index)

    series = df[col_name].copy()
    has_negatives = bool((series.dropna() < 0).any())

    # --- diurnal ---
    baseline = pd.Series(1.0, index=df.index)
    if use_diurnal and col_name not in DIURNAL_EXCLUDED and time_col in df.columns:
        series, baseline = diurnal_adjust(
            series, df[time_col], has_negatives
        )

    # --- semantic transform ---
    if use_transform:
        series = apply_semantic_transform(
            series, col_name, has_negatives, allow_missing=allow_missing
        )

    # --- winsorize ---
    ww = winsor_window if winsor_window is not None else 240
    series = rolling_winsorize(
        series, window=ww, allow_missing=allow_missing, is_target=is_target
    )

    return series, baseline
