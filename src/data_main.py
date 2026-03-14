import pandas as pd
import numpy as np
import os
from functools import reduce
from src import config
from typing import Literal

SKIP_VARS = {'hour', 'DOW', 't', 'date'}
DEFAULT_DIURNAL_EXCLUDED = SKIP_VARS | {'vix', 'sentiment'}


def diurnal_adjust(
    series: pd.Series, time_of_day_series: pd.Series,
    has_negatives: bool, window: int, min_periods: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Divide series by a rolling per-time-slot baseline.

    - Non-negative vars: baseline = rolling mean per slot.
    - Signed vars:       baseline = rolling std per slot.

    Returns (adjusted_series, baseline).
    """
    baseline = pd.Series(index=series.index, dtype=float)
    for slot, idx in time_of_day_series.groupby(time_of_day_series).groups.items():
        slot_series = series.loc[idx].sort_index()
        if not has_negatives:
            rolled = slot_series.rolling(window=window, min_periods=min_periods).mean().shift(1)
        else:
            rolled = slot_series.rolling(window=window, min_periods=min_periods).std().shift(1)
        baseline.loc[idx] = rolled
    baseline = baseline.fillna(1.0)
    return series / baseline, baseline


def apply_data_transform(
    series: pd.Series, col_name: str, has_negatives: bool, allow_missing: bool,
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

    if _col_matches('ret2', 'RV', 'turnover', 'bipow', 'effspread'):
        return np.sqrt(series)

    elif _col_matches('autocov'):
        return np.sign(series) * np.sqrt(np.abs(series))

    elif _col_matches('ret3'):
        return np.cbrt(series)

    elif _col_matches('ret4'):
        return np.power(series, 0.25)

    elif has_negatives or _col_matches('sumabsret'):
        if not allow_missing:
            return series.fillna(0.0)
        return series

    else:
        return np.log(series)


def rolling_winsorize(
    series: pd.Series, window: int, allow_missing: bool, is_target: bool,
) -> pd.Series:
    """
    Clip series to rolling 5th/95th quantile bounds.

    Uses nanquantile for allow_missing mode (except targets).
    """
    if allow_missing and not is_target:
        lower = series.rolling(window=window, min_periods=1).apply(
            lambda x: np.nanquantile(x, config.WINSOR_LOWER_Q), raw=True
        ).shift(1)
        upper = series.rolling(window=window, min_periods=1).apply(
            lambda x: np.nanquantile(x, config.WINSOR_UPPER_Q), raw=True
        ).shift(1)
    else:
        lower = series.rolling(window=window, min_periods=1).quantile(config.WINSOR_LOWER_Q).shift(1)
        upper = series.rolling(window=window, min_periods=1).quantile(config.WINSOR_UPPER_Q).shift(1)
    return series.clip(lower=lower, upper=upper)


def robust_transform(df: pd.DataFrame, col_name: str, time_col: str = "time_of_day",
                     diurnal_window=config.DIURNAL_WINDOW,
                     min_periods=config.DIURNAL_MIN_PERIODS,
                     use_transform=True, allow_missing=False,
                     use_diurnal=True, winsor_window=None,
                     is_target=False,
                     diurnal_excluded_cols=None):
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

    assert df.index.is_monotonic_increasing, (
        f"Index must be sorted before diurnal transform — "
        f"first offender at position {(df.index.to_series().diff() < 0).argmax()}"
    )

    # 1. Diurnal adjustment
    do_diurnal = use_diurnal and (col_name not in diurnal_excluded_cols)
    if do_diurnal:
        series, baseline = diurnal_adjust(
            series, df[time_col], has_negatives, diurnal_window, min_periods
        )
    else:
        baseline = pd.Series(1.0, index=df.index)

    # 2. Data-driven transform
    if use_transform:
        series = apply_data_transform(series, col_name, has_negatives, allow_missing)

    # 3. Winsorization
    if winsor_window is not None:
        series = rolling_winsorize(series, winsor_window, allow_missing, is_target)

    return series, baseline


def load_and_clean_base_data(hparams: dict, input_path: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Handles stitching, gridding, trimming, and base log-diurnal transforms.
    Returns a clean DataFrame ready for HAR feature engineering.

    hparams keys
    ------------
    use_transform_exog : bool  — apply data-driven transform to exog features (False for trees)
    use_diurnal        : bool  — diurnal-adjust exog features (False for trees)
    allow_missing      : bool  — NaN passthrough for XGBoost/LightGBM
    use_winsor         : bool  — winsorize features (False for XGBoost/LightGBM)
    winsor_window      : int   — rolling window for winsorization (default 240)
    """
    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.endswith('.parquet')]
        dataframes = [pd.read_parquet(os.path.join(input_path, f), engine="pyarrow") for f in files]
        data = reduce(lambda left, right: pd.merge(left, right, on='endbartime', how='outer'), dataframes)
    else:
        data = pd.read_parquet(input_path, engine="pyarrow")

    if 'endbartime' in data.columns:
        data = data.rename(columns={'endbartime': 't', 'sumret2': 'RV'})

    data['t'] = pd.to_datetime(data['t'])
    if data['t'].duplicated().any():
        data = data.drop_duplicates(subset=['t'], keep='last')

    end_date = data['t'].max().date()
    full_grid = pd.date_range(start=f"{config.START_DATE} 00:00", end=f"{end_date} 23:30", freq="30min")
    if len(full_grid) == 0:
        return pd.DataFrame(), []

    data = data.set_index('t').reindex(full_grid)
    data.index.name = 't'
    data = data.reset_index()

    # Drop target weekend hours and pre-start dates
    mask_friday_night   = (data['t'].dt.dayofweek == 4) & (data['t'].dt.time > pd.to_datetime("20:00").time())
    mask_saturday       = data['t'].dt.dayofweek == 5
    mask_sunday_morning = (data['t'].dt.dayofweek == 6) & (data['t'].dt.time < pd.to_datetime("18:30").time())
    mask_pre_start      = data['t'] < config.START_DATE
    data = data[~(mask_friday_night | mask_saturday | mask_sunday_morning | mask_pre_start)]

    exog_col_names = []
    if hparams.get("exog_cols") and str(hparams["exog_cols"]).lower() != "none":
        sep = '|' if '|' in hparams["exog_cols"] else ','
        exog_col_names = [c.strip() for c in hparams["exog_cols"].split(sep) if c.strip() in data.columns]
        for col in exog_col_names:
            if any(vix in col.lower() for vix in {'vvix', 'vix3m'}):
                data[col] = pd.to_numeric(data[col], errors='coerce')
            else:
                assert data[col].dtype != object, f"Unexpected object dtype on {col}"

    # --- Circuit Breaker Handling ---
    cb_dates = pd.to_datetime(['2020-03-09', '2020-03-12', '2020-03-16', '2020-03-18']).date
    mask_cb = data['t'].dt.date.isin(cb_dates) & (data['RV'] == 0.0)
    data.loc[mask_cb, 'RV'] = data['RV'].copy().where(~mask_cb).ffill()

    # --- Circuit Breaker Date Drop for Non-Moments Features ---
    def _is_moments_col(col):
        return col.startswith("sum") and "stock" not in col and "volume" not in col

    non_moments_exog = [c for c in exog_col_names if not _is_moments_col(c)]
    if non_moments_exog:
        mask_cb_rows = data['t'].dt.date.isin(cb_dates)
        n_dropped = mask_cb_rows.sum()
        if n_dropped:
            print(f"  [CB Drop] Dropping {n_dropped} rows on circuit-breaker dates "
                  f"(non-moments cols present: {non_moments_exog[:3]}{'...' if len(non_moments_exog) > 3 else ''})")
        data = data[~mask_cb_rows].reset_index(drop=True)
        hparams['cb_drop'] = True
    else:
        hparams['cb_drop'] = False

    # --- Read toggles ---
    allow_missing      = hparams.get('allow_missing', False)
    use_transform_exog = hparams.get('use_transform_exog', True)
    use_diurnal        = hparams.get('use_diurnal', True)
    use_winsor         = hparams.get('use_winsor', True)
    winsor_window      = hparams.get('winsor_window', 240) if use_winsor else None

    data['time_of_day'] = data['t'].dt.time

    # --- 1. Target: always clean, always transform + diurnal, model-invariant ---
    data['RV'] = data['RV'].ffill()
    data = data.dropna(subset=['RV'])

    data['adj_RV'], data['baseline_RV'] = robust_transform(
        data, 'RV', 'time_of_day',
        use_transform=True,
        use_diurnal=True,
        allow_missing=False,
        winsor_window=winsor_window,
        is_target=True,
    )
    cols_to_transform = ['adj_RV']

    if not exog_col_names:
        return data, cols_to_transform

    # --- 2. Overnight raw fills on exog before dropna ---
    # Fill overnight NaN slots with the per-slot rolling mean so robust_transform
    # sees no NaNs in those windows and produces identity (1.0) naturally after
    # diurnal division — no post-transform fill needed.
    OVERNIGHT_WINDOWS = {
        'ewstock':   ('20:30', '04:00'),
        'vwstock':   ('20:30', '04:00'),
        'voldemand': ('17:00', '10:00'),
    }

    for raw_col in exog_col_names:
        overnight_key = next((kw for kw in OVERNIGHT_WINDOWS if kw in raw_col), None)
        if overnight_key is None:
            continue

        t_start = pd.to_datetime(OVERNIGHT_WINDOWS[overnight_key][0]).time()
        t_end   = pd.to_datetime(OVERNIGHT_WINDOWS[overnight_key][1]).time()
        tod     = data['t'].dt.time
        in_overnight = (tod >= t_start) | (tod < t_end) if t_start > t_end \
                       else (tod >= t_start) & (tod < t_end)

        fill_mask = in_overnight & data[raw_col].isna()
        data.loc[fill_mask, raw_col] = 1.0

    # --- 3. Drop remaining NaN rows (pre-recording, intraday, vix pending) ---
    if not allow_missing:
        data = data.dropna(subset=exog_col_names)

    # --- 4. Transform exog features ---
    # Diurnal exclusions (sentiment, vix) are handled inside robust_transform
    # via diurnal_excluded_cols. Per-col diurnal flag only carries the global switch.
    for raw_col in exog_col_names:
        adj_col = f"adj_{raw_col}"

        data[adj_col], _ = robust_transform(
            data, raw_col, 'time_of_day',
            use_transform=use_transform_exog,
            allow_missing=allow_missing,
            use_diurnal=use_diurnal,
            winsor_window=winsor_window,
            is_target=False,
        )

        cols_to_transform.append(adj_col)

    return data, cols_to_transform