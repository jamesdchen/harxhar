import pandas as pd
import numpy as np
import os
from functools import reduce
from src import config
from typing import Literal

def robust_transform(df, col_name, time_col="time_of_day",
                     diurnal_window=config.DIURNAL_WINDOW,
                     min_periods=config.DIURNAL_MIN_PERIODS,
                     use_transform=True, allow_missing=False,
                     use_diurnal=True, winsor_window=None,
                     is_target=False,
                     diurnal_excluded_cols=None):
    """
    Applies diurnal adjustment, data-driven transform, then winsorization.

    Pipeline order:
        1. Diurnal adjustment  (on raw series)
           - signed vars (has_negatives): divide by rolling per-slot std
           - non-negative vars:           divide by rolling per-slot mean
        2. Transform           (determined by column type and data sign)
           - signed vars       : no further transform (diurnal already normalised)
           - *ret2* / *turnover*: sqrt
           - *ret4*            : fourth root
           - everything else   : log  (default)

    Parameters
    ----------
    diurnal_excluded_cols : set[str] or None
        Columns that must NOT be diurnally adjusted (e.g. structural integer
        features). Defaults to SKIP_VARS when None.
    is_target : bool
        If True: RV is strictly positive; clip to 1e-10 before log.
        Winsorization always uses standard quantile (model-invariant).
    winsor_window : int or None
        If None, winsorization is skipped.
    use_transform : bool
        Master switch for the transform step. If False, the data-driven
        transform (log, sqrt, fourth root, std) is skipped entirely.
    use_diurnal : bool
        Master switch for the diurnal step.
    allow_missing : bool
        If True, NaNs pass through for XGBoost native handling.
        If False, NaNs are filled with 0.0 after all steps.
    """
    SKIP_VARS = {'hour', 'DOW', 't', 'date'}

    if diurnal_excluded_cols is None:
        diurnal_excluded_cols = SKIP_VARS | {'vix', 'sentiment'}

    if col_name in SKIP_VARS:
        return df[col_name], pd.Series(0, index=df.index)

    series = pd.to_numeric(df[col_name], errors='coerce')

    # Detect sign on raw series before any adjustment.
    has_negatives = bool((series.dropna() < 0).any())

    # ------------------------------------------------------------------ #
    # 1. DIURNAL ADJUSTMENT (on raw series)
    # ------------------------------------------------------------------ #
    do_diurnal = use_diurnal and (col_name not in diurnal_excluded_cols)

    if do_diurnal:
        baseline = pd.Series(index=series.index, dtype=float)
        for slot, idx in df[time_col].groupby(df[time_col]).groups.items():
            slot_series = series.loc[idx].sort_index()
            if not has_negatives:
                rolled = slot_series.rolling(window=diurnal_window, min_periods=min_periods).mean().shift(1)
            else:
                rolled = slot_series.rolling(window=diurnal_window, min_periods=min_periods).std().shift(1)
            baseline.loc[idx] = rolled
        baseline = baseline.fillna(1.0)
        series = series / baseline
    else:
        baseline = pd.Series(1.0, index=df.index)

    # ------------------------------------------------------------------ #
    # 2. DATA-DRIVEN TRANSFORM
    # ------------------------------------------------------------------ #
    if use_transform:
        def _col_matches(*keywords):
            return any(kw in col_name for kw in keywords)

        if has_negatives:
            # Diurnal std-normalisation already handled the scale; no further
            # transform needed for signed residuals.
            if not allow_missing:
                series = series.fillna(0.0)

        elif _col_matches('ret2', 'sumret2', 'sumpret2', 'sumabsret', 'turnover'):
            series = np.sqrt(series)

        elif _col_matches('ret4', 'sumret4'):
            series = np.power(series, 0.25)

        else:
            # Default: log transform.
            # Zeros and negatives are already excluded by the has_negatives branch
            # and the diurnal step, so no clipping needed here.
            series = np.log(series)

    # ------------------------------------------------------------------ #
    # 3. WINSORIZATION (on transformed series)
    # ------------------------------------------------------------------ #
    if winsor_window is not None:
        if allow_missing and not is_target:
            lower = series.rolling(window=winsor_window, min_periods=1).apply(
                lambda x: np.nanquantile(x, 0.01), raw=True
            )
            upper = series.rolling(window=winsor_window, min_periods=1).apply(
                lambda x: np.nanquantile(x, 0.99), raw=True
            )
        else:
            lower = series.rolling(window=winsor_window, min_periods=1).quantile(0.01)
            upper = series.rolling(window=winsor_window, min_periods=1).quantile(0.99)
        series = series.clip(lower=lower, upper=upper)

    return series, baseline


def load_and_clean_base_data(hparams, input_path):
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
            data[col] = pd.to_numeric(data[col], errors='coerce')

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