"""Base data loading and cleaning."""

from __future__ import annotations

import os
import warnings
from functools import reduce

import pandas as pd

from src import config
from src.data.transforms import robust_transform
from src.log import get_logger

logger = get_logger(__name__)


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
        files = [f for f in os.listdir(input_path) if f.endswith(".parquet")]
        dataframes = [pd.read_parquet(os.path.join(input_path, f), engine="pyarrow") for f in files]
        data = reduce(lambda left, right: pd.merge(left, right, on="endbartime", how="outer"), dataframes)
    else:
        data = pd.read_parquet(input_path, engine="pyarrow")

    if "endbartime" in data.columns:
        data = data.rename(columns={"endbartime": "t", "sumret2": "RV"})

    data["t"] = pd.to_datetime(data["t"])
    if data["t"].duplicated().any():
        data = data.drop_duplicates(subset=["t"], keep="last")

    end_date = data["t"].max().date()
    full_grid = pd.date_range(start=f"{config.START_DATE} 00:00", end=f"{end_date} 23:30", freq="30min")
    if len(full_grid) == 0:
        return pd.DataFrame(), []

    data = data.set_index("t").reindex(full_grid)
    data.index.name = "t"
    data = data.reset_index()

    # Drop target weekend hours and pre-start dates
    mask_friday_night = (data["t"].dt.dayofweek == 4) & (data["t"].dt.time > pd.to_datetime("20:00").time())
    mask_saturday = data["t"].dt.dayofweek == 5
    mask_sunday_morning = (data["t"].dt.dayofweek == 6) & (data["t"].dt.time < pd.to_datetime("18:30").time())
    mask_pre_start = data["t"] < config.START_DATE
    data = data[~(mask_friday_night | mask_saturday | mask_sunday_morning | mask_pre_start)]

    exog_col_names = []
    if hparams.get("exog_cols") and str(hparams["exog_cols"]).lower() != "none":
        sep = "|" if "|" in hparams["exog_cols"] else ","
        exog_col_names = [c.strip() for c in hparams["exog_cols"].split(sep) if c.strip() in data.columns]
        for col in exog_col_names:
            if any(vix in col.lower() for vix in {"vvix", "vix3m"}):
                data[col] = pd.to_numeric(data[col], errors="coerce")
            else:
                if data[col].dtype == object:
                    raise ValueError(f"Unexpected object dtype on {col}")

    # --- Circuit Breaker Handling (market hours only: 09:30–16:00) ---
    cb_dates = pd.to_datetime(config.CIRCUIT_BREAKER_DATES).date
    market_open = pd.to_datetime("09:30").time()
    market_close = pd.to_datetime("16:00").time()
    tod = data["t"].dt.time
    in_market_hours = (tod >= market_open) & (tod <= market_close)
    mask_cb = data["t"].dt.date.isin(cb_dates) & (data["RV"] == 0.0) & in_market_hours
    data.loc[mask_cb, "RV"] = data["RV"].copy().where(~mask_cb).ffill()
    remaining_zeros = data["t"].dt.date.isin(cb_dates) & (data["RV"] == 0.0) & in_market_hours
    if remaining_zeros.any():
        zero_times = data.loc[remaining_zeros, "t"].dt.strftime("%Y-%m-%d %H:%M").tolist()
        warnings.warn(
            f"Circuit breaker ffill left {remaining_zeros.sum()} zero RV values at: {zero_times}",
            stacklevel=2,
        )

    # --- Circuit Breaker Date Drop for Non-Moments Features ---
    def _is_moments_col(col):
        return col.startswith("sum") and "stock" not in col and "volume" not in col

    non_moments_exog = [c for c in exog_col_names if not _is_moments_col(c)]
    if non_moments_exog:
        mask_cb_rows = data["t"].dt.date.isin(cb_dates)
        n_dropped = mask_cb_rows.sum()
        if n_dropped:
            logger.info(
                "[CB Drop] Dropping %d rows on circuit-breaker dates (non-moments cols present: %s%s)",
                n_dropped,
                non_moments_exog[:3],
                "..." if len(non_moments_exog) > 3 else "",
            )
        data = data[~mask_cb_rows].reset_index(drop=True)
        hparams["cb_drop"] = True
    else:
        hparams["cb_drop"] = False

    # --- Read toggles ---
    allow_missing = hparams.get("allow_missing", False)
    use_transform_exog = hparams.get("use_transform_exog", True)
    use_diurnal = hparams.get("use_diurnal", True)
    use_winsor = hparams.get("use_winsor", True)
    winsor_window = hparams.get("winsor_window", 240) if use_winsor else None

    data["time_of_day"] = data["t"].dt.time

    # --- 1. Target: always clean, always transform + diurnal, model-invariant ---
    data["RV"] = data["RV"].ffill()
    data = data.dropna(subset=["RV"])

    data["adj_RV"], data["baseline_RV"] = robust_transform(
        data,
        "RV",
        "time_of_day",
        use_transform=True,
        use_diurnal=True,
        allow_missing=False,
        winsor_window=winsor_window,
        is_target=True,
    )
    cols_to_transform = ["adj_RV"]

    if not exog_col_names:
        return data, cols_to_transform

    # --- 2. Overnight raw fills on exog before dropna ---
    OVERNIGHT_WINDOWS = {
        "ewstock": ("20:30", "04:00"),
        "vwstock": ("20:30", "04:00"),
        "voldemand": ("17:00", "10:00"),
    }

    for raw_col in exog_col_names:
        overnight_key = next((kw for kw in OVERNIGHT_WINDOWS if kw in raw_col), None)
        if overnight_key is None:
            continue

        t_start = pd.to_datetime(OVERNIGHT_WINDOWS[overnight_key][0]).time()
        t_end = pd.to_datetime(OVERNIGHT_WINDOWS[overnight_key][1]).time()
        tod = data["t"].dt.time
        in_overnight = (tod >= t_start) | (tod < t_end) if t_start > t_end else (tod >= t_start) & (tod < t_end)

        fill_mask = in_overnight & data[raw_col].isna()
        data.loc[fill_mask, raw_col] = 1.0

    # --- 3. Drop remaining NaN rows (pre-recording, intraday, vix pending) ---
    if not allow_missing:
        data = data.dropna(subset=exog_col_names)

    # --- 4. Transform exog features ---
    for raw_col in exog_col_names:
        adj_col = f"adj_{raw_col}"

        data[adj_col], _ = robust_transform(
            data,
            raw_col,
            "time_of_day",
            use_transform=use_transform_exog,
            allow_missing=allow_missing,
            use_diurnal=use_diurnal,
            winsor_window=winsor_window,
            is_target=False,
        )

        cols_to_transform.append(adj_col)

    return data, cols_to_transform
