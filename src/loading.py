# Auto-generated from notebooks/01_loading.ipynb. Do not edit by hand.

"""Standalone data loading module for volatility forecasting.

Loads raw parquet data, builds a 30-min grid, filters market hours,
and returns a clean DataFrame ready for downstream pipelines.

No imports from core/ or projects/ — only numpy, pandas, os, functools.
"""

from __future__ import annotations

import os
from functools import reduce

import pandas as pd

# ── Overnight fill windows (start, end) — wrap-around midnight ────────
OVERNIGHT_WINDOWS: dict[str, tuple[str, str]] = {
    "ewstock": ("20:30", "04:00"),
    "vwstock": ("20:30", "04:00"),
    "voldemand": ("17:00", "10:00"),
}


def parse_exog_cols(exog_str: str | None) -> list[str]:
    """Parse a pipe-separated exog column string into a list."""
    if not exog_str or exog_str.lower() == "none":
        return []
    return [c.strip() for c in exog_str.split("|") if c.strip()]


def apply_overnight_fills(df: pd.DataFrame, exog_cols: list[str]) -> None:
    """Fill NaN with 1.0 during overnight windows for specific exog columns."""
    tod = df["t"].dt.time
    for col in exog_cols:
        if col not in df.columns:
            continue
        name_lower = col.lower()
        overnight_key = next((kw for kw in OVERNIGHT_WINDOWS if kw in name_lower), None)
        if overnight_key is None:
            continue
        t_start = pd.Timestamp(f"1900-01-01 {OVERNIGHT_WINDOWS[overnight_key][0]}").time()
        t_end = pd.Timestamp(f"1900-01-01 {OVERNIGHT_WINDOWS[overnight_key][1]}").time()
        # Wrap-around midnight: time >= start OR time < end
        mask = (tod >= t_start) | (tod < t_end)
        df.loc[mask & df[col].isna(), col] = 1.0


# ── Feature groups ────────────────────────────────────────────────────
# Single source of truth for exogenous variable subgroup membership.
# Predicates encode the intent; lists are materialized at import time
# for cheap inspection. If new columns are added to the parquet data,
# extend ALL_FEATURES and verify the derived lists via the assertion.

ALL_FEATURES: list[str] = [
    "sumret",
    "sumabsret",
    "sumret3",
    "sumret4",
    "sumpret2",
    "sumbipow",
    "sumautocov",
    "sumvolume",
    "numobs",
    "sumret2_ewstock",
    "sumret3_ewstock",
    "sumret4_ewstock",
    "sumabsret_ewstock",
    "sumbipow_ewstock",
    "sumpret2_ewstock",
    "turnover_ewstock",
    "buyturnover_ewstock",
    "sellturnover_ewstock",
    "effspread_ewstock",
    "spread_ewstock",
    "sumret2_vwstock",
    "sumret3_vwstock",
    "sumret4_vwstock",
    "sumabsret_vwstock",
    "sumbipow_vwstock",
    "sumpret2_vwstock",
    "turnover_vwstock",
    "buyturnover_vwstock",
    "sellturnover_vwstock",
    "effspread_vwstock",
    "spread_vwstock",
    "stocktwits_attention",
    "stocktwits_sentiment",
    "stocktwits_sentcount",
    "vix",
    "vvix",
    "vix3m",
    "voldemand_spx_open_and_close",
    "voldemand_spx_open_only",
    "voldemand_all_open_and_close",
    "voldemand_all_open_only",
]


def _is_moment(f: str) -> bool:
    return f.startswith("sum") and "stock" not in f and "volume" not in f


def _is_liquidity(f: str) -> bool:
    return any(x in f for x in ("volume", "turnover", "spread", "numobs"))


SUBGROUPS: dict[str, list[str]] = {
    "baseline": [],
    "moments": [f for f in ALL_FEATURES if _is_moment(f)],
    "liquidity": [f for f in ALL_FEATURES if _is_liquidity(f)],
    "market_ew": [f for f in ALL_FEATURES if "ewstock" in f and not any(x in f for x in ("turnover", "spread"))],
    "market_vw": [f for f in ALL_FEATURES if "vwstock" in f and not any(x in f for x in ("turnover", "spread"))],
    "sentiment": [f for f in ALL_FEATURES if "stocktwits" in f],
    "implied_vol": [f for f in ALL_FEATURES if "vix" in f],
    "vol_demand": [f for f in ALL_FEATURES if "voldemand" in f],
    "all_features": ALL_FEATURES,
}


def get_bucket(name: str) -> list[str]:
    if name not in SUBGROUPS:
        raise KeyError(f"Unknown subgroup '{name}'. Valid: {sorted(SUBGROUPS.keys())}")
    return SUBGROUPS[name]


# ── Constants ──────────────────────────────────────────────────────────
START_DATE = "2005-01-01"
FRIDAY_CLOSE = "20:00"
SUNDAY_OPEN = "18:30"
FREQ = "30min"


def load_raw_data(data_path: str, allow_missing: bool = False) -> pd.DataFrame:
    """Load parquet data, grid to 30-min, filter market hours, clean NaNs.

    Parameters
    ----------
    data_path : str
        Path to a directory of .parquet files or a single .parquet file.
    allow_missing : bool
        If False (default), drop all rows with any remaining NaN after
        forward-filling the target column. If True, keep them.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns including ``t``, ``RV``,
        ``time_of_day``, and any additional feature columns from the
        parquet files.
    """
    # ── 1. Load parquet file(s) ────────────────────────────────────────
    if os.path.isfile(data_path):
        frames = [pd.read_parquet(data_path)]
    else:
        parquet_files = sorted(f for f in os.listdir(data_path) if f.endswith(".parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {data_path}")
        frames = [pd.read_parquet(os.path.join(data_path, f)) for f in parquet_files]

    # ── 2. Merge on endbartime (outer join) ────────────────────────────
    if len(frames) == 1:
        df = frames[0]
    else:
        df = reduce(
            lambda left, right: pd.merge(left, right, on="endbartime", how="outer"),
            frames,
        )

    # ── 3. Rename columns ─────────────────────────────────────────────
    rename_map = {}
    if "endbartime" in df.columns:
        rename_map["endbartime"] = "t"
    if "sumret2" in df.columns:
        rename_map["sumret2"] = "RV"
    df = df.rename(columns=rename_map)

    # ── 4. Convert t to datetime, drop duplicates ─────────────────────
    df["t"] = pd.to_datetime(df["t"])
    df = df.drop_duplicates(subset="t")

    # ── 5. Create 30-min grid and reindex ─────────────────────────────
    end_date = df["t"].max()
    grid = pd.date_range(start=START_DATE, end=end_date, freq=FREQ)
    df = df.set_index("t").reindex(grid).rename_axis("t").reset_index()

    # ── 6. Filter out market-closed hours ─────────────────────────────
    day_of_week = df["t"].dt.dayofweek  # Mon=0 … Sun=6
    time_of_day = df["t"].dt.time

    friday_close = pd.Timestamp(f"1900-01-01 {FRIDAY_CLOSE}").time()
    sunday_open = pd.Timestamp(f"1900-01-01 {SUNDAY_OPEN}").time()

    mask_friday_after_close = (day_of_week == 4) & (time_of_day > friday_close)
    mask_saturday = day_of_week == 5
    mask_sunday_before_open = (day_of_week == 6) & (time_of_day < sunday_open)

    closed_mask = mask_friday_after_close | mask_saturday | mask_sunday_before_open
    df = df[~closed_mask].reset_index(drop=True)

    # ── 7. Forward-fill RV, drop rows where RV is still NaN ──────────
    df["RV"] = df["RV"].ffill()
    df = df.dropna(subset=["RV"]).reset_index(drop=True)

    # ── 8. NaN handling for remaining columns ─────────────────────────
    if not allow_missing:
        df = df.dropna().reset_index(drop=True)

    # ── 9. Add time_of_day column ─────────────────────────────────────
    df["time_of_day"] = df["t"].dt.time

    return df
