"""Standalone data loading module for volatility forecasting.

Loads raw parquet data, builds a 30-min grid, filters market hours,
and returns a clean DataFrame ready for downstream pipelines.

No imports from core/ or projects/ — only numpy, pandas, os, functools.
"""

import os
from functools import reduce

import numpy as np
import pandas as pd

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
        parquet_files = sorted(
            f for f in os.listdir(data_path) if f.endswith(".parquet")
        )
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {data_path}")
        frames = [
            pd.read_parquet(os.path.join(data_path, f)) for f in parquet_files
        ]

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
