"""Shared fixtures for harxhar-ml tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_data(tmp_path):
    """Create a minimal synthetic parquet for pipeline testing."""
    rng = np.random.RandomState(42)
    n = 5000
    dates = pd.date_range("2006-01-01", periods=n, freq="30min")
    # Filter out weekends like the pipeline does
    mask = ~((dates.dayofweek == 5) | (dates.dayofweek == 6))
    dates = dates[mask][:3000]
    n = len(dates)

    df = pd.DataFrame(
        {
            "endbartime": dates,
            "sumret2": np.abs(rng.randn(n)) * 0.001 + 0.0001,
            "hour": dates.hour,
            "DOW": dates.dayofweek,
        }
    )
    path = tmp_path / "test.parquet"
    df.to_parquet(path, engine="pyarrow")
    return str(path)
