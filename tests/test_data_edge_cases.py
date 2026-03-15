"""Tests for edge cases in src/data.py."""

import numpy as np
import pandas as pd

from src.data import _resolve_lags, apply_data_transform


class TestApplyDataTransformEdgeCases:
    def test_ret3_cube_root(self):
        s = pd.Series([8.0, 27.0])
        result = apply_data_transform(s, "ret3_col", has_negatives=False, allow_missing=False)
        np.testing.assert_allclose(result.values, [2.0, 3.0])

    def test_ret4_fourth_root(self):
        s = pd.Series([16.0, 81.0])
        result = apply_data_transform(s, "ret4_col", has_negatives=False, allow_missing=False)
        np.testing.assert_allclose(result.values, [2.0, 3.0])

    def test_has_negatives_fillna(self):
        s = pd.Series([1.0, np.nan, 3.0])
        result = apply_data_transform(s, "signed_col", has_negatives=True, allow_missing=False)
        assert result.iloc[1] == 0.0

    def test_has_negatives_allow_missing(self):
        s = pd.Series([1.0, np.nan, 3.0])
        result = apply_data_transform(s, "signed_col", has_negatives=True, allow_missing=True)
        assert np.isnan(result.iloc[1])


class TestResolveLags:
    def test_har_geometric_sequence(self):
        lags = _resolve_lags("har", 125)
        assert lags == [1, 5, 25, 125]

    def test_har_partial(self):
        lags = _resolve_lags("har", 10)
        assert lags == [1, 5]

    def test_raw_consecutive(self):
        lags = _resolve_lags("raw", 5)
        assert lags == [1, 2, 3, 4, 5]
