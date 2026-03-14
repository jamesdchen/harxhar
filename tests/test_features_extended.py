"""Tests for PCATransform in src/features.py."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.features import PCATransform


class TestPCATransform:
    def test_fit_reduces_dimensions(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        pca = PCATransform(n_components=3)
        pca.fit(X)
        result = pca.transform(X)
        assert result.shape == (100, 3)

    def test_explained_variance_decreasing(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        pca = PCATransform(n_components=5)
        pca.fit(X)
        ev = pca.pca.explained_variance_
        assert np.all(ev[:-1] >= ev[1:])

    def test_transform_without_fit_raises(self):
        pca = PCATransform(n_components=3)
        X = np.random.randn(10, 5)
        with pytest.raises((NotFittedError, AttributeError)):
            pca.transform(X)

    def test_single_component(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        pca = PCATransform(n_components=1)
        pca.fit(X)
        result = pca.transform(X)
        assert result.shape == (50, 1)

    def test_roundtrip_low_rank(self):
        """PCA(n_components=2) on rank-2 data should capture nearly all variance."""
        rng = np.random.RandomState(42)
        basis = rng.randn(2, 5)
        coeffs = rng.randn(100, 2)
        X = coeffs @ basis  # rank-2 in 5D
        pca = PCATransform(n_components=2)
        pca.fit(X)
        assert sum(pca.pca.explained_variance_ratio_) > 0.99
