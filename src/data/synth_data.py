"""Moving Block Bootstrap for time series data augmentation."""

from __future__ import annotations

import numpy as np


class MovingBlockBootstrap:
    """Generate synthetic time series by sampling and concatenating overlapping blocks.

    The MBB preserves local temporal dependencies (autocorrelation, diurnal
    patterns) within each block while breaking long-range dependence between
    blocks — suitable for augmenting stationary or weakly-dependent series.

    Parameters
    ----------
    data : np.ndarray
        1-D or 2-D source array. For 2-D input, blocks are sampled along axis 0.
    block_size : int
        Length of each contiguous block (default 48, one trading day of 30-min bars).
    """

    def __init__(self, data: np.ndarray, block_size: int = 48):
        if data.ndim not in (1, 2):
            raise ValueError(f"data must be 1-D or 2-D, got {data.ndim}-D")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if block_size >= len(data):
            raise ValueError(f"block_size ({block_size}) must be < data length ({len(data)})")
        self.data = data
        self.block_size = block_size
        self._max_start = len(data) - block_size

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """Generate a synthetic dataset by randomly sampling and concatenating blocks.

        Parameters
        ----------
        n_samples : int
            Desired length of the output along axis 0.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Synthetic array of shape ``(n_samples,)`` (1-D input) or
            ``(n_samples, D)`` (2-D input).
        """
        rng = np.random.default_rng(seed)
        blocks: list[np.ndarray] = []
        total = 0
        while total < n_samples:
            start = rng.integers(0, self._max_start + 1)
            blocks.append(self.data[start : start + self.block_size])
            total += self.block_size
        return np.concatenate(blocks)[:n_samples]
