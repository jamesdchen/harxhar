"""DL-specific data utilities.

- MovingBlockBootstrap — synthetic time series generation by randomly sampling
  contiguous blocks (default 48 = one trading day) from source data.  Used for
  data augmentation in scaling-law experiments.
"""

__all__ = ["MovingBlockBootstrap"]


def __getattr__(name: str) -> object:
    if name == "MovingBlockBootstrap":
        from projects.dl.data.synth_data import MovingBlockBootstrap

        return MovingBlockBootstrap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
