"""CLI utilities shared across projects.

- metadata — git hash, timestamp, and experiment config tracking
- common — shared argument parser helpers for chunking and backtest CLIs
"""

__all__ = [
    "add_chunking_args",
    "build_metadata",
    "load_metadata",
    "save_metadata",
]


def __getattr__(name: str) -> object:
    _metadata_names = {"build_metadata", "save_metadata", "load_metadata"}
    if name in _metadata_names:
        from core.cli import metadata

        return getattr(metadata, name)

    if name == "add_chunking_args":
        from core.cli.common import add_chunking_args

        return add_chunking_args

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
