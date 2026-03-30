"""CLI argument parsing and execution.

Imports are lazy so that lightweight code paths do not pull in heavy
compute dependencies (numpy, tqdm, torch, …) that live behind
``executor.py``.
"""

__all__ = [
    "add_feature_args",
    "get_common_hparams",
    "get_common_parser",
    "main",
]


def __getattr__(name: str):
    _executor_names = {"add_feature_args", "get_common_hparams", "get_common_parser", "main"}
    if name in _executor_names:
        from projects.ml.cli import executor

        return getattr(executor, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
