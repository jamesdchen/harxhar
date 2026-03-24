"""CLI argument parsing, execution, and HPC submission.

Imports are lazy so that lightweight submit-only code paths (e.g.
``scripts/submit.py``) do not pull in heavy compute dependencies
(numpy, tqdm, torch, …) that live behind ``executor.py``.
"""

__all__ = [
    "add_feature_args",
    "get_common_hparams",
    "get_common_parser",
    "main",
    "ExperimentSpec",
    "add_common_submit_args",
    "build_extra_args",
    "submit_experiment",
    "submit_experiment_batch",
]


def __getattr__(name: str):
    # Executor (heavy) imports
    _executor_names = {"add_feature_args", "get_common_hparams", "get_common_parser", "main"}
    if name in _executor_names:
        from projects.ml.cli import executor
        return getattr(executor, name)

    # Submit (light) imports
    _submit_names = {
        "ExperimentSpec", "add_common_submit_args", "build_extra_args",
        "submit_experiment", "submit_experiment_batch",
    }
    if name in _submit_names:
        from projects.ml.cli import submit
        return getattr(submit, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
