"""CLI argument parsing, execution, and HPC submission."""

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

from projects.ml.cli.executor import add_feature_args as add_feature_args
from projects.ml.cli.executor import get_common_hparams as get_common_hparams
from projects.ml.cli.executor import get_common_parser as get_common_parser
from projects.ml.cli.executor import main as main
from projects.ml.cli.submit import ExperimentSpec as ExperimentSpec
from projects.ml.cli.submit import add_common_submit_args as add_common_submit_args
from projects.ml.cli.submit import build_extra_args as build_extra_args
from projects.ml.cli.submit import submit_experiment as submit_experiment
from projects.ml.cli.submit import submit_experiment_batch as submit_experiment_batch
