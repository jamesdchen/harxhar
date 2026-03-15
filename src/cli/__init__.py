"""CLI argument parsing, execution, and SLURM submission."""

from src.cli.executor import (
    get_common_parser,
    get_common_hparams,
    add_feature_args,
    main,
)
from src.cli.submit import (
    ExperimentSpec,
    submit_experiment,
    submit_experiment_batch,
    add_common_submit_args,
    build_extra_args,
)
