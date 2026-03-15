"""CLI argument parsing, execution, and SLURM submission."""

from src.cli.executor import (
    add_feature_args as add_feature_args,
)
from src.cli.executor import (
    get_common_hparams as get_common_hparams,
)
from src.cli.executor import (
    get_common_parser as get_common_parser,
)
from src.cli.executor import (
    main as main,
)
from src.cli.submit import (
    ExperimentSpec as ExperimentSpec,
)
from src.cli.submit import (
    add_common_submit_args as add_common_submit_args,
)
from src.cli.submit import (
    build_extra_args as build_extra_args,
)
from src.cli.submit import (
    submit_experiment as submit_experiment,
)
from src.cli.submit import (
    submit_experiment_batch as submit_experiment_batch,
)
