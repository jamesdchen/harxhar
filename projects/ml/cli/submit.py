"""
Shared experiment submission utilities.

Handles config.txt creation, env var construction, and array job submission.
All submission scripts delegate mechanical work here.  The actual scheduler
interaction is handled by pluggable backends (see projects.ml.cli.backends).
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from core.core.config import DEFAULT_RESULTS_DIR
from core.core.log import get_logger
from projects.ml.cli.backends import HPCBackend, get_backend
from projects.ml.cli.executor import add_feature_args
from projects.ml.cli.metadata import build_metadata, save_metadata

logger = get_logger(__name__)

# Resolve paths relative to the ml package root (four levels up from src/projects.ml/cli/submit.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TASKS_PER_ARRAY = 100
DEFAULT_TOTAL_CHUNKS = 100


@dataclasses.dataclass
class ExperimentSpec:
    exp_id: int
    exp_name: str
    model_type: str
    feature_type: str = "har"
    variables: list = dataclasses.field(default_factory=list)
    extra_args: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def write_config(exp_dir: str, spec: ExperimentSpec) -> None:
    """Write config.txt matching the format parse_config() in eval_utils.py expects."""
    with open(Path(exp_dir) / "config.txt", "w") as f:
        f.write(f"Experiment ID: {spec.exp_id}\n")
        f.write(f"Experiment Name: {spec.exp_name}\n")
        f.write(f"Model Type: {spec.model_type}\n")
        f.write(f"Features: {spec.feature_type}\n")
        f.write(f"Num Variables: {len(spec.variables)}\n")
        f.write(f"Variables: {spec.variables}\n")
        if spec.extra_args:
            f.write(f"Extra Args: {spec.extra_args}\n")


def short_model_name(model_type: str) -> str:
    mapping = {
        "xgboost": "xgb",
        "lightgbm": "lgb",
        "random_forest": "rf",
    }
    return mapping.get(model_type, model_type[:3])


def build_job_env(spec: ExperimentSpec, exp_dir: str, total_chunks: int) -> dict[str, str]:
    """Build the env dict that submit_carc.slurm expects."""
    env = os.environ.copy()
    env["TOTAL_CHUNKS"] = str(total_chunks)
    env["EXOG_COLS"] = "|".join(spec.variables) if spec.variables else "None"
    env["RESULT_DIR"] = exp_dir
    env["MODEL_TYPE"] = spec.model_type
    env["EXTRA_ARGS"] = spec.extra_args
    return env


def submit_experiment(
    spec,
    base_dir,
    total_chunks,
    tasks_per_array=DEFAULT_TASKS_PER_ARRAY,
    backend: HPCBackend | None = None,
):
    """Submit a single experiment: mkdir, write config, sbatch.

    If the experiment directory already has a ``.submitted`` marker,
    the experiment is skipped so that re-running is safe after a partial
    failure (e.g. hitting QOS limits).
    """
    dir_name = f"exp_{spec.exp_id}_{spec.model_type}_{spec.feature_type}_{spec.exp_name}"
    if spec.model_type == "naive":
        dir_name = f"exp_{spec.exp_id}_naive_baseline"
    exp_dir = str(Path(base_dir).resolve() / dir_name)

    submitted_marker = Path(exp_dir) / ".submitted"
    if submitted_marker.exists():
        logger.info("Skipping ID %d (already submitted): %s", spec.exp_id, dir_name)
        return exp_dir

    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    write_config(exp_dir, spec)
    save_metadata(exp_dir, build_metadata(spec.to_dict()))

    job_name = f"{short_model_name(spec.model_type)}_{spec.feature_type[:3]}_{spec.exp_id}"
    if spec.model_type == "naive":
        job_name = f"naive_{spec.exp_id}"

    n_vars = len(spec.variables)
    extra_tag = f" [{spec.extra_args}]" if spec.extra_args else ""
    logger.info(
        "Submitting ID %d: %s + %s - %s (%d vars)%s",
        spec.exp_id,
        spec.model_type.upper(),
        spec.feature_type.upper(),
        spec.exp_name.upper(),
        n_vars,
        extra_tag,
    )

    if backend is None:
        backend = get_backend("slurm")

    job_env = build_job_env(spec, exp_dir, total_chunks)
    backend.submit_array(job_name, total_chunks, tasks_per_array, job_env)
    submitted_marker.touch()
    return exp_dir


def _try_link_cached_naive(base_dir: str | Path) -> bool:
    """Symlink pre-computed naive results into *base_dir* if available.

    Returns True if the link was created (or already exists), False otherwise.
    """
    cached_naive = PROJECT_ROOT / "results" / "naive" / "exp_0_naive_baseline"
    link_path = Path(base_dir) / "exp_0_naive_baseline"

    # Already present (previous symlink or a real directory) — nothing to do.
    if link_path.exists():
        return True

    if cached_naive.is_dir():
        link_path.symlink_to(cached_naive.resolve())
        logger.info("Linked cached naive results: %s -> %s", link_path, cached_naive.resolve())
        return True

    return False


def submit_experiment_batch(
    specs,
    base_dir,
    total_chunks,
    tasks_per_array=DEFAULT_TASKS_PER_ARRAY,
    include_naive=True,
    backend: HPCBackend | None = None,
):
    """Submit a list of ExperimentSpecs, optionally prepending naive baseline."""
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    all_specs = list(specs)
    if include_naive:
        if _try_link_cached_naive(base_dir):
            logger.info("Reusing cached naive baseline from results/naive/.")
        else:
            logger.warning(
                "Cached naive results not found at results/naive/exp_0_naive_baseline. "
                "Submitting a new naive baseline job. Run 'scripts/submit.py naive' first to avoid this."
            )
            naive = ExperimentSpec(exp_id=0, exp_name="naive_baseline", model_type="naive", feature_type="har")
            all_specs.insert(0, naive)

    n_total = len(all_specs)
    logger.info("Submitting %d experiments to %s", n_total, base_dir)

    for spec in all_specs:
        submit_experiment(spec, base_dir, total_chunks, tasks_per_array, backend)

    # Mark this base_dir as needing aggregation
    (Path(base_dir) / ".needs_aggregation").touch()

    logger.info("All %d experiments submitted to %s.", n_total, base_dir)
    logger.info("Run 'python scripts/aggregate.py' to aggregate all pending results.")


def build_extra_args(feature_type, args):
    """Build the --features / --n-components / --ae-* / --horizon CLI string for harx.py."""
    parts = [f"--features {feature_type}"]
    if feature_type in ("pca", "ae"):
        parts.append(f"--n-components {args.n_components}")
    if feature_type == "ae":
        parts.append(f"--ae-alpha {args.ae_alpha}")
        parts.append(f"--ae-epochs {args.ae_epochs}")
        if args.ae_hidden:
            parts.append(f"--ae-hidden {args.ae_hidden}")
        if getattr(args, "ae_weights_path", None):
            parts.append(f"--ae-weights-path {args.ae_weights_path}")
    if args.train_window != 500:
        parts.append(f"--train-window {args.train_window}")
    horizon = getattr(args, "horizon", 1)
    if horizon > 1:
        parts.append(f"--horizon {horizon}")
    return " ".join(parts)


def add_common_submit_args(parser):
    """Add CLI arguments shared across all submission scripts."""
    parser.add_argument(
        "--result-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Base output directory for experiment results.",
    )
    parser.add_argument(
        "--total-chunks",
        type=int,
        default=DEFAULT_TOTAL_CHUNKS,
        help="Total number of dataset chunks to process.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="slurm",
        help="HPC backend to use for job submission (e.g. slurm, sge).",
    )
    add_feature_args(parser)
    parser.add_argument(
        "--no-naive",
        action="store_true",
        help="Skip submitting the naive baseline job.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Final forecast horizon H. Executor runs backtests for h=1..H.",
    )
    return parser
