"""
DL experiment submission via shared HPC backend.

Submits GPU backtest array jobs using the same backend infrastructure as ML.

Usage:
    python -m projects.dl.cli.submit --experiment patchts --total-chunks 10
    python -m projects.dl.cli.submit --experiment ae_ridge --total-chunks 20 --result-dir /scratch1/jc_905/ae_results
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from core.backends import get_backend
from core.core.log import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TASKS_PER_ARRAY = 100
DL_SLURM_SCRIPT = str(PROJECT_ROOT / "projects" / "dl" / "infra" / "slurm" / "submit_gpu.slurm")


def build_job_env(
    experiment: str,
    result_dir: str,
    total_chunks: int,
    *,
    input_path: str = "all30min",
    gpu_count: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    train_window: int | None = None,
    weights_dir: str | None = None,
) -> dict[str, str]:
    """Build the env dict that submit_gpu.slurm expects."""
    env = os.environ.copy()
    env["EXPERIMENT"] = experiment
    env["RESULT_DIR"] = result_dir
    env["TOTAL_CHUNKS"] = str(total_chunks)
    env["INPUT_PATH"] = input_path
    if gpu_count is not None:
        env["GPU_COUNT"] = str(gpu_count)
    if batch_size is not None:
        env["BATCH_SIZE"] = str(batch_size)
    if epochs is not None:
        env["EPOCHS"] = str(epochs)
    if learning_rate is not None:
        env["LEARNING_RATE"] = str(learning_rate)
    if train_window is not None:
        env["TRAIN_WINDOW"] = str(train_window)
    if weights_dir is not None:
        env["WEIGHTS_DIR"] = weights_dir
    return env


def submit_dl_experiment(
    experiment: str,
    result_dir: str,
    total_chunks: int,
    tasks_per_array: int = DEFAULT_TASKS_PER_ARRAY,
    backend_name: str = "slurm",
    **env_kwargs,
) -> str:
    """Submit a DL GPU backtest as a SLURM array job."""
    result_path = Path(result_dir).resolve()
    result_path.mkdir(parents=True, exist_ok=True)

    job_name = f"dl_{experiment}"
    job_env = build_job_env(experiment, str(result_path), total_chunks, **env_kwargs)

    logger.info(
        "Submitting %s: %d chunks to %s",
        experiment.upper(),
        total_chunks,
        result_path,
    )

    backend = get_backend(backend_name, script=DL_SLURM_SCRIPT)
    backend.submit_array(job_name, total_chunks, tasks_per_array, job_env)

    logger.info("Submitted %s (%d array tasks).", experiment, total_chunks)
    return str(result_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit DL GPU backtest as SLURM array job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["patchts", "ae_ridge"],
        required=True,
        help="Which DL experiment to run.",
    )
    parser.add_argument("--result-dir", type=str, required=True, help="Output directory for chunk results.")
    parser.add_argument("--total-chunks", type=int, default=10, help="Number of array tasks / chunks.")
    parser.add_argument("--tasks-per-array", type=int, default=DEFAULT_TASKS_PER_ARRAY, help="Max tasks per sbatch.")
    parser.add_argument("--backend", type=str, default="slurm", help="HPC backend (slurm, sge, dry-run).")
    parser.add_argument("--input-path", type=str, default="all30min", help="Data directory.")
    parser.add_argument("--gpu-count", type=int, default=None, help="GPUs per task.")
    parser.add_argument("--batch-size", type=int, default=None, help="Windows per batch.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--train-window", type=int, default=None, help="Training window size.")
    parser.add_argument("--weights-dir", type=str, default=None, help="AE weights directory (ae_ridge only).")

    args = parser.parse_args()

    env_kwargs = {}
    for key in ("input_path", "gpu_count", "batch_size", "epochs", "learning_rate", "train_window", "weights_dir"):
        val = getattr(args, key)
        if val is not None:
            env_kwargs[key] = val

    submit_dl_experiment(
        experiment=args.experiment,
        result_dir=args.result_dir,
        total_chunks=args.total_chunks,
        tasks_per_array=args.tasks_per_array,
        backend_name=args.backend,
        **env_kwargs,
    )


if __name__ == "__main__":
    main()
