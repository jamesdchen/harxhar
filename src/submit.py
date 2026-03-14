"""
Shared experiment submission utilities.

Handles config.txt creation, SLURM env var construction, and sbatch array submission.
All submission scripts delegate mechanical work here.
"""
import argparse
import dataclasses
import os
import subprocess
from pathlib import Path


SUBMISSION_SCRIPT = "slurm/submit_carc.slurm"
DEFAULT_TASKS_PER_ARRAY = 100
DEFAULT_TOTAL_CHUNKS = 100


@dataclasses.dataclass
class ExperimentSpec:
    exp_id: int
    exp_name: str
    model_type: str
    feature_type: str = "raw"
    variables: list = dataclasses.field(default_factory=list)
    extra_args: str = ""


def write_config(exp_dir, spec):
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


def short_model_name(model_type):
    mapping = {
        "xgboost": "xgb",
        "lightgbm": "lgb",
        "random_forest": "rf",
    }
    return mapping.get(model_type, model_type[:3])


def build_job_env(spec, exp_dir, total_chunks):
    """Build the env dict that submit_carc.slurm expects."""
    env = os.environ.copy()
    env["TOTAL_CHUNKS"] = str(total_chunks)
    env["EXOG_COLS"] = "|".join(spec.variables) if spec.variables else "None"
    env["RESULT_DIR"] = exp_dir
    env["MODEL_TYPE"] = spec.model_type
    env["EXTRA_ARGS"] = spec.extra_args
    return env


def submit_array(job_name, total_chunks, tasks_per_array, job_env,
                 slurm_script=SUBMISSION_SCRIPT):
    """Submit SLURM array job(s), chunking if tasks_per_array < total_chunks."""
    start_task = 1
    while start_task <= total_chunks:
        end_task = min(start_task + tasks_per_array - 1, total_chunks)
        task_range = f"{start_task}-{end_task}"
        cmd = ["sbatch", "--array", task_range, "--job-name", job_name, slurm_script]
        subprocess.run(cmd, env=job_env)
        start_task = end_task + 1


def submit_experiment(spec, base_dir, total_chunks,
                      tasks_per_array=DEFAULT_TASKS_PER_ARRAY,
                      slurm_script=SUBMISSION_SCRIPT):
    """Submit a single experiment: mkdir, write config, sbatch."""
    dir_name = f"exp_{spec.exp_id}_{spec.model_type}_{spec.feature_type}_{spec.exp_name}"
    if spec.model_type == "naive":
        dir_name = f"exp_{spec.exp_id}_naive_baseline"
    exp_dir = str(Path(base_dir).resolve() / dir_name)
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    write_config(exp_dir, spec)

    job_name = f"{short_model_name(spec.model_type)}_{spec.feature_type[:3]}_{spec.exp_id}"
    if spec.model_type == "naive":
        job_name = f"naive_{spec.exp_id}"

    n_vars = len(spec.variables)
    extra_tag = f" [{spec.extra_args}]" if spec.extra_args else ""
    print(
        f"--- Submitting ID {spec.exp_id}: {spec.model_type.upper()} + "
        f"{spec.feature_type.upper()} - {spec.exp_name.upper()} "
        f"({n_vars} vars){extra_tag} ---"
    )

    job_env = build_job_env(spec, exp_dir, total_chunks)
    submit_array(job_name, total_chunks, tasks_per_array, job_env, slurm_script)
    return exp_dir


def submit_experiment_batch(specs, base_dir, total_chunks,
                            tasks_per_array=DEFAULT_TASKS_PER_ARRAY,
                            include_naive=True,
                            slurm_script=SUBMISSION_SCRIPT):
    """Submit a list of ExperimentSpecs, optionally prepending naive baseline."""
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    all_specs = list(specs)
    if include_naive:
        naive = ExperimentSpec(exp_id=0, exp_name="naive_baseline",
                               model_type="naive", feature_type="raw")
        all_specs.insert(0, naive)

    n_total = len(all_specs)
    print(f"Submitting {n_total} experiments to {base_dir}...")

    for spec in all_specs:
        submit_experiment(spec, base_dir, total_chunks, tasks_per_array, slurm_script)

    # Mark this base_dir as needing aggregation
    (Path(base_dir) / ".needs_aggregation").touch()

    print(f"\nAll {n_total} experiments submitted to {base_dir}.")
    print("Run 'python scripts/aggregate.py' to aggregate all pending results.")


def build_extra_args(feature_type, args):
    """Build the --features / --n-components / --ae-* CLI string for harx.py."""
    parts = [f"--features {feature_type}"]
    if feature_type in ("pca", "ae"):
        parts.append(f"--n-components {args.n_components}")
    if feature_type == "ae":
        parts.append(f"--ae-alpha {args.ae_alpha}")
        parts.append(f"--ae-epochs {args.ae_epochs}")
        if args.ae_hidden:
            parts.append(f"--ae-hidden {args.ae_hidden}")
        if getattr(args, 'ae_weights_path', None):
            parts.append(f"--ae-weights-path {args.ae_weights_path}")
    if args.train_window != 500:
        parts.append(f"--train-window {args.train_window}")
    return " ".join(parts)


def add_common_submit_args(parser):
    """Add CLI arguments shared across all submission scripts."""
    parser.add_argument(
        "--result-dir", type=str, default=None,
        help="Base output directory for experiment results.",
    )
    parser.add_argument(
        "--total-chunks", type=int, default=DEFAULT_TOTAL_CHUNKS,
        help="Total number of dataset chunks to process.",
    )
    parser.add_argument(
        "--train-window", type=int, default=500,
        help="Training window in days (passed to harx.py).",
    )
    parser.add_argument(
        "--n-components", type=int, default=5,
        help="Number of latent components for --features pca and ae.",
    )
    parser.add_argument(
        "--ae-alpha", type=float, default=0.5,
        help="AE loss weight: alpha*recon + (1-alpha)*pred.",
    )
    parser.add_argument(
        "--ae-epochs", type=int, default=50,
        help="Training epochs per AE refit.",
    )
    parser.add_argument(
        "--ae-hidden", type=int, default=0,
        help="AE hidden layer width; 0 = auto (n_features // 2).",
    )
    parser.add_argument(
        "--ae-weights-path", type=str, default=None,
        help="Path to pre-trained AE weights .pt file (skip AE training on CPU).",
    )
    parser.add_argument(
        "--no-naive", action="store_true",
        help="Skip submitting the naive baseline job.",
    )
    return parser
