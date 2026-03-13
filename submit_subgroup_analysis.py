import argparse
import os
import subprocess
import numpy as np

# --- CONFIGURATION ---
TOTAL_CHUNKS = 100
TASKS_PER_ARRAY = 100
SUBMISSION_SCRIPT = "submit_carc.slurm"

# Original model set (backward-compatible default)
ORIGINAL_MODELS = ["ridge", "xgboost", "lightgbm", "random_forest"]
LAG_FEATURE_MODELS = ["pca_ridge", "ae_ridge"]
ALL_MODELS = ORIGINAL_MODELS + LAG_FEATURE_MODELS

# --- 1. DEFINE THE FEATURE UNIVERSE ---
FULL_FEATURE_STRING = (
    "endbartime|sumret|sumabsret|sumret3|sumret4|sumpret2|sumbipow|sumautocov|sumvolume|numobs|"
    "sumret2_ewstock|sumret3_ewstock|sumret4_ewstock|sumabsret_ewstock|sumbipow_ewstock|sumpret2_ewstock|"
    "turnover_ewstock|buyturnover_ewstock|sellturnover_ewstock|effspread_ewstock|spread_ewstock|"
    "sumret2_vwstock|sumret3_vwstock|sumret4_vwstock|sumabsret_vwstock|sumbipow_vwstock|sumpret2_vwstock|"
    "turnover_vwstock|buyturnover_vwstock|sellturnover_vwstock|effspread_vwstock|spread_vwstock|"
    "turnover_spy|buyturnover_spy|sellturnover_spy|"
    "stocktwits_attention|stocktwits_sentiment|stocktwits_sentcount|"
    "vix|vvix|vix3m|"
    "voldemand_spx_open_and_close|voldemand_spx_open_only|voldemand_all_open_and_close|voldemand_all_open_only|"
    "hour|DOW"
)

ALL_FEATURES = FULL_FEATURE_STRING.split("|")

# --- 2. DEFINE SUBGROUPS ---
SUBGROUPS = {
    "baseline": [],
    "moments": [f for f in ALL_FEATURES if f.startswith("sum") and "stock" not in f and "volume" not in f],
    "liquidity": [f for f in ALL_FEATURES if any(x in f for x in ["volume", "turnover", "spread"])],
    "market_ew": [f for f in ALL_FEATURES if "ewstock" in f],
    "market_vw": [f for f in ALL_FEATURES if "vwstock" in f],
    "sentiment": [f for f in ALL_FEATURES if "stocktwits" in f],
    "implied_vol": [f for f in ALL_FEATURES if "vix" in f],
    "vol_demand": [f for f in ALL_FEATURES if "voldemand" in f],
    "all_features": ALL_FEATURES
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit subgroup analysis experiments to Slurm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=ORIGINAL_MODELS,
        help=(
            f"Models to run. Use 'all' to run all models: {ALL_MODELS}. "
            f"Default: original set {ORIGINAL_MODELS}."
        ),
    )
    parser.add_argument(
        "--subgroups", nargs="+", default=["all"],
        help=(
            f"Subgroups to run. Use 'all' to run all defined subgroups: {list(SUBGROUPS)}. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--result-dir", type=str, default="results_ridge_subgroups",
        help="Base output directory for experiment results.",
    )
    parser.add_argument(
        "--total-chunks", type=int, default=TOTAL_CHUNKS,
        help="Total number of dataset chunks to process.",
    )
    parser.add_argument(
        "--train-window", type=int, default=500,
        help="Training window in days (passed to harx.py).",
    )
    # PCA / AE shared
    parser.add_argument(
        "--n-components", type=int, default=5,
        help="Number of latent components for pca_ridge and ae_ridge.",
    )
    # AE-specific
    parser.add_argument(
        "--ae-alpha", type=float, default=0.5,
        help="AE loss weight: alpha*recon + (1-alpha)*pred  (ae_ridge only).",
    )
    parser.add_argument(
        "--ae-epochs", type=int, default=50,
        help="Training epochs per AE refit  (ae_ridge only).",
    )
    parser.add_argument(
        "--ae-hidden", type=int, default=0,
        help="AE hidden layer width; 0 = auto (n_features // 2)  (ae_ridge only).",
    )
    parser.add_argument(
        "--no-naive", action="store_true",
        help="Skip submitting the naive baseline job.",
    )
    return parser.parse_args()


def resolve_models(models_arg):
    if len(models_arg) == 1 and models_arg[0] == "all":
        return ALL_MODELS
    return models_arg


def resolve_subgroups(subgroups_arg):
    if len(subgroups_arg) == 1 and subgroups_arg[0] == "all":
        return SUBGROUPS
    return {k: SUBGROUPS[k] for k in subgroups_arg if k in SUBGROUPS}


def build_extra_args(model_type, args):
    """Return a string of extra CLI flags to be forwarded to harx.py via EXTRA_ARGS."""
    parts = []
    if model_type in ("pca_ridge", "ae_ridge"):
        parts.append(f"--n-components {args.n_components}")
    if model_type == "ae_ridge":
        parts.append(f"--ae-alpha {args.ae_alpha}")
        parts.append(f"--ae-epochs {args.ae_epochs}")
        if args.ae_hidden:
            parts.append(f"--ae-hidden {args.ae_hidden}")
    if args.train_window != 500:
        parts.append(f"--train-window {args.train_window}")
    return " ".join(parts)


def short_model_name(model_type):
    mapping = {
        "xgboost": "xgb",
        "lightgbm": "lgb",
        "random_forest": "rf",
        "pca_ridge": "pca",
        "ae_ridge": "ae",
    }
    return mapping.get(model_type, model_type[:3])


def submit_array(job_name, total_chunks, tasks_per_array, job_env):
    start_task = 1
    while start_task <= total_chunks:
        end_task = min(start_task + tasks_per_array - 1, total_chunks)
        task_range = f"{start_task}-{end_task}"
        cmd = ["sbatch", "--array", task_range, "--job-name", job_name, SUBMISSION_SCRIPT]
        subprocess.run(cmd, env=job_env)
        start_task = end_task + 1


def main():
    args = parse_args()
    models_to_run = resolve_models(args.models)
    subgroups_to_run = resolve_subgroups(args.subgroups)
    total_chunks = args.total_chunks
    base_result_dir = args.result_dir

    total_experiments = len(subgroups_to_run) * len(models_to_run)
    print(
        f"Generating experiments for {len(subgroups_to_run)} subgroups "
        f"x {len(models_to_run)} models"
        + (" + Naive baseline" if not args.no_naive else "")
        + f"..."
    )
    os.makedirs(base_result_dir, exist_ok=True)

    # ==========================================
    # --- SUBMIT NAIVE BASELINE FIRST ---
    # ==========================================
    if not args.no_naive:
        naive_dir = os.path.abspath(os.path.join(base_result_dir, "exp_0_naive_baseline"))
        os.makedirs(naive_dir, exist_ok=True)

        with open(os.path.join(naive_dir, "config.txt"), "w") as f:
            f.write("Experiment ID: 0\n")
            f.write("Experiment Name: naive_baseline\n")
            f.write("Model Type: naive\n")
            f.write("Num Variables: 0\n")
            f.write("Variables: []\n")

        print(f"--- Submitting ID 0: NAIVE_BASELINE ---")

        job_env = os.environ.copy()
        job_env["TOTAL_CHUNKS"] = str(total_chunks)
        job_env["EXOG_COLS"] = "None"
        job_env["RESULT_DIR"] = naive_dir
        job_env["MODEL_TYPE"] = "naive"
        job_env["EXTRA_ARGS"] = ""

        submit_array("naive_0", total_chunks, TASKS_PER_ARRAY, job_env)

    # ==========================================
    # --- SUBMIT ML MODELS ---
    # ==========================================
    exp_id = 1
    for model_type in models_to_run:
        for exp_name, variables in subgroups_to_run.items():

            exog_str = "|".join(variables) if variables else "None"
            exp_dir = os.path.abspath(
                os.path.join(base_result_dir, f"exp_{exp_id}_{model_type}_{exp_name}")
            )
            os.makedirs(exp_dir, exist_ok=True)

            extra_args = build_extra_args(model_type, args)

            with open(os.path.join(exp_dir, "config.txt"), "w") as f:
                f.write(f"Experiment ID: {exp_id}\n")
                f.write(f"Experiment Name: {exp_name}\n")
                f.write(f"Model Type: {model_type}\n")
                f.write(f"Num Variables: {len(variables)}\n")
                f.write(f"Variables: {variables}\n")
                if extra_args:
                    f.write(f"Extra Args: {extra_args}\n")

            print(
                f"--- Submitting ID {exp_id}: {model_type.upper()} - {exp_name.upper()} "
                f"({len(variables)} vars)"
                + (f" [{extra_args}]" if extra_args else "")
                + " ---"
            )

            job_env = os.environ.copy()
            job_env["TOTAL_CHUNKS"] = str(total_chunks)
            job_env["EXOG_COLS"] = exog_str
            job_env["RESULT_DIR"] = exp_dir
            job_env["MODEL_TYPE"] = model_type
            job_env["EXTRA_ARGS"] = extra_args

            job_name = f"{short_model_name(model_type)}_{exp_id}"
            submit_array(job_name, total_chunks, TASKS_PER_ARRAY, job_env)

            exp_id += 1

    print(f"\nAll {total_experiments} experiments" + (" + Naive baseline" if not args.no_naive else "") + f" submitted to {base_result_dir}.")


if __name__ == "__main__":
    main()
