"""Aggregate ML experiment results: stitch chunks, compute metrics, calculate baseline deltas.

Auto-discovers experiment directories with ``.needs_aggregation`` markers, or
processes a specific ``--base_dir``.  Supports global, segmented, and
time-of-day filtering evaluation modes.

Run ``python projects/ml/scripts/aggregate.py --help`` for usage.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo_root))

import argparse  # noqa: E402
import glob  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

import pandas as pd  # noqa: E402

from core.evaluation.aggregation import build_segment_configs, print_and_save_summary  # noqa: E402
from core.evaluation.metrics import calculate_baseline_deltas  # noqa: E402

# Import the updated processor
from projects.ml.evaluation.aggregation import parse_config, process_single_experiment  # noqa: E402


def natural_sort_key(s):
    """Sorts strings containing numbers logically (e.g., exp_2 before exp_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]


def aggregate_base_dir(base_dir, eval_mode):
    """Aggregate all experiments in a single base_dir."""
    # 1. --- Route Logic & Dynamic Configurations ---
    title_str, out_filename, segment_configs = build_segment_configs(eval_mode)

    # Find Experiments
    search_path = os.path.join(base_dir, "exp_*")
    exp_dirs = sorted(glob.glob(search_path), key=natural_sort_key)

    print("=" * 150)
    print(f"Found {len(exp_dirs)} experiments in '{base_dir}' | Mode: {eval_mode.upper()}")
    if eval_mode == "filter_by_tod":
        print("Feature Active: Slicing global data by Time-of-Day (Morning, Midday, Closing, Overnight)")
    print("=" * 150)

    results = []

    # 2. --- Processing Loop ---
    for exp_dir in exp_dirs:
        exp_id, exp_name, model_type = parse_config(exp_dir)
        metadata = {"exp_id": exp_id, "experiment_name": exp_name, "model": model_type}

        # Call the agnostic processor imported from eval_utils
        results.extend(process_single_experiment(exp_dir, metadata, segment_configs))

    if not results:
        print("No valid results found. Exiting.")
        return

    # 3. --- Compile and Segment-Aware Delta Logic ---

    summary_df = calculate_baseline_deltas(pd.DataFrame(results))

    # 4. --- Final Table Output ---
    print_and_save_summary(summary_df, title_str, os.path.join(base_dir, out_filename))


def main(args):
    if args.base_dir:
        # Explicit mode: process the given directory regardless of marker
        aggregate_base_dir(args.base_dir, args.eval_mode)
        return

    # Auto-discover: find all base_dirs with .needs_aggregation marker
    marker_files = glob.glob("results/*/.needs_aggregation") + glob.glob("results_*/.needs_aggregation")
    if not marker_files:
        print("No pending results directories found. Nothing to aggregate.")
        return

    base_dirs = sorted(os.path.dirname(m) for m in marker_files)
    print(f"Found {len(base_dirs)} pending results directory(ies): {', '.join(base_dirs)}\n")

    for base_dir in base_dirs:
        aggregate_base_dir(base_dir, args.eval_mode)
        # Remove marker after successful aggregation
        marker_path = os.path.join(base_dir, ".needs_aggregation")
        os.remove(marker_path)
        print(f"Cleared .needs_aggregation for {base_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate Global/Segment Raw MSE, MAE & QLIKE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["global", "segments", "filter_by_tod"],
        default="global",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Process a specific directory. If omitted, auto-discovers all results/*/ with pending results.",
    )

    main(parser.parse_args())
