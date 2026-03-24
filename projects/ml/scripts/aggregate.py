import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent / "src"))
sys.path.insert(0, str(_scripts_dir.parent.parent / "core" / "src"))

import argparse  # noqa: E402
import glob  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

import pandas as pd  # noqa: E402

# Import the updated processor
from core.evaluation.aggregation import parse_config, process_single_experiment  # noqa: E402
from core.evaluation.metrics import calculate_baseline_deltas  # noqa: E402

TARGET_SEGMENTS = ["morning", "midday", "closing", "overnight"]

# Define exact boundaries for the memory slicer
TOD_BOUNDS = {
    "morning": {"start": "09:30", "end": "11:30"},
    "midday": {"start": "11:30", "end": "14:00"},
    "closing": {"start": "14:00", "end": "16:00"},
    "overnight": {"start": "16:00", "end": "09:30"},
}


def natural_sort_key(s):
    """Sorts strings containing numbers logically (e.g., exp_2 before exp_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]


def aggregate_base_dir(base_dir, eval_mode):
    """Aggregate all experiments in a single base_dir."""
    # 1. --- Route Logic & Dynamic Configurations ---
    if eval_mode == "segments":
        title_str = f"PRE-SEGMENTED FILES SUMMARY: {TARGET_SEGMENTS}"
        out_filename = "segment_results_summary.csv"
        segment_configs = [
            {"name": seg.upper(), "load_kwargs": {"require_suffixes": [seg], "ignore_suffixes": None}}
            for seg in TARGET_SEGMENTS
        ]

    elif eval_mode == "filter_by_tod":
        title_str = "GLOBAL DATA (Filtered into TOD Segments in Memory)"
        out_filename = "global_results_tod_filtered.csv"
        segment_configs = [
            {
                "name": f"GLOBAL_{seg.upper()}",
                "load_kwargs": {"require_suffixes": None, "ignore_suffixes": TARGET_SEGMENTS},
                "time_bounds": bounds,
            }
            for seg, bounds in TOD_BOUNDS.items()
        ]
    else:
        title_str = "GLOBAL SUMMARY (All Hours)"
        out_filename = "global_results_summary.csv"
        segment_configs = [
            {"name": "GLOBAL", "load_kwargs": {"require_suffixes": None, "ignore_suffixes": TARGET_SEGMENTS}}
        ]

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
    final_cols = [
        "exp_id",
        "model",
        "experiment_name",
        "segment",
        "horizon",
        "mse",
        "delta_mse",
        "oos_r2",
        "mae",
        "delta_mae",
        "qlike",
        "delta_qlike",
        "n_samples",
    ]
    final_cols = [c for c in final_cols if c in summary_df.columns]

    print("\n" + "=" * 175)
    print(title_str)
    print("=" * 165)

    formatters = {
        "mse": "{:.4e}".format,
        "delta_mse": "{:.4e}".format,
        "mae": "{:.4e}".format,
        "delta_mae": "{:.4e}".format,
        "qlike": "{:.6f}".format,
        "delta_qlike": "{:.6f}".format,
        "oos_r2": "{:.4%}".format,
    }

    pd.set_option("display.width", 1000)
    print(
        summary_df[final_cols].to_string(
            index=False, formatters={k: v for k, v in formatters.items() if k in final_cols}
        )
    )

    output_file = os.path.join(base_dir, out_filename)
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary to: {output_file}")


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
    parser = argparse.ArgumentParser(description="Aggregate Global/Segment Raw MSE, MAE & QLIKE")
    parser.add_argument("--eval-mode", type=str, choices=["global", "segments", "filter_by_tod"], default="global")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Process a specific directory. If omitted, auto-discovers all results/*/ with pending results.",
    )

    main(parser.parse_args())
