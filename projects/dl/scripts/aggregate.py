"""
Aggregate DL experiment results from flat result directories.

Usage:
    python -m projects.dl.scripts.aggregate results/dl_patchts_ctx480 results/dl_patchts_overlap
    python -m projects.dl.scripts.aggregate results/dl_patchts_ctx480 \\
        --baseline-dir results/model_comparison_v2/exp_1_xgboost_har_baseline
    python -m projects.dl.scripts.aggregate --auto
    python -m projects.dl.scripts.aggregate --auto --eval-mode filter_by_tod
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo_root))

import argparse  # noqa: E402
import glob  # noqa: E402

import pandas as pd  # noqa: E402

from core.evaluation.aggregation import (  # noqa: E402
    build_segment_configs,
    load_experiment_metadata,
    print_and_save_summary,
    process_single_experiment,
)
from core.evaluation.metrics import calculate_baseline_deltas  # noqa: E402


def _load_baseline_metrics(baseline_dir: str, segment_configs: list[dict]) -> list[dict]:
    """Load baseline experiment from a directory and compute its metrics."""
    baseline_path = Path(baseline_dir)
    if not baseline_path.exists():
        print(f"Warning: baseline directory '{baseline_dir}' not found. Deltas will be NaN.")
        return []

    metadata = {"exp_id": 0, "experiment_name": "baseline", "model": "baseline"}
    return process_single_experiment(baseline_path, metadata, segment_configs)


def aggregate_dl(result_dirs: list[Path], eval_mode: str, baseline_dir: str | None = None) -> None:
    """Aggregate one or more flat DL result directories."""
    title, out_file, segment_configs = build_segment_configs(eval_mode)

    print("=" * 150)
    print(f"DL Aggregation | {len(result_dirs)} experiment(s) | Mode: {eval_mode.upper()}")
    print("=" * 150)

    results = []

    # Load baseline first if provided
    if baseline_dir:
        baseline_results = _load_baseline_metrics(baseline_dir, segment_configs)
        results.extend(baseline_results)

    # Process each DL result directory as a single experiment
    for i, result_dir in enumerate(result_dirs, start=1):
        metadata = load_experiment_metadata(result_dir, fallback_exp_id=i)
        exp_results = process_single_experiment(result_dir, metadata, segment_configs)
        results.extend(exp_results)

    if not results:
        print("No valid results found. Exiting.")
        return

    summary_df = pd.DataFrame(results)

    # Apply baseline deltas if we have a baseline
    if baseline_dir:
        summary_df = calculate_baseline_deltas(summary_df)

    # Output
    if len(result_dirs) == 1:
        output_path = result_dirs[0] / out_file
    else:
        output_path = result_dirs[0].parent / f"dl_{out_file}"
    print_and_save_summary(summary_df, title, output_path)


def main(args: argparse.Namespace) -> None:
    if args.auto:
        # Auto-discover DL result directories
        dl_dirs = sorted(Path(p) for p in glob.glob("results/dl_*") if Path(p).is_dir())
        if not dl_dirs:
            print("No results/dl_* directories found.")
            return
        print(f"Auto-discovered {len(dl_dirs)} DL result dir(s): {', '.join(d.name for d in dl_dirs)}")
    else:
        dl_dirs = [Path(d) for d in args.dirs]
        missing = [d for d in dl_dirs if not d.exists()]
        if missing:
            print(f"Error: directories not found: {', '.join(str(d) for d in missing)}")
            return

    aggregate_dl(dl_dirs, args.eval_mode, args.baseline_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate DL experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dirs", nargs="*", help="DL result directories to aggregate")
    parser.add_argument("--auto", action="store_true", help="Auto-discover results/dl_* directories")
    parser.add_argument(
        "--eval-mode", choices=["global", "segments", "filter_by_tod"], default="global", help="Evaluation mode."
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Path to a baseline experiment directory for delta computation.",
    )
    main(parser.parse_args())
