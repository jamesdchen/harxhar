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

from core.evaluation.aggregation import process_single_experiment  # noqa: E402
from core.evaluation.metrics import calculate_baseline_deltas  # noqa: E402

TARGET_SEGMENTS = ["morning", "midday", "closing", "overnight"]

TOD_BOUNDS = {
    "morning": {"start": "09:30", "end": "11:30"},
    "midday": {"start": "11:30", "end": "14:00"},
    "closing": {"start": "14:00", "end": "16:00"},
    "overnight": {"start": "16:00", "end": "09:30"},
}


def _infer_metadata(result_dir: Path, exp_id: int) -> dict:
    """Infer experiment metadata from a flat DL result directory name."""
    name = result_dir.name
    # e.g. dl_patchts_ctx480 -> model=patchts, experiment_name=dl_patchts_ctx480
    parts = name.split("_")
    model = parts[1] if len(parts) >= 2 else name
    return {
        "exp_id": exp_id,
        "experiment_name": name,
        "model": model,
    }


def _build_segment_configs(eval_mode: str) -> tuple[str, str, list[dict]]:
    """Returns (title, output_filename, segment_configs) for the given mode."""
    if eval_mode == "segments":
        title = f"PRE-SEGMENTED FILES SUMMARY: {TARGET_SEGMENTS}"
        out_file = "segment_results_summary.csv"
        configs = [
            {"name": seg.upper(), "load_kwargs": {"require_suffixes": [seg], "ignore_suffixes": None}}
            for seg in TARGET_SEGMENTS
        ]
    elif eval_mode == "filter_by_tod":
        title = "GLOBAL DATA (Filtered into TOD Segments in Memory)"
        out_file = "global_results_tod_filtered.csv"
        configs = [
            {
                "name": f"GLOBAL_{seg.upper()}",
                "load_kwargs": {"require_suffixes": None, "ignore_suffixes": TARGET_SEGMENTS},
                "time_bounds": bounds,
            }
            for seg, bounds in TOD_BOUNDS.items()
        ]
    else:
        title = "GLOBAL SUMMARY (All Hours)"
        out_file = "global_results_summary.csv"
        configs = [{"name": "GLOBAL", "load_kwargs": {"require_suffixes": None, "ignore_suffixes": TARGET_SEGMENTS}}]
    return title, out_file, configs


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
    title, out_file, segment_configs = _build_segment_configs(eval_mode)

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
        metadata = _infer_metadata(result_dir, exp_id=i)
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

    print(f"\n{'=' * 175}")
    print(title)
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

    # Save to the first result directory (or parent)
    if len(result_dirs) == 1:
        output_path = result_dirs[0] / out_file
    else:
        output_path = result_dirs[0].parent / f"dl_{out_file}"
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved summary to: {output_path}")


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
    parser = argparse.ArgumentParser(description="Aggregate DL experiment results")
    parser.add_argument("dirs", nargs="*", help="DL result directories to aggregate")
    parser.add_argument("--auto", action="store_true", help="Auto-discover results/dl_* directories")
    parser.add_argument("--eval-mode", choices=["global", "segments", "filter_by_tod"], default="global")
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Path to a baseline experiment directory for delta computation.",
    )
    main(parser.parse_args())
