"""
Compare results across experiment directories.

Usage:
    python scripts/compare.py results/model_comparison results/subgroup_analysis
    python scripts/compare.py results/model_comparison --metric qlike --sort asc
    python scripts/compare.py results/*/global_results_summary.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.cli.metadata import load_metadata

METRIC_COLS = ["mse", "mae", "qlike", "oos_r2", "delta_mse", "delta_mae", "delta_qlike"]


def find_summary_csvs(paths: list[str]) -> list[Path]:
    """Find summary CSV files from directory or file paths."""
    csvs = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".csv":
            csvs.append(path)
        elif path.is_dir():
            # Look for summary CSVs in the directory
            for pattern in ["global_results_summary.csv", "segment_results_summary.csv", "*_summary.csv"]:
                found = list(path.glob(pattern))
                if found:
                    csvs.extend(found)
                    break
    return sorted(set(csvs))


def load_and_tag(csv_path: Path) -> pd.DataFrame:
    """Load a summary CSV and tag it with its source directory."""
    df = pd.read_csv(csv_path)
    df["source"] = csv_path.parent.name
    df["source_file"] = str(csv_path)

    # Try to load metadata for git info
    meta = load_metadata(csv_path.parent)
    if meta:
        df["git_hash"] = meta.get("git_short_hash", "")
        df["timestamp"] = meta.get("timestamp", "")[:19]
    return df


def compare(paths: list[str], metric: str, ascending: bool, top_n: int | None) -> None:
    """Compare experiments across result directories."""
    csvs = find_summary_csvs(paths)
    if not csvs:
        print("No summary CSV files found. Run 'python scripts/aggregate.py' first.")
        return

    print(f"Comparing {len(csvs)} result file(s):\n")
    for csv in csvs:
        print(f"  {csv}")
    print()

    frames = []
    for csv in csvs:
        try:
            frames.append(load_and_tag(csv))
        except Exception as e:
            print(f"  Warning: could not load {csv}: {e}")

    if not frames:
        print("No valid results to compare.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Determine display columns
    id_cols = ["source", "model", "experiment_name"]
    if "segment" in combined.columns:
        id_cols.append("segment")
    if "horizon" in combined.columns:
        id_cols.append("horizon")

    available_metrics = [c for c in METRIC_COLS if c in combined.columns]
    display_cols = id_cols + available_metrics
    if "n_samples" in combined.columns:
        display_cols.append("n_samples")

    display = combined[[c for c in display_cols if c in combined.columns]].copy()

    # Sort
    if metric in display.columns:
        display = display.sort_values(metric, ascending=ascending)
    else:
        print(f"Warning: metric '{metric}' not found, using default order.")

    if top_n:
        display = display.head(top_n)

    # Format
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
        display.to_string(
            index=False,
            formatters={k: v for k, v in formatters.items() if k in display.columns},
        )
    )
    print(f"\n({len(display)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare experiment results across directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="Result directories or summary CSV files to compare.")
    parser.add_argument("--metric", type=str, default="qlike", help="Metric to sort by.")
    parser.add_argument(
        "--sort",
        type=str,
        choices=["asc", "desc"],
        default="asc",
        help="Sort order (asc=best first for loss metrics, desc=best first for R²).",
    )
    parser.add_argument("--top", type=int, default=None, help="Show only top N results.")

    args = parser.parse_args()
    compare(args.paths, args.metric, ascending=(args.sort == "asc"), top_n=args.top)


if __name__ == "__main__":
    main()
