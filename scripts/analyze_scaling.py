"""Analyze scaling law experiment results and generate plots.

Usage:
    python -m scripts.analyze_scaling
    python -m scripts.analyze_scaling --results-dir results_scaling_laws
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze scaling law results.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_scaling_laws",
        help="Directory containing scaling_results.csv",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    csv_path = os.path.join(results_dir, "scaling_results.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} experiment runs from {csv_path}")

    # Aggregate across repeats
    summary = df.groupby("multiplier").agg(
        qlike_mean=("qlike", "mean"),
        qlike_std=("qlike", "std"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        n_windows=("n_train_windows", "first"),
    ).reset_index()
    print(summary.to_string(index=False))

    x = summary["multiplier"].values
    x_plot = np.where(x == 0, 0.5, x).astype(float)

    # --- Plot 1: QLIKE scaling curve ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(
        x_plot,
        summary["qlike_mean"],
        yerr=summary["qlike_std"],
        fmt="o-",
        capsize=5,
        linewidth=2.2,
        markersize=9,
        color="#1f77b4",
        ecolor="#aec7e8",
        elinewidth=1.5,
        label="QLIKE (mean +/- std)",
    )
    ax.set_xscale("log")
    ax.set_xticks(x_plot)
    ax.set_xticklabels(
        [f"{int(m)}x" if m > 0 else "0\n(real only)" for m in x],
        fontsize=11,
    )
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda val, pos: ""))
    ax.set_xlabel("Synthetic Data Multiplier (x real training size)", fontsize=13)
    ax.set_ylabel("Test QLIKE", fontsize=13)
    ax.set_title(
        "Scaling Laws: MBB Synthetic Data Augmentation for RV Forecasting",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    out1 = os.path.join(results_dir, "scaling_curve.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # --- Plot 2: All metrics ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric, label in zip(
        axes,
        ["qlike", "mse", "mae"],
        ["QLIKE", "MSE (adj scale)", "MAE (adj scale)"],
    ):
        ax.errorbar(
            x_plot,
            summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_std"],
            fmt="o-",
            capsize=4,
            linewidth=2,
            markersize=8,
            color="#1f77b4",
        )
        ax.set_xscale("log")
        ax.set_xticks(x_plot)
        ax.set_xticklabels(
            [f"{int(m)}x" if m > 0 else "0" for m in x],
            fontsize=10,
        )
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda val, pos: ""))
        ax.set_xlabel("Synthetic Multiplier", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Scaling Laws: All Metrics", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out2 = os.path.join(results_dir, "scaling_curve_all_metrics.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")

    # --- Summary table ---
    print("=" * 80)
    print("SCALING LAW RESULTS SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False, float_format="%.6f"))


if __name__ == "__main__":
    main()
