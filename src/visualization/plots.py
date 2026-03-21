"""Shared visualization functions for volatility forecast results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_timeseries_forecast(results, model_name, color="#1f77b4", n_tail=500):
    """Plot true vs predicted realized volatility (last n_tail points)."""
    fig, ax = plt.subplots(figsize=(14, 5))

    subset = results.iloc[-n_tail:]
    ax.plot(subset.index, subset["true_raw"], label="True RV", alpha=0.6, color="black")
    ax.plot(subset.index, subset["pred_raw"], label="Predicted RV", alpha=0.8, color=color)
    ax.set_title(f"{model_name}: Realized Volatility Forecast", fontsize=13, fontweight="bold")
    ax.set_ylabel("Realized Volatility")
    ax.set_xlabel("Sample Index")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_diagnostic_scatter(results, model_name, color="#1f77b4"):
    """Log-log scatter of true vs predicted volatility with 45-degree line."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(results["true_raw"], results["pred_raw"], alpha=0.3, s=10, color=color)
    lims = [results["true_raw"].min(), results["true_raw"].max()]
    ax.plot(lims, lims, "k-", alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True Volatility", fontsize=11)
    ax.set_ylabel("Predicted Volatility", fontsize=11)
    ax.set_title(f"{model_name}: Diagnostic Scatter", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_residual_histogram(results, model_name, color="#1f77b4"):
    """Histogram of prediction residuals (predicted - true)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    residuals = results["pred_raw"] - results["true_raw"]
    ax.hist(residuals, bins=100, alpha=0.7, color=color, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Residual (Predicted - True)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{model_name}: Residual Distribution", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_training_losses(loss_csv_path, model_name="Model", sample_chunks=10):
    """Plot per-epoch training losses from a loss CSV file.

    Parameters
    ----------
    loss_csv_path : str
        Path to CSV with columns [chunk, epoch, loss].
    model_name : str
        Label for the plot title.
    sample_chunks : int
        Max number of chunks to overlay individually. If there are more chunks,
        a random sample is drawn and the rest are shown as a faint aggregate.
    """
    df = pd.read_csv(loss_csv_path)

    chunks = df["chunk"].unique()
    n_chunks = len(chunks)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # --- Left: individual loss curves (sampled) ---
    ax = axes[0]
    if n_chunks <= sample_chunks:
        show_chunks = chunks
    else:
        rng = np.random.default_rng(42)
        show_chunks = rng.choice(chunks, size=sample_chunks, replace=False)

    for c in show_chunks:
        sub = df[df["chunk"] == c]
        ax.plot(sub["epoch"], sub["loss"], alpha=0.6, linewidth=0.8, label=f"chunk {c}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name}: Training Loss per Chunk (n={n_chunks})", fontweight="bold")
    if n_chunks <= sample_chunks:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Right: aggregate (mean +/- std across chunks) ---
    ax = axes[1]
    agg = df.groupby("epoch")["loss"].agg(["mean", "std"])
    ax.plot(agg.index, agg["mean"], color="black", linewidth=1.5, label="Mean")
    ax.fill_between(
        agg.index,
        agg["mean"] - agg["std"],
        agg["mean"] + agg["std"],
        alpha=0.25,
        color="steelblue",
        label="$\\pm$ 1 std",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name}: Aggregate Training Loss", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig
