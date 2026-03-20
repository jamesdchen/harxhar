"""Shared visualization functions for volatility forecast results."""

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
