"""Shared utilities for Colab/Jupyter notebooks."""

import os


def configure_cuda():
    """Set CUDA performance flags (TF32, matmul precision, memory)."""
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def verify_gpu():
    """Assert GPU is available and print device info."""
    import torch

    assert torch.cuda.is_available(), "No GPU detected! Change runtime to GPU."
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {name} ({vram:.1f} GB VRAM)")


def save_results(df, results_dir, filename):
    """Save a DataFrame to CSV and return the path."""
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, filename)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    return csv_path


def download_results(csv_path):
    """Download results via Colab files API, or print local path."""
    try:
        from google.colab import files

        files.download(csv_path)
    except ImportError:
        print(f"Not running on Colab. Results at: {os.path.abspath(csv_path)}")


def print_metrics(metrics):
    """Print QLIKE, MSE, MAE from a metrics dict."""
    print(f"QLIKE: {metrics.get('qlike', float('nan')):.4f}")
    print(f"MSE:   {metrics.get('mse', float('nan')):.6f}")
    print(f"MAE:   {metrics.get('mae', float('nan')):.6f}")
