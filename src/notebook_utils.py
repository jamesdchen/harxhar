"""Shared utilities for Colab/Jupyter notebooks."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# --- Drive status file paths ---
DRIVE_STATUS_DIR = "/content/drive/MyDrive/harxhar_status"
STATUS_FILENAME = "dl_runner.json"
STATUS_PATH = Path(DRIVE_STATUS_DIR) / STATUS_FILENAME


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
    torch.cuda.get_device_name(0)


# --- Status management for MCP-driven cell execution ---


def write_status(status: str, **kwargs) -> dict:
    """Create or update the Drive-persisted status JSON.

    Merges *kwargs* into the existing status dict (if any).  Always sets
    ``status`` and ``updated_at``.  On the first call also sets ``started_at``.
    Uses atomic write (temp file + rename) so Cell 7 can safely read mid-run.
    """
    existing = read_status() or {}
    existing.update(kwargs)
    existing["status"] = status
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    existing.setdefault("started_at", existing["updated_at"])

    os.makedirs(DRIVE_STATUS_DIR, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=DRIVE_STATUS_DIR, suffix=".tmp", prefix="status_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        os.replace(tmp_path, str(STATUS_PATH))
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return existing


def read_status() -> dict | None:
    """Read the Drive-persisted status JSON, or *None* if it does not exist."""
    try:
        with open(STATUS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def clear_status() -> None:
    """Delete the status JSON so each notebook session starts fresh."""
    with contextlib.suppress(FileNotFoundError):
        os.remove(STATUS_PATH)


def get_gpu_utilization() -> dict:
    """Query nvidia-smi and return a dict of GPU stats.

    Keys: ``gpu_name``, ``gpu_util_pct``, ``mem_used_mb``,
    ``mem_total_mb``, ``temp_c``.  Returns ``{"error": ...}`` on failure.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip()}
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "gpu_name": parts[0],
            "gpu_util_pct": int(parts[1]),
            "mem_used_mb": int(parts[2]),
            "mem_total_mb": int(parts[3]),
            "temp_c": int(parts[4]),
        }
    except Exception as exc:
        return {"error": str(exc)}


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


PROGRESS_PATH = "/content/harxhar_progress.json"


def read_progress() -> dict | None:
    """Read the live training progress JSON, or *None* if it does not exist."""
    try:
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def format_eta(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60
    return f"{hours:.1f}hr"


def recommend_nap(progress: dict) -> tuple[int, str]:
    """Return (nap_seconds, reason) based on training progress and stability.

    Adaptive logic:
    - Early in training (< 10% done): short naps (3 min) — pace still settling.
    - Stable pace (recent avg within 20% of overall avg): longer naps.
    - Unstable pace (> 20% deviation): shorter naps.
    - ETA > 30 min and stable: 15 min naps.
    - ETA 10–30 min and stable: 10 min naps.
    - ETA < 10 min: 3 min naps (finishing soon).
    """
    pct = progress.get("pct_complete", 0)
    eta = progress.get("eta_sec", 0)
    avg = progress.get("avg_chunk_sec", 1)
    recent = progress.get("recent_avg_chunk_sec", avg)

    # Pace stability: how much recent pace deviates from overall
    if avg > 0:
        pace_deviation = abs(recent - avg) / avg
    else:
        pace_deviation = 0

    stable = pace_deviation < 0.2

    if pct < 10:
        return 180, "early training — pace still settling"
    if eta < 600:  # < 10 min left
        return 180, "finishing soon"
    if eta < 1800:  # < 30 min left
        if stable:
            return 600, "stable pace, moderate time remaining"
        return 300, "pace fluctuating, moderate time remaining"
    # > 30 min left
    if stable:
        return 900, "stable pace, long run"
    return 600, "pace fluctuating, long run"


def print_metrics(metrics):
    """Print QLIKE, MSE, MAE from a metrics dict."""
    print(f"QLIKE: {metrics.get('qlike', float('nan')):.4f}")
    print(f"MSE:   {metrics.get('mse', float('nan')):.6f}")
    print(f"MAE:   {metrics.get('mae', float('nan')):.6f}")
