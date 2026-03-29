"""CLI entry points for GPU-based backtests.

- gpu_executor — single-run entry point: parses args, loads data, dispatches to
  the appropriate GPU backtest engine.  ``python -m projects.dl.cli.gpu_executor --help``
- lifecycle — submit + status tracking for SLURM array jobs.
  ``python -m projects.dl.cli.lifecycle --help``
"""
