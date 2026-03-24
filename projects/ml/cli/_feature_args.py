"""Lightweight feature-related CLI arguments.

Extracted from executor.py so that submit-only code paths can add
feature arguments to parsers without importing heavy compute
dependencies (numpy, tqdm, torch, …).
"""

from __future__ import annotations

import argparse


def add_feature_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add feature-related arguments shared between executor and submit parsers."""
    parser.add_argument("--train-window", type=int, default=500, help="Training window in days")
    parser.add_argument(
        "--n-components", type=int, default=5, help="Number of PCA/AE latent components (--features pca or ae)"
    )
    parser.add_argument(
        "--ae-alpha", type=float, default=0.5, help="AE loss weight: alpha*recon + (1-alpha)*pred (--features ae)"
    )
    parser.add_argument("--ae-epochs", type=int, default=50, help="Training epochs per AE refit (--features ae)")
    parser.add_argument(
        "--ae-hidden", type=int, default=0, help="AE hidden layer width; 0 = auto (n_features // 2) (--features ae)"
    )
    parser.add_argument(
        "--ae-weights-path", type=str, default=None, help="Path to pre-trained AE weights .pt file (--features ae)"
    )
    return parser
