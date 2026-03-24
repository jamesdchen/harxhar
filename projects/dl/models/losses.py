"""Loss functions for volatility forecasting models."""

import torch

from core.core import config as cfg


def functional_qlike_loss(h_pred, target_sqrt):
    """
    QLIKE parameterized in log-space for numerical stability.

    h_pred: model output = log(sigma^2_pred), shape (...,) or (..., H)
    target_sqrt: adj_RV (sqrt-space target from codebase pipeline), same shape

    L = sigma^2_true * exp(-h_pred) + h_pred
    dL/dh = -sigma^2_true * exp(-h) + 1   (always bounded, no log(0) or div-by-zero)

    For multi-horizon (H > 1), computes element-wise loss and averages over horizons.
    """
    target_sq = target_sqrt.float() ** 2
    h = h_pred.float()
    h = torch.clamp(h, min=cfg.QLIKE_CLAMP_MIN, max=cfg.QLIKE_CLAMP_MAX)
    per_element = target_sq * torch.exp(-h) + h
    # Average over horizon dimension if multi-output
    if per_element.dim() > 1:
        return per_element.mean(dim=-1)
    return per_element
