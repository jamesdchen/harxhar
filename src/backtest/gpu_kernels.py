"""Compiled training kernels for GPU backtesting."""

import torch
from torch.amp import autocast
from torch.func import functional_call, grad_and_value, vmap
from torch.optim.adamw import adamw

from src.core import config as cfg
from src.models.losses import functional_qlike_loss


def _make_train_kernel(batch_loss_fn, param_keys, num_epochs, base_lr):
    """Core training loop factory shared by all kernel builders.

    Parameters
    ----------
    batch_loss_fn : callable
        vmap-wrapped loss: (params_dict, buffers, X_batch, y_batch) -> losses.
    param_keys : list[str]
    num_epochs : int
    base_lr : float

    Returns
    -------
    train_loop : callable
        Returns (params, epoch_losses) where epoch_losses is a 1-D tensor
        of length num_epochs containing the mean loss at each epoch.
    """

    def train_loop(params, buffers, exp_avgs, exp_avg_sqs, step_tensors, X, y):
        epoch_losses = torch.zeros(num_epochs, device=X.device)

        for _i in range(1, num_epochs + 1):

            def mean_loss(p):
                with autocast("cuda"):
                    losses = batch_loss_fn(p, buffers, X, y)
                    return losses.mean()

            grads_dict, loss_val = grad_and_value(mean_loss)(params)
            epoch_losses[_i - 1] = loss_val.detach()

            grad_list = []
            found_inf = torch.tensor(False, device=X.device)

            for k in param_keys:
                g = grads_dict[k]
                g = torch.clamp(g, min=-cfg.GRAD_CLIP_BOUND, max=cfg.GRAD_CLIP_BOUND)
                grad_list.append(g)
                if not torch.isfinite(g).all():
                    found_inf = torch.tensor(True, device=X.device)

            if not found_inf:
                mutable_params = [params[k].clone() for k in param_keys]
                mutable_exp_avgs = [exp_avgs[k] for k in param_keys]
                mutable_exp_avg_sqs = [exp_avg_sqs[k] for k in param_keys]
                mutable_steps = [step_tensors[k] for k in param_keys]

                adamw(
                    params=mutable_params,
                    grads=grad_list,
                    exp_avgs=mutable_exp_avgs,
                    exp_avg_sqs=mutable_exp_avg_sqs,
                    max_exp_avg_sqs=[],
                    state_steps=mutable_steps,
                    amsgrad=False,
                    beta1=cfg.ADAMW_BETA1,
                    beta2=cfg.ADAMW_BETA2,
                    lr=base_lr,
                    weight_decay=cfg.ADAMW_WEIGHT_DECAY,
                    eps=cfg.NORM_EPS,
                    maximize=False,
                    foreach=False,
                    capturable=True,
                )
                params = {k: mutable_params[idx] for idx, k in enumerate(param_keys)}

        return params, epoch_losses

    return train_loop


def make_train_kernel(base_model, param_keys, num_epochs, base_lr):
    """Build a compiled training loop for PatchTS using QLIKE loss."""

    def compute_loss_stateless(params, buffers, x, y):
        x_in = x.unsqueeze(-1)
        h_pred = functional_call(base_model, (params, buffers), args=(x_in,), kwargs={})
        return functional_qlike_loss(h_pred, y)

    batch_loss_fn = vmap(compute_loss_stateless, in_dims=(0, None, 0, 0), randomness="different")
    return _make_train_kernel(batch_loss_fn, param_keys, num_epochs, base_lr)


def make_ae_train_kernel(base_model, param_keys, num_epochs, base_lr, alpha_recon):
    """Build a compiled AE training loop using hybrid MSE loss.

    Uses: alpha * MSE(recon, x) + (1-alpha) * MSE(pred, y).
    The base_model must be a LagAutoEncoder whose forward() returns
    (reconstructed, z, pred_rv).
    """

    def compute_loss_stateless(params, buffers, x, y):
        reconstructed, _z, pred_rv = functional_call(base_model, (params, buffers), args=(x,), kwargs={})
        recon_loss = ((reconstructed - x) ** 2).mean()
        pred_loss = ((pred_rv - y) ** 2).mean()
        return alpha_recon * recon_loss + (1.0 - alpha_recon) * pred_loss

    batch_loss_fn = vmap(compute_loss_stateless, in_dims=(0, None, 0, 0), randomness="different")
    return _make_train_kernel(batch_loss_fn, param_keys, num_epochs, base_lr)
