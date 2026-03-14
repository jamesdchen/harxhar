"""Loss functions and compiled training kernels for GPU backtesting."""

import torch
from torch.func import vmap, functional_call, grad
from torch.amp import autocast
from torch.optim.adamw import adamw
from src import config as cfg


def functional_qlike_loss(h_pred, target_sqrt):
    """
    QLIKE parameterized in log-space for numerical stability.

    h_pred: model output = log(sigma^2_pred), unconstrained real number
    target_sqrt: adj_RV (sqrt-space target from codebase pipeline)

    L = sigma^2_true * exp(-h_pred) + h_pred
    dL/dh = -sigma^2_true * exp(-h) + 1   (always bounded, no log(0) or div-by-zero)
    """
    target_sq = target_sqrt.squeeze().float() ** 2
    h = h_pred.squeeze().float()
    h = torch.clamp(h, min=cfg.QLIKE_CLAMP_MIN, max=cfg.QLIKE_CLAMP_MAX)
    return target_sq * torch.exp(-h) + h


def make_train_kernel(base_model, param_keys, num_epochs, base_lr):
    """Build a compiled training loop using functional API and vmap."""

    def compute_loss_stateless(params, buffers, x, y):
        x_in = x.unsqueeze(-1)
        h_pred = functional_call(base_model, (params, buffers), args=(x_in,), kwargs={})
        return functional_qlike_loss(h_pred, y)

    batch_loss_fn = vmap(compute_loss_stateless, in_dims=(0, None, 0, 0), randomness='different')

    def train_loop(params, buffers, exp_avgs, exp_avg_sqs, step_tensors, X, y):

        for i in range(1, num_epochs + 1):
            def mean_loss(p):
                with autocast('cuda'):
                    losses = batch_loss_fn(p, buffers, X, y)
                    return losses.mean()

            grads_dict = grad(mean_loss)(params)

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
                    capturable=True
                )
                params = {k: mutable_params[idx] for idx, k in enumerate(param_keys)}

        return params

    return train_loop


def make_ae_train_kernel(base_model, param_keys, num_epochs, base_lr,
                         alpha_recon):
    """
    Build a compiled AE training loop using functional API and vmap.

    Uses hybrid MSE loss: alpha * MSE(recon, x) + (1-alpha) * MSE(pred, y).
    The base_model must be a LagAutoEncoder whose forward() returns
    (reconstructed, z, pred_rv).

    Parameters
    ----------
    base_model : LagAutoEncoder
    param_keys : list[str]
    num_epochs : int
    base_lr : float
    alpha_recon : float
        Weight for reconstruction loss; (1-alpha_recon) weights prediction loss.
    """

    def compute_loss_stateless(params, buffers, x, y):
        # x: (train_win, n_features), y: (train_win,)
        reconstructed, _z, pred_rv = functional_call(
            base_model, (params, buffers), args=(x,), kwargs={})
        recon_loss = ((reconstructed - x) ** 2).mean()
        pred_loss = ((pred_rv - y) ** 2).mean()
        return alpha_recon * recon_loss + (1.0 - alpha_recon) * pred_loss

    batch_loss_fn = vmap(compute_loss_stateless, in_dims=(0, None, 0, 0),
                         randomness='different')

    def train_loop(params, buffers, exp_avgs, exp_avg_sqs, step_tensors, X, y):

        for i in range(1, num_epochs + 1):
            def mean_loss(p):
                with autocast('cuda'):
                    losses = batch_loss_fn(p, buffers, X, y)
                    return losses.mean()

            grads_dict = grad(mean_loss)(params)

            grad_list = []
            found_inf = torch.tensor(False, device=X.device)

            for k in param_keys:
                g = grads_dict[k]
                g = torch.clamp(g, min=-cfg.GRAD_CLIP_BOUND,
                                max=cfg.GRAD_CLIP_BOUND)
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
                    capturable=True
                )
                params = {k: mutable_params[idx]
                          for idx, k in enumerate(param_keys)}

        return params

    return train_loop
