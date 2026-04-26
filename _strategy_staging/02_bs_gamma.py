# export
"""Black-Scholes gamma helpers for the delta-hedged ATM straddle eval."""

import math

import numpy as np


def _bs_gamma(S: float, K: float, sigma: float, tau: float, r: float = 0.0) -> float:
    """Black-Scholes gamma for a European option (call or put — gamma is identical).

    Formula:
        d_1 = [ln(S / K) + (r + sigma^2 / 2) * tau] / (sigma * sqrt(tau))
        N'(x) = (1 / sqrt(2 * pi)) * exp(-x^2 / 2)
        gamma = N'(d_1) / (S * sigma * sqrt(tau))

    Parameters
    ----------
    S : float
        Spot price of the underlying (positive, same units as K).
    K : float
        Strike price (positive, same units as S).
    sigma : float
        Annualized volatility (positive; e.g., 0.18 for 18%).
    tau : float
        Time to expiry in years (non-negative; e.g., 1/252 for one trading day).
    r : float, default 0.0
        Annualized continuously-compounded risk-free rate.

    Returns
    -------
    float
        Gamma per unit underlying (the second derivative of the option price
        with respect to S). Same numeric value for the call and the put under
        the put-call parity gamma identity.

    Notes
    -----
    ATM-straddle code uses 2 * this for call+put gamma at ATM.

    Edge cases:
        tau == 0  -> 0.0 (option has expired; gamma is undefined but P&L code
                    treats it as zero by convention).
    """
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive, got S={S}, K={K}")

    if tau == 0:
        return 0.0

    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    n_prime_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return n_prime_d1 / (S * sigma * sqrt_tau)


def _bs_gamma_vec(
    S: np.ndarray,
    K: float,
    sigma: float,
    tau: np.ndarray,
    r: float = 0.0,
) -> np.ndarray:
    """Vectorized Black-Scholes gamma for the per-bar Gamma$ calculation in P&L.

    Computes gamma for each (S[i], tau[i]) pair given a single scalar strike,
    sigma, and r. Used by ``compute_delta_hedged_atm_straddle_pnl`` to evaluate
    Gamma$(b_k) = gamma(S(b_k), K, sigma_imp, tau_remaining(b_k)) * S(b_k)^2
    across every bar of a session in one shot.

    Parameters
    ----------
    S : np.ndarray
        Per-bar spot prices (positive).
    K : float
        Strike price (positive scalar).
    sigma : float
        Annualized volatility (positive scalar).
    tau : np.ndarray
        Per-bar time-to-expiry in years (non-negative; same shape as S).
    r : float, default 0.0
        Annualized continuously-compounded risk-free rate.

    Returns
    -------
    np.ndarray
        Per-bar gamma. Elements where tau[i] == 0 are 0.0; all other elements
        match the scalar ``_bs_gamma`` formula.

    Notes
    -----
    ATM-straddle code uses 2 * this for call+put gamma at ATM.
    """
    S_arr = np.asarray(S, dtype=float)
    tau_arr = np.asarray(tau, dtype=float)

    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if K <= 0:
        raise ValueError(f"K must be positive, got K={K}")
    if np.any(S_arr <= 0):
        raise ValueError("All elements of S must be positive")
    if np.any(tau_arr < 0):
        raise ValueError("All elements of tau must be non-negative")

    out = np.zeros_like(S_arr, dtype=float)
    # Only compute gamma where tau > 0; tau == 0 entries stay at the 0.0 init.
    mask = tau_arr > 0
    if not np.any(mask):
        return out

    S_m = S_arr[mask]
    tau_m = tau_arr[mask]
    sqrt_tau = np.sqrt(tau_m)
    d1 = (np.log(S_m / K) + (r + 0.5 * sigma * sigma) * tau_m) / (sigma * sqrt_tau)
    n_prime_d1 = np.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    out[mask] = n_prime_d1 / (S_m * sigma * sqrt_tau)
    return out
