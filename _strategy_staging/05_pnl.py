# export
"""P&L computation for the delta-hedged ATM straddle strategy and its
variance-swap diagnostic, plus standard strategy metrics.

This module assumes the helper functions `_bs_gamma` and `_session_bars` are
defined elsewhere in the same emitted `src/strategy_eval.py` module (see the
sibling staging files). Strategy code consumes the daily summary produced by
`filter_intraday_estimate` and the original chunk DataFrame; it depends on
implied vol only through the thin `IVProvider` protocol.
"""

# Imports (typing.Literal, numpy as np, pandas as pd) hoisted to 01_module_header.

# Annualization constants. 252 trading days * 48 thirty-minute bars per
# 24-hour day. The continuous 24/5 grid in `src/loading.py` motivates the
# 48-bar-per-day convention even though equity sessions are shorter.
_TRADING_DAYS_PER_YEAR: int = 252
_BARS_PER_DAY: int = 48
_DTAU_BAR: float = 1.0 / (_TRADING_DAYS_PER_YEAR * _BARS_PER_DAY)


def compute_delta_hedged_atm_straddle_pnl(
    daily_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    iv_provider,
    underlying_prices: pd.Series,
    *,
    strike_policy: str = "atm_at_open",
    cost_bps: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Delta-hedged ATM straddle P&L (PRIMARY strategy P&L).

    Simplifying assumptions
    -----------------------
    (i)   IV level is frozen for the day after the open observation. The
          single scalar returned by `iv_provider.get_atm_iv(D)` drives every
          bar of session `D`; intraday IV moves are not modeled. See
          ``writeup/future_work.md#STRAT-01``.
    (ii)  No skew, smile, or term structure — a single ATM IV scalar drives
          all gamma calcs. The function does not consult any surface
          (strike-, tenor-, or moneyness-dependent IV) anywhere. See
          ``writeup/future_work.md#STRAT-01``.
    (iii) Strike is fixed at ATM-at-open: `K = S(b_0(D))`. Other strike
          policies (rolling-ATM, fixed-strike, vol-targeted) are out of
          scope and raise ``NotImplementedError``. See
          ``writeup/future_work.md#STRAT-06``.
    (iv)  Delta hedge rebalances every bar (``hedge_freq=1``); ``r=0`` and
          no dividends. The discrete-bar P&L expression below is the
          frictionless 30-min-rebalance approximation of the continuous
          gamma-P&L integral; coarser hedging cadences inject un-hedged
          delta exposure into the realized P&L. See
          ``writeup/future_work.md#STRAT-04``.

    Algorithm (per trade_date `D` in ``daily_df``)
    ----------------------------------------------
    1. ``K = S(b_0(D))`` from ``underlying_prices`` at session-open bar.
    2. ``sigma_imp = iv_provider.get_atm_iv(D)``.
    3. ``tau_full_day = N_D / (252 * 48)`` where ``N_D = daily_df.expected_bars[D]``.
    4. ``position_sign = sign(daily_df.pred_var[D] - sigma_imp**2 * tau_full_day)``.
       Uses ``np.sign`` returning -1/0/+1.
    5. For each bar ``b_k`` in ``_session_bars(D, '16:00')``:
         - ``S_k = underlying_prices.loc[b_k]``
         - ``tau_remaining = (N_D - k) * Delta_tau_bar``
         - ``gamma_k = _bs_gamma(S_k, K, sigma_imp, tau_remaining)``
         - ``DollarGamma_k = 2 * gamma_k * S_k**2`` (factor 2 = call + put
           gamma at ATM straddle)
         - ``true_raw_bar`` = realized variance of ``b_k`` taken from
           ``chunk_df``: any row whose ``date`` (realization timestamp)
           equals ``b_k`` (``true_raw`` is model-conditioning-independent).
         - ``expected_var_bar = sigma_imp**2 * Delta_tau_bar``
         - ``pnl_bar = 0.5 * DollarGamma_k * (true_raw_bar - expected_var_bar) * position_sign``
    6. Sum bar P&L per ``D``. If ``position_sign != 0`` apply a transaction
       cost of ``cost_bps * 1e-4 * notional`` once per trade_date, where
       ``notional = K`` (a proxy — full-spec P&L would charge against the
       straddle premium, not the strike, but the strike-as-notional proxy is
       order-of-magnitude correct for ATM and avoids requiring an option
       pricer in the scaffold).

    Parameters
    ----------
    daily_df : pd.DataFrame
        Output of ``filter_intraday_estimate``'s daily summary. Required
        columns: ``trade_date``, ``pred_var``, ``real_var``, ``n_bars``,
        ``expected_bars``, ``full_day``.
    chunk_df : pd.DataFrame
        The original chunk DataFrame. Required columns: ``date`` (bar
        realization timestamp), ``true_raw``. Used to look up the per-bar
        realized variance.
    iv_provider : IVProvider
        Anything implementing ``get_atm_iv(trade_date) -> float`` returning
        annualized ATM IV at session open of ``trade_date``.
    underlying_prices : pd.Series
        Reconstructed underlying price series, indexed by bar end-timestamp.
        See ``_reconstruct_underlying_prices``.
    strike_policy : str, default 'atm_at_open'
        Only ``'atm_at_open'`` is supported. Anything else raises
        ``NotImplementedError`` referencing ``STRAT-06``.
    cost_bps : float, default 0.0
        Round-trip transaction cost in basis points of notional. Charged
        once per trade_date when ``position_sign != 0``.

    Returns
    -------
    daily_pnl_series : pd.Series
        Total daily P&L, indexed by ``trade_date``.
    bar_pnl_df : pd.DataFrame
        Per-bar diagnostic with columns ``trade_date``, ``bar_timestamp``,
        ``S``, ``K``, ``sigma_imp``, ``tau_remaining``, ``gamma``,
        ``dollar_gamma``, ``true_raw_bar``, ``expected_var_bar``,
        ``pnl_bar``, ``position_sign``, ``hedge_freq``.

    Notes
    -----
    The ``hedge_freq`` column in ``bar_pnl_df`` is currently always 1 by
    assumption (iv); it is included so the assumption travels with any
    saved diagnostic and a future relaxation (``STRAT-04``) does not change
    the schema.
    """
    if strike_policy != "atm_at_open":
        raise NotImplementedError(
            f"strike_policy={strike_policy!r} is not supported. Only "
            f"'atm_at_open' is implemented today; rolling-ATM, "
            f"fixed-strike, and vol-targeted variants are deferred. "
            f"See writeup/future_work.md#STRAT-06."
        )

    # Build a lookup: bar_timestamp -> true_raw. `true_raw` is realized and
    # model-conditioning-independent, so any matching row works; take the
    # first occurrence per bar timestamp.
    true_raw_by_bar: pd.Series = (
        chunk_df[["date", "true_raw"]].drop_duplicates(subset=["date"], keep="first").set_index("date")["true_raw"]
    )

    daily_pnl: dict[pd.Timestamp, float] = {}
    bar_rows: list[dict] = []

    for _, row in daily_df.iterrows():
        D: pd.Timestamp = pd.Timestamp(row["trade_date"])
        N_D: int = int(row["expected_bars"])
        pred_var_D: float = float(row["pred_var"])

        bars: pd.DatetimeIndex = _session_bars(D, "16:00")  # noqa: F821
        if len(bars) == 0:
            daily_pnl[D] = 0.0
            continue

        # 1. Strike at session open.
        b0: pd.Timestamp = bars[0]
        K: float = float(underlying_prices.loc[b0])

        # 2. ATM IV scalar for the day.
        sigma_imp: float = float(iv_provider.get_atm_iv(D))

        # 3. Full-day tau.
        tau_full_day: float = N_D * _DTAU_BAR

        # 4. Position sign at open.
        position_sign: float = float(np.sign(pred_var_D - sigma_imp * sigma_imp * tau_full_day))

        # 5. Per-bar P&L.
        day_pnl_sum: float = 0.0
        for k, b_k in enumerate(bars):
            try:
                S_k: float = float(underlying_prices.loc[b_k])
            except KeyError:
                # Missing underlying price for this bar; skip it. This
                # surfaces as a session-level partial P&L and is consistent
                # with the filter's `n_bars < expected_bars` partial-day
                # handling.
                continue
            tau_remaining: float = (N_D - k) * _DTAU_BAR
            gamma_k: float = float(
                _bs_gamma(S_k, K, sigma_imp, tau_remaining)  # noqa: F821
            )
            dollar_gamma: float = 2.0 * gamma_k * S_k * S_k

            true_raw_bar: float = float(true_raw_by_bar.loc[b_k]) if b_k in true_raw_by_bar.index else float("nan")
            expected_var_bar: float = sigma_imp * sigma_imp * _DTAU_BAR

            if np.isnan(true_raw_bar):
                pnl_bar: float = 0.0
            else:
                pnl_bar = 0.5 * dollar_gamma * (true_raw_bar - expected_var_bar) * position_sign

            day_pnl_sum += pnl_bar
            bar_rows.append(
                {
                    "trade_date": D,
                    "bar_timestamp": b_k,
                    "S": S_k,
                    "K": K,
                    "sigma_imp": sigma_imp,
                    "tau_remaining": tau_remaining,
                    "gamma": gamma_k,
                    "dollar_gamma": dollar_gamma,
                    "true_raw_bar": true_raw_bar,
                    "expected_var_bar": expected_var_bar,
                    "pnl_bar": pnl_bar,
                    "position_sign": position_sign,
                    "hedge_freq": 1,
                }
            )

        # 6. Transaction cost (once per trade_date).
        if position_sign != 0.0 and cost_bps != 0.0:
            notional: float = K  # proxy; documented in docstring.
            day_pnl_sum -= cost_bps * 1e-4 * notional

        daily_pnl[D] = day_pnl_sum

    daily_pnl_series: pd.Series = pd.Series(daily_pnl, name="pnl").sort_index()
    daily_pnl_series.index.name = "trade_date"

    bar_pnl_df: pd.DataFrame = pd.DataFrame(
        bar_rows,
        columns=[
            "trade_date",
            "bar_timestamp",
            "S",
            "K",
            "sigma_imp",
            "tau_remaining",
            "gamma",
            "dollar_gamma",
            "true_raw_bar",
            "expected_var_bar",
            "pnl_bar",
            "position_sign",
            "hedge_freq",
        ],
    )

    return daily_pnl_series, bar_pnl_df


def compute_variance_swap_pnl_diagnostic(
    daily_df: pd.DataFrame,
    iv_provider,
    *,
    rule: Literal["sign"] = "sign",
) -> pd.Series:
    """Path-INDEPENDENT diagnostic. Reported alongside delta-hedged P&L;
    large divergences flag path-dependence regimes (e.g., high-gamma days
    where S strays from strike then returns).

    For each trade_date `D` in ``daily_df``:

    - ``sigma_imp = iv_provider.get_atm_iv(D)``
    - ``implied_var = sigma_imp**2 * (N_D / (252 * 48))`` where
      ``N_D = daily_df.expected_bars[D]``
    - ``signal = np.sign(daily_df.pred_var[D] - implied_var)``
    - ``pnl = signal * (daily_df.real_var[D] - implied_var)``

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily summary from ``filter_intraday_estimate``. Required columns:
        ``trade_date``, ``pred_var``, ``real_var``, ``expected_bars``.
    iv_provider : IVProvider
        Anything implementing ``get_atm_iv(trade_date) -> float``.
    rule : {'sign'}
        Position-sizing rule. Only ``'sign'`` is implemented (binary
        long/short on ``sign(pred_var - implied_var)``).

    Returns
    -------
    pd.Series
        Daily P&L indexed by ``trade_date``.
    """
    if rule != "sign":
        raise NotImplementedError(f"rule={rule!r} is not supported; only 'sign' is implemented.")

    pnl: dict[pd.Timestamp, float] = {}
    for _, row in daily_df.iterrows():
        D: pd.Timestamp = pd.Timestamp(row["trade_date"])
        N_D: int = int(row["expected_bars"])
        sigma_imp: float = float(iv_provider.get_atm_iv(D))
        implied_var: float = sigma_imp * sigma_imp * (N_D * _DTAU_BAR)
        pred_var_D: float = float(row["pred_var"])
        real_var_D: float = float(row["real_var"])

        signal: float = float(np.sign(pred_var_D - implied_var))
        pnl[D] = signal * (real_var_D - implied_var)

    out: pd.Series = pd.Series(pnl, name="pnl_varswap").sort_index()
    out.index.name = "trade_date"
    return out


def compute_strategy_metrics(pnl: pd.Series) -> dict[str, float]:
    """Standard strategy metrics for a daily P&L series.

    Mirrors the shape of ``src/evaluation.py:calculate_metrics``: a flat
    dict of named scalar metrics suitable for JSON serialization. All
    metrics are scale-invariant ratios (Sharpe, hit rate, t-stat) or sums
    in the same units as the input ``pnl``.

    Parameters
    ----------
    pnl : pd.Series
        Daily P&L indexed by trade_date.

    Returns
    -------
    dict[str, float]
        Keys:

        - ``sharpe_annual`` : float
            Annualized Sharpe ratio: ``mean(pnl) / std(pnl) * sqrt(252)``
            using the sample std (ddof=1). NaN if ``std == 0`` or if fewer
            than two observations.
        - ``hit_rate`` : float
            Fraction of days with ``pnl > 0``. NaN if empty.
        - ``t_stat`` : float
            T-statistic of the mean P&L: ``mean(pnl) / SE(pnl)`` where
            ``SE = std(ddof=1) / sqrt(n)``. NaN if ``std == 0`` or if fewer
            than two observations.
        - ``cumulative_pnl`` : float
            Sum of ``pnl``.
        - ``max_drawdown`` : float
            Maximum peak-to-trough drawdown of the cumulative P&L curve,
            reported as a non-negative number (max of running peak minus
            running cumulative). 0.0 on an empty input.
        - ``n_days`` : int
            Number of observations in ``pnl``.
    """
    s: pd.Series = pd.Series(pnl).dropna()
    n: int = int(len(s))

    if n == 0:
        return {
            "sharpe_annual": float("nan"),
            "hit_rate": float("nan"),
            "t_stat": float("nan"),
            "cumulative_pnl": 0.0,
            "max_drawdown": 0.0,
            "n_days": 0,
        }

    arr: np.ndarray = s.to_numpy(dtype=float)
    mean_pnl: float = float(np.mean(arr))
    std_pnl: float = float(np.std(arr, ddof=1)) if n >= 2 else 0.0

    if std_pnl > 0.0 and n >= 2:
        sharpe_annual: float = mean_pnl / std_pnl * float(np.sqrt(252))
        se: float = std_pnl / float(np.sqrt(n))
        t_stat: float = mean_pnl / se
    else:
        sharpe_annual = float("nan")
        t_stat = float("nan")

    hit_rate: float = float(np.mean(arr > 0.0))
    cumulative_pnl: float = float(np.sum(arr))

    cum_curve: np.ndarray = np.cumsum(arr)
    running_peak: np.ndarray = np.maximum.accumulate(cum_curve)
    drawdowns: np.ndarray = running_peak - cum_curve
    max_drawdown: float = float(np.max(drawdowns)) if drawdowns.size > 0 else 0.0

    return {
        "sharpe_annual": sharpe_annual,
        "hit_rate": hit_rate,
        "t_stat": t_stat,
        "cumulative_pnl": cumulative_pnl,
        "max_drawdown": max_drawdown,
        "n_days": n,
    }
