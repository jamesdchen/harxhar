# Auto-generated from notebooks/06_strategy_eval.ipynb. Do not edit by hand.

"""
> **NO REAL IV DATA IS WIRED IN.** Every concrete number this module can
> produce today is computed against a synthetic implied-volatility provider.
> Resulting Sharpe / hit rate / cumulative P&L are plumbing-grade diagnostics
> only — not investment research, not a benchmark, not comparable to live
> option-trading results. Any output that names a dollar P&L is in normalized
> units (the underlying is reconstructed with an arbitrary baseline; see
> `STRAT-03`).

Strategy-level evaluation scaffold for harxhar volatility forecasts.

The primary P&L is a **delta-hedged ATM straddle**: a path-dependent eval
that consumes the per-bar realized variance and weights it by dollar-gamma
at the bar's underlying level. The diagnostic, reported alongside, is a
**variance-swap** P&L — path-independent, one-number-per-day, used to flag
divergences that signal path-dependence regimes.

Strategy code depends only on the thin :class:`IVProvider` protocol; the
shipped :class:`SyntheticIVProvider` is for plumbing tests only and is
labelled FAKE at every layer (class name, docstring, ``__init__`` warning,
``__repr__``). The :class:`OptionChainProvider` slot is a documented
``NotImplementedError`` stub awaiting real chain ingestion.

The centralized future-work tracker for harxhar lives at
``writeup/future_work.md``. Items relevant to this module:

- ``STRAT-01`` — surface dynamics / IV evolution. The scaffold freezes IV at
  session open and ignores intraday level changes, term structure, and skew.
  Replacing this requires a surface model (Heston / SVI / similar).
- ``STRAT-02`` — real option-chain ingestion (``OptionChainProvider``).
- ``STRAT-03`` — absolute SPY price level. The scaffold reconstructs the
  underlying from ``sumret`` with an arbitrary ``S0=100`` baseline; raw
  dollar magnitudes are in normalized units, ratio-based metrics are
  scale-invariant.
- ``STRAT-08`` — per-horizon QLIKE-vs-strategy-Sharpe consistency study.
  Once real IV / chain data lands, sanity-check that QLIKE-better
  forecasts tend toward higher Sharpe on this strategy eval. A research
  item, not a unit test; deferred until ``STRAT-01`` / ``STRAT-02``.

See ``writeup/future_work.md`` for the full list (``STRAT-01`` through
``STRAT-08``) and for the canonical "blocker / what changes when fixed"
notes per item. Module docstrings reference items by ID so a grep for
``STRAT-0N`` surfaces every still-stale callsite.
"""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import pandas as pd

H_BARS_PER_DAY: int = 48
"""Number of 30-min bars in a 24-hour day. The model emits this many horizons
at every issuance regardless of session length; the filter slices a
session-specific prefix."""


@runtime_checkable
class IVProvider(Protocol):
    """Pluggable implied-volatility provider.

    Returns the annualized ATM IV observed at the **session open** of
    ``trade_date``. By the scaffold's strong simplifying assumption, this
    single scalar is held constant for every bar of that day's session —
    i.e., the *level* of implied vol does not move intraday, and the
    *surface* (skew, term structure, smile) is not modeled at all. Only
    ``tau_remaining`` evolves bar-by-bar, which is mechanical (it is the
    deterministic shape of gamma's time-decay), not a market model.

    This assumption is **wrong in reality**: IV moves intraday and the
    smile / term-structure matter for any non-ATM or multi-day eval. It is
    acceptable here only for an ATM-at-open, single-day-tenor,
    scaffold-grade backtest because the dominant first-order P&L driver is
    ``(RV - IV**2)``, not surface evolution. Replacing this with a real
    surface model is an explicit known-future-work item — see
    ``writeup/future_work.md#STRAT-01``. Do NOT extend this interpretation
    to non-ATM strategies, multi-day holds, or skew trades without first
    implementing surface dynamics.

    The protocol is :func:`runtime_checkable` so notebook smoke cells can
    ``assert isinstance(provider, IVProvider)`` against any concrete
    implementation. A single-scalar-per-day contract keeps the interface
    minimal; a future option-chain / surface provider will replace this
    method with a richer one (date + strikes + tenors -> surface).
    """

    def get_atm_iv(self, trade_date: pd.Timestamp) -> float:
        """Return annualized ATM IV at the session open of ``trade_date``."""
        ...


class SyntheticIVProvider:
    """SYNTHETIC / FAKE IV. DO NOT USE FOR ANALYSIS. Plumbing tests only.

    Deterministic IV provider used to wire the strategy pipeline end-to-end
    in the absence of real option-chain data. Two construction modes:

    - ``sigma_constant`` (default 0.18): a single annualized vol, returned
      for every ``trade_date``.
    - ``sigma_series``: a per-trade-date :class:`pandas.Series` of
      annualized vols, indexed by :class:`pandas.Timestamp` keys. Lookups
      are by ``.loc[trade_date]``; missing dates raise :class:`KeyError`.

    Fakeness is signaled at every layer: the class name starts with
    ``Synthetic``, this docstring opens with the all-caps banner, the
    constructor emits a :class:`UserWarning`, and ``__repr__`` always
    prints with a ``FAKE`` prefix. See ``writeup/future_work.md#STRAT-01``
    and ``#STRAT-02`` for the path to a real provider.
    """

    def __init__(
        self,
        sigma_constant: float = 0.18,
        sigma_series: pd.Series | None = None,
    ) -> None:
        warnings.warn(
            "SyntheticIVProvider is FAKE — strategy outputs are not investment research",
            category=UserWarning,
            stacklevel=2,
        )
        self._sigma_constant: float = float(sigma_constant)
        self._sigma_series: pd.Series | None = sigma_series

    def __repr__(self) -> str:
        sigma_repr = self._sigma_constant if self._sigma_series is None else "series"
        return f"<FAKE SyntheticIVProvider sigma={sigma_repr}>"

    def get_atm_iv(self, trade_date: pd.Timestamp) -> float:
        if self._sigma_series is None:
            return self._sigma_constant
        try:
            value = self._sigma_series.loc[trade_date]
        except KeyError as exc:
            raise KeyError(f"SyntheticIVProvider: no sigma_series entry for trade_date={trade_date!r}") from exc
        return float(value)


class OptionChainProvider:
    """Real option-chain IV provider — STUB.

    Future contract: given a ``trade_date``, return the annualized ATM mid
    IV observed at session open from a real option chain. Concretely, the
    implementation will read a quote snapshot at (or immediately preceding)
    the session-open timestamp, locate the at-the-money strike on the
    nearest standard expiry, and report the mid of the bid-ask IV (or an
    OTM-symmetrized average of call and put IVs at the ATM strike).

    Candidate data sources to evaluate when this lands:

    - **OptionMetrics IvyDB US** — end-of-day chains with computed IVs;
      intraday is via IvyDB Intraday.
    - **CBOE DataShop** — historical quotes feed; raw bid/ask requires an
      IV computation step.
    - **ORATS** — pre-computed surfaces and per-strike IVs at multiple
      intraday snapshots.

    Source choice depends on subscription access and required intraday
    granularity. See ``writeup/future_work.md#STRAT-02`` for the tracking
    item; that ticket also gates ``STRAT-07`` (transaction-cost
    calibration) since chain bid-ask drives realistic ``cost_bps``.
    """

    def get_atm_iv(self, trade_date: pd.Timestamp) -> float:
        raise NotImplementedError(
            "OptionChainProvider.get_atm_iv is a stub. See "
            "writeup/future_work.md#STRAT-02 for the future contract. Candidate "
            "data sources: OptionMetrics IvyDB US, CBOE DataShop, ORATS."
        )


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


"""Trade-date bucketing, session-bar enumeration, and underlying-price reconstruction.

This module supplies the calendar primitives used by the strategy-eval scaffold:

- `_compute_trade_date`: maps each bar timestamp to the trade-date label of the
  trading session containing it, given a chosen trading-day boundary.
- `_session_bars`: enumerates the ordered list of 30-min bar realization
  timestamps making up a given session.
- `_reconstruct_underlying_prices`: reconstructs an underlying price series from
  the per-bar log-return sum (`sumret`) since the repo data has no absolute
  price column.

Conventions follow `src/loading.py`: bars are 30-min, continuous 24/5 from
Sunday 18:30 ET through Friday 20:00 ET, weekends dropped, all timestamps
tz-naive ET. Timestamps refer to the *end* of the bar (`endbartime`).
"""

from datetime import time as _time

import numpy as np
import pandas as pd

from .loading import FREQ, FRIDAY_CLOSE, START_DATE, SUBGROUPS, SUNDAY_OPEN, load_raw_data

# Legal trading-day-boundary values. The aggregator passes one of these strings.
_LEGAL_BOUNDARIES: tuple[str, ...] = ("16:00", "17:00", "18:30")

# 30-min bar duration as a pd.Timedelta.
_BAR: pd.Timedelta = pd.Timedelta(minutes=30)


def _parse_boundary(boundary: str) -> _time:
    """Validate and parse a boundary string into a `datetime.time`.

    Raises `ValueError` with the legal-values list if `boundary` is not one of
    the supported boundaries.
    """
    if boundary not in _LEGAL_BOUNDARIES:
        raise ValueError(f"Invalid trading_day_boundary {boundary!r}; legal values are {list(_LEGAL_BOUNDARIES)}")
    return pd.Timestamp(f"1900-01-01 {boundary}").time()


def _compute_trade_date(
    timestamps: pd.DatetimeIndex | pd.Series,
    boundary: str = "16:00",
) -> pd.DatetimeIndex:
    """Map bar timestamps to trade-date labels under a trading-day boundary.

    The trade-date label of bar `t` is the date of the trading session that
    contains `t`. Convention: a bar at exactly the boundary on day `D` belongs
    to `D`'s session; a bar strictly after the boundary belongs to the next
    trading session.

    For `boundary='16:00'` (default, equity-options convention):

    - `Mon 14:00 -> Mon` (intraday Mon, before boundary)
    - `Mon 16:00 -> Mon` (exactly at boundary; current session)
    - `Mon 16:30 -> Tue` (after boundary; next session)
    - `Sun 19:00 -> Mon` (Sunday-open bars belong to Monday's session)
    - `Fri 17:00 -> next Mon` (post-Fri-close tail belongs to next Monday)

    The returned `pd.DatetimeIndex` is tz-naive (matching the input convention
    in `src/loading.py`) and contains midnight-ET dates. The output length
    equals the input length; one label per input timestamp.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex or pd.Series
        Bar end-timestamps (`endbartime` per `src/loading.py`), tz-naive ET.
    boundary : str
        One of `'16:00'`, `'17:00'`, `'18:30'`. Anything else raises
        `ValueError` listing the legal values.

    Returns
    -------
    pd.DatetimeIndex
        Trade-date labels (midnight ET, tz-naive), one per input timestamp.
    """
    b: _time = _parse_boundary(boundary)
    idx: pd.DatetimeIndex = pd.DatetimeIndex(pd.Index(timestamps))

    times: np.ndarray = np.asarray(idx.time)
    weekdays: np.ndarray = np.asarray(idx.weekday)  # Mon=0 .. Sun=6
    base_dates: pd.DatetimeIndex = idx.normalize()

    # `after_boundary` is True iff the bar is strictly after the boundary on
    # its calendar day. A bar exactly at the boundary belongs to the same
    # day's session.
    after_boundary: np.ndarray = np.array([t > b for t in times], dtype=bool)

    # Compute per-row day shift (in calendar days) from `base_dates` to the
    # session label.
    shifts: np.ndarray = np.zeros(len(idx), dtype=np.int64)

    for i in range(len(idx)):
        wd = int(weekdays[i])
        ab = bool(after_boundary[i])
        if wd <= 3:  # Mon..Thu
            shifts[i] = 1 if ab else 0
        elif wd == 4:  # Fri
            # Pre-or-at-boundary on Friday -> Friday's session. Post-boundary
            # tail (Fri 16:30..20:00 under '16:00') -> next Monday (+3 days).
            shifts[i] = 3 if ab else 0
        elif wd == 5:  # Sat (should not occur on the 24/5 grid, but handle it)
            shifts[i] = 2  # -> next Monday
        else:  # wd == 6, Sun
            # Sunday bars (Sun 18:30 onward on the grid) belong to Monday's
            # session regardless of `after_boundary` (Sunday has no own
            # session under any of the supported boundaries).
            shifts[i] = 1

    return pd.DatetimeIndex(base_dates + pd.to_timedelta(shifts, unit="D"))


def _session_bars(
    trade_date: pd.Timestamp,
    boundary: str = "16:00",
) -> pd.DatetimeIndex:
    """Return the ordered bar end-timestamps of session `trade_date`.

    Bars are 30-min apart on the 24/5 grid (continuous Sun 18:30 ET .. Fri
    20:00 ET, weekends dropped) used by `src/loading.py`. The returned index
    is the list of bar realization-timestamps `b_0, b_1, ..., b_{N_D - 1}` for
    the session, in time order.

    Calendar logic mirrors `_compute_trade_date`: a candidate bar timestamp
    `t` is a session bar of `trade_date` iff `_compute_trade_date([t], boundary)`
    returns `trade_date`.

    Under `boundary='16:00'`:

    - Monday session: Sun 19:00 .. Mon 16:00 -> 43 bars.
    - Tue/Wed/Thu session: prev-day 16:30 .. today 16:00 -> 48 bars.
    - Fri session: Thu 16:30 .. Fri 16:00 -> 48 bars.

    Parameters
    ----------
    trade_date : pd.Timestamp
        A trade-date label (midnight ET, tz-naive). Must be a weekday on which
        a session exists under the chosen boundary.
    boundary : str
        One of `'16:00'`, `'17:00'`, `'18:30'`.

    Returns
    -------
    pd.DatetimeIndex
        Ordered bar end-timestamps belonging to the session. Empty if
        `trade_date` is not a valid session label.
    """
    _parse_boundary(boundary)  # validation only
    td: pd.Timestamp = pd.Timestamp(trade_date).normalize()

    # Build a generous candidate window around `td` and filter via
    # `_compute_trade_date`. The widest session shape is the Monday session
    # (Sun ~18:30 .. Mon end-of-boundary), so a window of [-2 days, +1 day]
    # always covers it. A +1-day cushion at the end captures any Friday-tail
    # bars that map back to next Monday.
    win_start: pd.Timestamp = td - pd.Timedelta(days=2)
    win_end: pd.Timestamp = td + pd.Timedelta(days=1)
    candidates: pd.DatetimeIndex = pd.date_range(start=win_start, end=win_end, freq=FREQ)

    # Apply the same 24/5 market-hours filter as `src/loading.py` so we never
    # return weekend or post-Friday-close bars.
    weekday: np.ndarray = np.asarray(candidates.weekday)
    tod: np.ndarray = np.asarray(candidates.time)
    fri_close = pd.Timestamp(f"1900-01-01 {FRIDAY_CLOSE}").time()
    sun_open = pd.Timestamp(f"1900-01-01 {SUNDAY_OPEN}").time()

    keep: np.ndarray = np.array(
        [not ((wd == 4 and t > fri_close) or (wd == 5) or (wd == 6 and t < sun_open)) for wd, t in zip(weekday, tod)],
        dtype=bool,
    )
    candidates = candidates[keep]

    # Bucket each candidate to its trade-date label, then keep only those
    # matching `td`.
    labels: pd.DatetimeIndex = _compute_trade_date(candidates, boundary=boundary)
    mask: np.ndarray = np.asarray(labels) == np.datetime64(td)
    return pd.DatetimeIndex(candidates[mask].sort_values())


def _reconstruct_underlying_prices(
    data_path: str,
    *,
    S0: float = 100.0,
) -> pd.Series:
    """Reconstruct the underlying price series from per-bar `sumret`.

    Absolute level is normalized; Sharpe / hit-rate / t-stat are scale-invariant.
    Raw P&L is in normalized units. See writeup/future_work.md#STRAT-03.

    The repo data carries no absolute price column. We reconstruct an
    underlying time series by exponentiating the cumulative sum of `sumret`
    (the per-bar log-return sum, member of `SUBGROUPS['moments']` in
    `core_stats.parquet` and surfaced by `load_raw_data`):

        S(t) = S0 * exp(cumsum(sumret))

    The first value of the returned series equals `S0` (the cumulative sum at
    `t_0` is taken to be 0; `sumret[t_0]` is treated as the return realized
    *into* `t_0`'s bar end-timestamp and is applied at the next step).

    Parameters
    ----------
    data_path : str
        Path passed through to `load_raw_data` (file or directory of parquet).
    S0 : float
        Arbitrary baseline level. Default `100.0`. The output is normalized;
        ratio metrics (Sharpe, hit rate, t-stat) are unaffected by this choice.

    Returns
    -------
    pd.Series
        Series of reconstructed prices indexed by bar end-timestamp `t`,
        named `'S'`. First value is `S0`.
    """
    df: pd.DataFrame = load_raw_data(data_path)
    if "sumret" not in df.columns:
        raise KeyError(
            "Column 'sumret' not found in loaded data; expected from SUBGROUPS['moments'] in core_stats.parquet."
        )
    if "sumret" not in SUBGROUPS["moments"]:
        # Defensive: keep the docstring's invariant honest.
        raise RuntimeError(
            "'sumret' missing from SUBGROUPS['moments']; "
            "loading.py contract changed — update _reconstruct_underlying_prices."
        )

    sumret: pd.Series = pd.Series(
        df["sumret"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(df["t"]),
        name="sumret",
    )

    # First bar's price is S0; subsequent bars accumulate the log-return sum
    # realized between consecutive bar-ends.
    log_levels: np.ndarray = np.empty(len(sumret), dtype=float)
    if len(log_levels) > 0:
        log_levels[0] = 0.0
        if len(log_levels) > 1:
            log_levels[1:] = np.cumsum(sumret.to_numpy()[1:])

    prices: pd.Series = pd.Series(
        S0 * np.exp(log_levels),
        index=sumret.index,
        name="S",
    )
    return prices


# Reference for downstream callers / readability; not strictly required.
_ = START_DATE


def filter_intraday_estimate(
    chunk_df: pd.DataFrame,
    *,
    trading_day_boundary: str = "16:00",
    summary_extract: Literal["session_open", "session_mid", "session_close"] = "session_open",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the per-issuance filter path and per-day summary from a chunk of model outputs.

    Returns ``(path_df, daily_df)``. See plan ``how-would-we-start-virtual-gadget.md``
    section "Aggregation Contract -- aggregate_to_daily" / "Filter Algorithm -- Exact
    Specification" for the canonical algorithm. This function follows that pseudocode
    line-for-line; if a true bug is found, fix it in the plan first.

    Key invariants the implementation MUST preserve (verbatim from the plan):

      INVARIANT 1: path_df rows exist only for (D, t_i) pairs where t_i was actually
        issued in the chunk. No fabrication of filter entries from earlier issuances.
        This is the load-bearing correctness property; everything else is bookkeeping.
      INVARIANT 2: For any path_df row at (D, t_i): remaining_pred is computed from
        pred_raw[t_i, h=1..N_D-i] only -- never from any other issuance.
      INVARIANT 3: observed_so_far for (D, t_i) is sum(true_raw[realization=b_k]) for
        k=0..i-1, sourced anywhere in the chunk -- this is permissible because true_raw
        is the realized value (model-conditioning-independent), not a model output.
      INVARIANT 4: The slice length N_D - i is computed from N_D (session-specific via
        the calendar) and i (issuance position). Off-by-one here is the most likely
        bug; smoke test 4 catches it.
      INVARIANT 5: Every drop is counted in attrs with a category-specific key, never
        silent.

    Note: this filter today supports only the open-time decision (the i=0 row
    drives ``daily_df.pred_var`` under ``summary_extract='session_open'``).
    The full ``path_df`` is produced for diagnostics; consuming the path for
    a continuously-rebalanced multi-period strategy is deferred — see
    ``writeup/future_work.md#STRAT-05``.
    """
    H = 48  # H_BARS_PER_DAY constant from the plan
    bar_duration = pd.Timedelta(minutes=30)

    # --- step 1: validate input (no silent dedup) ---
    required = {"date", "horizon", "true_raw", "pred_raw"}
    missing_cols = required - set(chunk_df.columns)
    if missing_cols:
        raise ValueError(
            f"chunk_df missing required columns: {sorted(missing_cols)}. Required schema: {sorted(required)}."
        )

    df = chunk_df.copy()

    # --- step 2: derive issuance_time from realization timestamp ---
    df["issuance_time"] = df["date"] - df["horizon"] * bar_duration
    df = df.sort_values(["issuance_time", "horizon"]).reset_index(drop=True)

    # No silent dedup: same (issuance_time, horizon) describes one prediction of one bar.
    if df.duplicated(subset=["issuance_time", "horizon"]).any():
        offenders = df[df.duplicated(subset=["issuance_time", "horizon"], keep=False)]
        raise ValueError(
            "Duplicate (issuance_time, horizon) rows in input. The filter does not "
            "dedup -- fix at the loader (chunk-stitching is the wrong layer to paper "
            f"over upstream nondeterminism / overlapping concat). Offending rows:\n"
            f"{offenders.head().to_string()}"
        )

    max_h = int(df["horizon"].max())
    if max_h != H:
        raise ValueError(
            f"Input does not satisfy the H={H} contract; max horizon = {max_h}. "
            f"The filter requires every issuance to emit horizons h=1..{H}. "
            f"Existing h=1-only trial dirs do NOT satisfy this contract -- wiring an "
            f"executor to emit H={H} is a separate downstream task. "
            f"See writeup/future_work.md and the notebook 'Input contract' section."
        )

    # --- step 3: bucket each issuance by trade_date under the configured boundary ---
    # An issuance at t_i predicts bars b_i, b_{i+1}, ... where b_i = t_i + bar.
    # The trade_date of an issuance is therefore the trade_date of its first
    # predicted bar (b_i), NOT of t_i itself. Without this correction, an
    # issuance landing exactly on the boundary (e.g., Mon 16:00) would be
    # assigned to the previous session even though it's the open issuance for
    # the next session.
    df["trade_date"] = _compute_trade_date(df["issuance_time"] + bar_duration, trading_day_boundary)

    # Build session bar lists once per trade_date.
    sessions: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for D in pd.unique(df["trade_date"]):
        sessions[D] = list(_session_bars(D, trading_day_boundary))

    # session_index i for each row: the integer i with b_i == issuance_time + B.
    def _idx(D: pd.Timestamp, t_i: pd.Timestamp) -> int:
        bars = sessions[D]
        target = t_i + bar_duration
        for k, b in enumerate(bars):
            if b == target:
                return k
        return -1

    df["session_index"] = [_idx(row.trade_date, row.issuance_time) for row in df.itertuples(index=False)]

    # --- step 4: validate per-session (summary index present + realization coverage) ---
    realized_set = set(df["date"])
    i_target_map = {"session_open": 0, "session_mid": None, "session_close": None}
    if summary_extract not in i_target_map:
        raise ValueError(f"summary_extract must be one of {list(i_target_map)}; got {summary_extract!r}.")

    valid_sessions: list[tuple[pd.Timestamp, int, int]] = []
    missing_summary_index_count = 0
    session_dropped_realization_gap_count = 0

    for D, bars in sessions.items():
        N_D = len(bars)
        if summary_extract == "session_open":
            i_t = 0
        elif summary_extract == "session_mid":
            i_t = N_D // 2
        else:  # session_close
            i_t = N_D - 1

        rows_D = df[df["trade_date"] == D]
        present_indices = set(int(x) for x in rows_D["session_index"].unique() if x >= 0)

        if i_t not in present_indices:
            missing_summary_index_count += 1
            continue

        missing_real = [k for k in range(i_t) if bars[k] not in realized_set]
        if missing_real:
            session_dropped_realization_gap_count += 1
            continue

        valid_sessions.append((D, N_D, i_t))

    # --- step 5: build path_df (one row per actually-present issuance per valid session) ---
    # Pre-index true_raw by realization timestamp -- first-match lookup only. This implements
    # INVARIANT 3 (true_raw sourced anywhere in chunk; first matching row's value is canonical).
    first_true_by_date: dict[pd.Timestamp, float] = {}
    for row in df.itertuples(index=False):
        if row.date not in first_true_by_date:
            first_true_by_date[row.date] = float(row.true_raw)

    path_rows: list[dict[str, object]] = []
    for D, N_D, _ in valid_sessions:
        bars = sessions[D]
        rows_D = df[df["trade_date"] == D]
        present_is = sorted(int(x) for x in rows_D["session_index"].unique() if x >= 0)
        for i in present_is:
            t_i = bars[i] - bar_duration
            # INVARIANT 3: observed_so_far over realized bars b_0..b_{i-1}.
            observed = float(np.sum([first_true_by_date[bars[k]] for k in range(i)])) if i > 0 else 0.0
            # INVARIANT 2: remaining_pred from THIS issuance's pred_raw at h=1..(N_D-i) only.
            rem_horizons = list(range(1, N_D - i + 1))
            sub = df[(df["issuance_time"] == t_i) & (df["horizon"].isin(rem_horizons))]
            if len(sub) != len(rem_horizons):
                missing_h = sorted(set(rem_horizons) - set(int(h) for h in sub["horizon"]))
                raise ValueError(
                    f"Issuance {t_i} on trade_date {D} is present but missing horizons "
                    f"{missing_h} in 1..{N_D - i}; violates the full-H emission contract "
                    f"(every issuance must emit h=1..{H}, of which 1..N_D-i is the in-session prefix)."
                )
            remaining = float(sub["pred_raw"].sum())
            est = observed + remaining
            path_rows.append(
                {
                    "trade_date": D,
                    "issuance_time": t_i,
                    "session_index": i,
                    "est_cum_pred_var": est,
                    "observed_so_far": observed,
                    "remaining_pred": remaining,
                    "remaining_bars": N_D - i,
                }
            )

    path_df = pd.DataFrame(
        path_rows,
        columns=[
            "trade_date",
            "issuance_time",
            "session_index",
            "est_cum_pred_var",
            "observed_so_far",
            "remaining_pred",
            "remaining_bars",
        ],
    )

    # --- step 6: build daily_df by extracting one row per session at i_target ---
    daily_rows: list[dict[str, object]] = []
    for D, N_D, i_t in valid_sessions:
        bars = sessions[D]
        path_for_D = path_df[(path_df["trade_date"] == D) & (path_df["session_index"] == i_t)]
        # INVARIANT 1: i_t guaranteed present by step-4 validation, so this is exactly one row.
        pred_var = float(path_for_D.iloc[0]["est_cum_pred_var"])
        present_bars = [b for b in bars if b in realized_set]
        n_bars = len(present_bars)
        real_var = float(np.sum([first_true_by_date[b] for b in present_bars])) if present_bars else 0.0
        full_day = n_bars == N_D
        daily_rows.append(
            {
                "trade_date": D,
                "pred_var": pred_var,
                "real_var": real_var,
                "n_bars": n_bars,
                "expected_bars": N_D,
                "full_day": full_day,
            }
        )

    daily_df = pd.DataFrame(
        daily_rows,
        columns=[
            "trade_date",
            "pred_var",
            "real_var",
            "n_bars",
            "expected_bars",
            "full_day",
        ],
    )

    # --- step 7: sort and tag attrs (INVARIANT 5: every drop counted) ---
    if not path_df.empty:
        path_df = path_df.sort_values(["trade_date", "session_index"]).reset_index(drop=True)
    if not daily_df.empty:
        daily_df = daily_df.sort_values("trade_date").reset_index(drop=True)

    partial_day_count = int((~daily_df["full_day"]).sum()) if not daily_df.empty else 0

    daily_df.attrs = {
        "H": H,
        "trading_day_boundary": trading_day_boundary,
        "summary_extract": summary_extract,
        "missing_summary_index_count": missing_summary_index_count,
        "session_dropped_realization_gap_count": session_dropped_realization_gap_count,
        "partial_day_count": partial_day_count,
    }

    return path_df, daily_df


"""P&L computation for the delta-hedged ATM straddle strategy and its
variance-swap diagnostic, plus standard strategy metrics.

This module assumes the helper functions `_bs_gamma` and `_session_bars` are
defined elsewhere in the same emitted `src/strategy_eval.py` module (see the
sibling staging files). Strategy code consumes the daily summary produced by
`filter_intraday_estimate` and the original chunk DataFrame; it depends on
implied vol only through the thin `IVProvider` protocol.
"""

from typing import Literal

import numpy as np
import pandas as pd

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
