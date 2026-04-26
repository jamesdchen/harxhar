# export
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

import math
import warnings
from datetime import time as _time
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .loading import FREQ, FRIDAY_CLOSE, START_DATE, SUBGROUPS, SUNDAY_OPEN, load_raw_data

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
