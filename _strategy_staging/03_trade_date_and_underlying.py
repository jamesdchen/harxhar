# export
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

# Imports hoisted to 01_module_header: datetime.time as _time, numpy as np,
# pandas as pd, and from .loading import FREQ / FRIDAY_CLOSE / START_DATE /
# SUBGROUPS / SUNDAY_OPEN / load_raw_data.

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
        [
            not ((wd == 4 and t > fri_close) or (wd == 5) or (wd == 6 and t < sun_open))
            for wd, t in zip(weekday, tod, strict=False)
        ],
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
