# export
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
