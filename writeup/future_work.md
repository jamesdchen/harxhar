# harxhar — Future Work Tracker

This is the durable tracker for known-but-deferred work across harxhar modules. Each entry has a stable ID; module docstrings cross-reference IDs (e.g., `"See writeup/future_work.md#STRAT-03"`) so the canonical description lives in one place. Deleting an entry should be followed by a grep for its ID to find and clean up stale references in code. New modules append entries with their own ID prefix (e.g., `EVAL-01`, `LOAD-01`) — no restructuring required.

## Summary

| ID | Module | Subject | Blocker |
|---|---|---|---|
| [STRAT-01](#strat-01--surface-dynamics--iv-evolution) | `src/strategy_eval.py` | Surface dynamics / IV evolution | Real chain data + research time on the surface model |
| [STRAT-02](#strat-02--real-option-chain-ingestion) | `src/strategy_eval.py` | Real option-chain ingestion | Data subscription / source decision |
| [STRAT-03](#strat-03--absolute-spy-price-level) | `src/strategy_eval.py` | Absolute SPY price level | None — implementable today |
| [STRAT-04](#strat-04--delta-hedge-rebalancing-frequency) | `src/strategy_eval.py` | Delta-hedge rebalancing frequency | Unblocked — purely implementation |
| [STRAT-05](#strat-05--continuously-rebalanced-multi-period-strategies) | `src/strategy_eval.py` | Continuously-rebalanced multi-period strategies | Unblocked but research-heavy |
| [STRAT-06](#strat-06--strike-policies-beyond-atm-at-open) | `src/strategy_eval.py` | Strike policies beyond ATM-at-open | Unblocked, additive |
| [STRAT-07](#strat-07--transaction-cost-calibration) | `src/strategy_eval.py` | Transaction cost calibration | Depends on `STRAT-02` |
| [STRAT-08](#strat-08--per-horizon-qlike-vs-strategy-sharpe-consistency-study) | `src/strategy_eval.py` | Per-horizon QLIKE-vs-strategy-Sharpe consistency study | Depends on `STRAT-01` / `STRAT-02` (real data) |

---

### STRAT-01 — Surface dynamics / IV evolution

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** The scaffold freezes IV at session open and ignores intraday level changes, term structure, and skew. Once strategies extend beyond same-day ATM (multi-day holds, OTM strikes, calendar spreads, skew trades), the frozen-ATM-IV assumption breaks.
- **Concrete change required:** `IVProvider` gains methods like `get_iv_surface(t, strikes, tenors)` and `get_atm_iv_intraday(t)`; `compute_delta_hedged_atm_straddle_pnl` uses bar-level IV; new strategies plug in. Likely first concrete model: a Heston-style stochastic vol or an SVI-fit static surface refitted nightly — discussion deferred until real chain data is in hand.
- **Blocker:** Real chain data + research time on the surface model.
- **Where flagged in code:** `IVProvider` protocol docstring (first paragraph), `compute_delta_hedged_atm_straddle_pnl` "Simplifying assumptions" block (item i), module docstring.

### STRAT-02 — Real option-chain ingestion

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** `OptionChainProvider` is a `NotImplementedError` stub today. The scaffold cannot produce strategy P&L against real implied vol until a chain provider lands.
- **Concrete change required:** Implement `OptionChainProvider` against one of the candidate sources: OptionMetrics IvyDB US, CBOE DataShop, ORATS. Honors the `IVProvider` protocol — date → ATM mid IV at session open.
- **Blocker:** Data subscription / source decision.
- **Where flagged in code:** `OptionChainProvider` stub docstring lists the candidate sources; module docstring.

### STRAT-03 — Absolute SPY price level

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** `_reconstruct_underlying_prices` builds the underlying via `S = S0 × exp(cumsum(sumret))` with `S0 = 100` because no actual price column lives in the repo data. Output JSON tags this with `"underlying_source": "reconstructed_from_sumret_S0_100"`. Sharpe, hit rate, t-stat, and any ratio-based metric are **scale-invariant** and unaffected. Raw dollar P&L magnitudes are in normalized units — a "$1 P&L" reported by the scaffold is `$1 / (real_S0 / 100)` of real-world dollar P&L. Anyone reading dollar magnitudes without remembering this will misinterpret them.
- **Concrete failure mode:** A stakeholder asks "what's the dollar P&L of this strategy on $10M notional" — answer is wrong by a factor of `real_S0 / 100`.
- **Concrete change required:** Wire a real $-denominated SPY series (yfinance, vendor feed, or a corrections file) into a new `underlying_source='real'` mode of `_reconstruct_underlying_prices`. Use real opens at session open as `S0` for that day; subsequent bars still rebuild via `cumsum(sumret)` to stay consistent with the rest of the data. No other strategy code changes — `K`, `Γ$`, and per-bar P&L all just inherit the new scale.
- **Blocker:** None, beyond "actually getting a SPY level series." Implementable today.
- **Where flagged in code:** `_reconstruct_underlying_prices` docstring (first paragraph), output-JSON warning string, and notebook banner.

### STRAT-04 — Delta-hedge rebalancing frequency

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** The current P&L computation implicitly assumes a delta hedge is rebalanced *every* 30-min bar. Each bar's contribution `(1/2) × Γ$ × (RV − IV²)Δt` is the discrete-time approximation of the continuous-hedge integral. Real options traders rebalance on event-based or cost-aware cadences (e.g., when delta drifts > X, or every N hours), not every bar. The scaffold's reported P&L corresponds to a frictionless 30-min-rebalance regime. Real strategies with sparser hedging will produce noisier P&L (un-hedged delta exposure between rebalances injects directional return into the eval); strategies with hedging too dense will pay too much in costs.
- **Concrete failure mode:** Comparing the scaffold's P&L to a real trader's monthly P&L and concluding the strategy is uneconomic when the real difference is hedging-cadence specification, not signal quality.
- **Concrete change required:** Add a `hedge_freq: int | 'event_band'` parameter to `compute_delta_hedged_atm_straddle_pnl`.
  - `hedge_freq=1` is the current behavior.
  - `hedge_freq=N` aggregates `N` bars between rebalances — un-hedged delta-driven P&L (`Δ × ΔS`) accumulates between rebalances and is added to the gamma term.
  - `'event_band'` triggers rebalances when `|Δ_current − Δ_at_last_rebalance| > threshold`.
  - Transaction costs are charged per rebalance (not per bar), so `cost_bps` becomes meaningful.
- **Blocker:** Unblocked — purely implementation; needs a clean derivation written into the docstring so the math is auditable.
- **Where flagged in code:** `compute_delta_hedged_atm_straddle_pnl` docstring (in the "Simplifying assumptions" block as item iv after the existing three), and the bar-level `bar_pnl_df` returned by the function carries a `hedge_freq` column reading 1 today so the assumption is in any saved diagnostic.

### STRAT-05 — Continuously-rebalanced multi-period strategies

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** The filter already produces an intraday path of forecasts; today only the `i=0` row drives an open-time decision. A follow-up uses the full path for intraday rebalancing.
- **Concrete change required:** Add a position-sizing rule, intraday cost model, and re-evaluation of the look-ahead lag. Consume the full `path_df` rather than only the session-open row.
- **Blocker:** Unblocked but research-heavy.
- **Where flagged in code:** `filter_intraday_estimate` docstring; module docstring.

### STRAT-06 — Strike policies beyond ATM-at-open

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** Strike policy is `ATM-at-open, frozen for the day` (`K = S(b_0(D))`). Other policies (rolling-ATM (re-strike each bar), fixed-strike (rolling expiries), vol-targeted notional) are all deferred.
- **Concrete change required:** Extend the `strike_policy` parameter on `compute_delta_hedged_atm_straddle_pnl` to handle `'rolling_atm'`, `'fixed_strike'`, `'vol_targeted'`. Each policy plugs in additively without disturbing the base ATM-at-open path.
- **Blocker:** Unblocked, additive.
- **Where flagged in code:** `compute_delta_hedged_atm_straddle_pnl` docstring (`strike_policy` parameter); module docstring.

### STRAT-07 — Transaction cost calibration

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** `cost_bps` defaults to 0; without real chain bid-ask values, transaction costs cannot be calibrated to a meaningful number.
- **Concrete change required:** Once chain data is available (via `STRAT-02`), derive realistic `cost_bps` from bid-ask spreads at the strikes/tenors used by each strategy.
- **Blocker:** Depends on `STRAT-02`.
- **Where flagged in code:** `compute_delta_hedged_atm_straddle_pnl` docstring (`cost_bps` parameter).

### STRAT-08 — Per-horizon QLIKE-vs-strategy-Sharpe consistency study

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** Once real numbers exist, sanity-check that QLIKE-better models tend toward higher Sharpe on the strategy eval. Not a unit test — a research item.
- **Concrete change required:** Run the strategy eval across all executors at multiple horizons against real IV data; rank by QLIKE and by strategy Sharpe; report rank correlation and any regimes where the two diverge.
- **Blocker:** Depends on `STRAT-01` / `STRAT-02` (real data) before the study is meaningful.
- **Where flagged in code:** Module docstring of `src/strategy_eval.py`.
