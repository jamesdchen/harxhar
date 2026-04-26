"""Contract-surface tests for src.strategy_eval.

Validates that the regenerated module (after `make pipeline-export`) exposes all
expected public/internal names with the right types and signatures, without
running any actual evaluation logic. Each former nested check in
scripts/validate_strategy_eval.py is its own test_* function so pytest reports
per-symbol pass/fail granularity.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import pytest

import src.strategy_eval as se


@pytest.fixture(autouse=True)
def _silence_warnings():
    """Mirror the original validator's warnings.catch_warnings() block."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _is_protocol(cls: Any) -> bool:
    """Heuristic check that `cls` is a typing.Protocol class."""
    return bool(getattr(cls, "_is_protocol", False) or getattr(cls, "_is_runtime_protocol", False))


def test_h_bars_per_day():
    val = se.H_BARS_PER_DAY
    assert isinstance(val, int), f"expected int, got {type(val).__name__}"
    assert val == 48, f"expected 48, got {val}"


def test_ivprovider_protocol():
    cls = se.IVProvider
    assert inspect.isclass(cls), "IVProvider is not a class"
    assert _is_protocol(cls), "IVProvider is not a typing.Protocol"


def test_synthetic_iv_provider():
    cls = se.SyntheticIVProvider
    assert inspect.isclass(cls), "SyntheticIVProvider is not a class"
    inst = cls()  # no-arg construction
    assert hasattr(inst, "get_atm_iv"), "SyntheticIVProvider missing get_atm_iv method"


def test_option_chain_provider():
    cls = se.OptionChainProvider
    assert inspect.isclass(cls), "OptionChainProvider is not a class"
    try:
        inst = cls()
    except TypeError:
        # If the class requires args, instantiate via __new__ (sibling-plan fix:
        # use object.__new__ to sidestep abstract/protocol __init__).
        inst = object.__new__(cls)
    assert hasattr(inst, "get_atm_iv"), "OptionChainProvider missing get_atm_iv method"

    raised_not_implemented = False
    try:
        try:
            inst.get_atm_iv()
        except TypeError:
            try:
                inst.get_atm_iv(None)
            except TypeError:
                inst.get_atm_iv(None, None)
    except NotImplementedError:
        raised_not_implemented = True
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"get_atm_iv raised {type(e).__name__}, expected NotImplementedError") from e
    assert raised_not_implemented, "get_atm_iv did not raise NotImplementedError"


def test_bs_gamma():
    fn = se._bs_gamma
    assert callable(fn), "_bs_gamma not callable"
    out = fn(100, 100, 0.2, 1 / 252)
    out_f = float(out)
    assert 0.0 < out_f < 1.0, f"expected float in (0, 1), got {out_f}"


def test_bs_gamma_vec():
    fn = se._bs_gamma_vec
    assert callable(fn), "_bs_gamma_vec not callable"


def test_compute_trade_date():
    fn = se._compute_trade_date
    assert callable(fn), "_compute_trade_date not callable"


def test_session_bars():
    fn = se._session_bars
    assert callable(fn), "_session_bars not callable"


def test_reconstruct_underlying_prices():
    fn = se._reconstruct_underlying_prices
    assert callable(fn), "_reconstruct_underlying_prices not callable"


def test_filter_intraday_estimate():
    fn = se.filter_intraday_estimate
    assert callable(fn), "filter_intraday_estimate not callable"
    sig = inspect.signature(fn)
    params = sig.parameters
    for required in ("chunk_df", "trading_day_boundary", "summary_extract"):
        assert required in params, f"filter_intraday_estimate missing parameter '{required}'"


def test_compute_delta_hedged_atm_straddle_pnl():
    fn = se.compute_delta_hedged_atm_straddle_pnl
    assert callable(fn), "compute_delta_hedged_atm_straddle_pnl not callable"


def test_compute_variance_swap_pnl_diagnostic():
    fn = se.compute_variance_swap_pnl_diagnostic
    assert callable(fn), "compute_variance_swap_pnl_diagnostic not callable"


def test_compute_strategy_metrics():
    fn = se.compute_strategy_metrics
    assert callable(fn), "compute_strategy_metrics not callable"
