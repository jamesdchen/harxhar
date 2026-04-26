"""Post-assembly validation harness for src/strategy_eval.py.

Validates that the regenerated module (after `make pipeline-export`) has all
expected public/internal names with the right types and signatures, without
running any actual evaluation logic.

Run from repo root:
    python scripts/validate_strategy_eval.py
"""

from __future__ import annotations

import inspect
import sys
import warnings
from collections.abc import Callable
from typing import Any


def _check(name: str, fn: Callable[[], None], results: list[tuple[str, bool, str]]) -> None:
    """Run a single check, capture pass/fail with reason."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return
    results.append((name, True, ""))
    print(f"[OK] {name}")


def _is_protocol(cls: Any) -> bool:
    """Heuristic check that `cls` is a typing.Protocol class."""
    return bool(getattr(cls, "_is_protocol", False) or getattr(cls, "_is_runtime_protocol", False))


def main() -> int:
    try:
        import src.strategy_eval as se
    except ImportError as e:
        print(
            "[FAIL] import src.strategy_eval: "
            "src/strategy_eval.py not found - run `make pipeline-export` first "
            f"(underlying error: {e})"
        )
        return 1

    results: list[tuple[str, bool, str]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # H_BARS_PER_DAY
        def check_h_bars() -> None:
            val = se.H_BARS_PER_DAY
            if not isinstance(val, int):
                raise TypeError(f"expected int, got {type(val).__name__}")
            if val != 48:
                raise ValueError(f"expected 48, got {val}")

        _check("H_BARS_PER_DAY", check_h_bars, results)

        # IVProvider Protocol
        def check_ivprovider() -> None:
            cls = se.IVProvider
            if not inspect.isclass(cls):
                raise TypeError("not a class")
            if not _is_protocol(cls):
                raise TypeError("not a typing.Protocol")

        _check("IVProvider", check_ivprovider, results)

        # SyntheticIVProvider
        def check_synthetic() -> None:
            cls = se.SyntheticIVProvider
            if not inspect.isclass(cls):
                raise TypeError("not a class")
            inst = cls()  # no-arg construction
            if not hasattr(inst, "get_atm_iv"):
                raise AttributeError("missing get_atm_iv method")

        _check("SyntheticIVProvider", check_synthetic, results)

        # OptionChainProvider
        def check_optionchain() -> None:
            cls = se.OptionChainProvider
            if not inspect.isclass(cls):
                raise TypeError("not a class")
            try:
                inst = cls()
            except TypeError:
                # If the class requires args, try to instantiate via __new__
                inst = cls.__new__(cls)
            if not hasattr(inst, "get_atm_iv"):
                raise AttributeError("missing get_atm_iv method")
            try:
                # Best-effort call - args unknown, try a few shapes
                try:
                    inst.get_atm_iv()
                except TypeError:
                    try:
                        inst.get_atm_iv(None)
                    except TypeError:
                        inst.get_atm_iv(None, None)
            except NotImplementedError:
                return
            except Exception as e:  # noqa: BLE001
                raise AssertionError(f"get_atm_iv raised {type(e).__name__}, expected NotImplementedError") from e
            raise AssertionError("get_atm_iv did not raise NotImplementedError")

        _check("OptionChainProvider", check_optionchain, results)

        # _bs_gamma
        def check_bs_gamma() -> None:
            fn = se._bs_gamma
            if not callable(fn):
                raise TypeError("not callable")
            out = fn(100, 100, 0.2, 1 / 252)
            out_f = float(out)
            if not (0.0 < out_f < 1.0):
                raise ValueError(f"expected float in (0, 1), got {out_f}")

        _check("_bs_gamma", check_bs_gamma, results)

        # _bs_gamma_vec
        def check_bs_gamma_vec() -> None:
            fn = se._bs_gamma_vec
            if not callable(fn):
                raise TypeError("not callable")

        _check("_bs_gamma_vec", check_bs_gamma_vec, results)

        # _compute_trade_date
        def check_compute_trade_date() -> None:
            fn = se._compute_trade_date
            if not callable(fn):
                raise TypeError("not callable")

        _check("_compute_trade_date", check_compute_trade_date, results)

        # _session_bars
        def check_session_bars() -> None:
            fn = se._session_bars
            if not callable(fn):
                raise TypeError("not callable")

        _check("_session_bars", check_session_bars, results)

        # _reconstruct_underlying_prices
        def check_reconstruct() -> None:
            fn = se._reconstruct_underlying_prices
            if not callable(fn):
                raise TypeError("not callable")

        _check("_reconstruct_underlying_prices", check_reconstruct, results)

        # filter_intraday_estimate
        def check_filter_intraday() -> None:
            fn = se.filter_intraday_estimate
            if not callable(fn):
                raise TypeError("not callable")
            sig = inspect.signature(fn)
            params = sig.parameters
            for required in ("chunk_df", "trading_day_boundary", "summary_extract"):
                if required not in params:
                    raise TypeError(f"missing parameter '{required}'")

        _check("filter_intraday_estimate", check_filter_intraday, results)

        # compute_delta_hedged_atm_straddle_pnl
        def check_straddle() -> None:
            fn = se.compute_delta_hedged_atm_straddle_pnl
            if not callable(fn):
                raise TypeError("not callable")

        _check("compute_delta_hedged_atm_straddle_pnl", check_straddle, results)

        # compute_variance_swap_pnl_diagnostic
        def check_varswap() -> None:
            fn = se.compute_variance_swap_pnl_diagnostic
            if not callable(fn):
                raise TypeError("not callable")

        _check("compute_variance_swap_pnl_diagnostic", check_varswap, results)

        # compute_strategy_metrics
        def check_metrics() -> None:
            fn = se.compute_strategy_metrics
            if not callable(fn):
                raise TypeError("not callable")

        _check("compute_strategy_metrics", check_metrics, results)

    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"VALIDATION: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
