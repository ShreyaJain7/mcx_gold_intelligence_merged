"""
Tests for the quant engine and attribution.

Run with ``pytest -q`` (or ``python tests/test_quant_engine.py`` if pytest is
not installed - a minimal runner at the bottom executes every ``test_*``).

The most important test is ``test_no_lookahead_truncation_invariance``: it
reruns the whole engine on data cut at an arbitrary date and asserts that
every residual, z-score and position on or before that date is *bit-for-bit*
identical to the full-history run. If any computation peeked at the future,
appending future rows would change the past and the test would fail.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.WARNING)

from src.attribution import compute_attribution  # noqa: E402
from src.config import CONTRACT_SPECS, CostModel, StrategyConfig  # noqa: E402
from src.quant_engine import (  # noqa: E402
    DataValidationError,
    build_pair_frame,
    normalize_prices,
    rolling_zscore,
    run_strategy,
    validate_bhavcopy,
)
from src.synthetic_data import generate_synthetic_bhavcopy  # noqa: E402

_CACHE: dict[str, pd.DataFrame] = {}


def market() -> pd.DataFrame:
    if "df" not in _CACHE:
        _CACHE["df"] = normalize_prices(validate_bhavcopy(generate_synthetic_bhavcopy()))
    return _CACHE["df"]


def _row(sym: str, close: float, date: str = "2025-01-10", exp: str = "2025-02-05") -> dict:
    return {"trade_date": pd.Timestamp(date), "symbol": sym, "expiry_date": pd.Timestamp(exp),
            "close": close, "volume": 100, "open_interest": 100}


# --------------------------------------------------------------------------- #
def test_normalization_factors():
    raw = pd.DataFrame([_row("GOLDM", 79_600.0), _row("GOLDGUINEA", 64_000.0),
                        _row("GOLDPETAL", 8_000.0), _row("GOLDTEN", 80_000.0)])
    out = normalize_prices(validate_bhavcopy(raw)).set_index("symbol")["norm_price"]
    assert np.isclose(out["GOLDM"], 79_600 / 10 * 999 / 995)
    assert np.isclose(out["GOLDGUINEA"], 64_000 / 8)
    assert np.isclose(out["GOLDPETAL"], 8_000)
    assert np.isclose(out["GOLDTEN"], 8_000)


def test_guinea_per_gram_edge_case_is_rebased():
    # GOLDGUINEA published per gram (8,000) instead of per 8 g (64,000).
    raw = pd.DataFrame([_row("GOLDM", 79_600.0), _row("GOLDPETAL", 8_000.0),
                        _row("GOLDGUINEA", 8_000.0)])
    out = normalize_prices(validate_bhavcopy(raw)).set_index("symbol")
    assert np.isclose(out.at["GOLDGUINEA", "norm_price"], 8_000)
    assert bool(out.at["GOLDGUINEA", "quote_basis_fixed"])
    # ``close`` must be restated on the per-8 g basis, since P&L is marked on it.
    assert np.isclose(out.at["GOLDGUINEA", "close"], 64_000)
    assert np.isclose(out.at["GOLDGUINEA", "close_raw"], 8_000)


def test_no_single_day_pnl_explosion():
    # Guards against unit errors reaching the P&L: a 200 g spread cannot
    # plausibly make or lose more than ~5% of one leg's notional in a day.
    df = market()
    for a, b in (("GOLDM", "GOLDGUINEA"), ("GOLDGUINEA", "GOLDPETAL"), ("GOLDM", "GOLDPETAL")):
        res = run_strategy(df, a, b)
        notional = 200 * df["norm_price"].median()
        assert res.daily["gross_pnl"].abs().max() < 0.05 * notional, (a, b)


def test_genuine_guinea_premium_is_not_rebased():
    # A 3% premium is a real dislocation, not a units error: leave it alone.
    raw = pd.DataFrame([_row("GOLDM", 79_600.0), _row("GOLDPETAL", 8_000.0),
                        _row("GOLDGUINEA", 64_000.0 * 1.03)])
    out = normalize_prices(validate_bhavcopy(raw)).set_index("symbol")
    assert not bool(out.at["GOLDGUINEA", "quote_basis_fixed"])


def test_validation_drops_bad_rows_and_rejects_bad_schema():
    raw = pd.DataFrame([_row("GOLDM", 79_600.0), _row("GOLDM", 79_600.0),   # duplicate
                        _row("GOLDM", 0.0, exp="2025-03-05"),               # zero close
                        _row("GOLDM", 1.0, date="2025-03-01", exp="2025-02-05"),  # post-expiry
                        _row("SILVER", 90_000.0)])                           # unsupported
    assert len(validate_bhavcopy(raw)) == 1
    try:
        validate_bhavcopy(raw.drop(columns="open_interest"))
    except DataValidationError:
        pass
    else:
        raise AssertionError("missing column not rejected")


def test_nearest_expiry_matching():
    df = market()
    pair = build_pair_frame(df, "GOLDM", "GOLDPETAL", roll_dte=2)
    # For every date, no other eligible GOLDPETAL expiry is closer to leg A's expiry.
    sample = pair.iloc[::37]
    petal = df[(df.symbol == "GOLDPETAL") & (df.days_to_expiry > 2)]
    for t, row in sample.iterrows():
        cands = petal.loc[petal.trade_date == t, "expiry_date"]
        best = (cands - row["expiry_date_a"]).abs().min()
        assert abs((row["expiry_date_b"] - row["expiry_date_a"])) == best


def test_zscore_uses_only_past_values():
    s = pd.Series(np.arange(100, dtype=float))
    z = rolling_zscore(s, window=10, min_periods=10)
    # mean at t=20 must be mean of 10..19, i.e. excluding 20 itself
    assert np.isclose(z["rolling_mean"].iloc[20], np.mean(np.arange(10, 20)))


def test_no_lookahead_truncation_invariance():
    df = market()
    cfg = StrategyConfig()
    full = run_strategy(df, "GOLDM", "GOLDGUINEA", cfg)
    for cutoff in ("2024-09-17", "2025-04-08", "2026-01-21"):
        cut = pd.Timestamp(cutoff)
        part = run_strategy(df[df.trade_date <= cut], "GOLDM", "GOLDGUINEA", cfg)
        cols = ["spread", "fitted_spread", "residual_spread", "zscore"]
        a = full.frame.loc[:cut, cols]
        b = part.frame.loc[:cut, cols]
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        pd.testing.assert_series_equal(full.daily.loc[:cut, "position"],
                                       part.daily.loc[:cut, "position"])
        pd.testing.assert_series_equal(full.daily.loc[:cut, "net_pnl"],
                                       part.daily.loc[:cut, "net_pnl"])


def test_future_shock_does_not_change_past_signals():
    df = market().copy()
    cfg = StrategyConfig()
    base = run_strategy(df, "GOLDM", "GOLDPETAL", cfg)
    shock_date = pd.Timestamp("2025-06-16")
    m = (df.symbol == "GOLDPETAL") & (df.trade_date >= shock_date)
    df.loc[m, "norm_price"] *= 1.05
    df.loc[m, "close"] *= 1.05
    shocked = run_strategy(df, "GOLDM", "GOLDPETAL", cfg)
    before = base.frame.index < shock_date
    pd.testing.assert_series_equal(base.frame.loc[before, "zscore"],
                                   shocked.frame.loc[before, "zscore"])


def test_execution_lag_respected():
    res = run_strategy(market(), "GOLDM", "GOLDGUINEA", StrategyConfig(execution_lag=1))
    t = res.trades
    assert len(t) > 0
    assert (t["entry_date"] > t["signal_date"]).all()


def test_cost_model_sides():
    cm = CostModel()
    buy = cm.order_cost("GOLDM", 80_000, 2, "buy")
    sell = cm.order_cost("GOLDM", 80_000, 2, "sell")
    notional = 80_000 * CONTRACT_SPECS["GOLDM"].multiplier * 2
    assert np.isclose(sell - buy, notional * (cm.ctt_sell_rate - cm.stamp_duty_buy_rate))
    assert cm.order_cost("GOLDM", 80_000, 2, "buy", illiquid=True) > buy
    assert cm.order_cost("GOLDM", 80_000, 0, "buy") == 0.0


def test_pnl_reconciles_between_daily_and_trades():
    res = run_strategy(market(), "GOLDM", "GOLDGUINEA")
    if res.daily["position"].iloc[-1] == 0:
        assert np.isclose(res.daily["net_pnl"].sum(), res.trades["net_pnl"].sum())


def test_attribution_recovers_known_alpha_beta():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=1500)
    gold = pd.Series(rng.normal(0.0005, 0.01, len(idx)), idx)
    strat = 0.0004 + 0.3 * gold + pd.Series(rng.normal(0, 0.002, len(idx)), idx)
    a = compute_attribution(strat, gold)
    assert abs(a.beta - 0.3) < 0.02
    assert abs(a.alpha_daily - 0.0004) < 0.0002
    assert a.alpha_pvalue < 0.01 and 0 <= a.r_squared <= 1


def test_attribution_insufficient_data_is_graceful():
    idx = pd.bdate_range("2024-01-01", periods=10)
    a = compute_attribution(pd.Series(0.001, idx), pd.Series(0.002, idx))
    assert np.isnan(a.beta) and a.note


# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # minimal runner when pytest is unavailable
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
