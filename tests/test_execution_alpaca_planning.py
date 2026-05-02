"""Unit tests for plan_orders -- the diff between current positions and targets.

We don't import the real Alpaca SDK here; the planning function only needs
the ``AccountSnapshot`` / ``Position`` dataclasses, not a live broker.
"""

from __future__ import annotations

from tradingagents.execution.alpaca import AccountSnapshot, Position, plan_orders
from tradingagents.execution.targets import compute_target_weights


def _account(equity=100_000.0, market_open=True):
    return AccountSnapshot(
        equity=equity,
        cash=equity,
        buying_power=equity,
        is_paper=True,
        market_open=market_open,
        timestamp="2026-05-02T13:45:00+00:00",
    )


def test_buy_from_flat_creates_buy_order():
    targets = compute_target_weights({"AAPL": "Buy"})  # 1 name, slot=1.0, Buy=1.0 -> 100%
    plan = plan_orders(
        account=_account(equity=10_000),
        positions={},
        target_weights=targets,
        prices={"AAPL": 200.0},
        min_trade_notional=25.0,
    )
    [order] = plan
    assert order.side == "buy"
    # 100% of $10k at $200 = 50 shares
    assert abs(order.qty - 50.0) < 1e-6
    assert order.reason == "target_diff"


def test_sell_collapses_existing_position():
    targets = compute_target_weights({"AAPL": "Sell"})
    plan = plan_orders(
        account=_account(equity=10_000),
        positions={"AAPL": Position("AAPL", qty=20.0, market_value=4000.0, avg_entry=200.0)},
        target_weights=targets,
        prices={"AAPL": 200.0},
        min_trade_notional=25.0,
    )
    [order] = plan
    assert order.side == "sell"
    assert abs(order.qty - 20.0) < 1e-6


def test_hold_emits_noop():
    targets = compute_target_weights({"AAPL": "Hold"})
    plan = plan_orders(
        account=_account(),
        positions={"AAPL": Position("AAPL", qty=5.0, market_value=1000.0, avg_entry=200.0)},
        target_weights=targets,
        prices={"AAPL": 200.0},
        min_trade_notional=25.0,
    )
    [order] = plan
    assert order.reason == "skip_hold"
    assert order.qty == 0.0


def test_dust_diff_is_skipped():
    targets = compute_target_weights({"AAPL": "Buy"})
    # current ~= target, so the diff is small.
    plan = plan_orders(
        account=_account(equity=10_000),
        positions={"AAPL": Position("AAPL", qty=49.99, market_value=9998.0, avg_entry=200.0)},
        target_weights=targets,
        prices={"AAPL": 200.0},
        min_trade_notional=25.0,
    )
    [order] = plan
    assert order.reason == "skip_min_notional"


def test_missing_price_skips_safely():
    targets = compute_target_weights({"AAPL": "Buy"})
    plan = plan_orders(
        account=_account(),
        positions={},
        target_weights=targets,
        prices={},  # no price
        min_trade_notional=25.0,
    )
    [order] = plan
    assert order.reason == "skip_no_price"
