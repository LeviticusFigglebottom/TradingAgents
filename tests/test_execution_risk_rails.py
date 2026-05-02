"""Unit tests for the hard risk rails."""

from __future__ import annotations

from tradingagents.execution.risk_rails import (
    RiskConfig,
    assert_paper_account,
    check_kill_switch,
    check_order,
)


def test_paper_only_blocks_live_account():
    cfg = RiskConfig(paper_only=True)
    assert assert_paper_account(account_is_paper=False, cfg=cfg) is not None


def test_paper_only_passes_paper_account():
    cfg = RiskConfig(paper_only=True)
    assert assert_paper_account(account_is_paper=True, cfg=cfg) is None


def test_kill_switch_trips_at_threshold():
    cfg = RiskConfig(max_daily_loss_pct=0.03)
    # Exactly 3% drawdown
    v = check_kill_switch(morning_equity=100_000, current_equity=97_000, cfg=cfg)
    assert v is not None and v.code == "DAILY_LOSS"


def test_kill_switch_quiet_at_smaller_drawdown():
    cfg = RiskConfig(max_daily_loss_pct=0.03)
    assert check_kill_switch(morning_equity=100_000, current_equity=98_000, cfg=cfg) is None


def test_blocklist_short_circuits():
    cfg = RiskConfig(blocklist=("XYZ",))
    violations = check_order(
        ticker="XYZ",
        proposed_notional=1000,
        proposed_post_trade_weight=0.05,
        current_gross_exposure=0.5,
        cfg=cfg,
    )
    codes = {v.code for v in violations}
    assert "BLOCKLIST" in codes


def test_max_weight_cap_violation():
    cfg = RiskConfig(max_weight_per_name=0.20)
    violations = check_order(
        ticker="AAPL",
        proposed_notional=10_000,
        proposed_post_trade_weight=0.30,
        current_gross_exposure=0.5,
        cfg=cfg,
    )
    assert any(v.code == "MAX_WEIGHT" for v in violations)


def test_min_notional_skips_dust_orders():
    cfg = RiskConfig(min_trade_notional=25.0)
    violations = check_order(
        ticker="AAPL",
        proposed_notional=10.0,
        proposed_post_trade_weight=0.05,
        current_gross_exposure=0.5,
        cfg=cfg,
    )
    assert any(v.code == "MIN_NOTIONAL" for v in violations)
