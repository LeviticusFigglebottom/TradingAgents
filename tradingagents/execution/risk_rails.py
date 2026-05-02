"""Hard risk limits enforced before any order is submitted.

Every check returns a ``RailViolation`` if it trips, or ``None`` if the
proposed order is allowed. The executor short-circuits on the first
violation: better to skip a trade than execute one that breaks a rail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class RailViolation:
    code: str
    message: str


@dataclass(frozen=True)
class RiskConfig:
    paper_only: bool = True
    max_weight_per_name: float = 0.20      # no single position > 20% of equity
    max_gross_exposure: float = 1.00       # never exceed 100% (no leverage)
    max_daily_loss_pct: float = 0.03       # 3% drawdown vs morning equity = halt
    min_trade_notional: float = 25.0       # don't submit orders smaller than $25
    blocklist: tuple = ()                  # tickers we refuse to touch


def assert_paper_account(account_is_paper: bool, cfg: RiskConfig) -> Optional[RailViolation]:
    if cfg.paper_only and not account_is_paper:
        return RailViolation(
            "NOT_PAPER",
            "Risk rail: paper_only=True but the configured Alpaca account is LIVE. Refusing.",
        )
    return None


def check_kill_switch(
    morning_equity: float, current_equity: float, cfg: RiskConfig
) -> Optional[RailViolation]:
    """Trip if today's drawdown exceeds the configured threshold."""
    if morning_equity <= 0:
        return None
    drawdown = (morning_equity - current_equity) / morning_equity
    if drawdown >= cfg.max_daily_loss_pct:
        return RailViolation(
            "DAILY_LOSS",
            f"Risk rail: daily drawdown {drawdown:.2%} >= cap {cfg.max_daily_loss_pct:.0%}. Halting.",
        )
    return None


def check_order(
    ticker: str,
    proposed_notional: float,
    proposed_post_trade_weight: float,
    current_gross_exposure: float,
    cfg: RiskConfig,
) -> List[RailViolation]:
    """Pre-trade checks for a single proposed order. Returns all violations."""
    violations: List[RailViolation] = []
    if ticker in cfg.blocklist:
        violations.append(RailViolation("BLOCKLIST", f"{ticker} is on the risk blocklist."))
    if proposed_post_trade_weight > cfg.max_weight_per_name + 1e-6:
        violations.append(
            RailViolation(
                "MAX_WEIGHT",
                f"{ticker}: post-trade weight {proposed_post_trade_weight:.2%} > cap "
                f"{cfg.max_weight_per_name:.0%}.",
            )
        )
    if current_gross_exposure > cfg.max_gross_exposure + 1e-6:
        violations.append(
            RailViolation(
                "MAX_GROSS",
                f"Gross exposure {current_gross_exposure:.2%} > cap "
                f"{cfg.max_gross_exposure:.0%}; refusing to add.",
            )
        )
    if abs(proposed_notional) < cfg.min_trade_notional:
        violations.append(
            RailViolation(
                "MIN_NOTIONAL",
                f"{ticker}: |notional| ${abs(proposed_notional):.2f} < min "
                f"${cfg.min_trade_notional:.2f}; skipping.",
            )
        )
    return violations
