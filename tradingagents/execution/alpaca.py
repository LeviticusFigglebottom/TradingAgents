"""Alpaca client wrapper: account state, position reconciliation, order placement.

This module is intentionally narrow: it exposes only what the runner needs
(account snapshot, position dict, submit a single market order). The
broker-specific SDK is imported lazily so unit tests can mock the entire
class without ``alpaca-py`` installed.

Paper-only by default. Switch to live by setting ``ALPACA_PAPER=false`` and
``RiskConfig.paper_only=False`` -- both must be flipped, on purpose.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    is_paper: bool
    market_open: bool
    timestamp: str


@dataclass(frozen=True)
class Position:
    ticker: str
    qty: float
    market_value: float
    avg_entry: float


@dataclass(frozen=True)
class OrderResult:
    ticker: str
    side: str          # "buy" or "sell"
    qty: float
    notional: float    # signed: positive = buy, negative = sell
    status: str        # "submitted", "skipped", "blocked", "error"
    detail: str        # human-readable explanation / broker order id
    submitted_at: str


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AlpacaClient:
    """Thin wrapper around alpaca-py's TradingClient.

    The wrapper deliberately does not retry or stream; one daily run is short
    enough that simple synchronous calls with explicit error reporting are
    easier to audit than implicit retries.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        paper: bool = True,
    ):
        if not api_key or not api_secret:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be set in the environment."
            )
        # Lazy import so the dependency is only required at runtime.
        from alpaca.trading.client import TradingClient

        self._paper = paper
        self._client = TradingClient(api_key, api_secret, paper=paper)

    # ------------------------------------------------------------------ reads

    @property
    def is_paper(self) -> bool:
        return self._paper

    def account(self) -> AccountSnapshot:
        acct = self._client.get_account()
        clock = self._client.get_clock()
        return AccountSnapshot(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            is_paper=self._paper,
            market_open=bool(clock.is_open),
            timestamp=_utcnow(),
        )

    def positions(self) -> Dict[str, Position]:
        out: Dict[str, Position] = {}
        for p in self._client.get_all_positions():
            out[p.symbol] = Position(
                ticker=p.symbol,
                qty=float(p.qty),
                market_value=float(p.market_value),
                avg_entry=float(p.avg_entry_price),
            )
        return out

    def latest_price(self, ticker: str) -> Optional[float]:
        """Last trade price via Alpaca Data API; falls back to None on failure."""
        try:
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestTradeRequest

            api_key = os.environ["ALPACA_API_KEY"]
            api_secret = os.environ["ALPACA_API_SECRET"]
            data = StockHistoricalDataClient(api_key, api_secret)
            resp = data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=ticker))
            trade = resp[ticker]
            return float(trade.price)
        except Exception as exc:  # noqa: BLE001 -- price is best-effort
            logger.warning("latest_price(%s) failed: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------ writes

    def submit_market_order(
        self,
        ticker: str,
        qty: float,
        side: str,
        *,
        time_in_force: str = "day",
    ) -> OrderResult:
        """Submit a fractional market order. Side must be ``buy`` or ``sell``."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        if side not in {"buy", "sell"}:
            raise ValueError(f"side must be buy/sell, got {side!r}")

        req = MarketOrderRequest(
            symbol=ticker,
            qty=round(abs(qty), 6),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC,
        )
        order = self._client.submit_order(req)
        return OrderResult(
            ticker=ticker,
            side=side,
            qty=float(qty),
            notional=0.0,  # broker fills at market; runner records intent separately
            status="submitted",
            detail=f"alpaca_order_id={getattr(order, 'id', None)}",
            submitted_at=_utcnow(),
        )


# ---------------------------------------------------------------------------
# Reconciliation: target weights -> orders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedOrder:
    ticker: str
    side: str           # "buy" or "sell"
    qty: float
    notional: float     # signed
    current_qty: float
    target_qty: float
    target_weight: float
    rating: str
    reason: str         # "target_diff", "skip_hold", "skip_min_notional", ...


def plan_orders(
    *,
    account: AccountSnapshot,
    positions: Dict[str, Position],
    target_weights,  # Dict[str, TargetWeight]
    prices: Dict[str, float],
    min_trade_notional: float,
) -> List[PlannedOrder]:
    """Compute the diff between current and target positions.

    ``Hold`` ratings are passed through as no-ops. ``Sell`` collapses the
    target to zero. Other ratings target a fixed equity weight; we convert
    that to a share count using the latest price and the account equity.
    """
    plan: List[PlannedOrder] = []
    equity = account.equity
    for ticker, tw in target_weights.items():
        price = prices.get(ticker)
        current = positions.get(ticker)
        current_qty = current.qty if current else 0.0
        rating = tw.rating

        if tw.target_weight is None:  # Hold
            plan.append(
                PlannedOrder(
                    ticker=ticker,
                    side="buy",
                    qty=0.0,
                    notional=0.0,
                    current_qty=current_qty,
                    target_qty=current_qty,
                    target_weight=(current.market_value / equity) if (current and equity) else 0.0,
                    rating=rating,
                    reason="skip_hold",
                )
            )
            continue

        if price is None or price <= 0:
            plan.append(
                PlannedOrder(
                    ticker=ticker,
                    side="buy",
                    qty=0.0,
                    notional=0.0,
                    current_qty=current_qty,
                    target_qty=current_qty,
                    target_weight=tw.target_weight,
                    rating=rating,
                    reason="skip_no_price",
                )
            )
            continue

        target_notional = tw.target_weight * equity
        target_qty = round(target_notional / price, 4)  # Alpaca supports fractional
        delta_qty = target_qty - current_qty
        delta_notional = delta_qty * price

        if abs(delta_notional) < min_trade_notional:
            plan.append(
                PlannedOrder(
                    ticker=ticker,
                    side="buy" if delta_qty >= 0 else "sell",
                    qty=0.0,
                    notional=delta_notional,
                    current_qty=current_qty,
                    target_qty=target_qty,
                    target_weight=tw.target_weight,
                    rating=rating,
                    reason="skip_min_notional",
                )
            )
            continue

        plan.append(
            PlannedOrder(
                ticker=ticker,
                side="buy" if delta_qty > 0 else "sell",
                qty=abs(delta_qty),
                notional=delta_notional,
                current_qty=current_qty,
                target_qty=target_qty,
                target_weight=tw.target_weight,
                rating=rating,
                reason="target_diff",
            )
        )
    return plan
