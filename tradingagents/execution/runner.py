"""End-to-end daily orchestrator.

Run flow:
    1. Snapshot Alpaca account; refuse if not paper (when paper_only=True).
    2. For each ticker in the watchlist:
         - Tag the trace handler with the ticker.
         - Call ``TradingAgentsGraph.propagate(ticker, today)``.
         - Parse the 5-tier rating from the PM's markdown.
         - Record the per-ticker verdict in the run summary.
    3. Convert ratings to target weights, plan orders, apply risk rails,
       submit orders to Alpaca.
    4. Write a run-summary JSON and an HTML dashboard.

Every artifact lands under ``results_dir / "live_runs" / <run_id>``:
    trace.jsonl       - every prompt/response/tool call (TraceCallbackHandler)
    summary.json      - account, ratings, planned orders, executed orders, errors
    dashboard.html    - rendered audit page
    <TICKER>/full_states_log_<date>.json  - framework's existing per-ticker dump
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.execution.alpaca import (
    AccountSnapshot,
    AlpacaClient,
    OrderResult,
    PlannedOrder,
    plan_orders,
)
from tradingagents.execution.observability import TraceCallbackHandler
from tradingagents.execution.risk_rails import (
    RailViolation,
    RiskConfig,
    assert_paper_account,
    check_kill_switch,
    check_order,
)
from tradingagents.execution.targets import compute_target_weights
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# The Magnificent 7. Adjust by passing a different watchlist to `run()`.
MAG7 = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA")


@dataclass
class TickerVerdict:
    ticker: str
    rating: Optional[str] = None
    decision_markdown: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


@dataclass
class RunSummary:
    run_id: str
    started_at: str
    finished_at: Optional[str] = None
    trade_date: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    account_pre: Optional[Dict[str, Any]] = None
    account_post: Optional[Dict[str, Any]] = None
    verdicts: List[TickerVerdict] = field(default_factory=list)
    planned_orders: List[Dict[str, Any]] = field(default_factory=list)
    executed_orders: List[Dict[str, Any]] = field(default_factory=list)
    rail_violations: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    dry_run: bool = True
    paper: bool = True

    def as_jsonable(self) -> Dict[str, Any]:
        # dataclass asdict handles nested dataclasses
        return asdict(self)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class LiveRunner:
    """Daily orchestrator. One instance per run."""

    def __init__(
        self,
        *,
        watchlist: tuple = MAG7,
        risk: Optional[RiskConfig] = None,
        config: Optional[Dict[str, Any]] = None,
        dry_run: bool = True,
        results_root: Optional[Path] = None,
    ):
        self.watchlist = tuple(watchlist)
        self.risk = risk or RiskConfig()
        self.dry_run = dry_run
        self.config = (config or DEFAULT_CONFIG).copy()

        self.run_started = datetime.now(timezone.utc)
        self.run_id = self.run_started.strftime("%Y%m%dT%H%M%SZ")
        self.trade_date = self.run_started.date().isoformat()

        results_root = Path(results_root or self.config["results_dir"])
        self.run_dir = results_root / "live_runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Per-ticker reports go under run_dir; framework's _log_state will
        # dump full_states_log_<date>.json for each ticker.
        self.config["results_dir"] = str(self.run_dir)

        self.trace = TraceCallbackHandler(self.run_dir / "trace.jsonl")
        self.summary = RunSummary(
            run_id=self.run_id,
            started_at=self.run_started.isoformat(timespec="seconds"),
            trade_date=self.trade_date,
            dry_run=dry_run,
            paper=self.risk.paper_only,
            config={
                "watchlist": list(self.watchlist),
                "llm_provider": self.config.get("llm_provider"),
                "deep_think_llm": self.config.get("deep_think_llm"),
                "quick_think_llm": self.config.get("quick_think_llm"),
                "max_debate_rounds": self.config.get("max_debate_rounds"),
                "risk": asdict(self.risk),
            },
        )

    # ------------------------------------------------------------------ public

    def run(self) -> RunSummary:
        try:
            self._run_inner()
        except Exception as exc:  # noqa: BLE001 -- top-level failsafe
            logger.exception("LiveRunner crashed")
            self.summary.errors.append(
                {"phase": "top_level", "error": repr(exc), "traceback": traceback.format_exc()}
            )
        finally:
            self.summary.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._write_summary()
            self._write_dashboard()
        return self.summary

    # ------------------------------------------------------------------ phases

    def _run_inner(self) -> None:
        self.trace.stage("run_start", run_id=self.run_id, trade_date=self.trade_date)

        broker = self._connect_broker()
        account = broker.account()
        self.summary.account_pre = asdict(account)

        rail = assert_paper_account(account.is_paper, self.risk)
        if rail:
            self.summary.rail_violations.append(asdict(rail))
            self.trace.stage("aborted_paper_check", reason=rail.message)
            logger.error(rail.message)
            return

        self.trace.stage(
            "account_snapshot",
            equity=account.equity,
            cash=account.cash,
            market_open=account.market_open,
            paper=account.is_paper,
        )

        ratings = self._reason_about_watchlist()
        if not ratings:
            self.trace.stage("no_ratings_aborting")
            return

        positions = broker.positions()
        prices: Dict[str, float] = {}
        for ticker in ratings:
            price = broker.latest_price(ticker)
            if price is not None:
                prices[ticker] = price

        targets = compute_target_weights(ratings)
        plan = plan_orders(
            account=account,
            positions=positions,
            target_weights=targets,
            prices=prices,
            min_trade_notional=self.risk.min_trade_notional,
        )
        self.summary.planned_orders = [asdict(p) for p in plan]
        self.trace.stage("orders_planned", count=len(plan))

        self._execute(broker, account, plan)

        self.summary.account_post = asdict(broker.account())
        self.trace.stage("run_end")

    def _connect_broker(self) -> AlpacaClient:
        api_key = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_API_SECRET", "")
        paper_env = os.environ.get("ALPACA_PAPER", "true").lower() not in {"0", "false", "no"}
        return AlpacaClient(api_key, api_secret, paper=paper_env)

    def _reason_about_watchlist(self) -> Dict[str, str]:
        """Run the agent pipeline for each ticker; return ticker -> rating."""
        graph = TradingAgentsGraph(
            debug=False,
            config=self.config,
            callbacks=[self.trace],
        )
        ratings: Dict[str, str] = {}
        for ticker in self.watchlist:
            self.trace.set_ticker(ticker)
            self.trace.stage("ticker_start", ticker=ticker)
            verdict = TickerVerdict(ticker=ticker)
            t0 = datetime.now(timezone.utc)
            try:
                _, decision_md = graph.propagate(ticker, self.trade_date)
                rating = parse_rating(decision_md)
                verdict.rating = rating
                verdict.decision_markdown = decision_md
                ratings[ticker] = rating
                self.trace.stage("ticker_verdict", ticker=ticker, rating=rating)
            except Exception as exc:  # noqa: BLE001
                logger.exception("propagate failed for %s", ticker)
                verdict.error = repr(exc)
                self.summary.errors.append(
                    {
                        "phase": "propagate",
                        "ticker": ticker,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                self.trace.stage("ticker_error", ticker=ticker, error=repr(exc))
            finally:
                verdict.duration_seconds = (
                    datetime.now(timezone.utc) - t0
                ).total_seconds()
                self.summary.verdicts.append(verdict)
                self.trace.set_ticker(None)
        return ratings

    def _execute(
        self,
        broker: AlpacaClient,
        account: AccountSnapshot,
        plan: List[PlannedOrder],
    ) -> None:
        equity = account.equity
        gross = sum(abs(p.notional) for p in plan if p.reason == "target_diff")
        gross_exposure = (gross / equity) if equity else 0.0

        for order in plan:
            if order.reason != "target_diff":
                self.trace.stage(
                    "order_skipped",
                    ticker=order.ticker,
                    reason=order.reason,
                    rating=order.rating,
                )
                continue

            post_weight = abs(order.target_qty) * (
                (order.notional / order.qty) if order.qty else 0.0
            ) / equity if equity else 0.0
            violations = check_order(
                ticker=order.ticker,
                proposed_notional=order.notional,
                proposed_post_trade_weight=post_weight,
                current_gross_exposure=gross_exposure,
                cfg=self.risk,
            )
            if violations:
                for v in violations:
                    self.summary.rail_violations.append({**asdict(v), "ticker": order.ticker})
                self.trace.stage(
                    "order_blocked",
                    ticker=order.ticker,
                    violations=[v.code for v in violations],
                )
                continue

            if self.dry_run or not account.market_open:
                reason = "dry_run" if self.dry_run else "market_closed"
                self.summary.executed_orders.append(
                    {
                        "ticker": order.ticker,
                        "side": order.side,
                        "qty": order.qty,
                        "notional": order.notional,
                        "status": "skipped",
                        "detail": reason,
                        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "rating": order.rating,
                    }
                )
                self.trace.stage("order_skipped_runtime", ticker=order.ticker, reason=reason)
                continue

            try:
                result = broker.submit_market_order(order.ticker, order.qty, order.side)
                rec = asdict(result)
                rec["rating"] = order.rating
                self.summary.executed_orders.append(rec)
                self.trace.stage(
                    "order_submitted",
                    ticker=order.ticker,
                    side=order.side,
                    qty=order.qty,
                    detail=result.detail,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("submit_market_order failed for %s", order.ticker)
                self.summary.executed_orders.append(
                    {
                        "ticker": order.ticker,
                        "side": order.side,
                        "qty": order.qty,
                        "notional": order.notional,
                        "status": "error",
                        "detail": repr(exc),
                        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "rating": order.rating,
                    }
                )
                self.summary.errors.append(
                    {
                        "phase": "submit_order",
                        "ticker": order.ticker,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                self.trace.stage("order_error", ticker=order.ticker, error=repr(exc))

    # ------------------------------------------------------------------ artifacts

    def _write_summary(self) -> None:
        path = self.run_dir / "summary.json"
        path.write_text(
            json.dumps(self.summary.as_jsonable(), indent=2, default=str), encoding="utf-8"
        )
        logger.info("Wrote run summary to %s", path)

    def _write_dashboard(self) -> None:
        # Imported lazily to keep the runtime path independent of any
        # missing template helpers.
        from tradingagents.execution.dashboard import render_dashboard

        path = self.run_dir / "dashboard.html"
        html = render_dashboard(self.run_dir, self.summary)
        path.write_text(html, encoding="utf-8")
        logger.info("Wrote dashboard to %s", path)
