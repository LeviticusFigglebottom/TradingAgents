"""Render a per-run HTML audit dashboard.

The dashboard answers, at a glance:

    1. Did the framework reason about every ticker? (verdict matrix)
    2. What did each agent say? (collapsible per-ticker, per-stage panels)
    3. What was the final rating per ticker, and what order followed?
    4. Were any errors raised, anywhere in the pipeline?
    5. What account state did we open and close on?

The page is a single self-contained HTML file (no JS framework, no external
fonts). Every prompt/response in the trace is shown verbatim under a
``<details>`` element so the dump is browsable but not visually overwhelming.
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from tradingagents.execution.runner import RunSummary


_CSS = """
body { font-family: -apple-system, system-ui, Segoe UI, sans-serif;
       margin: 0; padding: 24px; background: #0b0d10; color: #e6e6e6; max-width: 1200px; }
h1, h2, h3 { color: #ffffff; }
h1 { margin-top: 0; }
.section { background: #161a1f; padding: 16px 20px; border-radius: 8px; margin-bottom: 18px;
           border: 1px solid #262c34; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; font-size: 14px; }
.kv b { color: #9aa4b2; font-weight: 500; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #262c34; }
th { color: #9aa4b2; font-weight: 500; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; }
.t-buy { background: #0e3b1f; color: #6ee7a0; }
.t-overweight { background: #0e2c3b; color: #6ec8e7; }
.t-hold { background: #2c2f36; color: #cbd2dc; }
.t-underweight { background: #3b2a0e; color: #e7c46e; }
.t-sell { background: #3b0e0e; color: #ff7b7b; }
.t-error { background: #3b0e0e; color: #ff7b7b; }
.t-skip { background: #2c2f36; color: #9aa4b2; }
.t-ok { background: #0e3b1f; color: #6ee7a0; }
details { margin: 6px 0; background: #0f1318; padding: 8px 12px; border-radius: 6px;
          border: 1px solid #262c34; }
details > summary { cursor: pointer; user-select: none; color: #cbd2dc; }
details[open] > summary { color: #ffffff; margin-bottom: 8px; }
pre { white-space: pre-wrap; word-break: break-word; background: #0a0c10; padding: 10px;
      border-radius: 4px; border: 1px solid #20252b; max-height: 400px; overflow: auto;
      font-size: 12.5px; }
.error { color: #ff7b7b; }
.muted { color: #9aa4b2; }
.bar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
"""


def _tag(rating: str) -> str:
    cls = {
        "Buy": "t-buy",
        "Overweight": "t-overweight",
        "Hold": "t-hold",
        "Underweight": "t-underweight",
        "Sell": "t-sell",
    }.get(rating or "", "t-error")
    return f'<span class="tag {cls}">{html.escape(rating or "—")}</span>'


def _read_trace(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "trace.jsonl"
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _group_by_ticker(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_ticker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        key = ev.get("ticker") or "_global"
        by_ticker[key].append(ev)
    return by_ticker


def _render_event(ev: Dict[str, Any]) -> str:
    kind = ev.get("kind", "")
    payload = ev.get("payload", {}) or {}
    ts = html.escape(ev.get("ts", ""))
    title = f"[{ts}] {html.escape(kind)}"
    body = json.dumps(payload, indent=2, default=str)
    error_class = " class='error'" if kind.endswith("_error") else ""
    return (
        f"<details><summary{error_class}>{title}</summary>"
        f"<pre>{html.escape(body)}</pre></details>"
    )


def _verdict_row(v: Dict[str, Any]) -> str:
    rating = v.get("rating") or ("ERROR" if v.get("error") else "—")
    err = v.get("error")
    err_cell = f"<span class='error'>{html.escape(err)}</span>" if err else ""
    return (
        f"<tr><td><b>{html.escape(v['ticker'])}</b></td>"
        f"<td>{_tag(rating)}</td>"
        f"<td>{(v.get('duration_seconds') or 0):.1f}s</td>"
        f"<td>{err_cell}</td></tr>"
    )


def _order_row(o: Dict[str, Any]) -> str:
    status = o.get("status", "")
    status_cls = {
        "submitted": "t-ok",
        "skipped": "t-skip",
        "blocked": "t-error",
        "error": "t-error",
    }.get(status, "t-skip")
    rating = o.get("rating") or ""
    return (
        "<tr>"
        f"<td><b>{html.escape(o.get('ticker',''))}</b></td>"
        f"<td>{_tag(rating)}</td>"
        f"<td>{html.escape(o.get('side',''))}</td>"
        f"<td>{o.get('qty', 0):.4f}</td>"
        f"<td>${o.get('notional', 0):,.2f}</td>"
        f"<td><span class='tag {status_cls}'>{html.escape(status)}</span></td>"
        f"<td class='muted'>{html.escape(str(o.get('detail','')))}</td>"
        "</tr>"
    )


def _planned_row(p: Dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><b>{html.escape(p.get('ticker',''))}</b></td>"
        f"<td>{_tag(p.get('rating',''))}</td>"
        f"<td>{p.get('current_qty', 0):.4f}</td>"
        f"<td>{p.get('target_qty', 0):.4f}</td>"
        f"<td>{p.get('side','')}</td>"
        f"<td>{p.get('qty', 0):.4f}</td>"
        f"<td>${p.get('notional', 0):,.2f}</td>"
        f"<td>{(p.get('target_weight') or 0):.2%}</td>"
        f"<td class='muted'>{html.escape(p.get('reason',''))}</td>"
        "</tr>"
    )


def _ticker_states_block(run_dir: Path, ticker: str) -> str:
    """Inline the framework's full_states_log for the ticker if present."""
    candidates = list((run_dir / ticker).glob("**/full_states_log_*.json"))
    if not candidates:
        return "<p class='muted'>No saved state file for this ticker.</p>"
    parts = []
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for label, key in [
            ("Market Report", "market_report"),
            ("Sentiment Report", "sentiment_report"),
            ("News Report", "news_report"),
            ("Fundamentals Report", "fundamentals_report"),
            ("Bull/Bear Debate", "investment_debate_state"),
            ("Investment Plan (Research Manager)", "investment_plan"),
            ("Trader Plan", "trader_investment_decision"),
            ("Risk Debate", "risk_debate_state"),
            ("Final Decision (Portfolio Manager)", "final_trade_decision"),
        ]:
            section = data.get(key)
            if section is None:
                continue
            body = section if isinstance(section, str) else json.dumps(section, indent=2)
            parts.append(
                f"<details><summary><b>{html.escape(label)}</b></summary>"
                f"<pre>{html.escape(body)}</pre></details>"
            )
    return "\n".join(parts) if parts else "<p class='muted'>State file empty.</p>"


def render_dashboard(run_dir: Path, summary: "RunSummary") -> str:
    s = summary.as_jsonable()
    events = _read_trace(run_dir)
    grouped = _group_by_ticker(events)

    error_count = len(s.get("errors", []))
    rail_count = len(s.get("rail_violations", []))
    submitted = sum(1 for o in s.get("executed_orders", []) if o.get("status") == "submitted")

    head = f"""<!doctype html><html><head><meta charset="utf-8">
<title>TradingAgents run {html.escape(s['run_id'])}</title>
<style>{_CSS}</style></head><body>"""

    header = f"""
<h1>TradingAgents — Live Run <span class='muted'>{html.escape(s['run_id'])}</span></h1>
<div class='bar'>
  <span class='tag {'t-ok' if s.get('paper') else 't-error'}'>
    {'PAPER' if s.get('paper') else 'LIVE — DANGER'}
  </span>
  <span class='tag {'t-skip' if s.get('dry_run') else 't-ok'}'>
    {'DRY RUN' if s.get('dry_run') else 'EXECUTING'}
  </span>
  <span class='tag {'t-error' if error_count else 't-ok'}'>
    Errors: {error_count}
  </span>
  <span class='tag {'t-error' if rail_count else 't-ok'}'>
    Rail violations: {rail_count}
  </span>
  <span class='tag t-overweight'>Orders submitted: {submitted}</span>
</div>
"""

    cfg = s.get("config", {}) or {}
    cfg_section = f"""
<div class='section'><h2>Run config</h2>
<div class='kv'>
  <b>Trade date</b><span>{html.escape(s.get('trade_date') or '')}</span>
  <b>Started</b><span>{html.escape(s.get('started_at') or '')}</span>
  <b>Finished</b><span>{html.escape(s.get('finished_at') or '')}</span>
  <b>Watchlist</b><span>{html.escape(', '.join(cfg.get('watchlist', [])))}</span>
  <b>LLM provider</b><span>{html.escape(str(cfg.get('llm_provider')))}</span>
  <b>Deep model</b><span>{html.escape(str(cfg.get('deep_think_llm')))}</span>
  <b>Quick model</b><span>{html.escape(str(cfg.get('quick_think_llm')))}</span>
  <b>Debate rounds</b><span>{cfg.get('max_debate_rounds')}</span>
</div></div>"""

    pre = s.get("account_pre") or {}
    post = s.get("account_post") or {}
    account_section = f"""
<div class='section'><h2>Account</h2>
<table>
<tr><th></th><th>Pre-run</th><th>Post-run</th></tr>
<tr><td>Equity</td><td>${pre.get('equity', 0):,.2f}</td><td>${post.get('equity', 0):,.2f}</td></tr>
<tr><td>Cash</td><td>${pre.get('cash', 0):,.2f}</td><td>${post.get('cash', 0):,.2f}</td></tr>
<tr><td>Buying power</td><td>${pre.get('buying_power', 0):,.2f}</td><td>${post.get('buying_power', 0):,.2f}</td></tr>
<tr><td>Paper</td><td colspan='2'>{pre.get('is_paper')}</td></tr>
<tr><td>Market open</td><td colspan='2'>{pre.get('market_open')}</td></tr>
</table></div>"""

    verdicts_rows = "\n".join(_verdict_row(v) for v in s.get("verdicts", []))
    verdicts_section = f"""
<div class='section'><h2>Per-ticker verdicts</h2>
<table>
<tr><th>Ticker</th><th>Rating</th><th>Duration</th><th>Error</th></tr>
{verdicts_rows}
</table></div>"""

    planned_rows = "\n".join(_planned_row(p) for p in s.get("planned_orders", []))
    planned_section = f"""
<div class='section'><h2>Planned orders (target reconciliation)</h2>
<table>
<tr><th>Ticker</th><th>Rating</th><th>Cur qty</th><th>Tgt qty</th><th>Side</th><th>Δ qty</th><th>Δ notional</th><th>Tgt wt</th><th>Reason</th></tr>
{planned_rows}
</table></div>"""

    executed_rows = "\n".join(_order_row(o) for o in s.get("executed_orders", []))
    executed_section = f"""
<div class='section'><h2>Order outcomes</h2>
<table>
<tr><th>Ticker</th><th>Rating</th><th>Side</th><th>Qty</th><th>Notional</th><th>Status</th><th>Detail</th></tr>
{executed_rows or '<tr><td colspan=7 class=muted>No orders attempted.</td></tr>'}
</table></div>"""

    rail_html = ""
    if s.get("rail_violations"):
        rows = "".join(
            f"<tr><td>{html.escape(v.get('ticker',''))}</td>"
            f"<td>{html.escape(v.get('code',''))}</td>"
            f"<td>{html.escape(v.get('message',''))}</td></tr>"
            for v in s["rail_violations"]
        )
        rail_html = f"""
<div class='section'><h2 class='error'>Risk-rail violations</h2>
<table><tr><th>Ticker</th><th>Code</th><th>Message</th></tr>{rows}</table></div>"""

    error_html = ""
    if s.get("errors"):
        rows = "".join(
            f"<details><summary class='error'>{html.escape(str(e.get('phase')))}"
            f" — {html.escape(str(e.get('ticker') or ''))}: {html.escape(str(e.get('error')))}</summary>"
            f"<pre>{html.escape(str(e.get('traceback') or ''))}</pre></details>"
            for e in s["errors"]
        )
        error_html = f"<div class='section'><h2 class='error'>Errors</h2>{rows}</div>"

    # Per-ticker reasoning + trace events
    ticker_blocks = []
    seen_tickers = [v["ticker"] for v in s.get("verdicts", [])]
    for ticker in seen_tickers:
        ev_html = "\n".join(_render_event(e) for e in grouped.get(ticker, []))
        states_html = _ticker_states_block(run_dir, ticker)
        ticker_blocks.append(
            f"""<div class='section'><h3>{html.escape(ticker)}</h3>
<details><summary><b>Agent reports & debates</b></summary>{states_html}</details>
<details><summary><b>Raw trace events</b> ({len(grouped.get(ticker, []))})</summary>
{ev_html or '<p class=muted>No events captured.</p>'}
</details></div>"""
        )

    global_events = grouped.get("_global", [])
    global_html = (
        f"<div class='section'><h2>Run-level events</h2>"
        + "\n".join(_render_event(e) for e in global_events)
        + "</div>"
    )

    body = (
        head
        + header
        + cfg_section
        + account_section
        + verdicts_section
        + planned_section
        + executed_section
        + rail_html
        + error_html
        + "<h2>Per-ticker reasoning</h2>"
        + "\n".join(ticker_blocks)
        + global_html
        + "</body></html>"
    )
    return body
