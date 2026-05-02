"""Live execution layer: turns the framework's recommendations into orders.

Modules:
    observability  - callback that records every prompt/response/tool call to JSONL
    targets        - rating -> target portfolio weight mapping
    risk_rails     - hard limits applied before any order goes out
    alpaca         - Alpaca paper/live broker client + reconciliation
    dashboard      - per-run HTML report (ticker x agent matrix, prompts, verdicts)
    runner         - end-to-end daily run orchestrator
"""
