"""Unit tests for the trace callback handler."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tradingagents.execution.observability import TraceCallbackHandler


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_chat_model_start_writes_messages(tmp_path):
    h = TraceCallbackHandler(tmp_path / "trace.jsonl")
    h.set_ticker("AAPL")
    h.on_chat_model_start(
        {"name": "claude-haiku-4-5"},
        [[HumanMessage(content="hello")]],
    )
    [event] = _read(tmp_path / "trace.jsonl")
    assert event["kind"] == "llm_start"
    assert event["ticker"] == "AAPL"
    assert event["payload"]["model"] == "claude-haiku-4-5"
    assert event["payload"]["messages"][0][0]["content"] == "hello"


def test_llm_end_captures_text_and_usage(tmp_path):
    h = TraceCallbackHandler(tmp_path / "trace.jsonl")
    msg = AIMessage(content="response")
    msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    gen = ChatGeneration(message=msg)
    result = LLMResult(generations=[[gen]])
    h.on_llm_end(result)
    [event] = _read(tmp_path / "trace.jsonl")
    assert event["payload"]["text"] == "response"
    assert event["payload"]["usage"]["input_tokens"] == 10


def test_tool_calls_recorded(tmp_path):
    h = TraceCallbackHandler(tmp_path / "trace.jsonl")
    h.on_tool_start({"name": "get_stock_data"}, "AAPL,2026-05-02")
    h.on_tool_end("price=200")
    events = _read(tmp_path / "trace.jsonl")
    kinds = [e["kind"] for e in events]
    assert kinds == ["tool_start", "tool_end"]


def test_errors_are_collected(tmp_path):
    h = TraceCallbackHandler(tmp_path / "trace.jsonl")
    h.on_llm_error(RuntimeError("boom"))
    h.on_tool_error(ValueError("bad input"))
    assert len(h.errors) == 2
    assert h.errors[0]["payload"]["error"].startswith("RuntimeError")


def test_stage_markers_visible(tmp_path):
    h = TraceCallbackHandler(tmp_path / "trace.jsonl")
    h.stage("ticker_start", ticker="AAPL")
    h.stage("ticker_verdict", ticker="AAPL", rating="Buy")
    events = _read(tmp_path / "trace.jsonl")
    assert events[0]["kind"] == "stage"
    assert events[1]["payload"]["rating"] == "Buy"
