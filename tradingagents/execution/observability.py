"""Capture every LLM prompt, LLM response, and tool call to a JSONL trace.

This sits next to ``StatsCallbackHandler``: that one only counts; this one
records the full payloads so an operator can audit exactly what each agent
saw and said for every ticker on every run.

One ``TraceCallbackHandler`` is created per run. It writes one JSON object
per line to ``trace.jsonl``. The current ticker is set via
``set_ticker(symbol)`` so the runner can multiplex one handler across the
MAG7 loop and still keep events partitioned per symbol.

Event schema (all events share these fields):
    ts        - ISO-8601 UTC timestamp
    ticker    - symbol the runner is currently processing (or None)
    run_id    - opaque id distinguishing concurrent runs
    kind      - one of: llm_start, llm_end, llm_error, tool_start, tool_end,
                tool_error, chain_start, chain_end, chain_error, stage
    payload   - kind-specific dict

The dashboard generator reads this file plus the framework's own
``full_states_log_<date>.json`` and renders a per-run HTML audit page.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stringify_message(msg: Any) -> Dict[str, Any]:
    """Best-effort conversion of a LangChain message to a plain dict."""
    if isinstance(msg, BaseMessage):
        return {
            "type": msg.__class__.__name__,
            "content": _truncate(msg.content),
        }
    if isinstance(msg, dict):
        return {k: _truncate(v) for k, v in msg.items()}
    return {"type": type(msg).__name__, "content": _truncate(str(msg))}


_MAX_FIELD_CHARS = 32_000  # keep individual fields bounded so the trace stays grep-able


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_FIELD_CHARS:
        return value[:_MAX_FIELD_CHARS] + f"\n...[truncated {len(value) - _MAX_FIELD_CHARS} chars]"
    return value


class TraceCallbackHandler(BaseCallbackHandler):
    """Append every LLM prompt/response and tool call to a JSONL file.

    The handler is thread-safe so it can be reused across the MAG7 loop.
    Use ``set_ticker(symbol)`` before each ``propagate()`` call so events
    are tagged with the active symbol.
    """

    def __init__(self, trace_path: Path):
        super().__init__()
        self._lock = threading.Lock()
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any prior file at the same path so each run starts clean.
        self.trace_path.write_text("", encoding="utf-8")
        self._ticker: Optional[str] = None
        self.run_id = uuid.uuid4().hex[:12]
        self.errors: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ control

    def set_ticker(self, ticker: Optional[str]) -> None:
        with self._lock:
            self._ticker = ticker

    def stage(self, name: str, **payload: Any) -> None:
        """Emit an operator-visible stage marker (e.g. start/end of a ticker)."""
        self._emit("stage", {"name": name, **payload})

    # ------------------------------------------------------------------ writes

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        record = {
            "ts": _now(),
            "ticker": self._ticker,
            "run_id": self.run_id,
            "kind": kind,
            "payload": payload,
        }
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            if kind.endswith("_error"):
                self.errors.append(record)

    # ------------------------------------------------------------------ LLM hooks

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self._emit(
            "llm_start",
            {
                "model": (serialized or {}).get("name"),
                "prompts": [_truncate(p) for p in prompts],
            },
        )

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[Any]], **kwargs: Any
    ) -> None:
        self._emit(
            "llm_start",
            {
                "model": (serialized or {}).get("name"),
                "messages": [[_stringify_message(m) for m in seq] for seq in messages],
            },
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            generation = response.generations[0][0]
            text = getattr(generation, "text", None)
            usage = None
            if hasattr(generation, "message"):
                msg = generation.message
                text = getattr(msg, "content", text)
                usage = getattr(msg, "usage_metadata", None)
        except (IndexError, AttributeError, TypeError):
            text = None
            usage = None
        self._emit(
            "llm_end",
            {"text": _truncate(text) if text is not None else None, "usage": usage},
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._emit("llm_error", {"error": repr(error)})

    # ------------------------------------------------------------------ tool hooks

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        self._emit(
            "tool_start",
            {"tool": (serialized or {}).get("name"), "input": _truncate(input_str)},
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self._emit("tool_end", {"output": _truncate(str(output))})

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self._emit("tool_error", {"error": repr(error)})

    # ------------------------------------------------------------------ chain hooks (best-effort)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        name = (serialized or {}).get("name") or (kwargs.get("name") if kwargs else None)
        if not name:
            return  # don't spam with anonymous chain wrappers
        self._emit("chain_start", {"name": name})

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        # Skip free-form chain ends to keep trace size manageable; the
        # llm_end / tool_end events already carry the substantive payload.
        return

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        self._emit("chain_error", {"error": repr(error)})
