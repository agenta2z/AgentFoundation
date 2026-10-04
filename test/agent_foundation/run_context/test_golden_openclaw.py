"""Characterization goldens for ``OpenClawInferencer`` entries, sessions and retries.

Pins today's observable behaviour before the invocation-scoped refactor. The
gateway WebSocket (``_ws_connect``) and the pod transcript probe
(``run_subprocess``) are replaced by an in-process :class:`FakeGateway`; every
agent request and probe it serves is logged in order, together with the
RunContext active at that moment.

* Entry matrix: ``ainfer`` / ``infer`` / direct ``ainfer_streaming`` / direct
  ``infer_streaming``, as two bare calls on one instance and as one call with a
  host ``run_context``. Recorded per call: output, transport log,
  ``_maybe_initialize_session`` calls, kwargs reaching the public
  ``ainfer_streaming`` (none for sync streaming, whose bridge drives the
  pipeline), base ``_ainfer_single`` / ``_infer_single`` calls,
  ``active_session_id`` and the session ids initialized on the connection
  afterwards.
* Rate-limit retry inside ``_ainfer_with_retry``: continuation prompts, backoff
  delays handed to ``asyncio.sleep`` (made instant), recovery and exhaustion.
* Base ``max_retry`` / ``fallback_mode`` / ``fallback_inferencer`` are pinned
  at construction (a non-default is logged at WARNING) and a per-call
  ``fallback_mode`` is dropped the same way, so the one base ``_ainfer_single``
  attempt that ``ainfer`` / ``infer`` make never retries or falls back.
* A terminal failure after a successful call resets ``active_session_id``.
* Session initialization across two session ids (B19, fixed in P10: each session
  id is initialized once per connection).
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw import (
    openclaw_inferencer as oc_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer import (
    OpenClawInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    RunContext,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode

from ._golden import check_golden, Normalizer
from ._leaf_fixtures import ctx_view as _ctx_view, FakeGateway

ENTRIES = ("ainfer", "infer", "ainfer_streaming", "infer_streaming")
RATE_LIMIT = "429 rate limit exceeded"
HARD_ERROR = "invalid session state"
RETRY_SCRIPTS = {"recovers": [RATE_LIMIT] * 2, "exhausted": [RATE_LIMIT] * 3}


def _project(value: Any) -> Any:
    if isinstance(value, RunContext):
        return {"RunContext": _ctx_view(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__name__}>"


@attrs
class SpyOpenClaw(OpenClawInferencer):
    gateway: Any = attrib(default=None, kw_only=True)
    init_calls: list = attrib(factory=list, init=False)
    streaming_entry_kwargs: list = attrib(factory=list, init=False)
    base_single_calls: list = attrib(factory=list, init=False)

    async def _ws_connect(self) -> Any:
        return self.gateway.connect()

    async def _maybe_initialize_session(self, session_id: str) -> None:
        rec = {
            "session_id": session_id,
            "initialized_before": self._session_ready(session_id),
        }
        self.init_calls.append(rec)
        await super()._maybe_initialize_session(session_id)
        rec["initialized_after"] = self._session_ready(session_id)

    async def ainfer_streaming(self, inference_input, inference_config=None, **kwargs):
        self.streaming_entry_kwargs.append({k: _project(v) for k, v in kwargs.items()})
        async for chunk in super().ainfer_streaming(
            inference_input, inference_config, **kwargs
        ):
            yield chunk

    async def _ainfer_single(self, *args: Any, **kwargs: Any) -> Any:
        self.base_single_calls.append("_ainfer_single")
        return await super()._ainfer_single(*args, **kwargs)

    def _infer_single(self, *args: Any, **kwargs: Any) -> Any:
        self.base_single_calls.append("_infer_single")
        return super()._infer_single(*args, **kwargs)


@attrs
class FallbackStub(InferencerBase):
    calls: list = attrib(factory=list, init=False)

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        self.calls.append("_infer")
        return "fallback answer"

    async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
        self.calls.append("_ainfer")
        return "fallback answer"


def _make(monkeypatch, gateway: FakeGateway, **kwargs: Any) -> SpyOpenClaw:
    monkeypatch.setattr(oc_module, "run_subprocess", gateway.run_subprocess)
    return SpyOpenClaw(auth_token="tok", gateway=gateway, **kwargs)


def _instant_sleep(monkeypatch) -> List[float]:
    delays: List[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(oc_module.asyncio, "sleep", fake_sleep)
    return delays


async def _drain(agen: Any) -> List[str]:
    return [chunk async for chunk in agen]


def _invoke(inst: SpyOpenClaw, entry: str, prompt: str, kwargs: Dict) -> Any:
    if entry == "ainfer":
        return asyncio.run(inst.ainfer(prompt, **kwargs))
    if entry == "infer":
        return inst.infer(prompt, **kwargs)
    if entry == "ainfer_streaming":
        return asyncio.run(_drain(inst.ainfer_streaming(prompt, **kwargs)))
    return list(inst.infer_streaming(prompt, **kwargs))


def _error_view(exc: Optional[BaseException]) -> Optional[Dict[str, Any]]:
    if exc is None:
        return None
    return {"type": type(exc).__name__, "message": str(exc)}


def _outcome(fn: Callable[[], Any]) -> Dict[str, Any]:
    try:
        return {"output": fn()}
    except Exception as exc:
        raised = _error_view(exc)
        raised["cause"] = _error_view(exc.__cause__)
        return {"raised": raised}


def _record_call(inst: SpyOpenClaw, entry: str, prompt: str, kwargs: Dict) -> Dict:
    lists = (
        inst.gateway.log,
        inst.init_calls,
        inst.streaming_entry_kwargs,
        inst.base_single_calls,
    )
    marks = [len(lst) for lst in lists]
    result = _outcome(lambda: _invoke(inst, entry, prompt, kwargs))
    transport, inits, stream_kwargs, singles = (
        lst[mark:] for lst, mark in zip(lists, marks)
    )
    return {
        "entry": entry,
        "prompt": prompt,
        "call_kwargs": {k: _project(v) for k, v in kwargs.items()},
        **result,
        "transport": transport,
        "maybe_initialize_session": inits,
        "streaming_entry_kwargs": stream_kwargs,
        "base_single_calls": singles,
        "active_session_id_after": inst.active_session_id,
        "initialized_sessions_after": _initialized_sessions(inst),
    }


def _initialized_sessions(inst: SpyOpenClaw) -> List[Optional[str]]:
    """Every session id initialized on the connection, across its live-handle
    sets (each host branch's and the no-context backing)."""
    ready = set()
    for handles in inst._iter_live_handle_sets():
        ready |= handles.get("initialized_sessions") or set()
    return sorted(ready, key=str)


def _session_under(inst: SpyOpenClaw, ctx: RunContext) -> Optional[str]:
    token = enter_run(ctx)
    try:
        return inst.active_session_id
    finally:
        exit_run(token)


def _agent_requests(call: Dict) -> int:
    return sum(1 for c in call["transport"] if c["method"] == "agent")


@pytest.mark.parametrize("entry", ENTRIES)
def test_entry_two_bare_calls(entry, monkeypatch):
    """Two bare calls on one instance: one warm-up turn, then the same session."""
    inst = _make(monkeypatch, FakeGateway())
    calls = [
        _record_call(inst, entry, prompt, {})
        for prompt in ("first question", "second question")
    ]
    data = Normalizer().value({"calls": calls})
    check_golden(f"openclaw/entry_{entry}_two_bare", data)


@pytest.mark.parametrize("entry", ENTRIES)
def test_entry_host_run_context(entry, monkeypatch):
    """One call with a host ``run_context``. Every entry runs inside it: the
    transport sees ``/host`` and the session lands in the host branch's slot,
    so a sibling branch reads none."""
    inst = _make(monkeypatch, FakeGateway())
    host = RunContext.root().child("host")
    call = _record_call(inst, entry, "hosted question", {"run_context": host})
    call["active_session_id_under_sibling_ctx"] = _session_under(
        inst, RunContext.root().child("sibling")
    )
    data = Normalizer().value({"calls": [call]})
    check_golden(f"openclaw/entry_{entry}_host_ctx", data)


@pytest.mark.parametrize("script", sorted(RETRY_SCRIPTS))
@pytest.mark.parametrize("entry", ("ainfer", "infer"))
def test_rate_limit_retry(entry, script, monkeypatch):
    """B30: rate limits are retried only by ``_ainfer_with_retry`` (own
    ``max_retries`` / ``retry_delay`` / continuation prompt)."""
    sleeps = _instant_sleep(monkeypatch)
    gateway = FakeGateway(RETRY_SCRIPTS[script], transcripts=("main",))
    inst = _make(monkeypatch, gateway)
    call = _record_call(inst, entry, "original question", {})
    summary = {
        "agent_request_count": _agent_requests(call),
        "maybe_initialize_session_count": len(call["maybe_initialize_session"]),
        "backoff_sleeps": sleeps,
    }
    data = Normalizer().value({"summary": summary, "call": call})
    check_golden(f"openclaw/retry_{entry}_{script}", data)


@pytest.mark.parametrize("entry", ("ainfer", "infer"))
def test_base_retry_and_fallback_settings_ignored(entry, monkeypatch, caplog):
    """B30: base ``max_retry`` / ``fallback_mode`` / ``fallback_inferencer`` are
    pinned at construction with a WARNING; a non-rate-limit error propagates
    after one agent request."""
    sleeps = _instant_sleep(monkeypatch)
    fallback = FallbackStub()
    gateway = FakeGateway([HARD_ERROR] * 5, transcripts=("main",))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        inst = _make(
            monkeypatch,
            gateway,
            max_retry=3,
            fallback_mode=FallbackMode.ON_EXHAUSTED,
            fallback_inferencer=fallback,
        )
    norm = Normalizer()
    warnings = [
        {"level": r.levelname, "logger": r.name, "message": norm.text(r.getMessage())}
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ]
    call = _record_call(inst, entry, "doomed question", {})
    summary = {
        "construction_warnings": warnings,
        "agent_request_count": _agent_requests(call),
        "fallback_calls": fallback.calls,
        "backoff_sleeps": sleeps,
    }
    data = norm.value({"summary": summary, "call": call})
    check_golden(f"openclaw/ignored_base_settings_{entry}", data)


@pytest.mark.parametrize("entry", ("ainfer", "ainfer_streaming"))
def test_second_session_initialization(entry, monkeypatch):
    """B19 (fixed in P10): each session id is initialized once per connection,
    so the second session id gets ``_maybe_initialize_session``'s probe and
    warm-up turn too. A final bare call shows which session is resumed."""
    inst = _make(monkeypatch, FakeGateway())
    calls = [
        _record_call(inst, entry, f"{sid} question", {"session_id": sid})
        for sid in ("sess-alpha", "sess-beta")
    ]
    calls.append(_record_call(inst, entry, "bare follow-up", {}))
    data = Normalizer().value({"calls": calls})
    check_golden(f"openclaw/second_session_{entry}", data)


@pytest.mark.parametrize("entry", ("ainfer", "infer"))
def test_per_call_fallback_mode_dropped(entry, monkeypatch, caplog):
    """B30: a per-call ``fallback_mode`` would add a base retry; it is dropped
    with a WARNING, so a hard error still makes one agent request."""
    gateway = FakeGateway([HARD_ERROR] * 5, transcripts=("main",))
    inst = _make(monkeypatch, gateway)
    norm = Normalizer()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        call = _record_call(
            inst,
            entry,
            "doomed question",
            {"fallback_mode": FallbackMode.ON_FIRST_FAILURE},
        )
    warnings = [
        norm.text(r.getMessage())
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and "OpenClaw owns its retries" in r.getMessage()
    ]
    summary = {
        "agent_request_count": _agent_requests(call),
        "fallback_mode_warnings": warnings,
    }
    data = norm.value({"summary": summary, "call": call})
    check_golden(f"openclaw/per_call_fallback_mode_{entry}", data)


@pytest.mark.parametrize("entry", ENTRIES)
def test_failure_after_success(entry, monkeypatch):
    """A successful call, then one that fails without a retryable error. The
    failed call's ``active_session_id_after`` shows whether the session
    survives a terminal failure."""
    gateway = FakeGateway()
    inst = _make(monkeypatch, gateway)
    calls = [_record_call(inst, entry, "good question", {})]
    gateway.script = [HARD_ERROR]
    calls.append(_record_call(inst, entry, "doomed question", {}))
    data = Normalizer().value({"calls": calls})
    check_golden(f"openclaw/failure_after_success_{entry}", data)
