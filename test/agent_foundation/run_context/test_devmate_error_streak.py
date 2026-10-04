"""devmate's consecutive-error streak is session-scoped (plan v8 §13, P10; B33).

With ``session_mode=NEW_SESSION_ON_CONSECUTIVE_ERRORS`` devmate forces a new session
after ``consecutive_error_threshold`` failed calls. The streak is session state, read
and written through the session policy: under a host ctx each branch counts its own,
so failures in one branch never push another branch off its conversation; bare calls
count on the instance, as before. Only the async path counts failures (its post-hook
promotes ``success=False`` to an exception; inventory §1).
"""

from __future__ import annotations

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
    InferencerExecutionError,
    SessionMode,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import RunContext
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_response import (
    TerminalInferencerResponse,
)
from rich_python_utils.common_utils.function_helper import FallbackMode

from .test_cli_session_hooks import _call, _devmate

FAILED = TerminalInferencerResponse(output="", success=False, error="boom")


def _leaf(tmp_path, monkeypatch):
    devmate = _devmate(tmp_path, monkeypatch)
    devmate.session_mode = SessionMode.NEW_SESSION_ON_CONSECUTIVE_ERRORS
    devmate.consecutive_error_threshold = 2
    devmate.fallback_mode = FallbackMode.NEVER
    devmate.max_retry = 0
    return devmate


def _fail(devmate, kind, **kwargs):
    devmate.reply = FAILED
    with pytest.raises(InferencerExecutionError, match="boom"):
        _call(devmate, kind, **kwargs)
    devmate.reply = None


def _host(root):
    return RunContext.root(workspace=InferencerWorkspace(root=str(root)))


def test_failures_in_one_branch_leave_another_branch_on_its_session(
    tmp_path, monkeypatch
):
    kind = "async"
    devmate = _leaf(tmp_path, monkeypatch)
    first, second = _host(tmp_path / "a"), _host(tmp_path / "b")
    devmate.reply_session = "session-b"
    _call(devmate, kind, run_context=second)
    for _ in range(2):
        _fail(devmate, kind, run_context=first)
    devmate.calls.clear()
    _call(devmate, kind, run_context=second)
    assert devmate.calls == [("session-b", True)]


def test_bare_calls_keep_counting_on_the_instance(tmp_path, monkeypatch):
    kind = "async"
    devmate = _leaf(tmp_path, monkeypatch)
    devmate.reply_session = "session-x"
    _call(devmate, kind)
    for _ in range(2):
        _fail(devmate, kind)
    assert devmate._consecutive_error_count == 2
    devmate.calls.clear()
    _call(devmate, kind)
    assert devmate.calls == [(None, False)]
    assert devmate._consecutive_error_count == 0
