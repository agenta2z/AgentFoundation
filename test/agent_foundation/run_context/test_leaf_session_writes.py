"""Leaves record a streamed session through the session policy (plan v8 §13, P10 c5).

codex CLI and SDK, devmate SDK, metamate SDK and rovochat wrote a streamed session id
straight into ``_session_id``, the backing every host branch's cold read falls back
to, so a host fan-out branch could resume a sibling's conversation. They now write
``active_session_id`` (this branch's slot under a host ctx, the backing otherwise),
and metamate's and rovochat's conversation ids follow the same policy.
"""

from __future__ import annotations

import asyncio
import sys

from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_cli_inferencer import (
    CodexCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_sdk_inferencer import (
    MetamateSDKInferencer,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    RunContext,
)

from ._leaf_fixtures import FAKE_CODEX


def _codex(tmp_path):
    script = tmp_path / "fake_codex.py"
    script.write_text(FAKE_CODEX, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return CodexCliInferencer(
        codex_command=f"{sys.executable} {script}", target_path=str(work)
    )


def _host(root):
    return RunContext.root(workspace=InferencerWorkspace(root=str(root)))


def _session_under(inf, ctx):
    token = enter_run(ctx)
    try:
        return inf.active_session_id
    finally:
        exit_run(token)


def test_host_branches_keep_their_own_streamed_codex_session(tmp_path):
    codex = _codex(tmp_path)
    first, second = _host(tmp_path / "a"), _host(tmp_path / "b")
    asyncio.run(codex.ainfer("one a", run_context=first))
    asyncio.run(codex.ainfer("two b", run_context=second))
    assert _session_under(codex, first) == "thread-a"
    assert _session_under(codex, second) == "thread-b"
    assert codex.__dict__.get("_session_id") is None


def test_a_bare_codex_call_records_the_session_on_the_instance(tmp_path):
    codex = _codex(tmp_path)
    asyncio.run(codex.ainfer("bare c"))
    assert codex.active_session_id == "thread-c"
    assert codex.__dict__.get("_session_id") == "thread-c"


def test_a_conversation_id_is_branch_scoped_under_a_host_ctx():
    metamate = MetamateSDKInferencer()
    root = RunContext.root()
    token = enter_run(root)
    try:
        metamate._conversation_uuid = "conv-branch"
        assert metamate._conversation_uuid == "conv-branch"
    finally:
        exit_run(token)
    assert metamate.__dict__.get("_conversation_uuid") is None
    metamate._conversation_uuid = "conv-bare"
    assert metamate.__dict__.get("_conversation_uuid") == "conv-bare"
    metamate.reset_session()
    assert metamate._conversation_uuid is None
