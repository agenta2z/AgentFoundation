"""Connection-holding leaves keep one connection per host branch and drop them all
on disconnect (plan v8 §13, P10: per-family branch isolation and disconnect).

The claude_code SDK, codex SDK and rovodev serve leaves hold a live connection (an SDK
client, a Codex thread, a server process) in Tier-3 handles. Two independent host
roots each get their own connection and keep their own session; ``adisconnect`` at a
no-context boundary closes every branch's connection.
"""

from __future__ import annotations

import asyncio
import signal

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_sdk_inferencer import (
    ClaudeCodeSdkInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_sdk_inferencer import (
    CodexSdkInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_serve_inferencer import (
    RovoDevServeInferencer,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    enter_run,
    exit_run,
    RunContext,
)

from ._leaf_fixtures import (
    install_fake_claude_sdk,
    install_fake_codex_sdk,
    install_fake_rovodev_serve,
)


def _claude(tmp_path, monkeypatch, log):
    install_fake_claude_sdk(monkeypatch, log)
    return ClaudeCodeSdkInferencer(target_path=str(tmp_path))


def _codex(tmp_path, monkeypatch, log):
    install_fake_codex_sdk(monkeypatch, log)
    return CodexSdkInferencer(target_path=str(tmp_path))


def _serve(tmp_path, monkeypatch, log):
    install_fake_rovodev_serve(monkeypatch, log)
    return RovoDevServeInferencer(acli_path="acli", target_path=str(tmp_path))


# leaf -> (builder, log entry of one connection, log entry of one closed connection)
LEAVES = {
    "claude_sdk": (_claude, "connect", "disconnect"),
    "codex_sdk": (_codex, "thread_start", "close"),
    "rovodev_serve": (_serve, "spawn", "signal"),
}


def _host(root):
    return RunContext.root(workspace=InferencerWorkspace(root=str(root)))


def _session_under(inf, ctx):
    token = enter_run(ctx)
    try:
        return inf.active_session_id
    finally:
        exit_run(token)


def _run_two_branches_then_disconnect(inf, roots):
    """Both branches call, then the leaf disconnects with no context active, all
    on one event loop (the connections are loop-bound)."""

    async def main():
        for name, ctx in roots.items():
            await inf.ainfer(f"question {name}", run_context=ctx)
        sessions = {name: _session_under(inf, ctx) for name, ctx in roots.items()}
        assert active_run_context() is None
        await inf.adisconnect()
        return sessions

    return asyncio.run(main())


@pytest.mark.parametrize("leaf", sorted(LEAVES))
def test_each_branch_connects_once_and_disconnect_closes_them_all(
    leaf, tmp_path, monkeypatch
):
    build, opened, closed = LEAVES[leaf]
    log = []
    inf = build(tmp_path / "work", monkeypatch, log)
    roots = {name: _host(tmp_path / name) for name in ("a", "b")}
    _run_two_branches_then_disconnect(inf, roots)
    kinds = [entry[0] for entry in log]
    assert kinds.count(opened) == 2
    assert kinds.count(closed) == 2
    store = inf._get_live_handle_store()
    assert not any(
        branch.get(handle)
        for branch in store._by_path.values()
        for handle in ("client", "thread", "server_process", "http_client")
    )


@pytest.mark.parametrize("leaf", ["claude_sdk", "codex_sdk"])
def test_each_branch_keeps_its_own_session(leaf, tmp_path, monkeypatch):
    build, _, _ = LEAVES[leaf]
    inf = build(tmp_path / "work", monkeypatch, [])
    roots = {name: _host(tmp_path / name) for name in ("a", "b")}
    sessions = _run_two_branches_then_disconnect(inf, roots)
    assert sessions["a"] and sessions["b"] and sessions["a"] != sessions["b"]
    assert inf.__dict__.get("_session_id") is None


def test_the_serve_process_is_stopped_with_sigterm(tmp_path, monkeypatch):
    log = []
    inf = _serve(tmp_path / "work", monkeypatch, log)
    _run_two_branches_then_disconnect(inf, {"a": _host(tmp_path / "a")})
    assert ("signal", signal.SIGTERM) in log
