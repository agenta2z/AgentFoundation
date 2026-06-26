"""M7 log-path: a workspace-derived session logger follows the *live* run-context's
workspace at write time, instead of staying baked at whatever workspace the instance
was last assigned.

Root cause this guards against: M7 publishes a leaf's deep workspace into the ctx
(``workspace_override``) without calling the ``_workspace`` setter, so the setter-time
logger redirect never fires and the leaf's session log lands at a stale parent dir.
The fix re-bases the recorded workspace-relative path against the live ctx workspace
per write (``InferencerBase._log_path_override`` + the ``Debuggable.log`` seam), never
mutating the (possibly shared) logger.
"""

import glob
import os

from attr import attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    RunContext,
    enter_run,
    exit_run,
)


@attrs
class _Mock(InferencerBase):
    def _infer(self, inference_input, inference_config=None, **kw):
        return "x"

    async def _ainfer(self, inference_input, inference_config=None, **kw):
        return "x"


def _session_jsonls(root):
    return sorted(glob.glob(os.path.join(root, "logs", "**", "*.jsonl"), recursive=True))


def _ctx_with_workspace(ws):
    """A root ctx that publishes ``ws`` as the active node's workspace_override —
    the M7 publish-to-ctx path (no instance-setter call)."""
    ctx = RunContext.root(workspace=None)
    ctx.handles.set("workspace_override", ws)
    return ctx


def test_leaf_log_follows_ctx_workspace(tmp_path):
    """The session log lands in the *live ctx* workspace, not the baked one."""
    A = str(tmp_path / "A")
    B = str(tmp_path / "B")
    m = _Mock(workspace=InferencerWorkspace(root=A))
    # Tagged, baked at A.
    assert m._ws_log_relpaths == {"_workspace": os.path.join("logs", "session.jsonl")}

    tok = enter_run(_ctx_with_workspace(InferencerWorkspace(root=B)))
    try:
        m.log("under-ctx-B", log_type="test")
    finally:
        exit_run(tok)

    assert _session_jsonls(B), "leaf log must land in the live ctx workspace B"
    assert not _session_jsonls(A), "leaf log must NOT land at the baked workspace A"


def test_two_ctxs_isolated_on_one_shared_instance(tmp_path):
    """One shared instance, two contexts -> two independent deep log dirs (the M7
    payoff: no mutation, so concurrent branches can't clobber each other)."""
    A = str(tmp_path / "A")
    B = str(tmp_path / "B")
    base = str(tmp_path / "base")
    m = _Mock(workspace=InferencerWorkspace(root=base))

    for root in (A, B):
        tok = enter_run(_ctx_with_workspace(InferencerWorkspace(root=root)))
        try:
            m.log(f"hello-{root}", log_type="test")
        finally:
            exit_run(tok)

    assert _session_jsonls(A), "ctx A must get its own session log"
    assert _session_jsonls(B), "ctx B must get its own session log"
    assert not _session_jsonls(base), "nothing should land at the shared baked workspace"


def test_legacy_no_ctx_is_byte_identical(tmp_path):
    """No active context -> baked path used, unchanged."""
    A = str(tmp_path / "A")
    m = _Mock(workspace=InferencerWorkspace(root=A))
    m.log("legacy", log_type="test")
    assert _session_jsonls(A), "no-ctx log uses the baked workspace"


def test_setter_redirect_then_override_suppressed(tmp_path):
    """Reviewer/fixer path: a real ``_workspace`` setter re-bases the logger; under a
    ctx whose effective workspace already matches, the per-write override is a no-op."""
    A = str(tmp_path / "A")
    R = str(tmp_path / "role")
    m = _Mock(workspace=InferencerWorkspace(root=A))

    # switch_role-style: set the instance workspace via the real setter.
    m._workspace = InferencerWorkspace(root=R)
    wl = m.logger["_workspace"]
    assert wl.file_path == os.path.join(R, "logs", "session.jsonl"), "setter re-bases the logger"

    # Under a ctx whose effective workspace equals the instance workspace (R), the
    # override resolves to the same path -> suppressed (None).
    tok = enter_run(_ctx_with_workspace(InferencerWorkspace(root=R)))
    try:
        assert m._log_path_override("_workspace", wl) is None
        m.log("role-write", log_type="test")
    finally:
        exit_run(tok)
    assert _session_jsonls(R), "log lands in the role workspace"
    assert not _session_jsonls(A), "nothing left at the original workspace"


def test_two_distinct_tagged_loggers_both_follow_ctx(tmp_path):
    """Genericity: it is the *recorded relpath* that is re-based, not a hardcoded
    'session.jsonl' — a second workspace logger with a different filename follows too."""
    from rich_python_utils.io_utils.json_io import JsonLogger

    A = str(tmp_path / "A")
    B = str(tmp_path / "B")
    m = _Mock(workspace=InferencerWorkspace(root=A))

    # Add a second workspace-derived logger with a DIFFERENT relpath (audit.jsonl).
    audit_path = os.path.join(A, "logs", "audit.jsonl")
    m.logger["_audit"] = JsonLogger(file_path=audit_path, append=True)
    m._tag_ws_log_relpath("_audit", audit_path, InferencerWorkspace(root=A))
    assert m._ws_log_relpaths["_audit"] == os.path.join("logs", "audit.jsonl")

    tok = enter_run(_ctx_with_workspace(InferencerWorkspace(root=B)))
    try:
        m.log("both", log_type="test")
    finally:
        exit_run(tok)

    assert os.path.exists(os.path.join(B, "logs", "audit.jsonl")), "audit logger follows ctx (relpath, not hardcoded)"
    assert _session_jsonls(B), "session logger follows ctx too"
    assert not os.path.exists(os.path.join(A, "logs", "audit.jsonl")), "audit did not write at baked A"


def test_redirect_leaves_untagged_user_logger_untouched(tmp_path):
    """``_redirect_loggers_to_workspace`` re-bases only tagged loggers; a user's own
    JsonLogger (untagged) is not clobbered onto the workspace path."""
    from rich_python_utils.io_utils.json_io import JsonLogger

    A = str(tmp_path / "A")
    B = str(tmp_path / "B")
    user_path = str(tmp_path / "user_chosen.jsonl")
    m = _Mock(workspace=InferencerWorkspace(root=A))
    m.logger["_user"] = JsonLogger(file_path=user_path, append=True)  # untagged

    m._workspace = InferencerWorkspace(root=B)  # real setter -> redirect

    assert m.logger["_workspace"].file_path == os.path.join(B, "logs", "session.jsonl"), "tagged logger re-based"
    assert m.logger["_user"].file_path == user_path, "untagged user logger left untouched (no clobber)"


def test_deferred_logger_un_defers_under_ctx(tmp_path):
    """An inferencer constructed WITHOUT a workspace (deferred session logger)
    must still get a session log when it RUNS under a ctx that supplies the
    workspace only via ctx-publish — the BTA-aggregator / outer-review case.

    Before the fix the deferred logger was never un-deferred (no setter fires
    under M7 publish-to-ctx), so the node ran but emitted no session log.
    """
    import asyncio

    A = str(tmp_path / "A")
    m = _Mock()  # no workspace at construction -> deferred logger
    assert getattr(m, "_logger_awaiting_workspace", False) is True
    assert "_workspace" not in (m.logger if isinstance(m.logger, dict) else {})

    tok = enter_run(_ctx_with_workspace(InferencerWorkspace(root=A)))
    try:
        asyncio.run(m.ainfer("go"))
    finally:
        exit_run(tok)

    assert _session_jsonls(A), (
        "a deferred logger must un-defer and write a session log when the "
        "inferencer runs under a ctx-published workspace"
    )


def test_deferred_logger_no_ctx_is_byte_identical(tmp_path):
    """No active ctx -> the un-defer is a no-op (legacy byte-identical): a
    deferred instance with no workspace writes no session file."""
    import asyncio

    m = _Mock()  # deferred, no workspace, no ctx
    asyncio.run(m.ainfer("go"))  # must not crash and must not mint a logger
    assert getattr(m, "_logger_awaiting_workspace", False) is True
    assert "_workspace" not in (m.logger if isinstance(m.logger, dict) else {})


def test_deferred_logger_un_defers_via_bridge_entrypoint_override(tmp_path):
    """The CLI/streaming case: an inferencer that overrides ``ainfer`` with
    ``@bridge_entrypoint`` and calls ``_ainfer_single`` directly (like
    RovoDevCli/ClaudeCodeCli) BYPASSES ``InferencerBase.ainfer`` — but still
    converges at ``__ainfer_single_impl``, so a deferred logger un-defers there.
    This is the top-BTA-aggregator path."""
    import asyncio
    from agent_foundation.common.inferencers.run_context import bridge_entrypoint

    @attrs
    class _CliMock(InferencerBase):
        @bridge_entrypoint
        async def ainfer(self, inference_input, inference_config=None, **kw):
            return await self._ainfer_single(inference_input, inference_config, **kw)

        def _infer(self, inference_input, inference_config=None, **kw):
            return "x"

        async def _ainfer(self, inference_input, inference_config=None, **kw):
            return "x"

    A = str(tmp_path / "A")
    m = _CliMock()  # deferred at construction
    assert getattr(m, "_logger_awaiting_workspace", False) is True

    tok = enter_run(_ctx_with_workspace(InferencerWorkspace(root=A)))
    try:
        asyncio.run(m.ainfer("go"))
    finally:
        exit_run(tok)

    assert _session_jsonls(A), (
        "a CLI-override (bridge_entrypoint) deferred logger must un-defer at "
        "__ainfer_single_impl and write a session log"
    )
