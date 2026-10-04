"""The deferred workspace-logger un-defer is safe under ``parallel_infer`` threads."""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import RunContext
from agent_foundation.common.inferencers.run_context.bridge import enter_run, exit_run
from attr import attrs

N_THREADS = 8


@attrs(slots=False)
class _NoopBase(InferencerBase):
    def _infer(self, inp, cfg=None, **kw):
        return ""

    async def _ainfer(self, inp, cfg=None, **kw):
        return ""


@attrs(slots=False)
class _SlowWorkspaceBase(_NoopBase):
    """Widens the window between the deferred-flag check and the logger install."""

    def _workspace_under(self, ctx):
        time.sleep(0.02)
        return super()._workspace_under(ctx)


def _deferred(cls, tmp_path):
    inst = cls(logger="auto")
    assert inst._logger_awaiting_workspace is True
    caller = RunContext.root(
        workspace=InferencerWorkspace(root=str(tmp_path / "ws"))
    ).child("x")
    return inst, caller


def _undefer_under(inst, caller):
    token = enter_run(caller)
    try:
        inst._ensure_ctx_workspace_logger()
    finally:
        exit_run(token)


def test_undefer_does_not_break_a_concurrent_logger_iteration(tmp_path):
    inst, caller = _deferred(_NoopBase, tmp_path)
    pending = iter(inst.logger.items())
    next(pending)

    _undefer_under(inst, caller)

    list(pending)
    assert "_workspace" in inst.logger
    assert inst._logger_awaiting_workspace is False


def test_concurrent_undefer_creates_exactly_one_workspace_logger(tmp_path):
    inst, caller = _deferred(_SlowWorkspaceBase, tmp_path)
    created = []
    original = inst._add_workspace_logger

    def _counting_add(workspace):
        created.append(threading.get_ident())
        original(workspace)

    inst._add_workspace_logger = _counting_add
    barrier = threading.Barrier(N_THREADS)

    def _worker():
        barrier.wait()
        _undefer_under(inst, caller)

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        futures = [pool.submit(_worker) for _ in range(N_THREADS)]
    for future in futures:
        future.result()

    assert len(created) == 1
    ws = caller.workspace
    assert inst._ws_log_relpaths["_workspace"] == os.path.relpath(
        os.path.join(ws.logs_dir, "session.jsonl"), ws.root
    )
    assert inst._resolved_logger_configs["_workspace"] is not None
