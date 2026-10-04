"""A BTA call leases its checkpoint root (plan v8 §5.11, P8; invariant I4).

Taking the call record takes a non-blocking exclusive lock on
``<checkpoint_root>/.bta_execution.lock``. A second holder — another process, another
thread, another BTA instance on the same workspace — gets ``BtaWorkspaceBusyError``
before it reads or writes anything. The lease is registered first in the
invocation's ledger, so it spans every attempt, finalize and the call's own close,
and is released last. A BTA with no workspace and no ``checkpoint_dir`` persists
nothing and takes no lease.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
    BtaWorkspaceBusyError,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.bta_checkpoints import (
    CheckpointLease,
    LEASE_FILE,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    invocation_of,
    open_invocation,
)
from attr import attrib, attrs
from rich_python_utils.io_utils.file_lock import FileLock

KINDS = ("sync", "async")

HOLD_LOCK = """
import sys, time
sys.path[:0] = {path!r}
from rich_python_utils.io_utils.file_lock import FileLock
lock = FileLock({lock!r})
assert lock.acquire(timeout=0)
print("held", flush=True)
time.sleep(60)
"""


@attrs(slots=False)
class _Stage(InferencerBase):
    response: str = attrib(default="", kw_only=True)
    on_call: object = attrib(default=None, kw_only=True)
    fail: bool = attrib(default=False, kw_only=True)

    def _answer(self, inference_input):
        if self.on_call is not None:
            self.on_call()
        if self.fail:
            raise RuntimeError(f"{inference_input} failed")
        return self.response or f"w:{inference_input}"

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._answer(inference_input)

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._answer(inference_input)


def _bta(on_breakdown=None, **kwargs):
    return BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Stage(response="1. q0\n2. q1", on_call=on_breakdown),
        worker_inferencers=lambda sub_query, index: _Stage(),
        breakdown_format="numbered_list",
        disable_aggregator=True,
        **kwargs,
    )


def _ws(path):
    return InferencerWorkspace(root=str(path))


def _call(bta, kind, text="go"):
    if kind == "sync":
        return bta.infer(text)
    return asyncio.run(bta.ainfer(text))


def _held_elsewhere(root):
    probe = FileLock(os.path.join(str(root), LEASE_FILE))
    if probe.acquire(timeout=0):
        probe.release()
        return False
    return True


@pytest.mark.parametrize("kind", KINDS)
def test_another_process_holding_the_root_rejects_the_call_before_any_io(
    kind, tmp_path
):
    root = tmp_path / "checkpoints"
    root.mkdir()
    script = HOLD_LOCK.format(path=sys.path, lock=str(root / LEASE_FILE))
    holder = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    breakdowns = []
    try:
        assert holder.stdout.readline().strip() == "held"
        bta = _bta(on_breakdown=lambda: breakdowns.append(1), workspace=_ws(tmp_path))
        with pytest.raises(BtaWorkspaceBusyError):
            _call(bta, kind)
    finally:
        holder.kill()
        holder.wait()
    assert breakdowns == []
    assert sorted(os.listdir(root)) == [LEASE_FILE]
    assert _call(_bta(workspace=_ws(tmp_path)), kind) == ("w:q0", "w:q1")


def test_two_threads_on_distinct_btas_cannot_share_a_root(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def hold():
        entered.set()
        assert release.wait(10)

    first = _bta(on_breakdown=hold, workspace=_ws(tmp_path))
    outcome = {}
    thread = threading.Thread(target=lambda: outcome.update(first=first.infer("a")))
    thread.start()
    try:
        assert entered.wait(10)
        with pytest.raises(BtaWorkspaceBusyError):
            _bta(workspace=_ws(tmp_path)).infer("b")
    finally:
        release.set()
        thread.join(10)
    assert outcome["first"] == ("w:q0", "w:q1")


@pytest.mark.parametrize("kind", KINDS)
def test_a_checkpoint_dir_only_bta_takes_and_releases_the_lease(kind, tmp_path):
    held = []
    bta = _bta(
        on_breakdown=lambda: held.append(_held_elsewhere(tmp_path)),
        checkpoint_dir=str(tmp_path),
    )
    assert _call(bta, kind) == ("w:q0", "w:q1")
    assert held == [True]
    assert not _held_elsewhere(tmp_path)


class _ClosedLast:
    """A resource a stage registers in the BTA's ledger after the lease."""

    def __init__(self, root, events):
        self.root, self.events = root, events

    def close(self):
        self.events.append(("stage resource closed", _held_elsewhere(self.root)))


@pytest.mark.parametrize("kind", KINDS)
def test_the_lease_spans_retries_finalize_and_the_call_close(kind, tmp_path):
    root = tmp_path / "checkpoints"
    events = []
    attempts = []

    def breakdown():
        attempts.append(_held_elsewhere(root))
        if len(attempts) == 1:
            raise RuntimeError("first attempt fails")
        frame = invocation_of(bta)
        frame.ledger.register(_ClosedLast(root, events), "probe")

    bta = _bta(on_breakdown=breakdown, workspace=_ws(tmp_path), max_retry=2)
    finalize = bta._finalize_output

    def recording_finalize(response):
        events.append(("finalize", _held_elsewhere(root)))
        return finalize(response)

    bta._finalize_output = recording_finalize
    assert _call(bta, kind) == ("w:q0", "w:q1")
    assert attempts == [True, True]
    assert events == [("finalize", True), ("stage resource closed", True)]
    assert not _held_elsewhere(root)


def test_a_bta_with_no_root_takes_no_lease(monkeypatch):
    acquired = []
    real = CheckpointLease.acquire
    monkeypatch.setattr(
        CheckpointLease, "acquire", lambda self: acquired.append(self) or real(self)
    )
    bta = _bta()
    with open_invocation(bta) as frame:
        assert bta._bta_call().checkpoint_root is None
        assert len(frame.ledger) == 0
    assert acquired == []


@pytest.mark.parametrize("max_retry", (1, 3))
def test_a_busy_root_is_never_retried_or_archived(max_retry, tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    holder = FileLock(str(root / LEASE_FILE))
    assert holder.acquire(timeout=0)
    breakdowns = []
    try:
        bta = _bta(
            on_breakdown=lambda: breakdowns.append(1),
            workspace=_ws(tmp_path),
            max_retry=max_retry,
        )
        with pytest.raises(BtaWorkspaceBusyError):
            asyncio.run(bta.ainfer("go"))
    finally:
        holder.release()
    assert breakdowns == []
    assert sorted(os.listdir(root)) == [LEASE_FILE]
    assert not (tmp_path / ".attempts").exists()


def test_a_retry_archives_the_attempt_but_keeps_the_lease(tmp_path):
    """Before a BTA-level retry, U3c archives the failed attempt's checkpoints into
    ``.attempts/<n>/``; the lease file stays at the root and stays held, so the
    root is never free mid-call. (The retry itself re-raises: an orchestrator does
    not re-run its subtree, U4-A.)"""
    root = tmp_path / "checkpoints"

    def work():
        raise RuntimeError("the worker fails")

    bta = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Stage(response="1. q0"),
        worker_inferencers=lambda sub_query, index: _Stage(on_call=work, max_retry=1),
        breakdown_format="numbered_list",
        disable_aggregator=True,
        workspace=_ws(tmp_path),
        max_retry=2,
    )
    archive = bta._archive_workspace_for_retry
    archived = []

    def recording_archive(attempt):
        archive(attempt)
        archived.append((LEASE_FILE in os.listdir(root), _held_elsewhere(root)))

    bta._archive_workspace_for_retry = recording_archive
    with pytest.raises(RuntimeError, match="the worker fails"):
        asyncio.run(bta.ainfer("go"))
    assert archived and set(archived) == {(True, True)}
    attempt_dirs = sorted((tmp_path / ".attempts").iterdir())
    assert attempt_dirs and all(
        LEASE_FILE not in os.listdir(d / "checkpoints") for d in attempt_dirs
    )
    assert not _held_elsewhere(root)
