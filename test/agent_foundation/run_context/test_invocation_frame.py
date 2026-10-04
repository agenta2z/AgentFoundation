"""Invocation frames, runtime keys and the resource ledger (plan v8 §5.1, P3 c1)."""

import asyncio
import logging
import pickle

import attrs
import pytest
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    aopen_invocation,
    ConcurrentInvocationError,
    enter_run,
    exit_run,
    frame_for,
    invocation,
    invocation_of,
    InvocationCleanupError,
    InvocationContractError,
    NoInvocationError,
    open_invocation,
    ResourceLedger,
    RunContext,
    RuntimeKey,
)
from attr import attrs as attrs_decorator


class _Owner:
    pass


class _Recorder:
    """Closeable resource that appends its label to a shared log."""

    def __init__(self, log, label, method="adisconnect", fail=None):
        self.log = log
        self.label = label
        self._fail = fail
        if method == "adisconnect":
            self.adisconnect = self._aclose
        elif method == "aclose":
            self.aclose = self._aclose
        else:
            self.close = self._close

    async def _aclose(self):
        self._close()

    def _close(self):
        self.log.append(self.label)
        if self._fail is not None:
            raise self._fail


# -- RuntimeKey -------------------------------------------------------------


def test_runtime_key_hashes_by_identity():
    a = RuntimeKey("Owner.value")
    b = RuntimeKey("Owner.value")
    assert a != b
    assert len({a, b}) == 2
    assert a == a


def test_runtime_key_is_frozen_and_compat_read_only():
    key = RuntimeKey[int]("Owner.value", compat={"_last_value": ""})
    assert dict(key.compat) == {"_last_value": ""}
    with pytest.raises(TypeError):
        key.compat["_other"] = "x"
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        key.name = "renamed"


def test_runtime_key_compat_is_copied():
    source = {"_last_value": ""}
    key = RuntimeKey("Owner.value", compat=source)
    source["_later"] = "x"
    assert "_later" not in key.compat


# -- frame API --------------------------------------------------------------


def test_frame_get_put_discard_require():
    owner = _Owner()
    key = RuntimeKey[int]("Owner.value")
    with open_invocation(owner) as frame:
        assert frame.get(key) is None
        frame.put(key, 3)
        assert frame.get(key) == 3
        assert frame.require(key) == 3
        frame.discard(key)
        frame.discard(key)
        assert frame.get(key) is None
        with pytest.raises(KeyError, match="Owner.value is not set.*_Owner"):
            frame.require(key)


def test_frame_get_or_create_calls_factory_once():
    calls = []

    def factory():
        calls.append(1)
        return []

    key = RuntimeKey[list]("Owner.items", factory=factory)
    with open_invocation(_Owner()) as frame:
        first = frame.get_or_create(key)
        assert frame.get_or_create(key) is first
    assert calls == [1]


def test_frame_get_or_create_without_factory_raises():
    key = RuntimeKey[int]("Owner.value")
    with open_invocation(_Owner()) as frame:
        with pytest.raises(InvocationContractError, match="has no factory"):
            frame.get_or_create(key)


def test_frames_are_distinct_per_invocation():
    owner = _Owner()
    with open_invocation(owner) as first:
        pass
    with open_invocation(owner) as second:
        pass
    assert first is not second
    assert first.invocation_id != second.invocation_id
    assert "entry='infer'" in repr(first)


# -- frame_for / invocation_of -----------------------------------------------


def test_frame_for_inside_and_outside():
    owner = _Owner()
    assert frame_for(owner) is None
    with open_invocation(owner) as frame:
        assert frame_for(owner) is frame
        assert invocation_of(owner) is frame
        assert frame_for(_Owner()) is None
    assert frame_for(owner) is None


def test_invocation_of_without_frame_names_the_fix():
    with pytest.raises(NoInvocationError, match=r"_Owner has no open invocation.*"):
        invocation_of(_Owner())
    with pytest.raises(NoInvocationError, match=r"open_invocation\(inst\)"):
        invocation_of(_Owner())


def test_nested_frames_walk_by_owner_identity():
    outer_owner, inner_owner = _Owner(), _Owner()
    with open_invocation(outer_owner) as outer:
        with open_invocation(inner_owner) as inner:
            assert inner.parent is outer
            assert frame_for(outer_owner) is outer
            assert frame_for(inner_owner) is inner
        assert frame_for(inner_owner) is None


def test_same_owner_nested_resolves_innermost():
    owner = _Owner()
    with open_invocation(owner) as outer:
        with open_invocation(owner) as inner:
            assert frame_for(owner) is inner
        assert frame_for(owner) is outer


def test_async_frame_is_visible_to_child_tasks():
    owner = _Owner()

    async def main():
        async with aopen_invocation(owner) as frame:
            seen = await asyncio.create_task(_read_frame(owner))
            assert seen is frame
            assert frame.entry == "ainfer"

    async def _read_frame(o):
        return frame_for(o)

    asyncio.run(main())


# -- modes ------------------------------------------------------------------


def test_mode_no_ctx():
    with open_invocation(_Owner()) as frame:
        assert frame.mode == "no_ctx"
        assert frame.ctx is None


def test_mode_legacy():
    token = enter_run(None, default_workspace=InferencerWorkspace(root="/tmp/run"))
    try:
        with open_invocation(_Owner()) as frame:
            assert frame.mode == "legacy"
    finally:
        exit_run(token)


def test_mode_host():
    ctx = RunContext.root(workspace=InferencerWorkspace(root="/tmp/run"))
    token = enter_run(ctx)
    try:
        with open_invocation(_Owner()) as frame:
            assert frame.mode == "host"
            assert frame.ctx is ctx
    finally:
        exit_run(token)


# -- closed frames ----------------------------------------------------------


def test_task_outliving_the_call_sees_no_frame():
    owner = _Owner()
    log = []

    async def main():
        release = asyncio.Event()

        async def late():
            await release.wait()
            return frame_for(owner)

        async with aopen_invocation(owner) as frame:
            task = asyncio.create_task(late())
        assert frame.closed
        release.set()
        assert await task is None
        with pytest.raises(InvocationContractError, match="already closed"):
            frame.ledger.register(_Recorder(log, "late"), "late")

    asyncio.run(main())
    assert log == []


# -- ledger -----------------------------------------------------------------


def test_ledger_closes_in_reverse_order():
    log = []
    ledger = ResourceLedger()
    for label in ("a", "b", "c"):
        ledger.register(_Recorder(log, label), label)
    assert len(ledger) == 3
    assert asyncio.run(ledger.aclose()) == []
    assert log == ["c", "b", "a"]
    assert len(ledger) == 0


def test_ledger_prefers_adisconnect_then_aclose_then_close():
    class _All:
        def __init__(self):
            self.used = []

        async def adisconnect(self):
            self.used.append("adisconnect")

        async def aclose(self):
            self.used.append("aclose")

        def close(self):
            self.used.append("close")

    class _AcloseAndClose:
        def __init__(self):
            self.used = []

        async def aclose(self):
            self.used.append("aclose")

        def close(self):
            self.used.append("close")

    everything, partial = _All(), _AcloseAndClose()
    sync_only = _Recorder([], "sync", method="close")
    ledger = ResourceLedger()
    ledger.register(everything, "all")
    ledger.register(partial, "partial")
    ledger.register(sync_only, "sync")
    assert asyncio.run(ledger.aclose()) == []
    assert everything.used == ["adisconnect"]
    assert partial.used == ["aclose"]
    assert sync_only.log == ["sync"]


def test_ledger_attempts_every_close_and_reports_failures():
    log = []
    ledger = ResourceLedger()
    ledger.register(_Recorder(log, "a"), "first")
    ledger.register(_Recorder(log, "b", fail=ValueError("boom")), "second")
    ledger.register(_Recorder(log, "c", fail=OSError("gone")), "third")
    errors = asyncio.run(ledger.aclose())
    assert log == ["c", "b", "a"]
    assert errors == ["third: OSError: gone", "second: ValueError: boom"]


def test_ledger_close_is_idempotent_and_seals():
    log = []
    ledger = ResourceLedger()
    ledger.register(_Recorder(log, "a"), "a")
    asyncio.run(ledger.aclose())
    assert asyncio.run(ledger.aclose()) == []
    assert ledger.close_joined() == []
    assert log == ["a"]
    with pytest.raises(InvocationContractError, match="already closed"):
        ledger.register(_Recorder(log, "b"), "b")


def test_ledger_rejects_uncloseable_resource():
    with pytest.raises(InvocationContractError, match="object has none of"):
        ResourceLedger().register(object(), "thing")


def test_ledger_cancellation_still_closes_the_rest():
    log = []

    class _Cancelled:
        async def adisconnect(self):
            log.append("cancelled")
            raise asyncio.CancelledError

    ledger = ResourceLedger()
    ledger.register(_Recorder(log, "first"), "first")
    ledger.register(_Recorder(log, "failing", fail=ValueError("x")), "failing")
    ledger.register(_Cancelled(), "cancelled")

    # asyncio.run replaces a task's CancelledError with a fresh one, so the
    # exception is caught inside the coroutine to inspect its notes.
    async def main():
        try:
            await ledger.aclose()
        except asyncio.CancelledError as exc:
            return exc
        raise AssertionError("expected CancelledError")

    exc = asyncio.run(main())
    assert log == ["cancelled", "failing", "first"]
    assert exc.__notes__ == ["owned resource failed to close: failing: ValueError: x"]


# -- cleanup failures -------------------------------------------------------


def test_cleanup_failure_after_success_raises_with_result():
    owner = _Owner()
    log = []

    async def main():
        async with aopen_invocation(owner) as frame:
            frame.ledger.register(_Recorder(log, "ok"), "ok")
            frame.ledger.register(
                _Recorder(log, "bad", fail=RuntimeError("stuck")), "client"
            )
            frame.result = "answer"

    with pytest.raises(InvocationCleanupError) as info:
        asyncio.run(main())
    error = info.value
    assert error.result == "answer"
    assert error.errors == ("client: RuntimeError: stuck",)
    assert str(error) == (
        "1 owned resource(s) failed to close after a successful call: "
        "client: RuntimeError: stuck"
    )
    assert log == ["bad", "ok"]
    assert frame_for(owner) is None


def test_invocation_cleanup_error_pickles():
    error = InvocationCleanupError({"k": 1}, ("a: E: m",))
    restored = pickle.loads(pickle.dumps(error))
    assert restored.result == {"k": 1}
    assert restored.errors == ("a: E: m",)
    assert str(restored) == str(error)


def test_cleanup_errors_recorded_earlier_in_the_invocation_are_raised():
    with pytest.raises(InvocationCleanupError) as info:
        with open_invocation(_Owner()) as frame:
            frame.cleanup_errors.append("attempt worker: OSError: pipe")
            frame.result = 7
    assert info.value.result == 7
    assert info.value.errors == ("attempt worker: OSError: pipe",)


def test_failure_keeps_original_exception_and_notes_cleanup(caplog):
    owner = _Owner()
    log = []
    with caplog.at_level(logging.ERROR, logger=invocation.__name__):
        with pytest.raises(ValueError, match="inference failed") as info:
            with open_invocation(owner) as frame:
                frame.ledger.register(
                    _Recorder(log, "bad", method="close", fail=OSError("pipe")),
                    "proc",
                )
                raise ValueError("inference failed")
    assert info.value.__notes__ == [
        "owned resource failed to close: proc: OSError: pipe"
    ]
    assert log == ["bad"]
    assert frame.closed
    assert "_Owner.infer raised ValueError" in caplog.text
    assert "proc: OSError: pipe" in caplog.text


def test_failure_without_cleanup_errors_adds_no_note(caplog):
    with caplog.at_level(logging.ERROR, logger=invocation.__name__):
        with pytest.raises(ValueError) as info:
            with open_invocation(_Owner()):
                raise ValueError("x")
    assert not hasattr(info.value, "__notes__")
    assert caplog.text == ""


# -- sync entry inside a running loop ----------------------------------------


def test_sync_invocation_inside_running_loop_closes_async_resource():
    log = []

    def sync_entry():
        with open_invocation(_Owner()) as frame:
            frame.ledger.register(_Recorder(log, "client"), "client")
            frame.result = "done"
        return frame.result

    async def main():
        return sync_entry()

    assert asyncio.run(main()) == "done"
    assert log == ["client"]


def test_sync_invocation_with_empty_ledger_skips_the_joined_bridge(monkeypatch):
    def _unexpected(coro):
        coro.close()
        raise AssertionError("run_async_joined must not run for an empty ledger")

    monkeypatch.setattr(invocation, "run_async_joined", _unexpected)
    with open_invocation(_Owner()) as frame:
        frame.result = 1


# -- contract errors are never retried --------------------------------------


@attrs_decorator
class _ContractLeaf(InferencerBase):
    calls: int = 0

    def _infer(self, x, inference_config=None, **kw):
        self.calls += 1
        raise ConcurrentInvocationError("overlap")

    async def _ainfer(self, x, inference_config=None, **kw):
        self.calls += 1
        raise ConcurrentInvocationError("overlap")


def test_contract_error_is_not_retried_sync():
    leaf = _ContractLeaf(max_retry=3)
    with pytest.raises(ConcurrentInvocationError):
        leaf.infer("q")
    assert leaf.calls == 1


def test_contract_error_is_not_retried_async():
    leaf = _ContractLeaf(max_retry=3)
    with pytest.raises(ConcurrentInvocationError):
        asyncio.run(leaf.ainfer("q"))
    assert leaf.calls == 1
