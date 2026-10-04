"""``publish_result`` / ``read_result`` and the declared bare-compat fields (plan v8
§5.4, P3 c10).

A key declares the documented getter fields its value backs (``RuntimeKey.compat``).
Inside the owner's invocation a published value is a frame component; in a
non-host invocation the declared fields are written onto the owner when the
invocation closes, whatever its outcome. Host invocations never touch them. A
frameless publish is a no-op under a host ctx and writes the fields at once
otherwise; a frameless read raises.
"""

from __future__ import annotations

import asyncio

import attrs as attrs_mod
import pytest
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import (
    aopen_invocation,
    declared_compat_fields,
    declared_runtime_keys,
    enter_run,
    exit_run,
    frame_for,
    mint_root,
    NoInvocationError,
    open_invocation,
    publish_result,
    read_result,
    RunContext,
    RuntimeKey,
)
from attr import attrib, attrs

KINDS = ("sync", "async")


@attrs_mod.frozen
class _Result:
    text: str
    code: int


class _Owner:
    RESULT = RuntimeKey(
        "_Owner.result", compat={"_last_text": "text", "_last_code": "code"}
    )
    PLAIN = RuntimeKey("_Owner.plain", compat={"_last_plain": ""})
    PRIVATE = RuntimeKey("_Owner.private")

    _last_text = "previous"
    _last_code = -1
    _last_plain = None


class _SubOwner(_Owner):
    EXTRA = RuntimeKey("_SubOwner.extra", compat={"_last_extra": ""})


VALUE = _Result("hello", 7)


def _host():
    return RunContext.root().child("leaf")


def _fields(owner):
    return (owner._last_text, owner._last_code)


# -- inside an invocation ------------------------------------------------------


def test_a_host_invocation_publishes_the_component_and_never_the_fields():
    owner, ctx = _Owner(), _host()
    token = enter_run(ctx)
    try:
        with open_invocation(owner) as frame:
            publish_result(owner, _Owner.RESULT, VALUE)
            assert read_result(owner, _Owner.RESULT) is VALUE
            assert frame.pending_compat == {}
    finally:
        exit_run(token)
    assert _fields(owner) == ("previous", -1)


@pytest.mark.parametrize("mode", ("legacy", "no_ctx"))
def test_a_bare_invocation_writes_the_fields_when_it_closes(mode):
    owner = _Owner()
    token = enter_run(mint_root()) if mode == "legacy" else None
    try:
        with open_invocation(owner) as frame:
            assert frame.mode == mode
            publish_result(owner, _Owner.RESULT, VALUE)
            assert read_result(owner, _Owner.RESULT) is VALUE
            assert _fields(owner) == ("previous", -1)
    finally:
        if token is not None:
            exit_run(token)
    assert _fields(owner) == ("hello", 7)


def test_bare_fields_are_written_on_failure_too():
    owner = _Owner()
    with pytest.raises(RuntimeError, match="boom"):
        with open_invocation(owner):
            publish_result(owner, _Owner.RESULT, VALUE)
            raise RuntimeError("boom")
    assert _fields(owner) == ("hello", 7)


def test_bare_fields_are_written_on_cancellation_too():
    owner = _Owner()

    async def call(started):
        async with aopen_invocation(owner):
            publish_result(owner, _Owner.RESULT, VALUE)
            started.set()
            await asyncio.sleep(10)

    async def main():
        started = asyncio.Event()
        task = asyncio.create_task(call(started))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert _fields(owner) == ("hello", 7)


def test_only_the_fields_this_invocation_published_are_written():
    owner = _Owner()
    owner._last_plain = "kept"
    with open_invocation(owner):
        publish_result(owner, _Owner.RESULT, VALUE)
        publish_result(owner, _Owner.PRIVATE, "no compat")
    assert owner._last_plain == "kept"
    assert "_Owner.private" not in vars(owner)


def test_a_compat_mapping_to_the_empty_name_writes_the_value_itself():
    owner = _Owner()
    with open_invocation(owner):
        publish_result(owner, _Owner.PLAIN, ["v"])
    assert owner._last_plain == ["v"]


def test_publishing_none_writes_none_into_every_declared_field():
    owner = _Owner()
    with open_invocation(owner):
        publish_result(owner, _Owner.RESULT, None)
    assert _fields(owner) == (None, None)


# -- without an invocation ------------------------------------------------------


def test_a_frameless_publish_under_a_host_ctx_is_a_no_op():
    owner, ctx = _Owner(), _host()
    token = enter_run(ctx)
    try:
        publish_result(owner, _Owner.RESULT, VALUE)
    finally:
        exit_run(token)
    assert _fields(owner) == ("previous", -1)


@pytest.mark.parametrize("mode", ("legacy", "no_ctx"))
def test_a_frameless_bare_publish_writes_the_fields_at_once(mode):
    owner = _Owner()
    token = enter_run(mint_root()) if mode == "legacy" else None
    try:
        publish_result(owner, _Owner.RESULT, VALUE)
        assert _fields(owner) == ("hello", 7)
    finally:
        if token is not None:
            exit_run(token)


def test_a_frameless_read_raises_and_names_the_fix():
    with pytest.raises(NoInvocationError) as caught:
        read_result(_Owner(), _Owner.RESULT)
    message = str(caught.value)
    assert "_Owner.result" in message
    assert "_Owner" in message
    assert "open_invocation(inst)" in message


def test_another_owners_frame_does_not_count():
    owner, other = _Owner(), _Owner()
    with open_invocation(other):
        with pytest.raises(NoInvocationError):
            read_result(owner, _Owner.RESULT)
        publish_result(owner, _Owner.RESULT, VALUE)
        assert frame_for(owner) is None
    assert _fields(owner) == ("hello", 7)
    assert _fields(other) == ("previous", -1)


# -- declarations -------------------------------------------------------------


def test_declared_keys_are_collected_along_the_mro():
    assert set(declared_runtime_keys(_SubOwner)) == {
        _Owner.RESULT,
        _Owner.PLAIN,
        _Owner.PRIVATE,
        _SubOwner.EXTRA,
    }
    assert declared_compat_fields(_Owner) == {"_last_text", "_last_code", "_last_plain"}
    assert declared_compat_fields(_SubOwner) == declared_compat_fields(_Owner) | {
        "_last_extra"
    }


# -- through a public entry ------------------------------------------------------


@attrs
class _Leaf(InferencerBase):
    RESULT = RuntimeKey("_Leaf.result", compat={"_last_result": ""})

    _last_result = attrib(default=None, init=False)
    fail: bool = attrib(default=False, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        publish_result(self, self.RESULT, f"r:{inference_input}")
        if self.fail:
            raise RuntimeError("after publishing")
        return read_result(self, self.RESULT)

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input, inference_config, **kwargs)


def _call(leaf, kind, text, **kwargs):
    if kind == "sync":
        return leaf.infer(text, **kwargs)
    return asyncio.run(leaf.ainfer(text, **kwargs))


@pytest.mark.parametrize("kind", KINDS)
def test_a_bare_call_updates_the_getter_field_and_a_host_call_does_not(kind):
    leaf = _Leaf()
    assert _call(leaf, kind, "a") == "r:a"
    assert leaf._last_result == "r:a"
    assert _call(leaf, kind, "b", run_context=_host()) == "r:b"
    assert leaf._last_result == "r:a"


@pytest.mark.parametrize("kind", KINDS)
def test_a_failed_bare_call_still_updates_the_getter_field(kind):
    leaf = _Leaf(fail=True)
    with pytest.raises(RuntimeError, match="after publishing"):
        _call(leaf, kind, "a")
    assert leaf._last_result == "r:a"
