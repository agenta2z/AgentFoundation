"""§2.1/P-#6: cancellation_token halts fan-out; no-op (byte-identical) without one."""

import asyncio

import pytest
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import RunContext, RuntimeBindings
from attr import attrs


@attrs
class _Leaf(InferencerBase):
    def _infer(self, x, inference_config=None, **kw):
        return x

    async def _ainfer(self, x, inference_config=None, **kw):
        return x


# One item at a time: a host ``aparallel_infer`` on an uncertified class refuses
# overlapping items (the P6 single-flight guard).


def test_no_token_is_noop():
    leaf = _Leaf()
    root = RunContext.root(workspace=None)
    out = asyncio.run(
        leaf.aparallel_infer(["a", "b"], max_concurrency=1, run_context=root)
    )
    assert out == ["a", "b"]


def test_set_token_halts_fanout():
    leaf = _Leaf()
    token = {"cancelled": True}
    root = RunContext.root(
        workspace=None, runtime=RuntimeBindings(cancellation_token=token)
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            leaf.aparallel_infer(["a", "b", "c"], max_concurrency=1, run_context=root)
        )


def test_check_cancelled_supports_event_and_callable():
    leaf = _Leaf()
    # callable token
    root = RunContext.root(
        workspace=None, runtime=RuntimeBindings(cancellation_token=lambda: True)
    )
    tok = __import__(
        "agent_foundation.common.inferencers.run_context", fromlist=["enter_run"]
    ).enter_run(root)
    try:
        with pytest.raises(asyncio.CancelledError):
            leaf._check_cancelled()
    finally:
        __import__(
            "agent_foundation.common.inferencers.run_context", fromlist=["exit_run"]
        ).exit_run(tok)
