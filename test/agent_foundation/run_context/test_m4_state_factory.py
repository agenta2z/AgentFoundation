"""M4: state_factory populates ctx.node.call (typed or dict); None is a byte-identical no-op."""

import asyncio

from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import MultiFlowState, RunContext


@attrs
class _SF(InferencerBase):
    def _infer(self, inference_input, inference_config=None, **kw):
        return "ok"

    async def _ainfer(self, inference_input, inference_config=None, **kw):
        return "ok"


def test_state_factory_populates_typed_node_call():
    inf = _SF(state_factory=lambda inp: MultiFlowState(winner_idx=99))
    root = RunContext.root(workspace=None)
    inf.infer("x", run_context=root)
    node = root._store.node("/")
    assert isinstance(node.call, MultiFlowState) and node.call.winner_idx == 99


def test_state_factory_populates_dict_node_call_async():
    inf = _SF(state_factory=lambda inp: {"seed": inp})
    root = RunContext.root(workspace=None)
    asyncio.run(inf.ainfer("hello", run_context=root))
    assert root._store.node("/").call == {"seed": "hello"}


def test_state_factory_none_is_noop():
    inf = _SF()  # no state_factory
    root = RunContext.root(workspace=None)
    inf.infer("x", run_context=root)
    assert root._store.node("/").call is None


def test_state_factory_populates_once_not_overwritten_on_reentry():
    calls = {"n": 0}

    def factory(inp):
        calls["n"] += 1
        return {"built": calls["n"]}

    inf = _SF(state_factory=factory)
    root = RunContext.root(workspace=None)
    inf.infer("x", run_context=root)
    inf.infer("x", run_context=root)  # same node already has .call -> not rebuilt
    assert calls["n"] == 1
    assert root._store.node("/").call == {"built": 1}


def test_state_factory_no_ctx_legacy_is_noop():
    # No explicit run_context -> legacy mint, discarded; must not raise.
    inf = _SF(state_factory=lambda inp: {"x": 1})
    assert inf.infer("x") == "ok"
