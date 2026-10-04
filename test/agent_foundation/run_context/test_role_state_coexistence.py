"""B12: a context-scoped ``switch_role`` and a class's typed call state share one
node without displacing each other, whichever comes first."""

import pytest
from agent_foundation.common.inferencers.run_context import (
    BTAState,
    DualState,
    enter_run,
    exit_run,
    MultiFlowState,
    RoleState,
    RunContext,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrs

STATE_CLASSES = [MultiFlowState, DualState, BTAState]


@attrs
class _Templated(TemplatedInferencerBase):
    def _infer(self, inference_input, inference_config=None, **kw):
        return "x"


def _leaf(state_cls):
    return _Templated(template_key="default", state_factory=lambda _input: state_cls())


def _assert_both_kept(root, leaf, state_cls):
    node = root._store.node("/")
    assert isinstance(node.call, state_cls)
    assert isinstance(node.role_state, RoleState)
    assert node.role_state.template_key == "review"
    assert leaf._effective_role()[0] == "review"


@pytest.mark.parametrize("state_cls", STATE_CLASSES)
def test_switch_before_call_keeps_typed_call_state(state_cls):
    leaf, root = _leaf(state_cls), RunContext.root(workspace=None)
    token = enter_run(root)
    try:
        leaf.switch_role("reviewer", template_key="review")
        assert leaf.infer("q") == "x"
        _assert_both_kept(root, leaf, state_cls)
    finally:
        exit_run(token)


@pytest.mark.parametrize("state_cls", STATE_CLASSES)
def test_switch_after_call_keeps_typed_call_state(state_cls):
    leaf, root = _leaf(state_cls), RunContext.root(workspace=None)
    token = enter_run(root)
    try:
        assert leaf.infer("q") == "x"
        call = root._store.node("/").call
        leaf.switch_role("reviewer", template_key="review")
        assert root._store.node("/").call is call
        _assert_both_kept(root, leaf, state_cls)
    finally:
        exit_run(token)
