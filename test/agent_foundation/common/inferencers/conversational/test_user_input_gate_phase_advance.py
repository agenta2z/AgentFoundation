"""Regression: interactive widget responses must advance requires_user_input
SOP phases.

Bug: after the user answered Phase 0a's compound widget (clarification +
single_choice), the model_optimization SOP stayed stuck at "Phase 0a (idle)"
and never advanced to Phase 0b. Root cause: the interactive conversation-tool
collection path never opened ``user_input_gate_passed``, so
``_check_phase_completion`` could not satisfy Path 2 (gate +
``requires user input`` directive) — the *only* viable completion path for
phases driven by clarification/single_choice/compound widgets (Path 1 needs
action-tool execution; Path 3 needs declared phase outputs, which these phases
do not parse). The yolo path *did* open the gate, so only interactive runs hung.

Fix: ``ConversationalInferencer._open_user_input_gate_if_satisfied`` opens the
gate when the user supplies what the phase asked for, withholding it only for a
*declined* confirmation (preserving the prior confirmation-gate semantics).

These tests exercise the gate helper + ``_check_phase_completion`` against the
real ``model_optimization`` SOP, independent of the (env-fragile) full-CI
fixtures.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (  # noqa: E501
    ConversationalInferencer as CI,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (  # noqa: E501
    ConversationToolType,
)
from agent_foundation.resources.tools.sop.executor import build_sop_state


def _fresh_state():
    state, err = build_sop_state("model_optimization")
    if err:
        pytest.skip(f"model_optimization SOP unavailable: {err}")
    return state


def _fake_ci(state):
    """Minimal stand-in exposing exactly what the two methods under test use."""
    f = SimpleNamespace(
        sop_state=state,
        _auto_shutdown_on_sop_complete=False,
        request_shutdown=lambda: None,
        _is_affirmative_response=CI._is_affirmative_response,
    )

    def _update_prior_context(**kw):
        if "sop_state" in kw:
            f.sop_state = kw.pop("sop_state")

    f.update_prior_context = _update_prior_context
    return f


def _tool(tool_type, var):
    return SimpleNamespace(tool_type=tool_type, output_vars=[var])


def _completed_ids(state):
    return [r.phase if hasattr(r, "phase") else r for r in state.completed_phases]


def test_bug_repro_no_gate_stays_stuck():
    """Without opening the gate, Phase 0a never advances (the reported bug)."""
    state = _fresh_state()
    assert state.current_phase == "0a"
    CI._check_phase_completion(_fake_ci(state))
    assert state.current_phase == "0a", "expected STUCK at 0a without the gate"
    assert _completed_ids(state) == []


def test_interactive_compound_advances_0a_to_0b():
    """Answering Phase 0a's compound widget advances 0a -> 0b (the fix)."""
    f = _fake_ci(_fresh_state())
    tools = [
        _tool(ConversationToolType.CLARIFICATION, "workflow_target_path"),
        _tool(ConversationToolType.SINGLE_CHOICE, "workflow_modeling_artifacts_mode"),
    ]
    collected = {
        "workflow_target_path": "/repo/generative_recommenders",
        "workflow_modeling_artifacts_mode": "auto_discover",
    }
    CI._open_user_input_gate_if_satisfied(f, tools, collected)
    assert f.sop_state.user_input_gate_passed is True
    CI._check_phase_completion(f)
    assert f.sop_state.current_phase == "0b"
    assert "0a" in _completed_ids(f.sop_state)
    # gate is consumed on advance
    assert f.sop_state.user_input_gate_passed is False


def test_full_interactive_chain_reaches_phase_1():
    """0a (compound) -> 0b (single_choice) -> Phase 1 (action-tool phase)."""
    f = _fake_ci(_fresh_state())
    CI._open_user_input_gate_if_satisfied(
        f,
        [
            _tool(ConversationToolType.CLARIFICATION, "workflow_target_path"),
            _tool(ConversationToolType.SINGLE_CHOICE, "workflow_modeling_artifacts_mode"),
        ],
        {"workflow_target_path": "/repo", "workflow_modeling_artifacts_mode": "auto_discover"},
    )
    CI._check_phase_completion(f)
    assert f.sop_state.current_phase == "0b"

    CI._open_user_input_gate_if_satisfied(
        f,
        [_tool(ConversationToolType.SINGLE_CHOICE, "evolution_strategy")],
        {"evolution_strategy": "holistic"},
    )
    CI._check_phase_completion(f)
    assert f.sop_state.current_phase == "1"


@pytest.mark.parametrize("answer", ["yes", "proceed", "YES", " Proceed "])
def test_affirmative_confirmation_opens_gate(answer):
    f = _fake_ci(_fresh_state())
    CI._open_user_input_gate_if_satisfied(
        f, [_tool(ConversationToolType.CONFIRMATION, "ok")], {"ok": answer}
    )
    assert f.sop_state.user_input_gate_passed is True


@pytest.mark.parametrize("answer", ["no", "cancel", "", "nope"])
def test_declined_confirmation_withholds_gate(answer):
    """A declined confirmation must NOT advance the phase (preserved semantics)."""
    f = _fake_ci(_fresh_state())
    CI._open_user_input_gate_if_satisfied(
        f, [_tool(ConversationToolType.CONFIRMATION, "ok")], {"ok": answer}
    )
    assert f.sop_state.user_input_gate_passed is False


def test_empty_or_missing_collected_is_noop():
    f = _fake_ci(_fresh_state())
    tools = [_tool(ConversationToolType.CLARIFICATION, "x")]
    CI._open_user_input_gate_if_satisfied(f, tools, {})
    CI._open_user_input_gate_if_satisfied(f, tools, None)
    assert f.sop_state.user_input_gate_passed is False


def test_compound_with_declined_confirmation_withholds_gate():
    """If a compound bundle contains a declined confirmation, withhold the gate."""
    f = _fake_ci(_fresh_state())
    tools = [
        _tool(ConversationToolType.CLARIFICATION, "path"),
        _tool(ConversationToolType.CONFIRMATION, "ok"),
    ]
    CI._open_user_input_gate_if_satisfied(f, tools, {"path": "/repo", "ok": "no"})
    assert f.sop_state.user_input_gate_passed is False
