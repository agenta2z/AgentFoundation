"""Unit tests for the SOP phase-progression fixes (unittest.TestCase style so
the fbcode ``python_unittest`` runner collects them; mirrors the sibling
``test_output_guardrail.py``).

Covers, against the REAL model_optimization SOP where a graph is needed:
  * SOPState.__post_init__ — set-typed maps survive a to_dict/from_dict (resume)
    round-trip, so ``required <= executed`` never degrades to ``list <= set``.
  * Fix B — forward-only async dispatch writer + SOPState.completed_phase_ids().
  * Fix C — record required conversation tools (Defect C: a fresh run otherwise
    stalls at Phase 0a) + the full 0a->0b->1 interactive chain (the oracle).
  * Guard A (Fix 4) — a required ACTION tool must run before the phase completes.
  * Fix A-residual — Strategy 3 (outputs) is gated by ``required <= executed``.
  * Fix D — derived phase-maps recomputed from the fresh graph on reattach.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (  # noqa: E501
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (  # noqa: E501
    ConversationalInferencer as CI,
)
from agent_foundation.common.workflow.sop_state import SOPState
from agent_foundation.resources.tools.sop.executor import build_sop_state


def _tool(tool_type, var):
    return SimpleNamespace(tool_type=tool_type, output_vars=[var])


class _CITestBase(unittest.TestCase):
    def _fresh_state(self):
        state, err = build_sop_state("model_optimization")
        if err:
            self.skipTest(f"model_optimization SOP unavailable: {err}")
        return state

    def _fake_ci(self, state):
        """Post-Phase K fake CI — includes SOPController with pre-populated state.

        CI's SOP methods delegate to `self.sop_controller`; the fake needs a
        real controller for the delegated methods to run against.
        """
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_controller import (
            SOPController,
        )

        prior_context: dict = {}
        ctrl = SOPController(
            prior_context_reader=lambda: prior_context,
            request_shutdown=lambda: None,
        )
        ctrl.sop_state = state

        f = SimpleNamespace(
            sop_state=state,
            sop_controller=ctrl,
            prior_context=prior_context,
            _auto_shutdown_on_sop_complete=False,
            request_shutdown=lambda: None,
            _is_affirmative_response=CI._is_affirmative_response,
        )

        def _update_prior_context(**kw):
            prior_context.update(kw)
            if "sop_state" in kw:
                f.sop_state = kw["sop_state"]
                ctrl.sop_state = kw["sop_state"]

        f.update_prior_context = _update_prior_context
        f._record_answered_required_conv_tools = (
            lambda tools: CI._record_answered_required_conv_tools(f, tools)
        )
        return f


class SOPStateInvariantTest(unittest.TestCase):
    def test_post_init_coerces_serialized_lists_to_sets(self):
        st = SOPState.from_dict(
            {
                "sop_name": "x",
                "phase_required_tools": {
                    "2": ["research_propose"],
                    "0a": ["clarification", "single_choice"],
                },
                "phase_executed_tools": {"1": ["understand_codebase"]},
            }
        )
        self.assertEqual(st.phase_required_tools["2"], {"research_propose"})
        self.assertIsInstance(st.phase_required_tools["0a"], set)
        self.assertIsInstance(st.phase_executed_tools["1"], set)
        # The subset guard the resume path relies on now works (no TypeError):
        self.assertFalse(
            st.phase_required_tools["2"] <= st.phase_executed_tools.get("2", set())
        )
        st.phase_executed_tools.setdefault("2", set()).add("research_propose")
        self.assertTrue(st.phase_required_tools["2"] <= st.phase_executed_tools["2"])

    def test_completed_phase_ids_handles_strings_and_objects(self):
        st = SOPState(completed_phases=["0a", SimpleNamespace(phase="0b"), "1"])
        self.assertEqual(st.completed_phase_ids(), ["0a", "0b", "1"])

    def test_outline_marks_completed_phases_however_they_are_stored(self):
        phases = [SimpleNamespace(id=i, name=f"Step {i}") for i in "01234"]
        st = SOPState(
            current_phase="3",
            completed_phases=["0", SimpleNamespace(phase="1"), {"phase": "2"}],
            sop=SimpleNamespace(phases=phases),
        )
        self.assertEqual(st.completed_phase_ids(), ["0", "1", "2"])
        self.assertEqual(
            st.sop_outline.splitlines(),
            [
                "0 Step 0 ✓ (done)",
                "1 Step 1 ✓ (done)",
                "2 Step 2 ✓ (done)",
                "3 Step 3 ▶ (current)",
                "4 Step 4",
            ],
        )


class FixBForwardOnlyTest(_CITestBase):
    def test_round_trip_keeps_sets(self):
        revived = SOPState.from_dict(self._fresh_state().to_dict())
        self.assertTrue(revived.phase_required_tools)
        for v in revived.phase_required_tools.values():
            self.assertIsInstance(v, set)

    def test_mark_async_phase_running_forward_only(self):
        st = self._fresh_state()
        st.current_phase = "2b"
        st.completed_phases = ["0a", "0b", "1", "1b", "2"]
        f = self._fake_ci(st)
        # research_propose -> phase "2", already completed -> must NOT regress.
        CI._mark_async_tool_phase_running(f, "research_propose")
        self.assertEqual(st.current_phase, "2b")
        # Forward move still works when the target phase is not completed.
        st.current_phase = "1b"
        st.completed_phases = ["0a", "0b", "1"]
        CI._mark_async_tool_phase_running(f, "research_propose")
        self.assertEqual(st.current_phase, "2")

    def test_mark_async_phase_running_unknown_tool_is_noop(self):
        st = self._fresh_state()
        st.current_phase = "1"
        CI._mark_async_tool_phase_running(self._fake_ci(st), "not_a_tool")
        self.assertEqual(st.current_phase, "1")


class FixCConversationRequiredTest(_CITestBase):
    def test_oracle_full_interactive_chain_reaches_phase_1(self):
        """0a (compound) -> 0b (single_choice) -> Phase 1. FAILS pre-Fix-C
        (Defect C: conversation tools never recorded -> Guard A blocks)."""
        f = self._fake_ci(self._fresh_state())
        CI._open_user_input_gate_if_satisfied(
            f,
            [
                _tool(ConversationToolType.CLARIFICATION, "workflow_target_path"),
                _tool(
                    ConversationToolType.SINGLE_CHOICE,
                    "workflow_modeling_artifacts_mode",
                ),
            ],
            {
                "workflow_target_path": "/repo",
                "workflow_modeling_artifacts_mode": "auto_discover",
            },
        )
        # BOTH required conv tools recorded for 0a (single_choice is shared with
        # 0b, so a tool_phase_map check would miss it — must use the required set).
        self.assertEqual(
            f.sop_state.phase_executed_tools.get("0a"),
            {"clarification", "single_choice"},
        )
        CI._check_phase_completion(f)
        self.assertEqual(f.sop_state.current_phase, "0b")

        CI._open_user_input_gate_if_satisfied(
            f,
            [_tool(ConversationToolType.SINGLE_CHOICE, "evolution_strategy")],
            {"evolution_strategy": "holistic"},
        )
        CI._check_phase_completion(f)
        self.assertEqual(f.sop_state.current_phase, "1")

    def test_proposal_selection_completes_2b_to_3(self):
        f = self._fake_ci(self._fresh_state())
        f.sop_state.current_phase = "2b"
        f.sop_state.completed_phases = ["0a", "0b", "1", "1b", "2"]
        CI._open_user_input_gate_if_satisfied(
            f,
            [_tool(ConversationToolType.PROPOSAL_SELECTION, "selected_proposals_ids")],
            {"selected_proposals_ids": "H1,H3"},
        )
        self.assertIn(
            "proposal_selection", f.sop_state.phase_executed_tools.get("2b", set())
        )
        CI._check_phase_completion(f)
        self.assertEqual(f.sop_state.current_phase, "3")


class GuardATest(_CITestBase):
    def test_phase2_waits_for_research_propose(self):
        f = self._fake_ci(self._fresh_state())
        f.sop_state.current_phase = "2"
        f.sop_state.completed_phases = ["0a", "0b", "1", "1b"]
        # Goal is a single_choice widget; single_choice is NOT in phase 2's
        # required set ({research_propose}) -> nothing recorded for phase 2.
        CI._open_user_input_gate_if_satisfied(
            f,
            [_tool(ConversationToolType.SINGLE_CHOICE, "goal")],
            {"goal": "use the proposed goal"},
        )
        CI._check_phase_completion(f)
        self.assertEqual(f.sop_state.current_phase, "2")  # not completed by widget
        # research_propose completes (Strategy 1 records it) -> phase advances.
        CI._check_phase_completion(f, tool_name="research_propose")
        self.assertEqual(f.sop_state.current_phase, "2b")


class FixAResidualTest(unittest.TestCase):
    def test_outputs_do_not_complete_before_required_tool(self):
        phase = SimpleNamespace(
            id="X",
            name="Synthetic",
            directives=["requires user input"],
            outputs=["out1"],
        )
        st = SOPState(sop_name="synthetic", current_phase="X")
        st.sop = SimpleNamespace(phases=[phase])
        st.completed_phases = []
        st.phase_outputs = {"out1": "value"}  # output present
        st.phase_required_tools = {"X": {"toolX"}}
        st.phase_executed_tools = {}
        st.user_input_gate_passed = False  # isolate Strategy 3
        # Post-Phase K: use a real SOPController for the delegator.
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_controller import (
            SOPController,
        )

        ctrl = SOPController(request_shutdown=lambda: None)
        ctrl.sop_state = st
        f = SimpleNamespace(
            sop_state=st,
            sop_controller=ctrl,
            _auto_shutdown_on_sop_complete=False,
            request_shutdown=lambda: None,
        )
        CI._check_phase_completion(f)
        self.assertEqual(st.current_phase, "X")
        self.assertEqual(st.completed_phase_ids(), [])


class FixDReattachTest(unittest.TestCase):
    def test_reload_sop_definition_recomputes_stale_maps(self):
        st = SOPState(sop_name="model_optimization", current_phase="2b")
        st.sop = None
        st.phase_required_tools = {"stale_phase": {"stale_tool"}}
        st.tool_phase_map = {"stale_tool": "stale_phase"}
        st.completed_phases = ["0a", "0b", "1", "1b", "2"]
        st.phase_executed_tools = {"1": {"understand_codebase"}}
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.sop_controller import (
            SOPController,
        )

        _fake = SimpleNamespace(sop_controller=SOPController())
        try:
            CI._reload_sop_definition(_fake, st)
        except Exception as e:  # registry unavailable in this env
            self.skipTest(f"model_optimization SOP unavailable for reload: {e}")
        self.assertEqual(st.tool_phase_map.get("research_propose"), "2")
        self.assertEqual(st.tool_phase_map.get("proposal_selection"), "2b")
        self.assertNotIn("stale_tool", st.tool_phase_map)
        self.assertEqual(st.phase_required_tools.get("2"), {"research_propose"})
        # Runtime progress untouched:
        self.assertEqual(st.completed_phases, ["0a", "0b", "1", "1b", "2"])
        self.assertEqual(st.phase_executed_tools, {"1": {"understand_codebase"}})


class ApplyWidgetAnswerDeterminismTest(unittest.IsolatedAsyncioTestCase):
    """`_apply_widget_answer` — the shared, emit-free single-tool decode used by
    BOTH the live path and pending-widget recovery. Post-Phase F: dispatches
    through the handler registry; test uses a real ``default_registry()`` +
    a minimal fake CI that provides the surface the handlers touch.
    """

    def _fake_ci(self):
        from types import MappingProxyType

        from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_protocol import (
            HandlerContext,
        )
        from agent_foundation.common.inferencers.agentic_inferencers.conversational.handlers import (
            default_registry,
        )

        published: dict = {}
        registry = default_registry()

        f = SimpleNamespace(
            _last_conv_nested_bindings={},
            _next_action_tool_overrides=None,
            _next_turn_variables=None,
            _next_dashboard_directives=None,
            prompt_renderer=None,
            tool_executor=None,
            interactive=None,
            tool_registry=None,
            prior_context={},
            _session_root=lambda: "",
            handler_registry=registry,
        )
        f.set_session_variables = lambda mapping, tool_type=None: published.update(
            mapping
        )
        f._published = published
        f._resolve_tool_name = lambda name: name
        f._build_handler_context = lambda: HandlerContext(
            prior_context=MappingProxyType({}),
            prompt_renderer=None,
            tool_executor=None,
            interactive=None,
            action_tools=None,
            tool_registry=None,
            resolve_tool_name=None,
            handler_registry=registry,
        )
        return f

    async def test_proposal_selection_publishes_and_returns_joined(self):
        f = self._fake_ci()
        tool = SimpleNamespace(
            tool_type=ConversationToolType.PROPOSAL_SELECTION,
            output_vars=["selected_proposals_ids"],
            choices=None,
            metadata={},
        )
        out = await CI._apply_widget_answer(
            f, tool, {"selected_proposals": ["P1", "P3"]}
        )
        self.assertEqual(out, "P1,P3")
        self.assertEqual(f._published.get("selected_proposals_ids"), "P1,P3")

    async def test_confirmation_sets_pending_and_returns_choice(self):
        f = self._fake_ci()
        tool = SimpleNamespace(
            tool_type=ConversationToolType.CONFIRMATION, output_vars=[], choices=None
        )
        out = await CI._apply_widget_answer(
            f,
            tool,
            {"choice": "yes", "param_overrides": {"x": 1}, "variables": {"v": 2}},
        )
        self.assertEqual(out, "yes")
        self.assertEqual(f._next_action_tool_overrides, {"x": 1})
        self.assertEqual(f._next_turn_variables, {"v": 2})

    async def test_freetext_finalizes_and_publishes(self):
        f = self._fake_ci()
        tool = SimpleNamespace(
            tool_type=ConversationToolType.CLARIFICATION,
            output_vars=["p"],
            choices=None,
            expected_input_type="free_text",
            prefix="",
            allow_multiple_input=False,
            serialization="auto",
        )
        # user_input is a dict wrapping a string -> free-text finalize path.
        out = await CI._apply_widget_answer(f, tool, {"user_input": "hello world"})
        self.assertEqual(out, "hello world")
        self.assertEqual(f._published.get("p"), "hello world")

    async def test_none_answer_returns_none(self):
        self.assertIsNone(
            await CI._apply_widget_answer(self._fake_ci(), _tool("x", "v"), None)
        )


if __name__ == "__main__":
    unittest.main()
