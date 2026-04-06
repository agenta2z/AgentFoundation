

"""Integration test: SOP-driven prompt rotation via ConversationalInferencer.

Verifies that as state transitions happen, the rendered prompt's
{{ workflow_nextstep_guidance }} section shifts to show the correct available phases.
Uses a mock template and SOP — no real LLM calls.
"""

import os
import tempfile
import unittest

from agent_foundation.common.inferencers.agentic_inferencers.conversational.prompt_rendering import (
    JinjaPromptRenderer,
)
from rich_python_utils.common_objects.workflow.stategraph import (
    StateGraphTracker,
)
from rich_python_utils.string_utils.formatting.template_manager.sop_manager import (
    SOPManager,
)

# ---------------------------------------------------------------------------
# Mock template and SOP
# ---------------------------------------------------------------------------

MOCK_TEMPLATE = """\
You are {{ employee.name }}. Your role is {{ employee.role }}.

{{ workflow_nextstep_guidance }}

Current target: {{ target_path }}
"""

MOCK_SOP = """\
## Phase 0 [initial]: Setup `target_path` and `strategy`

Configure the workspace.

**Tools** [__must__]:
- /set-target-path <path>

## Phase 1 [__depends on__ Phase 0]: Investigation `findings`

Investigate the codebase at {target_path}.

## Phase 2 [__depends on__ Phase 1]: Proposal `proposal`

Generate research proposals.

## Phase 3 [__depends on__ Phase 2; __go to__ Phase 2 __if__ `continue`]: Evaluate and decide to `continue`

Decide whether to continue.
"""

MOCK_VARIABLES = """\
employee:
  name: TestBot
  role: a test AI agent
"""


class TestSOPPromptIntegration(unittest.TestCase):
    """Test that SOP state transitions rotate the prompt's nextstep_guidance."""

    def setUp(self):
        """Create a temp directory with mock template, SOP, and variables."""
        self.tmpdir = tempfile.mkdtemp()

        # Create template directory structure
        template_dir = os.path.join(self.tmpdir, "conversation", "main")
        os.makedirs(template_dir)
        variables_dir = os.path.join(template_dir, "_variables", "workflow")
        os.makedirs(variables_dir)

        # Write mock files
        with open(os.path.join(template_dir, "initial.jinja2"), "w") as f:
            f.write(MOCK_TEMPLATE)
        with open(os.path.join(variables_dir, "sop.md"), "w") as f:
            f.write(MOCK_SOP)
        with open(os.path.join(template_dir, ".initial.variables.yaml"), "w") as f:
            f.write(MOCK_VARIABLES)

        self.renderer = JinjaPromptRenderer(
            template_dir=self.tmpdir,
            template_path="conversation/main/initial.jinja2",
        )

        # Load the SOP
        sop_path = self.renderer.find_sop_file()
        self.assertIsNotNone(sop_path, "SOP file should be found")
        self.sop = SOPManager.load(sop_path)

    def _render_with_state(self, tracker, user_message="what next?"):
        """Render the template with state from the tracker."""
        template_vars = self.renderer.template_variables
        nextstep_guidance = SOPManager.render_guidance(
            tracker, self.sop, context={"target_path": tracker.state_outputs.get("target_path", "not set")},
        )
        feed = {
            **template_vars,
            "workflow_nextstep_guidance": nextstep_guidance,
            "target_path": tracker.state_outputs.get("target_path", "not set"),
        }
        return self.renderer.render(feed)

    # -- Scenario tests --

    def test_idle_state_shows_setup(self):
        """When no phases completed, guidance shows Phase 0 (Setup)."""
        tracker = StateGraphTracker(graph=self.sop)
        rendered = self._render_with_state(tracker)

        self.assertIn("You are TestBot", rendered)
        self.assertIn("a test AI agent", rendered)
        self.assertIn("Setup", rendered)
        self.assertIn("/set-target-path", rendered)
        self.assertNotIn("Investigation", rendered)

    def test_after_setup_shows_investigation(self):
        """After Phase 0 completes with outputs, guidance shifts to Phase 1."""
        tracker = StateGraphTracker(graph=self.sop)
        tracker.complete("0", target_path="/fbcode/myproject", strategy="default")

        rendered = self._render_with_state(tracker)

        self.assertIn("Investigation", rendered)
        self.assertIn("/fbcode/myproject", rendered)
        self.assertNotIn("Setup", rendered)

    def test_after_investigation_shows_proposal(self):
        """After Phase 1 completes, guidance shifts to Phase 2."""
        tracker = StateGraphTracker(graph=self.sop)
        tracker.complete("0", target_path="/fbcode", strategy="default")
        tracker.complete("1", findings="done")

        rendered = self._render_with_state(tracker)

        self.assertIn("Proposal", rendered)
        self.assertNotIn("Investigation", rendered)

    def test_goto_cycle_returns_to_proposal(self):
        """After Phase 3 with continue=True, goto re-enables Phase 2."""
        tracker = StateGraphTracker(graph=self.sop)
        tracker.complete("0", target_path="/fbcode", strategy="default")
        tracker.complete("1", findings="done")
        tracker.complete("2", proposal="plan A")
        tracker.complete("3", **{"continue": True})

        rendered = self._render_with_state(tracker)

        # Phase 2 should be available again (goto re-enabled it)
        self.assertIn("Proposal", rendered)

    def test_goto_no_cycle_when_false(self):
        """After Phase 3 with continue=False, no goto — all complete."""
        tracker = StateGraphTracker(graph=self.sop)
        tracker.complete("0", target_path="/fbcode", strategy="default")
        tracker.complete("1", findings="done")
        tracker.complete("2", proposal="plan A")
        tracker.complete("3", **{"continue": False})

        rendered = self._render_with_state(tracker)

        # Should show "All phases complete"
        self.assertIn("complete", rendered.lower())
        self.assertNotIn("Proposal", rendered)

    def test_missing_outputs_shows_incomplete(self):
        """If Phase 0 is in completed but outputs missing, shows incomplete."""
        tracker = StateGraphTracker(
            graph=self.sop,
            completed_states=["0"],
            state_outputs={},  # target_path and strategy NOT set
        )

        rendered = self._render_with_state(tracker)

        self.assertIn("incomplete", rendered)
        self.assertIn("`target_path`", rendered)

    def test_running_state_shows_in_progress(self):
        """While a phase is running, guidance shows 'in progress'."""
        tracker = StateGraphTracker(graph=self.sop)
        tracker.complete("0", target_path="/fbcode", strategy="default")
        tracker.start("1")

        rendered = self._render_with_state(tracker)

        self.assertIn("In progress", rendered)
        self.assertIn("Investigation", rendered)

    def test_error_state_shows_error(self):
        """When a phase fails, guidance shows error message."""
        tracker = StateGraphTracker(graph=self.sop)
        tracker.complete("0", target_path="/fbcode", strategy="default")
        tracker.start("1")
        tracker.fail("1", "connection timeout")

        rendered = self._render_with_state(tracker)

        self.assertIn("Error occurred", rendered)

    def test_employee_identity_from_variables(self):
        """Employee name/role from .variables.yaml renders correctly."""
        tracker = StateGraphTracker(graph=self.sop)
        rendered = self._render_with_state(tracker)

        self.assertIn("You are TestBot", rendered)
        self.assertIn("a test AI agent", rendered)

    def test_prompt_changes_across_full_lifecycle(self):
        """Walk through the full lifecycle and verify prompt changes at each step."""
        tracker = StateGraphTracker(graph=self.sop)

        # Step 1: Idle
        r1 = self._render_with_state(tracker)
        self.assertIn("Setup", r1)

        # Step 2: Setup complete
        tracker.complete("0", target_path="/fbcode", strategy="default")
        r2 = self._render_with_state(tracker)
        self.assertIn("Investigation", r2)
        self.assertNotIn("Setup", r2)

        # Step 3: Investigation complete
        tracker.complete("1", findings="architecture documented")
        r3 = self._render_with_state(tracker)
        self.assertIn("Proposal", r3)
        self.assertNotIn("Investigation", r3)

        # Step 4: Proposal complete
        tracker.complete("2", proposal="use flash attention")
        r4 = self._render_with_state(tracker)
        self.assertIn("Evaluate", r4)

        # Step 5: Evaluate with continue=True → back to Proposal
        tracker.complete("3", **{"continue": True})
        r5 = self._render_with_state(tracker)
        self.assertIn("Proposal", r5)  # goto re-enabled Phase 2

        # Step 6: Second cycle — clear continue flag, complete Phase 2 again
        # In a real system, the new Phase 3 run resets continue.
        # Here we clear it before the new cycle.
        tracker.state_outputs.pop("continue", None)
        # Remove "2" from completed so it can be re-completed
        if "2" in tracker.completed_states:
            tracker.completed_states.remove("2")
        tracker.complete("2", proposal="use HSTU v2")
        r6 = self._render_with_state(tracker)
        self.assertIn("Evaluate", r6)

        # Step 7: Evaluate with continue=False → done
        # Remove "3" so it can be re-completed
        if "3" in tracker.completed_states:
            tracker.completed_states.remove("3")
        tracker.complete("3", **{"continue": False})
        r7 = self._render_with_state(tracker)
        self.assertIn("complete", r7.lower())

        # Verify all 7 renders are distinct
        renders = [r1, r2, r3, r4, r5, r6, r7]
        # r3 and r5 may be similar (both show Proposal), r4 and r6 similar (both show Evaluate)
        # but r1, r2, r7 should be unique
        self.assertNotEqual(r1, r2)
        self.assertNotEqual(r2, r3)
        self.assertNotEqual(r1, r7)


if __name__ == "__main__":
    unittest.main()
