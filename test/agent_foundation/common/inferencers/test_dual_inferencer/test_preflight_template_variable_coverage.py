"""Pre-flight test: verify all template variables in plan/main/review.jinja2 and
plan/main/followup.jinja2 are populated when Dual invokes a leaf via the
Phase 2 modern path (extra_feed + leaf._build_template_feed).

This test exists to resolve the confusion from the misleading ❌ table in the
pre-flight report. The ❌ marks meant "NOT in Dual's feed dict" — but that is
CORRECT because those variables come from the LEAF's own _build_template_feed
machinery (template_variables, _inject_mode_flags_and_content, output_path).

Specifically this test verifies:
  1. Variables populated by Dual's extra_feed:
       main_response, prior_output_path, reviewer_response, round_index,
       input, counter_feedback (review only)
  2. Variables populated by the leaf's _build_template_feed (NOT Dual):
       output_path            → from resolve_output_path()
       instructions.modes.*  → from _inject_mode_flags_and_content()
       employee.*             → from template_variables via load_variables()
       task_preamble          → from template_variables via load_variables()
       task_instructions      → from template_variables via load_variables()
  3. Reserved-key protection (input, __template_space__ cannot be in extra_feed)
  4. Dual's extra_feed merges AFTER leaf's template_variables but BEFORE
     output_path/input (sacrosanct last)
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock


class _RecordingTemplateManager:
    """Records the final merged feed dict passed to Jinja rendering."""

    def __init__(self):
        self.last_feed: Dict[str, Any] = {}
        self.last_key: Optional[str] = None
        self.default_template = ""

    def __call__(self, key, *, active_template_root_space=None, **feed):
        self.last_key = key
        self.last_feed = dict(feed)
        # Return a placeholder that includes all key names for inspection
        return f"RENDERED:{key}:keys={sorted(feed.keys())}"

    def get_raw_template(self, key, **kwargs):
        return f"<<{key}>>"

    def load_variables(self, variable_specs=None, **kwargs):
        """Simulate variable loading — returns the spec values as content."""
        if not variable_specs:
            return {}
        result = {}
        for k, v in variable_specs.items():
            if v:
                result[k] = f"LOADED_VAR:{v}"
        return result

    def _cascade_load_variable(self, path, name, root_space, tmpl_type):
        """Simulate mode instruction loading."""
        return f"MODE_CONTENT:{name}"


class _FullFeaturedLeaf:
    """Simulates a RovoDevCliInferencer-like leaf with full template machinery.

    Manually implements the key parts of TemplatedInferencerBase._build_template_feed
    to simulate what happens in production — without requiring the full import chain.
    """

    def __init__(
        self,
        template_manager,
        template_key="followup",
        template_root_space="plan",
        template_variables=None,
        template_extra_feed=None,
        modes=None,
        output_path_value="/task/fixer/outputs/output.md",
    ):
        self.template_manager = template_manager
        self.template_key = template_key
        self.template_root_space = template_root_space
        self.template_variables = template_variables or {"task_preamble": "general"}
        self.template_extra_feed = template_extra_feed or {
            "employee": {"name": "Alice", "role": "Engineer"},
        }
        self.modes = modes or {"deep_mode": True, "elegant_mode": True}
        self._output_path_value = output_path_value

    def resolve_output_path(self):
        return self._output_path_value

    def _build_template_feed(self, inference_input, *, extra_feed=None):
        """Faithful simulation of TemplatedInferencerBase._build_template_feed.

        Priority (lowest to highest):
        1. template_variables via load_variables
        2. template_extra_feed
        3. extra_feed (per-call, from Dual)
        4. __template_space__ (sacrosanct)
        5. mode flags (via _inject_mode_flags_and_content)
        6. output_path (sacrosanct)
        7. input = inference_input (most sacrosanct)
        """
        # Guard: reserved keys in extra_feed
        if extra_feed:
            PROTECTED = {"input", "__template_space__"}
            collisions = PROTECTED & extra_feed.keys()
            if collisions:
                raise ValueError(
                    f"extra_feed contains reserved key(s): {sorted(collisions)}"
                )

        feed: Dict[str, Any] = {}

        # Step 1: load template_variables
        if self.template_variables and self.template_manager:
            resolved = self.template_manager.load_variables(
                variable_specs=self.template_variables
            )
            feed.update(resolved)

        # Step 2: template_extra_feed (class-level overrides)
        feed.update(self.template_extra_feed)

        # Step 3: per-call extra_feed (from Dual) — wins over class-level
        if extra_feed:
            feed.update(extra_feed)

        # Step 4: __template_space__ (sacrosanct)
        if self.template_root_space:
            feed["__template_space__"] = self.template_root_space

        # Step 5: mode flags
        self._inject_mode_flags(feed)

        # Step 6: output_path (sacrosanct)
        op = self.resolve_output_path()
        if op:
            feed["output_path"] = op

        # Step 7: input (most sacrosanct — set last)
        feed["input"] = inference_input

        return feed

    def _inject_mode_flags(self, feed):
        """Simulate _inject_mode_flags_and_content."""
        for mode_name, enabled in self.modes.items():
            feed[f"enable_{mode_name}"] = bool(enabled)
            if enabled and self.template_manager:
                content = self.template_manager._cascade_load_variable(
                    "instructions/modes", mode_name, self.template_root_space, "main"
                )
                instructions = feed.setdefault("instructions", {})
                if isinstance(instructions, dict):
                    modes_dict = instructions.setdefault("modes", {})
                    modes_dict[mode_name] = content

    def _render_prompt(self, inference_input, *, extra_feed=None):
        """Simulate TemplatedInferencerBase._render_prompt."""
        if self.template_manager is None:
            return inference_input
        feed = self._build_template_feed(inference_input, extra_feed=extra_feed)
        return self.template_manager(
            self.template_key,
            active_template_root_space=self.template_root_space,
            **feed,
        )


class TestTemplateVariableCoverageReview(unittest.TestCase):
    """Verify plan/main/review.jinja2 variables are all populated.

    Variables in review.jinja2:
      {{ employee.name }} / {{ employee.role }}  — from template_extra_feed
      {{ task_preamble }}                         — from template_variables
      {{ task_instructions }}                     — from template_variables (optional)
      {{ instructions.modes.deep_mode }}          — from _inject_mode_flags
      {{ instructions.modes.elegant_mode }}       — from _inject_mode_flags
      {{ output_path }}                           — from resolve_output_path()
      {{ input }}                                 — from inference_input
      {{ main_response }}                         — from Dual's extra_feed
      {{ prior_output_path }}                     — from Dual's extra_feed
      {{ round_index }}                           — from Dual's extra_feed
      {{ counter_feedback }}                      — from Dual's extra_feed (optional)
    """

    def setUp(self):
        self.tm = _RecordingTemplateManager()
        self.leaf = _FullFeaturedLeaf(
            template_manager=self.tm,
            template_key="review",
            template_root_space="plan",
            template_variables={"task_preamble": "general", "task_instructions": ""},
            template_extra_feed={
                "employee": {"name": "Alice", "role": "Software Engineer"},
            },
        )
        # Simulate what Dual._build_review_feed puts in extra_feed
        # (after stripping "input" and "__template_space__")
        self.dual_extra_feed = {
            "proposal": "BASE_PROPOSAL_TEXT",
            "main_response": "BASE_PROPOSAL_TEXT",
            "prior_output_path": "/task/base/outputs/final_deliverables/output.md",
            "round_index": 0,
            "iteration": 1,
            "attempt": 1,
        }

    def test_all_required_review_variables_populated(self):
        """The combined leaf+Dual feed must cover all review template variables."""
        self.leaf._render_prompt("USER_REQUEST", extra_feed=self.dual_extra_feed)
        feed = self.tm.last_feed

        # From Dual's extra_feed:
        self.assertIn("main_response", feed, "main_response missing from feed")
        self.assertIn("prior_output_path", feed, "prior_output_path missing")
        self.assertIn("round_index", feed, "round_index missing")
        self.assertIn("input", feed, "input missing")
        self.assertEqual(feed["input"], "USER_REQUEST")

        # From leaf's template_extra_feed:
        self.assertIn("employee", feed, "employee missing from feed")
        self.assertIsInstance(feed["employee"], dict)
        self.assertEqual(feed["employee"]["name"], "Alice")

        # From leaf's template_variables via load_variables():
        self.assertIn("task_preamble", feed, "task_preamble missing from feed")

        # From leaf's _inject_mode_flags:
        self.assertIn("instructions", feed, "instructions missing (mode flags)")
        self.assertIn("modes", feed["instructions"])
        self.assertIn("deep_mode", feed["instructions"]["modes"])
        self.assertIn("elegant_mode", feed["instructions"]["modes"])
        self.assertIn("enable_deep_mode", feed)
        self.assertIn("enable_elegant_mode", feed)

        # From resolve_output_path():
        self.assertIn("output_path", feed, "output_path missing")
        self.assertEqual(feed["output_path"], "/task/fixer/outputs/output.md")

    def test_dual_extra_feed_wins_over_class_template_extra_feed(self):
        """If Dual sends a key that also exists in template_extra_feed, Dual wins."""
        leaf = _FullFeaturedLeaf(
            template_manager=self.tm,
            template_extra_feed={"proposal": "CLASS_LEVEL_PROPOSAL"},
        )
        leaf._render_prompt("X", extra_feed={"proposal": "DUAL_PROPOSAL"})
        self.assertEqual(self.tm.last_feed["proposal"], "DUAL_PROPOSAL")

    def test_input_is_sacrosanct_always_last(self):
        """{{ input }} must always equal the inference_input, regardless of extra_feed."""
        # extra_feed cannot clobber input (reserved-key guard)
        with self.assertRaises(ValueError):
            self.leaf._render_prompt("REAL_INPUT", extra_feed={"input": "ATTACK"})

        # Without the collision, input is always the inference_input
        self.leaf._render_prompt("REAL_INPUT", extra_feed=self.dual_extra_feed)
        self.assertEqual(self.tm.last_feed["input"], "REAL_INPUT")

    def test_output_path_is_sacrosanct(self):
        """output_path comes from resolve_output_path, not from extra_feed."""
        self.leaf._render_prompt("X", extra_feed=self.dual_extra_feed)
        self.assertEqual(
            self.tm.last_feed["output_path"],
            "/task/fixer/outputs/output.md",
        )


class TestTemplateVariableCoverageFollowup(unittest.TestCase):
    """Verify plan/main/followup.jinja2 variables are all populated.

    Variables in followup.jinja2:
      {{ employee.name }} / {{ employee.role }}  — from template_extra_feed
      {{ task_preamble }}                         — from template_variables
      {{ task_instructions }}                     — from template_variables
      {{ instructions.modes.elegant_mode }}       — from _inject_mode_flags
      {{ output_path }}                           — from resolve_output_path()
      {{ input }}                                 — from inference_input
      {{ main_response }}                         — from Dual's extra_feed
      {{ prior_output_path }}                     — from Dual's extra_feed
      {{ reviewer_response }}                     — from Dual's extra_feed
      {{ round_index }}                           — from Dual's extra_feed
    """

    def setUp(self):
        self.tm = _RecordingTemplateManager()
        self.leaf = _FullFeaturedLeaf(
            template_manager=self.tm,
            template_key="followup",
            template_root_space="plan",
            template_variables={"task_preamble": "general"},
            template_extra_feed={
                "employee": {"name": "Alice", "role": "Software Engineer"},
            },
        )
        # Simulate what Dual._build_followup_feed puts in extra_feed
        self.dual_extra_feed = {
            "proposal": "BASE_PROPOSAL_TEXT",
            "main_response": "BASE_PROPOSAL_TEXT",
            "prior_output_path": "/task/base/outputs/final_deliverables/output.md",
            "reviewer_response": '{"issues": [], "verdict": "APPROVE"}',
            "round_index": 1,
            "iteration": 1,
            "attempt": 1,
            "enable_counter_feedback": False,
            "issues": "[]",
            "reasoning": "Looks good",
        }

    def test_all_required_followup_variables_populated(self):
        """The combined leaf+Dual feed must cover all followup template variables."""
        self.leaf._render_prompt("USER_REQUEST", extra_feed=self.dual_extra_feed)
        feed = self.tm.last_feed

        # From Dual's extra_feed:
        self.assertIn("main_response", feed)
        self.assertIn("prior_output_path", feed)
        self.assertIn("reviewer_response", feed)
        self.assertIn("round_index", feed)
        self.assertIn("input", feed)
        self.assertEqual(feed["input"], "USER_REQUEST")

        # From leaf's template_extra_feed:
        self.assertIn("employee", feed)
        self.assertIsInstance(feed["employee"], dict)

        # From leaf's template_variables:
        self.assertIn("task_preamble", feed)

        # From _inject_mode_flags (elegant_mode is specifically in followup.jinja2):
        self.assertIn("instructions", feed)
        self.assertIn("elegant_mode", feed["instructions"]["modes"])
        self.assertIn("enable_elegant_mode", feed)

        # From resolve_output_path:
        self.assertIn("output_path", feed)

    def test_prior_output_path_is_non_empty_when_base_has_deliverable(self):
        """prior_output_path from Dual's feed is the base inferencer's deliverable."""
        self.leaf._render_prompt("X", extra_feed=self.dual_extra_feed)
        self.assertEqual(
            self.tm.last_feed["prior_output_path"],
            "/task/base/outputs/final_deliverables/output.md",
        )

    def test_reviewer_response_populated(self):
        """reviewer_response from Dual's feed is passed to the fixer template."""
        self.leaf._render_prompt("X", extra_feed=self.dual_extra_feed)
        self.assertIn("reviewer_response", self.tm.last_feed)
        self.assertNotEqual(self.tm.last_feed["reviewer_response"], "None")

    def test_empty_prior_output_path_when_no_deliverable(self):
        """When base has no deliverable, prior_output_path is empty string (not None)."""
        feed_no_path = dict(self.dual_extra_feed)
        feed_no_path["prior_output_path"] = ""  # Dual sends "" not None
        self.leaf._render_prompt("X", extra_feed=feed_no_path)
        # Template's {% if prior_output_path %} guard will suppress the cp instruction
        self.assertEqual(self.tm.last_feed["prior_output_path"], "")


class TestMergePriority(unittest.TestCase):
    """Verify the exact merge priority: template_variables < template_extra_feed < extra_feed < sacrosanct."""

    def setUp(self):
        self.tm = _RecordingTemplateManager()

    def test_full_priority_chain(self):
        """All four layers stack correctly."""
        leaf = _FullFeaturedLeaf(
            template_manager=self.tm,
            template_variables={"shared_key": "from_template_variables"},
            template_extra_feed={
                "shared_key": "from_class_extra_feed",   # wins over template_variables
                "class_only": "class_level_value",
            },
        )
        extra = {
            "shared_key": "from_dual_extra_feed",        # wins over class extra_feed
            "dual_only": "dual_level_value",
        }
        leaf._render_prompt("MY_INPUT", extra_feed=extra)
        feed = self.tm.last_feed

        # extra_feed wins over class template_extra_feed
        self.assertEqual(feed["shared_key"], "from_dual_extra_feed")
        # class_only preserved
        self.assertEqual(feed["class_only"], "class_level_value")
        # dual_only present
        self.assertEqual(feed["dual_only"], "dual_level_value")
        # sacrosanct: input always wins last
        self.assertEqual(feed["input"], "MY_INPUT")
        # sacrosanct: output_path always set from resolve_output_path
        self.assertIn("output_path", feed)


if __name__ == "__main__":
    unittest.main()
