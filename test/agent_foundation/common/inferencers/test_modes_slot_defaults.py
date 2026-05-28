"""Tests for modes support in InferencerTemplateDefaults and template rendering.

Covers:
- modes dict merge in apply_to() (SLOT_DEFAULTS → YAML node)
- AGGREGATION_DEFAULTS has deep_mode=False, elegant_mode=True
- User YAML override wins over SLOT_DEFAULTS modes
- Template rendering: implementation (initial, review, followup), deep_research, plan review
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from jinja2 import Undefined as _JinjaUndefined

from agent_foundation.common.inferencers.template_defaults import (
    AGGREGATION_DEFAULTS,
    InferencerTemplateDefaults,
    InferencerTemplateVersionDefaults,
)


class _ChainableUndefined(_JinjaUndefined):
    """Allow chained attribute access on undefined variables."""

    def __getattr__(self, _name):
        return _ChainableUndefined()

    def __str__(self):
        return ""

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False


_AF_ROOT = Path(__file__).resolve().parents[4] / "src" / "agent_foundation"
_TEMPLATES_DIR = _AF_ROOT / "resources" / "prompt_templates"

_DEEP_MODE_CONTENT = "Spawn as many agents as possible"
_ELEGANT_MODE_CONTENT = "elegant, proper solution"


def _render(root_space: str, template_key: str, **extra_feed) -> str:
    template_dir = str(_TEMPLATES_DIR / root_space / "main")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        undefined=_ChainableUndefined,
    )
    tmpl = env.get_template(f"{template_key}.jinja2")
    feed = {
        "context": {"user_request_with_task_preamble": "Test request."},
        "user_request_with_task_preamble": "Test request.",
        "main_response": "Test proposal.",
        "prior_output_path": "",
        "round_index": 0,
        "counter_feedback": "",
        "task_instructions": "",
        "output_path": "/tmp/test.md",
        "reviewer_response": "",
    }
    feed.update(extra_feed)
    return tmpl.render(**feed)


# ---------------------------------------------------------------------------
# SLOT_DEFAULTS modes merge
# ---------------------------------------------------------------------------


class TestModesMergeInApplyTo(unittest.TestCase):

    def test_modes_merged_into_node(self):
        defaults = InferencerTemplateDefaults(
            modes={"deep_mode": False, "elegant_mode": True},
        )
        node = {}
        defaults.apply_to(node)
        self.assertEqual(node["modes"], {"deep_mode": False, "elegant_mode": True})

    def test_user_yaml_override_wins(self):
        defaults = InferencerTemplateDefaults(
            modes={"deep_mode": False, "elegant_mode": True},
        )
        node = {"modes": {"deep_mode": True}}
        defaults.apply_to(node)
        self.assertTrue(node["modes"]["deep_mode"])
        self.assertTrue(node["modes"]["elegant_mode"])

    def test_empty_modes_skips_merge(self):
        defaults = InferencerTemplateDefaults()
        node = {}
        defaults.apply_to(node)
        self.assertNotIn("modes", node)

    def test_version_defaults_passes_modes_through(self):
        defaults = InferencerTemplateVersionDefaults(
            template_version="aggregation",
            modes={"deep_mode": False},
        )
        node = {}
        defaults.apply_to(node)
        self.assertEqual(node["modes"], {"deep_mode": False})


class TestAggregationDefaultsModes(unittest.TestCase):

    def test_aggregation_defaults_has_modes(self):
        self.assertEqual(AGGREGATION_DEFAULTS.modes, {"deep_mode": False})

    def test_apply_to_sets_deep_mode_false(self):
        node = {}
        AGGREGATION_DEFAULTS.apply_to(node)
        self.assertFalse(node["modes"]["deep_mode"])


# ---------------------------------------------------------------------------
# Template rendering with modes
# ---------------------------------------------------------------------------


class TestImplementationInitialModes(unittest.TestCase):

    def test_deep_mode_rendered_when_enabled(self):
        rendered = _render(
            "implementation", "initial",
            enable_deep_mode=True,
            instructions={"modes": {"deep_mode": _DEEP_MODE_CONTENT}},
        )
        self.assertIn(_DEEP_MODE_CONTENT, rendered)

    def test_elegant_mode_rendered_when_enabled(self):
        rendered = _render(
            "implementation", "initial",
            enable_deep_mode=False,
            enable_elegant_mode=True,
            instructions={"modes": {"elegant_mode": _ELEGANT_MODE_CONTENT}},
        )
        self.assertIn(_ELEGANT_MODE_CONTENT, rendered)

    def test_modes_absent_when_disabled(self):
        rendered = _render(
            "implementation", "initial",
            enable_deep_mode=False,
            enable_elegant_mode=False,
        )
        self.assertNotIn(_DEEP_MODE_CONTENT, rendered)
        self.assertNotIn(_ELEGANT_MODE_CONTENT, rendered)


class TestImplementationReviewModes(unittest.TestCase):

    def test_deep_mode_rendered(self):
        rendered = _render(
            "implementation", "review",
            enable_deep_mode=True,
            instructions={"modes": {"deep_mode": _DEEP_MODE_CONTENT}},
        )
        self.assertIn(_DEEP_MODE_CONTENT, rendered)

    def test_no_elegant_mode_in_review(self):
        rendered = _render(
            "implementation", "review",
            enable_deep_mode=True,
            enable_elegant_mode=True,
            instructions={"modes": {
                "deep_mode": _DEEP_MODE_CONTENT,
                "elegant_mode": _ELEGANT_MODE_CONTENT,
            }},
        )
        self.assertIn(_DEEP_MODE_CONTENT, rendered)
        self.assertNotIn(_ELEGANT_MODE_CONTENT, rendered)


class TestImplementationFollowupModes(unittest.TestCase):

    def test_both_modes_rendered_when_enabled(self):
        rendered = _render(
            "implementation", "followup",
            enable_deep_mode=True,
            enable_elegant_mode=True,
            instructions={"modes": {
                "deep_mode": _DEEP_MODE_CONTENT,
                "elegant_mode": _ELEGANT_MODE_CONTENT,
            }},
        )
        self.assertIn(_DEEP_MODE_CONTENT, rendered)
        self.assertIn(_ELEGANT_MODE_CONTENT, rendered)


class TestDeepResearchModes(unittest.TestCase):

    def test_both_modes_rendered(self):
        rendered = _render(
            "deep_research", "initial",
            enable_deep_mode=True,
            enable_elegant_mode=True,
            instructions={"modes": {
                "deep_mode": _DEEP_MODE_CONTENT,
                "elegant_mode": _ELEGANT_MODE_CONTENT,
            }},
        )
        self.assertIn(_DEEP_MODE_CONTENT, rendered)
        self.assertIn(_ELEGANT_MODE_CONTENT, rendered)

    def test_modes_absent_when_disabled(self):
        rendered = _render(
            "deep_research", "initial",
            enable_deep_mode=False,
            enable_elegant_mode=False,
        )
        self.assertNotIn(_DEEP_MODE_CONTENT, rendered)
        self.assertNotIn(_ELEGANT_MODE_CONTENT, rendered)


class TestPlanReviewNoElegantMode(unittest.TestCase):

    def test_elegant_mode_removed_from_plan_review(self):
        rendered = _render(
            "plan", "review",
            enable_deep_mode=True,
            enable_elegant_mode=True,
            instructions={"modes": {
                "deep_mode": _DEEP_MODE_CONTENT,
                "elegant_mode": _ELEGANT_MODE_CONTENT,
            }},
        )
        self.assertIn(_DEEP_MODE_CONTENT, rendered)
        self.assertNotIn(_ELEGANT_MODE_CONTENT, rendered)

    def test_plan_initial_still_has_elegant(self):
        rendered = _render(
            "plan", "initial",
            enable_deep_mode=True,
            enable_elegant_mode=True,
            instructions={"modes": {
                "deep_mode": _DEEP_MODE_CONTENT,
                "elegant_mode": _ELEGANT_MODE_CONTENT,
            }},
        )
        self.assertIn(_DEEP_MODE_CONTENT, rendered)
        self.assertIn(_ELEGANT_MODE_CONTENT, rendered)

    def test_plan_followup_still_has_elegant(self):
        rendered = _render(
            "plan", "followup",
            enable_deep_mode=False,
            enable_elegant_mode=True,
            instructions={"modes": {"elegant_mode": _ELEGANT_MODE_CONTENT}},
        )
        self.assertIn(_ELEGANT_MODE_CONTENT, rendered)


if __name__ == "__main__":
    unittest.main()
