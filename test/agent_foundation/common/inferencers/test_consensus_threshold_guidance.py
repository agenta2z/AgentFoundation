"""Tests for ConsensusConfig threshold guidance methods and feed injection.

Verifies that approve_hint() and approval_guidance() produce correct,
concise strings that stay in sync with the consensus enforcement logic,
that _build_review_feed / _build_followup_feed inject them, and that
the actual review.jinja2 templates render them without Jinja2 leakage.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)


def _mock_inferencer():
    inf = MagicMock()
    inf.ainfer = AsyncMock(return_value="response")
    inf.aconnect = AsyncMock()
    inf.adisconnect = AsyncMock()
    return inf


# ---------------------------------------------------------------------------
# ConsensusConfig helper methods
# ---------------------------------------------------------------------------


class TestAcceptableAndBlockingSeverities(unittest.TestCase):

    def test_default_cosmetic_threshold(self):
        config = ConsensusConfig()
        self.assertEqual(config.acceptable_severities(), ("NONE", "COSMETIC"))
        self.assertEqual(
            config.blocking_severities(), ("MINOR", "MAJOR", "CRITICAL")
        )

    def test_minor_threshold(self):
        config = ConsensusConfig(consensus_threshold="MINOR")
        self.assertEqual(
            config.acceptable_severities(), ("NONE", "COSMETIC", "MINOR")
        )
        self.assertEqual(config.blocking_severities(), ("MAJOR", "CRITICAL"))

    def test_major_threshold(self):
        config = ConsensusConfig(consensus_threshold="MAJOR")
        self.assertEqual(
            config.acceptable_severities(),
            ("NONE", "COSMETIC", "MINOR", "MAJOR"),
        )
        self.assertEqual(config.blocking_severities(), ("CRITICAL",))

    def test_critical_threshold_nothing_blocks(self):
        config = ConsensusConfig(consensus_threshold="CRITICAL")
        self.assertEqual(
            config.acceptable_severities(),
            ("NONE", "COSMETIC", "MINOR", "MAJOR", "CRITICAL"),
        )
        self.assertEqual(config.blocking_severities(), ())

    def test_none_threshold(self):
        config = ConsensusConfig(consensus_threshold="NONE")
        self.assertEqual(config.acceptable_severities(), ("NONE",))
        self.assertEqual(
            config.blocking_severities(),
            ("COSMETIC", "MINOR", "MAJOR", "CRITICAL"),
        )

    def test_custom_severity_levels(self):
        config = ConsensusConfig(
            severity_levels=("LOW", "MEDIUM", "HIGH"),
            consensus_threshold="MEDIUM",
        )
        self.assertEqual(config.acceptable_severities(), ("LOW", "MEDIUM"))
        self.assertEqual(config.blocking_severities(), ("HIGH",))


class TestApproveHint(unittest.TestCase):
    """approve_hint picks whichever list is shorter for conciseness."""

    def test_cosmetic_uses_acceptable_list(self):
        config = ConsensusConfig(consensus_threshold="COSMETIC")
        self.assertEqual(
            config.approve_hint(),
            "true if only observing NONE/COSMETIC issues",
        )

    def test_minor_uses_blocking_list(self):
        config = ConsensusConfig(consensus_threshold="MINOR")
        self.assertEqual(
            config.approve_hint(), "true if no MAJOR/CRITICAL issues"
        )

    def test_major_uses_blocking_list(self):
        config = ConsensusConfig(consensus_threshold="MAJOR")
        self.assertEqual(config.approve_hint(), "true if no CRITICAL issues")

    def test_critical_all_acceptable(self):
        config = ConsensusConfig(consensus_threshold="CRITICAL")
        self.assertEqual(
            config.approve_hint(),
            "true (all severity levels are acceptable)",
        )

    def test_none_threshold_tie_uses_acceptable(self):
        config = ConsensusConfig(consensus_threshold="NONE")
        self.assertEqual(
            config.approve_hint(), "true if only observing NONE issues"
        )

    def test_custom_levels_blocking_shorter(self):
        config = ConsensusConfig(
            severity_levels=("LOW", "MEDIUM", "HIGH"),
            consensus_threshold="MEDIUM",
        )
        self.assertEqual(config.approve_hint(), "true if no HIGH issues")


class TestApprovalGuidance(unittest.TestCase):

    def test_cosmetic_threshold(self):
        guidance = ConsensusConfig().approval_guidance()
        self.assertIn("at most COSMETIC severity", guidance)
        self.assertIn("NONE or COSMETIC", guidance)
        self.assertIn("MINOR, MAJOR, CRITICAL", guidance)
        self.assertIn("must NOT approve", guidance)

    def test_minor_threshold(self):
        guidance = ConsensusConfig(
            consensus_threshold="MINOR"
        ).approval_guidance()
        self.assertIn("at most MINOR severity", guidance)
        self.assertIn("NONE or COSMETIC or MINOR", guidance)
        self.assertIn("MAJOR, CRITICAL", guidance)

    def test_critical_threshold_no_restrictions(self):
        guidance = ConsensusConfig(
            consensus_threshold="CRITICAL"
        ).approval_guidance()
        self.assertIn("no severity restrictions", guidance)


# ---------------------------------------------------------------------------
# Feed dict injection
# ---------------------------------------------------------------------------


class TestReviewFeedContainsThresholdStrings(unittest.TestCase):

    def test_default_threshold(self):
        dual = DualInferencer(
            base_inferencer=_mock_inferencer(),
            review_inferencer=_mock_inferencer(),
            consensus_config=ConsensusConfig(consensus_threshold="COSMETIC"),
        )
        feed = dual._build_review_feed(
            "input", "proposal", None, inference_config={}
        )
        self.assertEqual(feed["consensus_threshold"], "COSMETIC")
        self.assertIn("NONE/COSMETIC", feed["approve_hint"])
        self.assertIn("at most COSMETIC", feed["approval_guidance"])

    def test_custom_threshold_via_inference_config(self):
        dual = DualInferencer(
            base_inferencer=_mock_inferencer(),
            review_inferencer=_mock_inferencer(),
        )
        custom = ConsensusConfig(consensus_threshold="MINOR")
        feed = dual._build_review_feed(
            "input",
            "proposal",
            None,
            inference_config={"consensus_config": custom},
        )
        self.assertEqual(feed["consensus_threshold"], "MINOR")
        self.assertIn("MAJOR/CRITICAL", feed["approve_hint"])

    def test_no_inference_config_uses_instance_default(self):
        dual = DualInferencer(
            base_inferencer=_mock_inferencer(),
            review_inferencer=_mock_inferencer(),
            consensus_config=ConsensusConfig(consensus_threshold="MAJOR"),
        )
        feed = dual._build_review_feed("input", "proposal", None)
        self.assertEqual(feed["consensus_threshold"], "MAJOR")
        self.assertIn("CRITICAL", feed["approve_hint"])


class TestFollowupFeedContainsThresholdStrings(unittest.TestCase):

    def test_followup_feed_has_threshold_keys(self):
        dual = DualInferencer(
            base_inferencer=_mock_inferencer(),
            review_inferencer=_mock_inferencer(),
            consensus_config=ConsensusConfig(consensus_threshold="COSMETIC"),
        )
        feed = dual._build_followup_feed(
            "input", "proposal", {"issues": []}, {}
        )
        self.assertEqual(feed["consensus_threshold"], "COSMETIC")
        self.assertIn("approve_hint", feed)
        self.assertIn("approval_guidance", feed)


# ---------------------------------------------------------------------------
# Template rendering — verify {{ approve_hint }} and {{ approval_guidance }}
# render correctly in the actual review.jinja2 files.
# ---------------------------------------------------------------------------

_AF_ROOT = Path(__file__).resolve().parents[4] / "src" / "agent_foundation"
_TEMPLATES_DIR = _AF_ROOT / "resources" / "prompt_templates"


from jinja2 import Undefined as _JinjaUndefined


class _ChainableUndefined(_JinjaUndefined):
    """Jinja2 Undefined subclass that allows chained attribute access.

    ``{{ instructions.behavior.x }}`` renders as "" instead of raising
    UndefinedError, matching how TemplateManager resolves _variables/
    subdirectory templates in production.
    """

    def __getattr__(self, _name):
        return _ChainableUndefined()

    def __str__(self):
        return ""

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False


def _render_review_template(root_space: str, feed: dict) -> str:
    """Render a review.jinja2 template with Jinja2 directly.

    Uses a ChainableUndefined so nested template variables like
    ``{{ instructions.behavior.file_reading_efficiency }}`` (normally
    resolved from _variables/ subdirectories by TemplateManager) render
    as empty strings instead of raising UndefinedError.
    """
    from jinja2 import Environment, FileSystemLoader

    template_dir = str(_TEMPLATES_DIR / root_space / "main")
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(template_dir)

    env = Environment(
        loader=FileSystemLoader(template_dir),
        undefined=_ChainableUndefined,
    )
    tmpl = env.get_template("review.jinja2")
    return tmpl.render(**feed)


def _minimal_review_feed(root_space: str, **overrides) -> dict:
    """Build the minimum feed dict needed to render a review template."""
    config = overrides.pop(
        "_config", ConsensusConfig(consensus_threshold="COSMETIC")
    )
    base = {
        "main_response": "The implementation report placeholder.",
        "prior_output_path": "",
        "round_index": 0,
        "iteration": 1,
        "attempt": 1,
        "counter_feedback": "",
        "task_instructions": "",
        "output_path": "/tmp/test_output.md",
        # Consensus threshold guidance (injected by _build_review_feed)
        "consensus_threshold": str(config.consensus_threshold),
        "approve_hint": config.approve_hint(),
        "approval_guidance": config.approval_guidance(),
    }
    if root_space == "implementation":
        base["user_request_with_task_preamble"] = "Test user request."
    elif root_space == "plan":
        base["context"] = {
            "user_request_with_task_preamble": "Test user request.",
        }
    elif root_space == "analysis":
        base["context"] = {
            "user_request_with_task_preamble": "Test user request.",
        }
    base.update(overrides)
    return base


class TestReviewTemplateRendersThresholdGuidance(unittest.TestCase):
    """Render actual review.jinja2 templates and verify threshold text appears."""

    def _assert_rendered_contains_threshold(self, root_space, config=None):
        if config is None:
            config = ConsensusConfig()
        feed = _minimal_review_feed(root_space, _config=config)
        rendered = _render_review_template(root_space, feed)

        self.assertIn(
            config.approve_hint(),
            rendered,
            f"{root_space}/review.jinja2 should contain approve_hint",
        )
        self.assertNotIn(
            "{{ approve_hint }}",
            rendered,
            f"Raw Jinja2 {{ approve_hint }} leaked in {root_space}/review.jinja2",
        )
        self.assertNotIn(
            "{{ approval_guidance }}",
            rendered,
            f"Raw Jinja2 {{ approval_guidance }} leaked in {root_space}/review.jinja2",
        )
        return rendered

    def test_implementation_review_default_threshold(self):
        rendered = self._assert_rendered_contains_threshold("implementation")
        self.assertIn("NONE/COSMETIC", rendered)
        self.assertIn("at most COSMETIC", rendered)

    def test_plan_review_default_threshold(self):
        rendered = self._assert_rendered_contains_threshold("plan")
        self.assertIn("NONE/COSMETIC", rendered)
        self.assertIn("at most COSMETIC", rendered)

    def test_analysis_review_default_threshold(self):
        rendered = self._assert_rendered_contains_threshold("analysis")
        self.assertIn("NONE/COSMETIC", rendered)

    def test_implementation_review_minor_threshold(self):
        config = ConsensusConfig(consensus_threshold="MINOR")
        rendered = self._assert_rendered_contains_threshold(
            "implementation", config
        )
        self.assertIn("MAJOR/CRITICAL", rendered)
        self.assertIn("at most MINOR", rendered)

    def test_plan_review_minor_threshold(self):
        config = ConsensusConfig(consensus_threshold="MINOR")
        rendered = self._assert_rendered_contains_threshold("plan", config)
        self.assertIn("MAJOR/CRITICAL", rendered)

    def test_no_hardcoded_approve_contradiction(self):
        """The old contradictory hint 'no CRITICAL/MAJOR issues' is gone."""
        for space in ("implementation", "plan", "analysis"):
            feed = _minimal_review_feed(space)
            try:
                rendered = _render_review_template(space, feed)
            except FileNotFoundError:
                continue
            self.assertNotIn(
                "true if no CRITICAL/MAJOR issues",
                rendered,
                f"{space}/review.jinja2 still has the old hardcoded hint",
            )


class TestLegacyRenderPathThresholdInjection(unittest.TestCase):
    """Verify _build_review_prompt (legacy path) renders threshold text."""

    def test_legacy_review_prompt_contains_threshold(self):
        dual = DualInferencer(
            base_inferencer=_mock_inferencer(),
            review_inferencer=_mock_inferencer(),
            review_prompt=(
                "Review this: {{ main_response }}. "
                "Approve hint: {{ approve_hint }}. "
                "Guidance: {{ approval_guidance }}"
            ),
            consensus_config=ConsensusConfig(consensus_threshold="COSMETIC"),
        )
        rendered = dual._build_review_prompt(
            inference_input="test request",
            proposal="test proposal",
            counter_feedback=None,
            inference_config={},
        )
        self.assertIn("NONE/COSMETIC", rendered)
        self.assertIn("at most COSMETIC", rendered)
        self.assertNotIn("{{ approve_hint }}", rendered)

    def test_legacy_review_prompt_with_custom_threshold(self):
        config = ConsensusConfig(consensus_threshold="MAJOR")
        dual = DualInferencer(
            base_inferencer=_mock_inferencer(),
            review_inferencer=_mock_inferencer(),
            review_prompt="Approve: {{ approve_hint }}",
            consensus_config=config,
        )
        rendered = dual._build_review_prompt(
            inference_input="test",
            proposal="proposal",
            counter_feedback=None,
            inference_config={},
        )
        self.assertIn("true if no CRITICAL issues", rendered)


# ---------------------------------------------------------------------------
# End-to-end: exercise _step_review_impl call path (catches NameError
# regressions where inference_config is not in scope).
# ---------------------------------------------------------------------------


def _make_approved_review_json():
    """Return a review JSON string that approves immediately."""
    import json
    return "```json\n" + json.dumps({
        "approve": True,
        "overall_severity": "NONE",
        "issues": [],
        "reasoning": "Looks good.",
    }) + "\n```"


def _make_mock_inferencer_for_e2e(response):
    """Mock inferencer that works with the full ainfer() path."""
    inf = MagicMock()
    inf.ainfer = AsyncMock(return_value=response)
    inf.infer = MagicMock(return_value=response)
    inf.aconnect = AsyncMock()
    inf.adisconnect = AsyncMock()
    inf.supports_prompt_rendering = False
    inf._workspace = None
    inf.id = "mock"
    return inf


class TestEndToEndReviewPathNoNameError(unittest.TestCase):
    """Exercise the full ainfer() → _step_review_impl → _build_review_feed
    path to catch NameError regressions (e.g., referencing 'inference_config'
    when it's not in _step_review_impl's scope)."""

    def _run(self, coro):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_ainfer_completes_without_name_error(self):
        review_json = _make_approved_review_json()
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer_for_e2e("proposal text"),
            review_inferencer=_make_mock_inferencer_for_e2e(review_json),
            consensus_config=ConsensusConfig(consensus_threshold="COSMETIC"),
        )
        try:
            result = self._run(dual.ainfer("test request"))
        except NameError as e:
            self.fail(
                f"NameError in review path — likely inference_config "
                f"not in scope in _step_review_impl: {e}"
            )

    def test_ainfer_with_custom_threshold(self):
        review_json = _make_approved_review_json()
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer_for_e2e("proposal"),
            review_inferencer=_make_mock_inferencer_for_e2e(review_json),
            consensus_config=ConsensusConfig(consensus_threshold="MINOR"),
        )
        try:
            result = self._run(dual.ainfer("test"))
        except NameError as e:
            self.fail(f"NameError with custom threshold: {e}")


if __name__ == "__main__":
    unittest.main()
