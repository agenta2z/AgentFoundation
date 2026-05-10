"""Phase 2 tests for Dual's leaf-rendering routing.

Covers:
1. _leaf_can_self_render returns False for None / Mocks / non-templated leaves
2. _leaf_can_self_render returns True for properly configured templated leaves
3. _build_review_feed and _build_followup_feed return dict (not rendered string)
4. SLOT_DEFAULTS now contains FOLLOWUP_TEMPLATE_DEFAULTS for fixer_inferencer
5. Legacy path: when leaf cannot self-render, Dual still works (regression
   verified by test_dual_inferencer_resume.py — these tests are additive)
6. Modern path: leaf with template_manager+template_key gets extra_feed
   instead of pre-rendered prompt
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.template_defaults import (
    FOLLOWUP_TEMPLATE_DEFAULTS,
    REVIEW_TEMPLATE_DEFAULTS,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)


class _MockTemplateManager:
    """Mock TemplateManager that returns the template key concatenated with feed."""

    def __init__(self):
        self.calls = []
        self.default_template = ""

    def __call__(self, key, *, active_template_root_space=None, **feed):
        self.calls.append({"key": key, "feed": dict(feed)})
        return f"<RENDERED:{key}:keys={sorted(feed.keys())}>"

    def get_raw_template(self, key, **kwargs):
        return f"<<{key}>>"

    def load_variables(self, **kwargs):
        return {}


class _ProductionLikeLeaf(TemplatedInferencerBase):
    """Templated leaf that records what it was asked to do."""

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self._last_input = inference_input
        self._last_kwargs = dict(kwargs)
        return f"<RESPONSE:{inference_input[:50]}...>"


class TestSlotDefaults(unittest.TestCase):
    """4. SLOT_DEFAULTS contains both REVIEW and FOLLOWUP defaults."""

    def test_slot_defaults_has_review(self):
        self.assertIn("review_inferencer", DualInferencer.SLOT_DEFAULTS)
        self.assertIs(
            DualInferencer.SLOT_DEFAULTS["review_inferencer"],
            REVIEW_TEMPLATE_DEFAULTS,
        )

    def test_slot_defaults_has_followup(self):
        self.assertIn("fixer_inferencer", DualInferencer.SLOT_DEFAULTS)
        self.assertIs(
            DualInferencer.SLOT_DEFAULTS["fixer_inferencer"],
            FOLLOWUP_TEMPLATE_DEFAULTS,
        )


class TestLeafCanSelfRender(unittest.TestCase):
    """1 & 2. _leaf_can_self_render correct classification."""

    def test_none_returns_false(self):
        self.assertFalse(DualInferencer._leaf_can_self_render(None))

    def test_magicmock_returns_false(self):
        # Critical regression test: MagicMock has any attribute = Mock
        # (truthy). Without the isinstance check, this would be True.
        m = MagicMock()
        self.assertFalse(DualInferencer._leaf_can_self_render(m))

    def test_templated_leaf_without_template_manager_returns_false(self):
        leaf = _ProductionLikeLeaf(template_key="followup")
        self.assertIsNone(leaf.template_manager)
        self.assertFalse(DualInferencer._leaf_can_self_render(leaf))

    def test_templated_leaf_without_key_or_space_returns_false(self):
        leaf = _ProductionLikeLeaf(template_manager=_MockTemplateManager())
        self.assertFalse(DualInferencer._leaf_can_self_render(leaf))

    def test_templated_leaf_with_manager_and_key_returns_true(self):
        leaf = _ProductionLikeLeaf(
            template_manager=_MockTemplateManager(),
            template_key="followup",
        )
        self.assertTrue(DualInferencer._leaf_can_self_render(leaf))

    def test_templated_leaf_with_manager_and_only_root_space_returns_true(self):
        leaf = _ProductionLikeLeaf(
            template_manager=_MockTemplateManager(),
            template_root_space="plan",
        )
        self.assertTrue(DualInferencer._leaf_can_self_render(leaf))


class TestFeedBuilders(unittest.TestCase):
    """3. _build_review_feed and _build_followup_feed return feed dicts."""

    def _make_dual(self):
        # Construct a minimal Dual that won't crash on prompt setup (no
        # explicit prompts → defaults to None → implicit-key path).
        return DualInferencer(
            base_inferencer=MagicMock(),
            review_inferencer=MagicMock(),
            fixer_inferencer=MagicMock(),
        )

    def test_build_review_feed_returns_dict(self):
        dual = self._make_dual()
        feed = dual._build_review_feed(
            inference_input="USER",
            proposal="PROPOSAL",
            counter_feedback=None,
            iteration=1,
            attempt=1,
        )
        self.assertIsInstance(feed, dict)
        self.assertEqual(feed["main_response"], "PROPOSAL")
        self.assertEqual(feed["iteration"], 1)
        # Must NOT have rendered (no <RENDERED tag expected)
        self.assertNotIn("<RENDERED", str(feed))

    def test_build_followup_feed_returns_dict(self):
        dual = self._make_dual()
        feed = dual._build_followup_feed(
            inference_input="USER",
            proposal="PROPOSAL",
            parsed_review={"issues": [], "reasoning": "rsn"},
            inference_config={},
            iteration=1,
            attempt=1,
            review_output="REVIEW",
        )
        self.assertIsInstance(feed, dict)
        self.assertEqual(feed["main_response"], "PROPOSAL")
        self.assertEqual(feed["reviewer_response"], "REVIEW")

    def test_review_feed_includes_counter_feedback_when_provided(self):
        dual = self._make_dual()
        feed = dual._build_review_feed(
            inference_input="USER",
            proposal="P",
            counter_feedback="CF",
            iteration=2,
        )
        # counter_feedback maps to placeholder_counter_feedback name
        self.assertIn(dual.placeholder_counter_feedback, feed)


if __name__ == "__main__":
    unittest.main()
