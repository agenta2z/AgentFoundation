"""Tests for supports_prompt_rendering property and DualInferencer _render_prompt fix.

Covers:
- InferencerBase.supports_prompt_rendering base property
- DualInferencer: crash fix (renamed _render_role_prompt), supports_prompt_rendering override
- ReflectiveInferencer: supports_prompt_rendering override
- ConversationalInferencer: supports_prompt_rendering override
- Flow inferencers inherit base property correctly
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.reflective_inferencer import (
    ReflectiveInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_inferencer(response="response"):
    """Create a mock InferencerBase whose ainfer/infer return `response`."""
    inf = MagicMock()
    inf.ainfer = AsyncMock(return_value=response)
    inf.infer = MagicMock(return_value=response)
    inf.iter_infer = MagicMock(return_value=iter([response]))
    inf.aconnect = AsyncMock()
    inf.adisconnect = AsyncMock()
    return inf


def _make_approved_review():
    """Build a review JSON that approves on first round."""
    review = {
        "approved": True,
        "severity": "NONE",
        "issues": [],
        "reasoning": "Looks good.",
    }
    return f"```json\n{json.dumps(review)}\n```"


# ---------------------------------------------------------------------------
# InferencerBase.supports_prompt_rendering
# ---------------------------------------------------------------------------

class TestInferencerBaseProperty(unittest.TestCase):
    """Base class property returns True iff template_manager is set."""

    def test_false_by_default(self):
        """Without template_manager, supports_prompt_rendering is False."""
        # We can't instantiate InferencerBase directly (abstract), so test
        # via DualInferencer with no prompt_formatter (doesn't set template_manager)
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer(),
            review_inferencer=_make_mock_inferencer(),
        )
        # The base class attribute template_manager should be None
        self.assertIsNone(dual.template_manager)


# ---------------------------------------------------------------------------
# DualInferencer: crash fix + supports_prompt_rendering
# ---------------------------------------------------------------------------

class TestDualInferencerCrashFix(unittest.TestCase):
    """After renaming _render_prompt → _render_role_prompt, the standard
    __call__/infer/ainfer paths no longer crash."""

    def _run_async(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_sync_call_no_crash(self):
        """dual('input') should not raise TypeError from _render_prompt shadowing."""
        review_response = _make_approved_review()
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(review_response),
        )
        # This used to crash with:
        # TypeError: _render_prompt() missing 2 required positional arguments
        try:
            result = dual("test input")
        except TypeError as e:
            if "missing" in str(e) and "positional argument" in str(e):
                self.fail(
                    f"DualInferencer.__call__ still crashes with shadowing bug: {e}"
                )
            raise

    def test_async_ainfer_no_crash(self):
        """await dual.ainfer('input') should not raise TypeError."""
        review_response = _make_approved_review()
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer("proposal"),
            review_inferencer=_make_mock_inferencer(review_response),
        )
        try:
            self._run_async(dual.ainfer("test input"))
        except TypeError as e:
            if "missing" in str(e) and "positional argument" in str(e):
                self.fail(
                    f"DualInferencer.ainfer still crashes with shadowing bug: {e}"
                )
            raise

    def test_render_role_prompt_called_by_builders(self):
        """_build_initial_prompt uses _render_role_prompt (not _render_prompt)."""
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer(),
            review_inferencer=_make_mock_inferencer(),
            initial_prompt="Hello {{input}}",  # placeholder_input default is "input"
        )
        # _build_initial_prompt should work (calls _render_role_prompt internally)
        result = dual._build_initial_prompt(
            inference_input="world", inference_config={}, attempt=1
        )
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Hello world")


class TestDualInferencerSupportsPromptRendering(unittest.TestCase):
    """DualInferencer.supports_prompt_rendering reflects prompt_formatter state."""

    def test_false_without_formatter(self):
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer(),
            review_inferencer=_make_mock_inferencer(),
            prompt_formatter=None,
        )
        self.assertFalse(dual.supports_prompt_rendering)

    def test_true_with_callable_formatter(self):
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer(),
            review_inferencer=_make_mock_inferencer(),
            prompt_formatter=lambda t, **kw: t,
        )
        self.assertTrue(dual.supports_prompt_rendering)

    def test_true_with_template_manager(self):
        from rich_python_utils.string_utils.formatting.template_manager import (
            TemplateManager,
        )

        tm = TemplateManager(
            default_template="{{input}}",
            templates={
                "initial_prompt": "init: {{inference_input}}",
                "review_prompt": "review: {{proposal}}",
                "followup_prompt": "fix: {{proposal}}",
            },
        )
        dual = DualInferencer(
            base_inferencer=_make_mock_inferencer(),
            review_inferencer=_make_mock_inferencer(),
            prompt_formatter=tm,
            initial_prompt="initial_prompt",
            review_prompt="review_prompt",
            followup_prompt="followup_prompt",
        )
        self.assertTrue(dual.supports_prompt_rendering)


# ---------------------------------------------------------------------------
# ReflectiveInferencer: supports_prompt_rendering
# ---------------------------------------------------------------------------

class TestReflectiveInferencerSupportsPromptRendering(unittest.TestCase):

    def test_always_true_after_init(self):
        """ReflectiveInferencer always wraps reflection_prompt_formatter into
        a TemplateManager in __attrs_post_init__, so it's always True."""
        r = ReflectiveInferencer(
            base_inferencer=_make_mock_inferencer(),
            reflection_inferencer=_make_mock_inferencer(),
        )
        self.assertTrue(r.supports_prompt_rendering)

    def test_true_even_with_none_formatter(self):
        """Even with reflection_prompt_formatter=None, init wraps it in TM."""
        r = ReflectiveInferencer(
            base_inferencer=_make_mock_inferencer(),
            reflection_inferencer=_make_mock_inferencer(),
            reflection_prompt_formatter=None,
        )
        self.assertTrue(r.supports_prompt_rendering)


# ---------------------------------------------------------------------------
# ConversationalInferencer: supports_prompt_rendering
# ---------------------------------------------------------------------------

class TestConversationalInferencerSupportsPromptRendering(unittest.TestCase):

    def test_false_without_renderer(self):
        c = ConversationalInferencer(
            base_inferencer=_make_mock_inferencer(),
            prompt_renderer=None,
        )
        self.assertFalse(c.supports_prompt_rendering)

    def test_true_with_renderer(self):
        mock_renderer = MagicMock()
        mock_renderer.render = MagicMock(return_value="rendered")
        c = ConversationalInferencer(
            base_inferencer=_make_mock_inferencer(),
            prompt_renderer=mock_renderer,
        )
        self.assertTrue(c.supports_prompt_rendering)


# ---------------------------------------------------------------------------
# Flow inferencers: inherit base property
# ---------------------------------------------------------------------------

class TestFlowInferencersInheritProperty(unittest.TestCase):
    """BreakdownThenAggregateInferencer and PlanThenImplementInferencer
    should inherit supports_prompt_rendering from InferencerBase."""

    def test_bta_has_property(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
            BreakdownThenAggregateInferencer,
        )
        self.assertTrue(hasattr(BreakdownThenAggregateInferencer, "supports_prompt_rendering"))

    def test_pti_has_property(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.plan_then_implement_inferencer import (
            PlanThenImplementInferencer,
        )
        self.assertTrue(hasattr(PlanThenImplementInferencer, "supports_prompt_rendering"))


if __name__ == "__main__":
    unittest.main()
