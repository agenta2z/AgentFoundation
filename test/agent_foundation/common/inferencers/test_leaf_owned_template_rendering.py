"""Phase 1 tests for leaf-owned template rendering refactor.

Covers:
1. extra_feed merges into _build_template_feed at correct precedence
2. extra_feed reserved-key guard raises ValueError on collision
3. _render_prompt accepts extra_feed kwarg
4. _ainfer_single / _infer_single conditional pass:
   - extra_feed=None → byte-identical to legacy form (no kwarg passed)
   - extra_feed={...} → kwarg forwarded to _render_prompt
5. ConversationalInferencer-style override (no extra_feed kwarg) does NOT
   crash with TypeError when extra_feed=None (Round-7 fix)
6. render_only short-circuits LLM call and returns rendered prompt
7. extra_feed and render_only are NOT leaked to _ainfer/_infer
8. FOLLOWUP_TEMPLATE_DEFAULTS constant exists and is exported
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import MagicMock

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.template_defaults import (
    FOLLOWUP_TEMPLATE_DEFAULTS,
    REVIEW_TEMPLATE_DEFAULTS,
)
from agent_foundation.common.inferencers.template_constants import KEY_FOLLOWUP
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)


# ─── Test fixtures ────────────────────────────────────────────────────────


class _RecordingTemplateManager:
    """Mock template manager that records what it was called with."""

    def __init__(self, return_value="RENDERED:{key}:{feed}"):
        self.return_value = return_value
        self.calls = []
        self.default_template = ""

    def __call__(self, key, *, active_template_root_space=None, **feed):
        self.calls.append({
            "key": key,
            "active_template_root_space": active_template_root_space,
            "feed": dict(feed),
        })
        return self.return_value.format(key=key, feed=feed)

    def get_raw_template(self, key, **kwargs):
        # Return non-empty so loud-failure check passes
        return f"<<{key}>>"

    def load_variables(self, **kwargs):
        return {}


class _StubLeaf(TemplatedInferencerBase):
    """Minimal templated leaf for testing. _infer just echoes the input."""

    def _infer(self, inference_input, inference_config=None, **kwargs):
        # Record what made it past the rendering layer
        self._last_infer_input = inference_input
        self._last_infer_kwargs = dict(kwargs)
        return f"INFER_RESPONSE:{inference_input}"


class _LegacyOverrideLeaf(TemplatedInferencerBase):
    """Mimics ConversationalInferencer: overrides _render_prompt WITHOUT
    declaring extra_feed kwarg. Must not crash when conditional-pass
    code path triggers extra_feed=None.
    """

    def _render_prompt(self, current_message: str) -> str:  # type: ignore[override]
        return f"LEGACY_RENDERED:{current_message}"

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return f"LEGACY_INFER:{inference_input}"


# ─── Tests ────────────────────────────────────────────────────────────────


class TestExtraFeedMerging(unittest.TestCase):
    """1. extra_feed merges with correct precedence."""

    def _make_leaf(self, **overrides):
        kwargs = {
            "template_manager": _RecordingTemplateManager(),
            "template_root_space": "plan",
            "template_key": "followup",
            "template_extra_feed": {"class_level": "from_class"},
        }
        kwargs.update(overrides)
        return _StubLeaf(**kwargs)

    def test_extra_feed_merges_into_feed(self):
        leaf = self._make_leaf()
        feed = leaf._build_template_feed("USER_INPUT", extra_feed={"per_call": "x"})
        self.assertEqual(feed["per_call"], "x")
        self.assertEqual(feed["class_level"], "from_class")
        self.assertEqual(feed["input"], "USER_INPUT")

    def test_extra_feed_overrides_class_extra_feed(self):
        leaf = self._make_leaf()
        feed = leaf._build_template_feed("X", extra_feed={"class_level": "overridden"})
        self.assertEqual(feed["class_level"], "overridden")

    def test_extra_feed_does_not_override_input_via_late_set(self):
        # `input` is set AFTER extra_feed merge — verifies sacrosanct precedence
        leaf = self._make_leaf()
        # Use a non-protected key here; we test the protected case in TestReservedKeyGuard
        feed = leaf._build_template_feed("REAL_INPUT", extra_feed={"another": "y"})
        self.assertEqual(feed["input"], "REAL_INPUT")

    def test_no_extra_feed_works_unchanged(self):
        leaf = self._make_leaf()
        feed = leaf._build_template_feed("X")
        self.assertEqual(feed["input"], "X")
        self.assertEqual(feed["class_level"], "from_class")
        self.assertNotIn("per_call", feed)


class TestReservedKeyGuard(unittest.TestCase):
    """2. extra_feed reserved-key guard."""

    def _make_leaf(self):
        return _StubLeaf(
            template_manager=_RecordingTemplateManager(),
            template_root_space="plan",
            template_key="followup",
        )

    def test_input_key_collision_raises(self):
        leaf = self._make_leaf()
        with self.assertRaises(ValueError) as ctx:
            leaf._build_template_feed("X", extra_feed={"input": "ATTACK"})
        self.assertIn("reserved key", str(ctx.exception))
        self.assertIn("input", str(ctx.exception))

    def test_template_space_key_collision_raises(self):
        leaf = self._make_leaf()
        with self.assertRaises(ValueError) as ctx:
            leaf._build_template_feed("X", extra_feed={"__template_space__": "EVIL"})
        self.assertIn("reserved key", str(ctx.exception))

    def test_non_reserved_keys_pass_through(self):
        leaf = self._make_leaf()
        # No exception
        feed = leaf._build_template_feed("X", extra_feed={"output_path": "/tmp/x"})
        self.assertEqual(feed["output_path"], "/tmp/x")


class TestRenderPromptKwarg(unittest.TestCase):
    """3. _render_prompt accepts extra_feed kwarg and forwards to feed."""

    def test_render_prompt_with_extra_feed(self):
        tm = _RecordingTemplateManager()
        leaf = _StubLeaf(
            template_manager=tm,
            template_root_space="plan",
            template_key="followup",
        )
        result = leaf._render_prompt("USER", extra_feed={"context_path": "/foo"})
        # The recording TM saw the merged feed
        self.assertEqual(len(tm.calls), 1)
        feed = tm.calls[0]["feed"]
        self.assertEqual(feed["context_path"], "/foo")
        self.assertEqual(feed["input"], "USER")

    def test_render_prompt_without_extra_feed(self):
        tm = _RecordingTemplateManager()
        leaf = _StubLeaf(
            template_manager=tm,
            template_root_space="plan",
            template_key="followup",
        )
        result = leaf._render_prompt("USER")
        feed = tm.calls[0]["feed"]
        self.assertEqual(feed["input"], "USER")
        self.assertNotIn("context_path", feed)


class TestConditionalKwargPass(unittest.TestCase):
    """4 & 5. Conditional pass — backward-compat with legacy overrides.

    The Round-7 fix: when extra_feed=None, _*_single MUST call
    _render_prompt(input) without the extra_feed kwarg, or legacy
    overrides like ConversationalInferencer crash.
    """

    def test_legacy_override_without_extra_feed_does_not_crash(self):
        """The critical Round-7 regression test. Sync path."""
        leaf = _LegacyOverrideLeaf(
            # No template_manager — _render_prompt is overridden anyway
        )
        # Default call (no extra_feed) — must not crash
        result = leaf._infer_single("HELLO")
        self.assertEqual(result, "LEGACY_INFER:LEGACY_RENDERED:HELLO")

    def test_legacy_override_async_without_extra_feed_does_not_crash(self):
        """Round-7 regression — async path."""
        leaf = _LegacyOverrideLeaf()
        result = asyncio.run(leaf._ainfer_single("HELLO"))
        self.assertEqual(result, "LEGACY_INFER:LEGACY_RENDERED:HELLO")

    def test_legacy_override_with_extra_feed_crashes_loudly(self):
        """Sanity: if a caller PASSES extra_feed={...} to a legacy leaf,
        we expect a TypeError (not a silent ignore). This documents the
        contract: legacy overrides only work for legacy callers."""
        leaf = _LegacyOverrideLeaf()
        with self.assertRaises(TypeError):
            leaf._infer_single("X", extra_feed={"some_key": "v"})


class TestExtraFeedFlowsThrough(unittest.TestCase):
    """4. extra_feed flows through _*_single → _render_prompt → feed."""

    def test_extra_feed_reaches_template_manager_via_infer_single(self):
        tm = _RecordingTemplateManager()
        leaf = _StubLeaf(
            template_manager=tm,
            template_root_space="plan",
            template_key="followup",
        )
        leaf._infer_single("USER", extra_feed={"prior_path": "/p"})
        self.assertEqual(tm.calls[0]["feed"]["prior_path"], "/p")

    def test_extra_feed_reaches_template_manager_via_ainfer_single(self):
        tm = _RecordingTemplateManager()
        leaf = _StubLeaf(
            template_manager=tm,
            template_root_space="plan",
            template_key="followup",
        )
        asyncio.run(leaf._ainfer_single("USER", extra_feed={"prior_path": "/p"}))
        self.assertEqual(tm.calls[0]["feed"]["prior_path"], "/p")


class TestRenderOnly(unittest.TestCase):
    """6. render_only short-circuits the LLM call."""

    def test_render_only_returns_rendered_and_skips_infer(self):
        tm = _RecordingTemplateManager(return_value="RENDERED_PROMPT")
        leaf = _StubLeaf(
            template_manager=tm,
            template_root_space="plan",
            template_key="followup",
        )
        result = leaf._infer_single("USER", render_only=True)
        self.assertEqual(result, "RENDERED_PROMPT")
        # _infer must NOT have been called
        self.assertFalse(hasattr(leaf, "_last_infer_input"))

    def test_render_only_async(self):
        tm = _RecordingTemplateManager(return_value="RENDERED_PROMPT")
        leaf = _StubLeaf(
            template_manager=tm,
            template_root_space="plan",
            template_key="followup",
        )
        result = asyncio.run(leaf._ainfer_single("USER", render_only=True))
        self.assertEqual(result, "RENDERED_PROMPT")
        self.assertFalse(hasattr(leaf, "_last_infer_input"))

    def test_render_only_with_extra_feed(self):
        """render_only + extra_feed together — pre-render with custom feed."""
        tm = _RecordingTemplateManager(return_value="RENDERED")
        leaf = _StubLeaf(
            template_manager=tm,
            template_root_space="plan",
            template_key="followup",
        )
        result = leaf._infer_single(
            "USER", extra_feed={"context": "C"}, render_only=True
        )
        self.assertEqual(tm.calls[0]["feed"]["context"], "C")
        self.assertEqual(result, "RENDERED")


class TestNoLeakageToInfer(unittest.TestCase):
    """7. extra_feed and render_only are NOT leaked to _infer/_ainfer."""

    def test_extra_feed_not_in_infer_kwargs(self):
        tm = _RecordingTemplateManager()
        leaf = _StubLeaf(
            template_manager=tm,
            template_root_space="plan",
            template_key="followup",
        )
        leaf._infer_single(
            "USER",
            extra_feed={"X": "Y"},
            unrelated_kwarg="passes_through",
        )
        # _infer DID receive unrelated_kwarg but NOT extra_feed
        self.assertIn("unrelated_kwarg", leaf._last_infer_kwargs)
        self.assertNotIn("extra_feed", leaf._last_infer_kwargs)
        self.assertNotIn("render_only", leaf._last_infer_kwargs)

    def test_render_only_not_in_infer_kwargs_when_false(self):
        tm = _RecordingTemplateManager()
        leaf = _StubLeaf(
            template_manager=tm,
            template_root_space="plan",
            template_key="followup",
        )
        leaf._infer_single("USER", render_only=False)
        self.assertNotIn("render_only", leaf._last_infer_kwargs)


class TestFollowupTemplateDefaultsConstant(unittest.TestCase):
    """8. FOLLOWUP_TEMPLATE_DEFAULTS exists and points to KEY_FOLLOWUP."""

    def test_constant_exists(self):
        self.assertIsNotNone(FOLLOWUP_TEMPLATE_DEFAULTS)

    def test_constant_template_key(self):
        # Apply to a stub dict and verify it cascades the right key
        node = {"_target_": "x"}
        FOLLOWUP_TEMPLATE_DEFAULTS.apply_to(node)
        self.assertEqual(node.get("template_key"), KEY_FOLLOWUP)

    def test_constant_does_not_overwrite_existing_template_key(self):
        # Per apply_to() contract: only fills if not already set
        node = {"_target_": "x", "template_key": "explicit"}
        FOLLOWUP_TEMPLATE_DEFAULTS.apply_to(node)
        self.assertEqual(node["template_key"], "explicit")


class TestWorkspaceFeedVariables(unittest.TestCase):
    """workspace_root and workspace_outputs are injected into feed when
    _workspace is set and inferencer has local file access."""

    def _make_leaf(self, has_local_access=True, workspace=None):
        leaf = _StubLeaf(
            template_manager=_RecordingTemplateManager(),
            template_root_space="implementation",
            template_key="initial",
        )
        if has_local_access:
            leaf.has_local_access = True
        if workspace is not None:
            leaf._workspace = workspace
        return leaf

    def _make_workspace(self, root="/tmp/test_ws"):
        ws = MagicMock()
        ws.root = root
        return ws

    def test_workspace_vars_present_when_workspace_set(self):
        leaf = self._make_leaf(workspace=self._make_workspace("/tmp/test_ws"))
        feed = leaf._build_template_feed("input")
        self.assertEqual(feed["workspace_root"], "/tmp/test_ws")
        self.assertEqual(feed["workspace_outputs"], "/tmp/test_ws/outputs")

    def test_workspace_vars_absent_when_no_workspace(self):
        leaf = self._make_leaf(workspace=None)
        feed = leaf._build_template_feed("input")
        self.assertNotIn("workspace_root", feed)
        self.assertNotIn("workspace_outputs", feed)

    def test_workspace_vars_absent_when_no_local_access(self):
        leaf = self._make_leaf(has_local_access=False, workspace=self._make_workspace())
        feed = leaf._build_template_feed("input")
        self.assertNotIn("workspace_root", feed)
        self.assertNotIn("workspace_outputs", feed)

    def test_workspace_outputs_is_always_outputs_subdir(self):
        leaf = self._make_leaf(workspace=self._make_workspace("/data/runs/run_001"))
        feed = leaf._build_template_feed("input")
        self.assertTrue(feed["workspace_outputs"].endswith("/outputs"))
        self.assertEqual(feed["workspace_outputs"], "/data/runs/run_001/outputs")


if __name__ == "__main__":
    unittest.main()
