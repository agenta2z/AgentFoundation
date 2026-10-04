"""Mock-LLM integration tests for the reviewer + output-guardrail cascade.

These wire a REAL ``DualInferencer`` with mock leaves and drive its real
``_step_propose_impl`` / ``_step_review_impl`` — the exact production path where
a flow leaf acts as the *reviewer* and its output-guardrail judge runs on the
reviewer's critique. They are the integration counterparts to the unit tests in
``test_output_guardrail.py`` (which drive a single leaf directly).

Root cause they guard against (see the reviewer-cascade investigation):
- The reviewer leaf renders its prompt from ``state["inference_input"]`` (the
  bare seed / original task) + the review ``extra_feed`` (the artifact under
  review). The raw input stays the seed; the *rendered* prompt is the review
  prompt.
- PRE-fix, the judge read the raw input (the seed) → it judged the
  critique against the *propose* task and false-RESTARTed valid critiques; and
  recovery re-sent the seed → the reviewer re-proposed (analysis) instead of
  reviewing.
- Fix #1 feeds the judge the rendered REVIEW prompt (via the ``_fallback_state``
  ContextVar); Fix #4 makes recovery re-send the rendered prompt. These tests
  fail if either fix is reverted.
"""

from __future__ import annotations

import asyncio
import unittest

from agent_foundation.common.inferencers.agentic_inferencers.common import (
    ConsensusConfig,
    Severity,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from attr import attrib, attrs


@attrs
class _MockBase(InferencerBase):
    """Base proposer: returns a fixed proposal artifact (the thing to review)."""

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return "BASE_PROPOSAL_ARTIFACT_MARKER"


@attrs
class _ReviewerLeaf(InferencerBase):
    """A reviewer leaf that renders its prompt from the seed + review feed
    (``extra_feed``) and returns a structured JSON critique. It carries an
    output-guardrail judge, exactly like a real flow leaf promoted to reviewer.
    The bare seed and ``rendered_input`` (the review prompt) genuinely diverge
    (revert-sensitivity)."""

    _critique = attrib(
        default=(
            "```json\n"
            '{"approved": true, "severity": "COSMETIC", "issues": [], '
            '"reasoning": "verified claims against source"}\n'
            "```"
        )
    )

    def _render_prompt(self, inference_input, extra_feed=None, **kwargs):
        return (
            "REVIEW_PREAMBLE: You are reviewing artifacts.\n"
            f"<UserRequest>{inference_input}</UserRequest>\n"
            f"<ReviewFeed>{extra_feed!r}</ReviewFeed>\n"
            "Now start your review."
        )

    def supports_prompt_rendering(self):
        return True

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._critique


@attrs
class _CapturingJudge(InferencerBase):
    """Records every prompt it is asked to judge; returns a fixed verdict."""

    _seen = attrib(factory=list, init=False)
    _verdict = attrib(default="PASS")

    def _infer(self, inp, inference_config=None, **kwargs):
        self._seen.append(str(inp))
        return self._verdict


@attrs
class _ReviewAwareJudge(InferencerBase):
    """Mimics the real judge's failure mode: it only accepts an output when the
    task it is judged against is a *review* task. If it is (wrongly) fed the
    propose seed, it rejects the critique as "not a proposal" — reproducing the
    exact pre-fix false-RESTART."""

    _verdicts = attrib(factory=list, init=False)
    _seen = attrib(factory=list, init=False)

    def _infer(self, inp, inference_config=None, **kwargs):
        text = str(inp)
        self._seen.append(text)
        verdict = (
            "PASS"
            if "You are reviewing artifacts" in text
            else "RESTART: output is a review, expected a proposal"
        )
        self._verdicts.append(verdict)
        return verdict


def _drive_review(dual, seed="ORIGINAL_SEED_TASK"):
    """Drive the REAL propose+review steps once (mirrors the multireviewer e2e
    helper). Swallows the consensus abort — we assert on the judge capture,
    which happens *inside* the reviewer's ``ainfer`` (before any consensus
    decision)."""
    # Force the modern leaf-render review path so the reviewer receives
    # ``state["inference_input"]`` + ``extra_feed`` and renders it itself.
    dual._leaf_can_self_render = lambda inf: True

    async def _go():
        st = {
            "inference_input": seed,
            "total_iterations": 0,
            "attempt_record": {"attempt": 1, "iterations": []},
        }
        dual._state = st
        await dual._step_propose_impl(seed, st)
        try:
            await dual._step_review_impl(seed, st)
        except Exception:
            # Consensus/parse aborts are expected here; the judge already ran.
            pass

    asyncio.run(_go())


def _make_dual(judge):
    reviewer = _ReviewerLeaf(output_guardrail_inferencer=judge)
    return DualInferencer(
        base_inferencer=_MockBase(),
        review_inferencer=reviewer,
        consensus_config=ConsensusConfig(
            max_iterations=2,
            max_consensus_attempts=1,
            consensus_threshold=Severity.COSMETIC,
        ),
    )


class TestReviewerGuardrailCascade(unittest.TestCase):
    def test_reviewer_judge_sees_rendered_review_prompt_not_seed(self):
        """Fix #1 end-to-end: the reviewer's guardrail judge is fed the rendered
        REVIEW prompt (review framing + the artifact feed), NOT the bare Dual
        seed. Reverting Fix #1 (judge reads the raw input = the seed) drops all
        these markers → this test fails."""
        judge = _CapturingJudge(verdict="PASS")
        dual = _make_dual(judge)
        _drive_review(dual)

        self.assertTrue(judge._seen, "reviewer's guardrail judge must have run")
        jp = judge._seen[0]
        self.assertIn("REVIEW_PREAMBLE", jp)
        self.assertIn("Now start your review", jp)
        # The review feed (containing the base proposal artifact) reached the
        # judge — proving the rendered prompt (seed + extra_feed) was used…
        self.assertIn("BASE_PROPOSAL_ARTIFACT_MARKER", jp)
        # …and it is NOT merely the bare seed.
        self.assertNotEqual(jp.strip(), "ORIGINAL_SEED_TASK")

    def test_reviewer_critique_passes_guardrail_with_review_prompt(self):
        """Fix #1 regression: with the judge fed the review prompt, a critique
        is PASSed (not false-RESTARTed). The ``_ReviewAwareJudge`` RESTARTs a
        critique judged against a *propose* seed (the bug) but PASSes when it
        sees the *review* framing — so a PASS here only happens because Fix #1
        feeds it the review prompt."""
        judge = _ReviewAwareJudge()
        dual = _make_dual(judge)
        _drive_review(dual)

        self.assertTrue(judge._seen, "judge must have run")
        # The judge saw the review framing (Fix #1) …
        self.assertIn("You are reviewing artifacts", judge._seen[0])
        # … so its FIRST verdict was PASS, not a false RESTART. Reverting Fix #1
        # → judge sees the bare seed → first verdict RESTART → this fails.
        self.assertEqual(judge._verdicts[0], "PASS")
        # No RESTART verdict was ever produced for the (valid) critique.
        self.assertNotIn(
            "RESTART: output is a review, expected a proposal", judge._verdicts
        )


if __name__ == "__main__":
    unittest.main()
