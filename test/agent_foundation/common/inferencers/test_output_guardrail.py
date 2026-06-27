"""Unit tests for the output guardrail (LLM-based quality judge).

Covers:
- output_guardrail_inferencer=None (default): zero behavioral change
- Guardrail judge returns "PASS": output accepted normally
- Guardrail judge returns "FAIL: reason": output rejected → recovery fires
- _parse_guardrail_verdict overridable by subclass
- Guardrail fail-open: judge exception → output accepted (no crash)
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase


@attrs
class _StubInferencer(InferencerBase):
    """A minimal leaf inferencer for testing."""

    _responses = attrib(factory=list, init=False)
    _call_count = attrib(default=0, init=False)

    def set_responses(self, responses):
        self._responses = list(responses)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._responses):
            r = self._responses[idx]
            if isinstance(r, Exception):
                raise r
            return r
        return f"response_{idx}"


@attrs
class _StubJudge(InferencerBase):
    """A stub guardrail judge that returns configurable verdicts."""

    _verdict = attrib(default="PASS")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._verdict


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestGuardrailDisabledByDefault(unittest.TestCase):
    def test_no_guardrail_accepts_normally(self):
        inf = _StubInferencer()
        inf.set_responses(["good output"])
        result = inf.infer("input")
        self.assertEqual(str(result), "good output")

    def test_no_guardrail_async(self):
        inf = _StubInferencer()
        inf.set_responses(["good output"])
        result = _run(inf.ainfer("input"))
        self.assertEqual(str(result), "good output")


class TestGuardrailAccepts(unittest.TestCase):
    def test_pass_verdict_accepts(self):
        judge = _StubJudge(verdict="PASS")
        inf = _StubInferencer(output_guardrail_inferencer=judge)
        inf.set_responses(["good output"])
        result = inf.infer("input")
        self.assertEqual(str(result), "good output")

    def test_pass_verdict_async(self):
        judge = _StubJudge(verdict="PASS")
        inf = _StubInferencer(output_guardrail_inferencer=judge)
        inf.set_responses(["good output"])
        result = _run(inf.ainfer("input"))
        self.assertEqual(str(result), "good output")


class TestGuardrailRejects(unittest.TestCase):
    def test_fail_verdict_triggers_retry(self):
        """When the judge says FAIL, the inferencer retries via recovery."""
        judge = _StubJudge(verdict="FAIL: output is just narration")
        inf = _StubInferencer(
            output_guardrail_inferencer=judge,
            max_retry=2,
        )
        inf.set_responses(["bad narration", "also bad", "still bad"])
        # All attempts rejected → should exhaust retries and raise/return default
        # With default_return_or_raise=None and all attempts failing validation,
        # execute_with_retry raises the last exception.
        with self.assertRaises(Exception):
            inf.infer("input")
        # The inferencer was called multiple times (retried)
        self.assertGreater(inf._call_count, 1)

    def test_fail_then_pass_accepts_second(self):
        """First attempt rejected, second attempt passes the judge."""
        call_count = [0]

        @attrs
        class _FlipJudge(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                call_count[0] += 1
                return "FAIL: too short" if call_count[0] == 1 else "PASS"

        judge = _FlipJudge()
        inf = _StubInferencer(
            output_guardrail_inferencer=judge,
            max_retry=3,
        )
        inf.set_responses(["bad narration", "good complete plan"])
        result = inf.infer("input")
        self.assertEqual(str(result), "good complete plan")
        self.assertEqual(inf._call_count, 2)


class TestGuardrailFailOpen(unittest.TestCase):
    def test_judge_exception_accepts_output(self):
        """If the judge itself crashes, fail-open: accept the output."""

        @attrs
        class _CrashingJudge(InferencerBase):
            def _infer(self, inp, inference_config=None, **kw):
                raise RuntimeError("judge crashed")

        judge = _CrashingJudge()
        inf = _StubInferencer(output_guardrail_inferencer=judge)
        inf.set_responses(["some output"])
        result = inf.infer("input")
        self.assertEqual(str(result), "some output")


class TestParseGuardrailVerdict(unittest.TestCase):
    def test_pass_variants(self):
        inf = _StubInferencer()
        self.assertTrue(inf._parse_guardrail_verdict("PASS"))
        self.assertTrue(inf._parse_guardrail_verdict("pass"))
        self.assertTrue(inf._parse_guardrail_verdict("PASS - looks good"))

    def test_handler_routing(self):
        """Verdict returns handler name strings for different failure modes."""
        inf = _StubInferencer()
        self.assertEqual(inf._parse_guardrail_verdict("RESTART: completely wrong"), "restart")
        self.assertEqual(inf._parse_guardrail_verdict("RETRY_WITH_REFERENCE: incomplete"), "retry_with_reference")
        self.assertEqual(inf._parse_guardrail_verdict("CONTINUE: truncated"), "continue")
        self.assertEqual(inf._parse_guardrail_verdict("FAIL: generic"), "retry_with_reference")

    def test_overridable(self):
        """Subclass can override the verdict parser."""

        @attrs
        class _CustomVerdict(_StubInferencer):
            def _parse_guardrail_verdict(self, judge_response):
                return "approved" if "APPROVED" in str(judge_response).upper() else "restart"

        inf = _CustomVerdict()
        self.assertEqual(inf._parse_guardrail_verdict("APPROVED: ok"), "approved")
        self.assertEqual(inf._parse_guardrail_verdict("REJECTED: bad"), "restart")


class TestJudgePromptRendering(unittest.TestCase):
    def test_render_guardrail_prompt_contains_input_and_output(self):
        inf = _StubInferencer()
        inf._last_inference_input = "original task request"
        prompt = inf._render_guardrail_prompt("the agent's output text")
        self.assertIn("original task request", prompt)
        self.assertIn("the agent's output text", prompt)
        self.assertIn("PASS", prompt)
        self.assertIn("RESTART", prompt)
        self.assertIn("RETRY_WITH_REFERENCE", prompt)
        self.assertIn("CONTINUE", prompt)


if __name__ == "__main__":
    unittest.main()
