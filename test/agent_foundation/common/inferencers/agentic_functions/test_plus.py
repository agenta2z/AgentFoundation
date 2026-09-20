# pyre-strict

"""Canonical end-to-end: the code-agentic hybrid ``plus``.

Exercises the deterministic fast path (no inference runs), the escalation path
(inference runs when the body cannot compute deterministically), and the
``ParseError`` -> repair-retry -> ``fallback`` path.
"""

from __future__ import annotations

import unittest
from typing import Any

from agent_foundation.common.inferencers.agentic_functions import (
    agentic_function,
    escalate,
)

from ._helpers import FakeInferencer


def _make_plus(fake: FakeInferencer, **extra: Any) -> Any:
    @agentic_function(
        inferencer=lambda: fake,
        template_string="Add {{ x }} and {{ y }}; reply with the number only.",
        **extra,
    )
    def plus(x: Any, y: Any) -> float:
        try:
            return float(x) + float(y)
        except (TypeError, ValueError):
            return escalate()

    return plus


class PlusTest(unittest.TestCase):
    def test_deterministic_fast_path_runs_no_inference(self) -> None:
        fake = FakeInferencer("7")
        plus = _make_plus(fake)
        self.assertEqual(plus(3, 4), 7.0)
        self.assertEqual(fake.calls, 0)
        self.assertEqual(plus.last_call.path, "precheck")

    def test_escalates_to_inference_when_body_cannot_compute(self) -> None:
        fake = FakeInferencer("7")
        plus = _make_plus(fake)
        self.assertEqual(plus("three", "four"), 7.0)
        self.assertEqual(fake.calls, 1)
        self.assertEqual(plus.last_call.path, "agentic")

    def test_parse_error_triggers_repair_retry_then_fallback(self) -> None:
        fake = FakeInferencer("banana")
        plus = _make_plus(fake, parse_max_retries=1, fallback=-1.0)
        self.assertEqual(plus("a", "b"), -1.0)
        self.assertEqual(fake.calls, 2)  # parse_max_retries=1 -> exactly 2 attempts
        self.assertIn("[Retry]", fake.prompts[1])  # repair note in the 2nd prompt
        self.assertEqual(plus.last_call.attempts, 2)
        self.assertEqual(plus.last_call.stages_used, ("fallback",))

    def test_ellipsis_body_always_escalates(self) -> None:
        fake = FakeInferencer("42")

        @agentic_function(
            inferencer=lambda: fake,
            template_string="Add {{ x }} and {{ y }}.",
        )
        def plus2(x: Any, y: Any) -> float: ...

        self.assertEqual(plus2(1, 2), 42.0)
        self.assertEqual(fake.calls, 1)


if __name__ == "__main__":
    unittest.main()
