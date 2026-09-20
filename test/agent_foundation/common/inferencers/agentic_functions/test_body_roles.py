# pyre-strict

"""The signature-driven body-role model (A2/A4).

Covers the three roles (pre-attempt / post-parser / mixed), every escalation
signal, the tri-state ``escalate_on_none`` decoration-time guard, and the
``Agentic`` carrier's feed/parser overrides.
"""

from __future__ import annotations

import unittest
from typing import Any, Optional

from agent_foundation.common.inferencers.agentic_functions import (
    Agentic,
    agentic_function,
    AgenticFunctionConfigurationError,
    escalate,
    ESCALATE,
    FromInference,
)

from ._helpers import FakeInferencer


class FastPathTest(unittest.TestCase):
    def test_pre_attempt_real_value_runs_no_inference(self) -> None:
        fake = FakeInferencer("SHOULD-NOT-BE-USED")

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        def f(x: Any) -> str:
            return "deterministic"

        self.assertEqual(f("in"), "deterministic")
        self.assertEqual(fake.calls, 0)
        self.assertEqual(f.last_call.path, "precheck")

    def test_falsey_non_none_returns_never_escalate(self) -> None:
        # A helper per case gives each closure its own scope (the loop variable
        # would otherwise be captured by reference — flake8 B023).
        def _check(value: Any) -> None:
            fake = FakeInferencer("X")

            @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
            def f(x: Any) -> Any:
                return value

            self.assertEqual(f("in"), value)
            self.assertEqual(fake.calls, 0, f"escalated on falsey {value!r}")

        for value in (0, "", [], False):
            _check(value)


class EscalationSignalTest(unittest.TestCase):
    def test_escalate_via_ESCALATE(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        def f(x: Any) -> Any:
            return ESCALATE

        self.assertEqual(f("in"), "ok")
        self.assertEqual(fake.calls, 1)

    def test_escalate_via_none_with_default_escalate_on_none(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        def f(x: Any) -> str:
            return None  # -> str does not admit None, so this escalates

        self.assertEqual(f("in"), "ok")
        self.assertEqual(fake.calls, 1)

    def test_escalate_via_not_implemented(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        def f(x: Any) -> str:
            raise NotImplementedError

        self.assertEqual(f("in"), "ok")
        self.assertEqual(fake.calls, 1)

    def test_non_signal_exception_propagates_without_inference(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        def f(x: Any) -> str:
            raise ValueError("a real bug must not become a silent LLM call")

        with self.assertRaises(ValueError):
            f("in")
        self.assertEqual(fake.calls, 0)


class EscalateOnNoneTriStateTest(unittest.TestCase):
    def test_optional_return_without_flag_raises_at_decoration(self) -> None:
        fake = FakeInferencer("ok")

        def _define() -> Any:
            @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
            def f(x: Any) -> Optional[int]:
                return None

            return f

        with self.assertRaises(AgenticFunctionConfigurationError):
            _define()

    def test_optional_return_with_explicit_false_returns_none(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(
            inferencer=lambda: fake,
            template_string="{{ x }}",
            escalate_on_none=False,
        )
        def f(x: Any) -> Optional[int]:
            return None

        self.assertIsNone(f("in"))
        self.assertEqual(fake.calls, 0)


class AgenticCarrierTest(unittest.TestCase):
    def test_feed_overrides_a_param_variable(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        def f(x: Any) -> Any:
            return Agentic(feed={"x": "OVERRIDE"})

        f("original")
        self.assertEqual(fake.prompts[0], "OVERRIDE")

    def test_parser_override_replaces_stage1(self) -> None:
        fake = FakeInferencer("raw-ignored")

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        def f(x: Any) -> Any:
            return escalate(parser=lambda out: "PARSED")

        self.assertEqual(f("in"), "PARSED")


class MixedBodyTest(unittest.TestCase):
    def _make(self, fake: FakeInferencer) -> Any:
        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        def f(x: Any, *, response: Any = FromInference) -> str:
            if response is FromInference:
                return escalate() if x == "hard" else f"det:{x}"
            return f"parsed:{response.text}"

        return f

    def test_mixed_fast_path(self) -> None:
        fake = FakeInferencer("ok")
        f = self._make(fake)
        self.assertEqual(f("easy"), "det:easy")
        self.assertEqual(fake.calls, 0)

    def test_mixed_escalates_then_parses_pass2(self) -> None:
        fake = FakeInferencer("ok")
        f = self._make(fake)
        self.assertEqual(f("hard"), "parsed:ok")
        self.assertEqual(fake.calls, 1)


if __name__ == "__main__":
    unittest.main()
