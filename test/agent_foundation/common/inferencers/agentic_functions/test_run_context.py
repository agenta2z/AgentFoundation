# pyre-strict

"""RunContext composition (A5): a child derived under an active context, a
per-call slot that keeps a loop from colliding, a caller-supplied context used
verbatim, and per-task ``.last_call`` isolation across concurrent asyncio tasks."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from agent_foundation.common.inferencers.agentic_functions import (
    agentic_function,
    escalate,
)
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    enter_run,
    exit_run,
    RunContext,
)

from ._helpers import FakeInferencer


class RunContextDerivationTest(unittest.TestCase):
    def test_child_derived_under_active_context(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(
            inferencer=lambda: fake,
            template_string="{{ x }}",
            run_context_slot="review",
        )
        def f(x: Any) -> str:
            return escalate()

        token = enter_run(RunContext.root(workspace=None))
        try:
            f("in")
        finally:
            exit_run(token)

        self.assertEqual(fake.run_contexts[0].path, "/review")
        self.assertIsNone(active_run_context())  # nothing leaked past teardown

    def test_loop_with_per_call_slot_does_not_collide(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(
            inferencer=lambda: fake,
            template_string="{{ i }}",
            run_context_slot=lambda i: f"q{i}",
        )
        def f(i: Any) -> str:
            return escalate()

        token = enter_run(RunContext.root(workspace=None))
        try:
            for i in range(3):
                f(i)
        finally:
            exit_run(token)

        self.assertEqual([rc.path for rc in fake.run_contexts], ["/q0", "/q1", "/q2"])


class CallerRunContextTest(unittest.TestCase):
    def test_caller_supplied_run_context_is_used_verbatim(self) -> None:
        fake = FakeInferencer("ok")
        supplied = RunContext.root(workspace=None).child("caller_supplied")

        @agentic_function(
            inferencer=lambda: fake,
            template_string="{{ x }}",
            infer_kwargs={"run_context": supplied},
        )
        def f(x: Any) -> str:
            return escalate()

        f("in")  # no active context entered — the caller's ctx short-circuits
        self.assertIs(fake.run_contexts[0], supplied)


class TraceIsolationTest(unittest.TestCase):
    def test_last_call_isolated_across_concurrent_tasks(self) -> None:
        # A per-call reply so each task's trace carries a distinct raw_text; if
        # last_call were a shared instance attr the sleep(0) interleave would let
        # the tasks clobber each other and a task would read the other's value.
        fake = FakeInferencer(lambda n: f"r{n}")

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        async def f(x: Any) -> str:
            return escalate()

        async def worker(x: Any) -> tuple[Any, Any]:
            r = await f(x)
            await asyncio.sleep(0)  # yield so the other task interleaves
            last = f.last_call
            assert last is not None
            return last.raw_text, r

        async def main() -> Any:
            return await asyncio.gather(worker("a"), worker("b"))

        for raw_text, r in asyncio.run(main()):
            self.assertEqual(raw_text, r)


if __name__ == "__main__":
    unittest.main()
