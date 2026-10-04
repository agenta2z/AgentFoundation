# pyre-strict

"""RunContext composition (A5): a child derived under an active context, a
per-call slot that keeps a loop from colliding, a caller-supplied context used as
the parent of the call's own child, the session reset scoped to the call's
context, per-task ``.last_call`` isolation across concurrent asyncio tasks, and a
per-call copy of an uncertified inferencer under a host context."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any, ClassVar

from agent_foundation.common.inferencers.agentic_functions import (
    agentic_function,
    escalate,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    enter_run,
    exit_run,
    RunContext,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from attr import attrs

from ._helpers import FakeInferencer, SessionFake


class _ScopedSessionFake(SessionFake):
    """Records the run-context path active when ``reset_session()`` runs."""

    def __init__(self, response: Any = "ok") -> None:
        super().__init__(response)
        self.reset_paths: list[Any] = []

    def reset_session(self) -> None:
        super().reset_session()
        ctx = active_run_context()
        self.reset_paths.append(None if ctx is None else ctx.path)


@attrs
class _SessionStream(StreamingInferencerBase):
    """A real session-bearing leaf: records the session it would resume, then
    opens a new one, so a reset that misses the call's slot shows up as resume.
    Certified, so an agentic function shares the one instance under a host."""

    _HOST_PURE_CERTIFIED: ClassVar[bool] = True

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self.seen_sessions: list[Any] = []

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kw: Any
    ) -> str:
        self.seen_sessions.append(self.active_session_id)
        self.active_session_id = f"s{len(self.seen_sessions)}"
        return "ok"

    async def _ainfer_streaming(self, prompt: Any, **kwargs: Any) -> Any:
        yield "ok"


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
    """A caller-supplied context is the parent: the call runs at its own child, so
    it never shares the caller's path while the caller's own call is live."""

    def _decorate(self, fake: Any, supplied: RunContext, is_async: bool) -> Any:
        opts: dict[str, Any] = {
            "inferencer": lambda: fake,
            "template_string": "{{ x }}",
            "infer_kwargs": {"run_context": supplied},
        }
        if is_async:

            @agentic_function(**opts)
            async def af(x: Any) -> str:
                return escalate()

            return af

        @agentic_function(**opts)
        def f(x: Any) -> str:
            return escalate()

        return f

    def test_caller_supplied_run_context_is_the_parent(self) -> None:
        fake = FakeInferencer("ok")
        supplied = RunContext.root(workspace=None).child("caller_supplied")
        self._decorate(fake, supplied, is_async=False)("in")
        self.assertEqual(fake.run_contexts[0].path, "/caller_supplied/f")

    def test_caller_supplied_run_context_is_the_parent_async(self) -> None:
        fake = FakeInferencer("ok")
        supplied = RunContext.root(workspace=None).child("caller_supplied")
        asyncio.run(self._decorate(fake, supplied, is_async=True)("in"))
        self.assertEqual(fake.run_contexts[0].path, "/caller_supplied/af")

    def test_caller_supplied_run_context_wins_over_the_active_one(self) -> None:
        fake = FakeInferencer("ok")
        supplied = RunContext.root(workspace=None).child("caller_supplied")
        f = self._decorate(fake, supplied, is_async=False)
        token = enter_run(RunContext.root(workspace=None).child("active"))
        try:
            f("in")
        finally:
            exit_run(token)
        self.assertEqual(fake.run_contexts[0].path, "/caller_supplied/f")


class SessionResetScopeTest(unittest.TestCase):
    """The pre-call session reset runs under the call's own context, so it clears
    the slot the call resumes from, never the caller's."""

    def test_reset_runs_under_the_call_context(self) -> None:
        fake = _ScopedSessionFake("ok")

        @agentic_function(
            inferencer=lambda: fake,
            template_string="{{ x }}",
            run_context_slot="judge",
        )
        def f(x: Any) -> str:
            return escalate()

        token = enter_run(RunContext.root(workspace=None).child("caller"))
        try:
            f("in")
            self.assertEqual(active_run_context().path, "/caller")
        finally:
            exit_run(token)
        self.assertEqual(fake.reset_paths, ["/caller/judge"])

    def test_reset_runs_under_the_call_context_async(self) -> None:
        fake = _ScopedSessionFake("ok")

        @agentic_function(
            inferencer=lambda: fake,
            template_string="{{ x }}",
            run_context_slot="judge",
        )
        async def f(x: Any) -> str:
            return escalate()

        async def main() -> None:
            token = enter_run(RunContext.root(workspace=None).child("caller"))
            try:
                await f("in")
                self.assertEqual(active_run_context().path, "/caller")
            finally:
                exit_run(token)

        asyncio.run(main())
        self.assertEqual(fake.reset_paths, ["/caller/judge"])

    def test_reset_without_any_context_binds_none(self) -> None:
        fake = _ScopedSessionFake("ok")

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        def f(x: Any) -> str:
            return escalate()

        f("in")
        self.assertEqual(fake.reset_paths, [None])
        self.assertIsNone(active_run_context())

    def test_reused_leaf_never_resumes_the_previous_call_under_a_host(self) -> None:
        leaf = _SessionStream()

        @agentic_function(
            inferencer=lambda: leaf,
            template_string="{{ x }}",
            run_context_slot="judge",
        )
        def f(x: Any) -> str:
            return escalate()

        token = enter_run(RunContext.root(workspace=None))
        try:
            f("a")
            f("b")
        finally:
            exit_run(token)
        self.assertEqual(leaf.seen_sessions, [None, None])


@attrs
class _UncertifiedLeaf(InferencerBase):
    """Records which instance served each call; yields once mid-call, so two
    gathered calls overlap."""

    ran_on: ClassVar[list[Any]] = []

    def _infer(
        self, inference_input: Any, inference_config: Any = None, **kw: Any
    ) -> str:
        _UncertifiedLeaf.ran_on.append(self)
        return "ok"

    async def _ainfer(
        self, inference_input: Any, inference_config: Any = None, **kw: Any
    ) -> str:
        _UncertifiedLeaf.ran_on.append(self)
        await asyncio.sleep(0.01)
        return "ok"


class UncertifiedInferencerTest(unittest.TestCase):
    """An inferencer not certified host-pure runs one host invocation at a time,
    so under a host context each call runs on its own copy of it."""

    def setUp(self) -> None:
        _UncertifiedLeaf.ran_on.clear()
        self.leaf = _UncertifiedLeaf()

        @agentic_function(
            inferencer=lambda: self.leaf,
            template_string="{{ x }}",
            run_context_slot="judge",
        )
        async def judge(x: Any) -> str:
            return escalate()

        self.judge = judge

    def test_overlapping_host_calls_each_run_on_their_own_copy(self) -> None:
        """Like fan-out workers, each judging its task at its own path."""
        root = RunContext.root(workspace=None)

        async def worker(name: str) -> Any:
            token = enter_run(root.child(name))
            try:
                return await self.judge(name)
            finally:
                exit_run(token)

        async def main() -> list[Any]:
            return await asyncio.gather(worker("w0"), worker("w1"))

        self.assertEqual(asyncio.run(main()), ["ok", "ok"])
        first, second = _UncertifiedLeaf.ran_on
        self.assertIsNot(first, second)
        self.assertNotIn(self.leaf, _UncertifiedLeaf.ran_on)

    def test_a_call_without_a_host_shares_the_inferencer(self) -> None:
        asyncio.run(self.judge("a"))
        token = enter_run(RunContext.root(workspace=None, legacy_mint=True))
        try:
            asyncio.run(self.judge("b"))
        finally:
            exit_run(token)
        self.assertEqual(_UncertifiedLeaf.ran_on, [self.leaf, self.leaf])


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
