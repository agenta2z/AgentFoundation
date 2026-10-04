# pyre-strict

"""Decorator wiring: polymorphic/lazy/cached inferencer build, method binding,
per-call kwarg forwarding, session-reset between calls, the async path using
``ainfer``, and ``validate_agentic_function`` preflight surfacing."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agent_foundation.common.inferencers.agentic_functions import (
    agentic_function,
    AgenticFunctionConfigurationError,
    escalate,
    validate_agentic_function,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    RunContext,
)
from rich_python_utils.string_utils.formatting.template_manager.template_manager import (
    jinjia_template_format,
    TemplateManager,
)

from ._helpers import AsyncOnlyFake, FakeInferencer, PreflightFake, SessionFake


def _dir_manager(body: str, *, space: str = "conversation") -> TemplateManager:
    """A manager whose only template lives at ``<space>/main/initial.jinja2``.

    Pass a ``space`` other than ``conversation`` to build a manager that is
    populated but cannot resolve the key the caller will ask for.
    """
    d = Path(tempfile.mkdtemp(prefix="agentic_tm_decorator_"))
    (d / space / "main").mkdir(parents=True, exist_ok=True)
    (d / space / "main" / "initial.jinja2").write_text(body, encoding="utf-8")
    return TemplateManager(
        templates=str(d),
        active_template_root_space="conversation",
        active_template_type="main",
        template_formatter=jinjia_template_format,
    )


class InferencerWiringTest(unittest.TestCase):
    def test_unknown_inferencer_kwargs_raise_on_first_resolution(self) -> None:
        @agentic_function(
            inferencer_kwargs={"definitely_not_a_field": 1}, template_string="{{ x }}"
        )
        def f(x: Any) -> str:
            return escalate()

        with self.assertRaises(AgenticFunctionConfigurationError):
            _ = f.inferencer

    def test_spec_and_kwargs_conflict_raises_at_decoration(self) -> None:
        fake = FakeInferencer("ok")
        with self.assertRaises(AgenticFunctionConfigurationError):

            @agentic_function(
                inferencer=lambda: fake,
                inferencer_kwargs={"a": 1},
                template_string="{{ x }}",
            )
            def f(x: Any) -> str:
                return escalate()

    def test_factory_built_lazily_and_cached_across_calls(self) -> None:
        calls = {"n": 0}

        def factory() -> Any:
            calls["n"] += 1
            return FakeInferencer("ok")

        @agentic_function(inferencer=factory, template_string="{{ x }}")
        def f(x: Any) -> str:
            return escalate()

        self.assertEqual(calls["n"], 0)  # not built at decoration
        f("a")
        f("b")
        self.assertEqual(calls["n"], 1)  # one build, reused

    def test_inferencer_property_returns_built_instance(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        def f(x: Any) -> str:
            return escalate()

        self.assertIs(f.inferencer, fake)


class MethodBindingTest(unittest.TestCase):
    def test_self_excluded_from_feed(self) -> None:
        fake = FakeInferencer("ok")

        class C:
            @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
            def m(self, x: Any) -> str:
                return escalate()

        C().m("v")
        self.assertEqual(fake.prompts[0], "v")  # `self` never reached the feed


class PerCallArgsTest(unittest.TestCase):
    def test_with_inference_args_shares_provider_and_forwards_kwargs(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        def f(x: Any) -> str:
            return escalate()

        g = f.with_inference_args(temperature=0.0)
        self.assertIs(g.provider, f.provider)  # same cached inferencer
        g("a")
        self.assertEqual(fake.infer_kwargs[0], {"temperature": 0.0})

    def test_with_rebuilds_a_distinct_provider(self) -> None:
        @agentic_function(template_string="{{ x }}")
        def f(x: Any) -> str:
            return escalate()

        g = f.with_(model_id="some-model")
        self.assertIsNot(g.provider, f.provider)


class SessionResetTest(unittest.TestCase):
    def test_session_bearing_inferencer_reset_before_every_inference(self) -> None:
        fake = SessionFake("ok")

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        def f(x: Any) -> str:
            return escalate()

        f("a")
        f("b")
        self.assertEqual(fake.calls, 2)
        self.assertEqual(fake.resets, 2)  # no conversation bleed across calls


class AsyncPathTest(unittest.TestCase):
    def test_async_wrapper_uses_ainfer_not_sync_infer(self) -> None:
        fake = AsyncOnlyFake("ok")  # sync infer() raises AssertionError

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        async def f(x: Any) -> str:
            return escalate()

        self.assertEqual(asyncio.run(f("a")), "ok")
        self.assertEqual(fake.calls, 1)


class ValidateAgenticFunctionTest(unittest.TestCase):
    def test_preflight_problems_surface_as_configuration_error(self) -> None:
        fake = PreflightFake(problems=["missing GraphQL client"])

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        async def f(x: Any) -> str:
            return escalate()

        with self.assertRaises(AgenticFunctionConfigurationError):
            asyncio.run(validate_agentic_function(f))

    def test_clean_preflight_passes(self) -> None:
        fake = PreflightFake(problems=[])

        @agentic_function(inferencer=lambda: fake, template_string="{{ x }}")
        async def f(x: Any) -> str:
            return escalate()

        asyncio.run(validate_agentic_function(f))  # no raise

    def test_rejects_non_agentic_function(self) -> None:
        with self.assertRaises(AgenticFunctionConfigurationError):
            asyncio.run(validate_agentic_function(lambda x: x))


class ManagerTemplateTest(unittest.TestCase):
    """A ``TemplateManager`` can back a decorated function end to end."""

    def test_prompt_is_rendered_from_the_manager(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(
            inferencer=lambda: fake,
            template=_dir_manager("Hello, {{ name }}!"),
            template_key="initial",
        )
        def f(name: Any) -> str:
            return escalate()

        self.assertEqual(f("Alice"), "ok")
        self.assertEqual(fake.prompts[0], "Hello, Alice!")

    def test_unresolved_key_fails_before_any_inference(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(
            inferencer=lambda: fake,
            template=_dir_manager("unused", space="other"),
            template_key="initial",
        )
        def f(name: Any) -> str:
            return escalate()

        with self.assertRaises(AgenticFunctionConfigurationError):
            f("Alice")
        self.assertEqual(fake.calls, 0)  # the model was never reached


class TracePersistTest(unittest.TestCase):
    """The trace-persist artifact (Design Y): a JSON dump lands in the call's
    run-context child workspace, carrying the typed ``result`` and the redacted
    ``redacted_args`` (assembled at write-time from the feed, never a trace field
    — the trace stays structurally argument-free; see test_trace.py)."""

    def _judge(self, fake: FakeInferencer) -> Any:
        @agentic_function(
            inferencer=lambda: fake,
            template_string="{{ task }}",
            run_context_slot="code_scope_judge",
            redact_arguments=("secret",),
        )
        def f(task: Any, secret: Any) -> str:
            return escalate()

        return f

    def test_call_writes_trace_artifact_under_child_slot(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = self._judge(FakeInferencer("ok"))
            token = enter_run(RunContext.root(workspace=InferencerWorkspace(root=d)))
            try:
                self.assertEqual(f("find the ranking model", "shh"), "ok")
            finally:
                exit_run(token)

            path = os.path.join(
                d,
                "children",
                "code_scope_judge",
                "artifacts",
                "agentic_function_trace.json",
            )
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(payload["result"], "ok")
            self.assertEqual(payload["path"], "agentic")
            # The unredacted input is in the artifact...
            self.assertEqual(payload["redacted_args"]["task"], "find the ranking model")
            # ...but a redacted argument never is.
            self.assertNotIn("secret", payload["redacted_args"])

    def test_no_workspace_context_is_a_silent_noop(self) -> None:
        f = self._judge(FakeInferencer("ok"))
        token = enter_run(RunContext.root(workspace=None))
        try:
            # No workspace on the context → nothing to write, no error.
            self.assertEqual(f("t", "shh"), "ok")
        finally:
            exit_run(token)

    def test_no_active_context_is_a_silent_noop(self) -> None:
        f = self._judge(FakeInferencer("ok"))
        # No active run context → rc is None → nothing to write, no error.
        self.assertEqual(f("t", "shh"), "ok")


if __name__ == "__main__":
    unittest.main()
