# pyre-strict

"""Co-located template resolution, rendering, variable validation, and the
whitebox redaction check via the decorator's ``.render()``."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from agent_foundation.common.inferencers.agentic_functions import (
    agentic_function,
    AgenticFunctionConfigurationError,
    escalate,
)
from agent_foundation.common.inferencers.agentic_functions.rendering import (
    ManagerTemplateSource,
    render,
    resolve_template_source,
    resolve_template_text,
    TextTemplateSource,
    validate_template_variables,
)
from rich_python_utils.string_utils.formatting.template_manager.template_manager import (
    jinjia_template_format,
    TemplateManager,
)

from ._helpers import FakeInferencer


def _fn() -> None: ...


def _manager_with(body: str, *, name: str = "initial") -> TemplateManager:
    """A dir-backed manager holding one template at ``conversation/main/<name>``."""
    d = Path(tempfile.mkdtemp(prefix="agentic_tm_"))
    (d / "conversation" / "main").mkdir(parents=True, exist_ok=True)
    (d / "conversation" / "main" / f"{name}.jinja2").write_text(body, encoding="utf-8")
    return TemplateManager(
        templates=str(d),
        active_template_root_space="conversation",
        active_template_type="main",
        template_formatter=jinjia_template_format,
    )


class ResolveTemplateTextTest(unittest.TestCase):
    def test_template_string_returns_text_and_no_path(self) -> None:
        text, source = resolve_template_text(_fn, template_string="{{ x }}")
        self.assertEqual(text, "{{ x }}")
        self.assertIsNone(source)

    def test_both_provided_raises(self) -> None:
        with self.assertRaises(AgenticFunctionConfigurationError):
            resolve_template_text(_fn, template="a.jinja2", template_string="{{ x }}")

    def test_neither_provided_raises(self) -> None:
        with self.assertRaises(AgenticFunctionConfigurationError):
            resolve_template_text(_fn)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(AgenticFunctionConfigurationError):
            resolve_template_text(_fn, template="__no_such_template__.jinja2")


class RenderTest(unittest.TestCase):
    def test_render_substitutes(self) -> None:
        self.assertEqual(render("{{ x }}", {"x": "V"}), "V")

    def test_missing_variable_renders_empty(self) -> None:
        self.assertEqual(render("[{{ missing }}]", {}), "[]")


class ValidateTemplateVariablesTest(unittest.TestCase):
    def test_all_declared_ok(self) -> None:
        validate_template_variables("{{ x }} {{ y }}", {"x", "y"})

    def test_undeclared_raises(self) -> None:
        with self.assertRaises(AgenticFunctionConfigurationError):
            validate_template_variables("{{ x }} {{ z }}", {"x"})

    def test_helper_globals_allowed(self) -> None:
        validate_template_variables("{{ currentDate }}", set())


class RedactionTest(unittest.TestCase):
    def test_redacted_argument_never_reaches_the_feed(self) -> None:
        fake = FakeInferencer("ok")

        @agentic_function(
            inferencer=lambda: fake,
            template_string="user={{ user }} secret={{ secret }}",
            redact_arguments=("secret",),
        )
        def f(user: Any, secret: Any) -> str:
            return escalate()

        self.assertEqual(f.render("alice", "PASSWORD"), "user=alice secret=")


class ResolveTemplateSourceTest(unittest.TestCase):
    """The source is chosen by the *type* of ``template``, never by inspecting
    a string's contents."""

    def test_template_string_gives_an_inline_text_source(self) -> None:
        source = resolve_template_source(_fn, template_string="{{ x }}")
        self.assertIsInstance(source, TextTemplateSource)
        self.assertEqual(source.render({"x": "V"}), "V")

    def test_pathlike_is_accepted_and_resolved_as_a_file(self) -> None:
        # Reaching the file-read error (rather than "unsupported spec") is what
        # proves a PathLike took the file branch instead of being rejected.
        with self.assertRaises(AgenticFunctionConfigurationError) as cm:
            resolve_template_source(_fn, template=Path("__no_such__.jinja2"))
        self.assertIn("not found or unreadable", str(cm.exception))

    def test_manager_gives_a_manager_source(self) -> None:
        tm = _manager_with("Hello, {{ name }}!")
        source = resolve_template_source(_fn, template=tm, template_key="initial")
        self.assertIsInstance(source, ManagerTemplateSource)

    def test_unsupported_type_raises(self) -> None:
        with self.assertRaises(AgenticFunctionConfigurationError) as cm:
            resolve_template_source(_fn, template=123)
        self.assertIn("unsupported template spec of type int", str(cm.exception))

    def test_manager_without_key_or_root_space_raises(self) -> None:
        tm = _manager_with("Hello!")
        with self.assertRaises(AgenticFunctionConfigurationError) as cm:
            resolve_template_source(_fn, template=tm)
        self.assertIn("template_key", str(cm.exception))

    def test_manager_options_without_a_manager_raise(self) -> None:
        with self.assertRaises(AgenticFunctionConfigurationError) as cm:
            resolve_template_source(_fn, template_string="{{ x }}", template_key="k")
        self.assertIn("template_key", str(cm.exception))
        self.assertIn("TemplateManager", str(cm.exception))

    def test_both_or_neither_still_raises(self) -> None:
        with self.assertRaises(AgenticFunctionConfigurationError):
            resolve_template_source(_fn, template="a.jinja2", template_string="{{ x }}")
        with self.assertRaises(AgenticFunctionConfigurationError):
            resolve_template_source(_fn)


class ManagerTemplateSourceTest(unittest.TestCase):
    def test_renders_through_the_manager(self) -> None:
        tm = _manager_with("Hello, {{ name }}!")
        source = resolve_template_source(_fn, template=tm, template_key="initial")
        source.validate(frozenset({"name"}))
        self.assertEqual(source.render({"name": "Alice"}), "Hello, Alice!")

    def test_unresolved_key_raises_instead_of_silently_using_the_default(self) -> None:
        tm = _manager_with("ok", name="placeholder")
        source = resolve_template_source(_fn, template=tm, template_key="initial")
        with self.assertRaises(AgenticFunctionConfigurationError) as cm:
            source.validate(frozenset())
        self.assertIn("'initial'", str(cm.exception))

    def test_the_hazard_the_guard_defends_against(self) -> None:
        # Upstream contract with strict_lookup=False (the default): an
        # unresolved key renders EMPTY rather than raising. That silence is
        # precisely why validate() probes with a sentinel default.
        tm = _manager_with("ok", name="placeholder")
        self.assertEqual(tm("initial", feed={}), "")

    def test_feed_is_passed_as_data_not_splatted(self) -> None:
        # A parameter named like one of TemplateManager.__call__'s own keywords
        # must stay a template variable; splatting would bind it to the manager
        # keyword instead and the variable would render empty.
        tm = _manager_with("v={{ master_version }} name={{ name }}")
        source = resolve_template_source(_fn, template=tm, template_key="initial")
        rendered = source.render({"master_version": "USER", "name": "Bob"})
        self.assertEqual(rendered, "v=USER name=Bob")

    def test_non_string_render_result_raises(self) -> None:
        class _IteratorManager:
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                return iter(["a", "b"])

        source = ManagerTemplateSource(_IteratorManager(), template_key="k")
        with self.assertRaises(AgenticFunctionConfigurationError) as cm:
            source.render({})
        self.assertIn("not str", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
