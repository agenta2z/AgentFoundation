"""Regression test for the ``_render_prompt`` non-empty postcondition (L2).

Locks in the principled invariant introduced after the production incident
in ``server_20260615_194631_8e0863a8`` turn_002, where a misconfigured
``TemplateManagerPromptRenderer`` silently returned an empty string, which
then propagated to the rovodev CLI backend and hung for 120s waiting on
non-empty input.

After the fix, an explicit caller-injected renderer that can't resolve its
template MUST raise ``RuntimeError`` from ``_render_prompt`` with full
diagnostic context, NOT silently produce an empty rendered prompt.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest
from attr import attrs

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.template_manager_renderer import (
    TemplateManagerPromptRenderer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from rich_python_utils.string_utils.formatting.template_manager.template_manager import (
    TemplateManager,
)


@attrs(slots=False)
class _MockBase(InferencerBase):
    """Minimal backend that satisfies CI's base-inferencer slot."""

    def _infer(self, inp, cfg=None, **kw):  # type: ignore[override]
        return "ok"

    async def _ainfer(self, inp, cfg=None, **kw):  # type: ignore[override]
        return "ok"


class TestRenderPromptNonEmptyInvariant(unittest.TestCase):
    """L2: ``_render_prompt`` must raise (not silently return '') when the
    injected renderer cannot produce a non-empty rendered prompt."""

    def test_raises_on_empty_rendered_prompt_from_missing_template(self) -> None:
        # ── Arrange: a templates dir with SOME content but NO
        # ``conversation/main/initial.*`` — the exact misconfiguration
        # observed in production (OpenStartup's prompt_templates/ had
        # task_breakdown/, plan/, implementation/, deep_research/ but no
        # conversation/ subdir).
        td = Path(tempfile.mkdtemp(prefix="test_render_prompt_invariant_"))
        # Drop in a placeholder template under a different namespace so
        # TemplateManager doesn't reject the dir as empty at construction.
        (td / "other_space" / "main").mkdir(parents=True, exist_ok=True)
        (td / "other_space" / "main" / "placeholder.jinja2").write_text(
            "ok", encoding="utf-8"
        )

        bad_renderer = TemplateManagerPromptRenderer(
            template_manager=TemplateManager(
                templates=str(td),
                active_template_root_space="conversation",
                active_template_type="main",
            ),
            template_key="initial",
        )

        ci = ConversationalInferencer(
            base_inferencer=_MockBase(),
            prompt_renderer=bad_renderer,
        )

        # ── Act + Assert: render must raise loudly, not return ''.
        with self.assertRaises(RuntimeError) as cm:
            ci._render_prompt("hello")

        # The error message must carry actionable diagnostic context so the
        # developer can fix the misconfiguration without instrumenting the
        # lookup chain.
        message = str(cm.exception)
        self.assertIn("empty", message.lower(),
                      "Error must explain that the rendered prompt is empty.")
        self.assertIn("template_key", message,
                      "Error must include the failing template_key.")
        self.assertIn("initial", message,
                      "Error must include the actual failing template key value.")
        self.assertIn("templates", message,
                      "Error must guide the reader toward the TemplateManager configuration.")

    def test_passes_when_renderer_produces_non_empty_prompt(self) -> None:
        """Sanity: with the default (working) renderer, no exception is raised."""
        ci = ConversationalInferencer(base_inferencer=_MockBase())
        # Should NOT raise — the CI's default renderer points at AF's own
        # canonical conversation/main/initial.jinja2 and produces a non-empty
        # rendered prompt.
        rendered = ci._render_prompt("hello")
        self.assertTrue(
            rendered and rendered.strip(),
            "Default renderer should produce a non-empty rendered prompt.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
