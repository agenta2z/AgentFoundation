#!/usr/bin/env python3
"""Example 04 — Custom Recovery Templates and Disabling Recovery.

This example demonstrates how to customize or disable the recovery prompt
templates that the streaming inferencer uses for cache-based recovery.

Recovery taxonomy (Option C): UPDATE edits/completes the prior output in place and
is the only mode with a template (`recovery/update.jinja2`); RETRY is a plain re-run
of the original input and has NO template (`recovery/retry.jinja2` was retired).

The system provides several levels of customization:

    Level 1: Template override (easiest)
        Place your own `recovery/update.jinja2` in your template directory. Since
        your template root has higher priority than the built-in defaults, your
        template wins automatically. (RETRY has no template to override — it is
        always a plain re-run.)

    Level 2: Key override
        Set `fallback_recovery_template_key = "my_recovery"` on your subclass.
        The system will look for `my_recovery/update.jinja2` instead of
        `recovery/update.jinja2`.

    Level 3: Method override (most control)
        Override `_render_recovery_prompt()` to do anything you want — use
        a different template engine, hardcode strings, call an external API, etc.

    Level 4: Disable entirely
        Set `use_default_prompt_templates=False`. UPDATE then falls through to a
        plain re-run since no template is available (RETRY always plain-re-runs).

Expected terminal output:

    === Demo 1: Default recovery templates (built-in) ===
    UPDATE is the only mode with a built-in template; RETRY is a plain re-run.
    UPDATE recovery prompt starts with: "Your previous attempt produced real, on-topic..."
    RETRY renders: None  -> plain re-run of the original input

    === Demo 2: Custom _render_recovery_prompt override ===
    Custom recovery prompt: "[CUSTOM] Task: Explain X. Previous partial: Once upon..."
    Result: "Response using custom prompt format"

    === Demo 3: Disabled templates (falls through to a plain re-run) ===
    _render_recovery_prompt returned None
    -> UPDATE/RETRY modes fall through to a plain re-run

    === Demo 4: Using render_recovery_prompt() standalone ===
    Rendered UPDATE template (edit-in-place / re-emit-inline; contains input + prior partial).
    RETRY has no standalone template (retired) — it is a plain re-run.

Run:
    python examples/agent_foundation/common/inferencers/recovery/example_04_custom_recovery_templates.py
"""

import asyncio
import os
import sys

# --- Path setup ---
_script_dir = os.path.dirname(os.path.abspath(__file__))
_agent_root = os.path.normpath(
    os.path.join(_script_dir, "..", "..", "..", "..", "..", "..")
)
for _sub in ("AgentFoundation/src", "RichPythonUtils/src"):
    _p = os.path.normpath(os.path.join(_agent_root, _sub))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Any, AsyncIterator, Optional

from agent_foundation.common.inferencers.prompt_templates import render_recovery_prompt
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    FallbackInferMode,
    StreamingInferencerBase,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode


# ---------------------------------------------------------------------------
# Base mock that always crashes on primary, succeeds on recovery
# ---------------------------------------------------------------------------


@attrs
class MockStreamingBase(StreamingInferencerBase):
    """Base mock that crashes on primary _ainfer, returns response on recovery."""

    mock_response: str = attrib(default="Mock response")
    _call_count: int = attrib(default=0, init=False, repr=False)

    async def _ainfer_streaming(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        yield self.mock_response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        self._call_count += 1
        if self._call_count == 1:
            raise RuntimeError("Primary failed")
        return self.mock_response

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.mock_response

    async def adisconnect(self):
        pass


# ---------------------------------------------------------------------------
# Demo 2: Subclass with custom _render_recovery_prompt
# ---------------------------------------------------------------------------


@attrs
class CustomPromptInferencer(MockStreamingBase):
    """Overrides _render_recovery_prompt to use a completely custom format."""

    def _render_recovery_prompt(self, mode, prompt, partial_output):
        """Custom recovery prompt — no Jinja2, no templates, just a string."""
        custom = f"[CUSTOM] Task: {prompt}. Previous partial: {partial_output[:30]}..."
        print(f'  Custom recovery prompt: "{custom}"')
        return custom


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------


def separator(title: str):
    print(f"\n{'=' * 3} {title} {'=' * 3}")


async def main():
    print("Recovery Template Customization Demo")
    print("=" * 60)

    # ── Demo 1: Default built-in templates ──────────────────────────────
    separator("Demo 1: Default recovery templates (built-in)")
    print(
        "  UPDATE is the only mode with a built-in template; RETRY is a plain re-run."
    )

    inf = MockStreamingBase(
        mock_response="Fresh model response using built-in template",
        fallback_infer_mode=FallbackInferMode.UPDATE,
        max_retry=2,
        fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        min_retry_wait=0,
        max_retry_wait=0,
        # use_default_prompt_templates=True is the default
    )

    # UPDATE renders the built-in recovery/update.jinja2 template.
    rendered = inf._render_recovery_prompt(
        FallbackInferMode.UPDATE, "Explain X", "Partial output here"
    )
    print(f'  UPDATE recovery prompt starts with: "{rendered[:60]}..."')
    print(
        "  (Uses Jinja2 template from resources/prompt_templates/recovery/update.jinja2)"
    )
    # RETRY has no template — it returns None, so the caller does a plain re-run.
    retry_rendered = inf._render_recovery_prompt(
        FallbackInferMode.RETRY, "Explain X", "Partial output here"
    )
    print(f"  RETRY renders: {retry_rendered!r}  -> plain re-run of the original input")

    # ── Demo 2: Custom _render_recovery_prompt ──────────────────────────
    separator("Demo 2: Custom _render_recovery_prompt override")

    inf2 = CustomPromptInferencer(
        mock_response="Response using custom prompt format",
        max_retry=2,
        fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        min_retry_wait=0,
        max_retry_wait=0,
    )

    # Show what the custom prompt looks like
    rendered = inf2._render_recovery_prompt(
        FallbackInferMode.UPDATE, "Explain X", "Once upon a time in a land"
    )

    # ── Demo 3: Disabled templates ──────────────────────────────────────
    separator("Demo 3: Disabled templates (falls through to a plain re-run)")

    inf3 = MockStreamingBase(
        mock_response="Plain re-run -- original prompt sent as-is",
        use_default_prompt_templates=False,  # Disable templates!
        fallback_infer_mode=FallbackInferMode.RETRY,
        max_retry=2,
        fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        min_retry_wait=0,
        max_retry_wait=0,
    )

    # With templates disabled, even UPDATE (which normally has a template) returns
    # None → plain re-run. (RETRY returns None regardless of this setting.)
    result = inf3._render_recovery_prompt(
        FallbackInferMode.UPDATE, "Explain X", "Partial output"
    )
    if result is None:
        print("  _render_recovery_prompt returned None")
        print("  -> UPDATE/RETRY modes fall through to a plain re-run")
        print("  -> The original prompt is sent as-is, cache is ignored")
    else:
        print(f"  Unexpected: got result: {result}")

    # ── Demo 4: Using render_recovery_prompt() standalone ───────────────
    separator("Demo 4: Using render_recovery_prompt() standalone")
    print("  The render_recovery_prompt() function can be used independently")
    print("  of any inferencer — useful for testing or custom pipelines.")
    print()

    update_result = render_recovery_prompt(
        "recovery/update",
        prompt="Write a poem",
        partial_output="Here is some partial text",
    )
    print("  Rendered UPDATE template:")
    for line in update_result.strip().split("\n"):
        print(f"    {line}")

    print()
    print("  RETRY has no standalone template — recovery/retry.jinja2 was retired;")
    print("  RETRY is a plain re-run of the original input (nothing to render).")

    # Summary
    separator("Summary of customization levels")
    print("  1. Template override: place recovery/update.jinja2 in YOUR template dir")
    print("     -> Your root has higher priority than the built-in defaults")
    print("  2. Key override: set fallback_recovery_template_key = 'my_recovery'")
    print("     -> System looks for my_recovery/update.jinja2 instead")
    print("  3. Method override: override _render_recovery_prompt() entirely")
    print("     -> Full control, no template system involved")
    print("  4. Disable: set use_default_prompt_templates=False")
    print("     -> UPDATE falls through to a plain re-run (RETRY always does)")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    asyncio.run(main())
