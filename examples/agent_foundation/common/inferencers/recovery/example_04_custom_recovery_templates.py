#!/usr/bin/env python3
"""Example 04 — Custom Recovery Templates and Disabling Recovery.

This example demonstrates how to customize or disable the recovery prompt
templates that the streaming inferencer uses for cache-based recovery.

The system provides three levels of customization:

    Level 1: Template override (easiest)
        Place your own `recovery/continue.jinja2` or `recovery/reference.jinja2`
        in your template directory. Since your template root has higher priority
        than the built-in defaults, your templates win automatically.

    Level 2: Key override
        Set `fallback_recovery_template_key = "my_recovery"` on your subclass.
        The system will look for `my_recovery/continue.jinja2` instead of
        `recovery/continue.jinja2`.

    Level 3: Method override (most control)
        Override `_render_recovery_prompt()` to do anything you want — use
        a different template engine, hardcode strings, call an external API, etc.

    Level 4: Disable entirely
        Set `use_default_prompt_templates=False`. CONTINUE and REFERENCE modes
        will fall through to RESTART (plain retry) since no templates are available.

Expected terminal output:

    === Demo 1: Default recovery templates (built-in) ===
    Using built-in Jinja2 template for REFERENCE mode...
    Recovery prompt starts with: "A previous attempt at this task was interrupted..."
    Result: "Fresh model response using built-in template"

    === Demo 2: Custom _render_recovery_prompt override ===
    Custom recovery prompt: "[CUSTOM] Task: Explain X. Previous partial: Once upon..."
    Result: "Response using custom prompt format"

    === Demo 3: Disabled templates (falls through to RESTART) ===
    No recovery template available — falling through to RESTART.
    Result: "Plain restart — original prompt sent as-is"

    === Demo 4: Using render_recovery_prompt() standalone ===
    Rendered CONTINUE template:
      "The previous response was interrupted...
       ---BEGIN PARTIAL OUTPUT---
       Here is some partial text
       ---END PARTIAL OUTPUT---
       Continue from where the response was interrupted..."

    Rendered REFERENCE template:
      "A previous attempt at this task was interrupted...
       ---BEGIN PARTIAL (REFERENCE ONLY)---
       Here is some partial text
       ---END PARTIAL---
       The task is: Write a poem"

Run:
    python examples/agent_foundation/common/inferencers/recovery/example_04_custom_recovery_templates.py
"""

import asyncio
import os
import sys

# --- Path setup ---
_script_dir = os.path.dirname(os.path.abspath(__file__))
_agent_root = os.path.normpath(os.path.join(_script_dir, "..", "..", "..", "..", "..", ".."))
for _sub in ("AgentFoundation/src", "RichPythonUtils/src"):
    _p = os.path.normpath(os.path.join(_agent_root, _sub))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Any, AsyncIterator, Optional

from attr import attrib, attrs

from agent_foundation.common.inferencers.streaming_inferencer_base import (
    FallbackInferMode,
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.prompt_templates import (
    render_recovery_prompt,
)
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
        print(f"  Custom recovery prompt: \"{custom}\"")
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
    print("  Using built-in Jinja2 template for REFERENCE mode...")

    inf = MockStreamingBase(
        mock_response="Fresh model response using built-in template",
        fallback_infer_mode=FallbackInferMode.REFERENCE,
        max_retry=2,
        fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        min_retry_wait=0, max_retry_wait=0,
        # use_default_prompt_templates=True is the default
    )

    # Manually test what the template renders
    rendered = inf._render_recovery_prompt(
        FallbackInferMode.REFERENCE, "Explain X", "Partial output here"
    )
    print(f"  Recovery prompt starts with: \"{rendered[:60]}...\"")
    print(f"  (Uses Jinja2 template from resources/prompt_templates/recovery/reference.jinja2)")

    # ── Demo 2: Custom _render_recovery_prompt ──────────────────────────
    separator("Demo 2: Custom _render_recovery_prompt override")

    inf2 = CustomPromptInferencer(
        mock_response="Response using custom prompt format",
        max_retry=2,
        fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        min_retry_wait=0, max_retry_wait=0,
    )

    # Show what the custom prompt looks like
    rendered = inf2._render_recovery_prompt(
        FallbackInferMode.CONTINUE, "Explain X", "Once upon a time in a land"
    )

    # ── Demo 3: Disabled templates ──────────────────────────────────────
    separator("Demo 3: Disabled templates (falls through to RESTART)")

    inf3 = MockStreamingBase(
        mock_response="Plain restart -- original prompt sent as-is",
        use_default_prompt_templates=False,  # Disable templates!
        fallback_infer_mode=FallbackInferMode.REFERENCE,
        max_retry=2,
        fallback_mode=FallbackMode.ON_FIRST_FAILURE,
        min_retry_wait=0, max_retry_wait=0,
    )

    result = inf3._render_recovery_prompt(
        FallbackInferMode.REFERENCE, "Explain X", "Partial output"
    )
    if result is None:
        print("  _render_recovery_prompt returned None")
        print("  -> CONTINUE/REFERENCE modes fall through to RESTART (plain retry)")
        print("  -> The original prompt is sent as-is, cache is ignored")
    else:
        print(f"  Unexpected: got result: {result}")

    # ── Demo 4: Using render_recovery_prompt() standalone ───────────────
    separator("Demo 4: Using render_recovery_prompt() standalone")
    print("  The render_recovery_prompt() function can be used independently")
    print("  of any inferencer — useful for testing or custom pipelines.")
    print()

    continue_result = render_recovery_prompt(
        "recovery/continue",
        prompt="Write a poem",
        partial_output="Here is some partial text",
    )
    print("  Rendered CONTINUE template:")
    for line in continue_result.strip().split("\n"):
        print(f"    {line}")

    print()

    reference_result = render_recovery_prompt(
        "recovery/reference",
        prompt="Write a poem",
        partial_output="Here is some partial text",
    )
    print("  Rendered REFERENCE template:")
    for line in reference_result.strip().split("\n"):
        print(f"    {line}")

    # Summary
    separator("Summary of customization levels")
    print("  1. Template override: place recovery/continue.jinja2 in YOUR template dir")
    print("     -> Your root has higher priority than the built-in defaults")
    print("  2. Key override: set fallback_recovery_template_key = 'my_recovery'")
    print("     -> System looks for my_recovery/continue.jinja2 instead")
    print("  3. Method override: override _render_recovery_prompt() entirely")
    print("     -> Full control, no template system involved")
    print("  4. Disable: set use_default_prompt_templates=False")
    print("     -> CONTINUE/REFERENCE fall through to RESTART")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    asyncio.run(main())
