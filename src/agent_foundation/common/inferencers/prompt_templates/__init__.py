"""Recovery prompt templates for streaming inferencer fallback.

Provides a lazy-initialized TemplateManager for rendering recovery prompts
when streaming inference is interrupted and needs cache-based recovery.

Public API:
    render_recovery_prompt(template_key, prompt, partial_output) — Render by key.
    DEFAULT_RECOVERY_DIR — Path to the default recovery template directory.
"""

from pathlib import Path

DEFAULT_RECOVERY_DIR = str(
    Path(__file__).resolve().parent.parent / "resources" / "prompt_templates"
)

_RECOVERY_TM = None


def render_recovery_prompt(
    template_key: str, prompt: str, partial_output: str
) -> str:
    """Render a recovery prompt template by key.

    Uses a lazy-initialized standalone TemplateManager backed by the
    default recovery templates in ``resources/prompt_templates/``.

    Args:
        template_key: Slash-separated key, e.g. ``"recovery/continue"``.
        prompt: The original prompt/task text.
        partial_output: The cached partial output from the failed attempt.

    Returns:
        The rendered recovery prompt string.
    """
    global _RECOVERY_TM
    if _RECOVERY_TM is None:
        from rich_python_utils.string_utils.formatting.template_manager import (
            TemplateManager,
        )

        _RECOVERY_TM = TemplateManager(
            templates=DEFAULT_RECOVERY_DIR,
            active_template_type=None,
        )
    return _RECOVERY_TM(template_key, prompt=prompt, partial_output=partial_output)


__all__ = [
    "DEFAULT_RECOVERY_DIR",
    "render_recovery_prompt",
]
