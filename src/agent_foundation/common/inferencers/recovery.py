"""Recovery prompt rendering for streaming inferencer fallback.

Provides a lazy-initialized TemplateManager for rendering recovery prompts
when streaming inference is interrupted and needs cache-based recovery.
"""

from typing import Optional

from agent_foundation.common.inferencers.constants.paths import DEFAULT_RECOVERY_DIR

_RECOVERY_TM = None


def render_recovery_prompt(
    template_key: str,
    prompt: str,
    partial_output: str,
    has_local_access: bool = True,
    reason: Optional[str] = None,
) -> str:
    """Render a recovery prompt template by key.

    Uses a lazy-initialized standalone TemplateManager backed by the
    default recovery templates in ``resources/prompt_templates/``.

    Args:
        template_key: Slash-separated key, e.g. ``"recovery/update"``.
        prompt: The original prompt/task text. Rendered into the template's
            ``{{ agent_prompt }}`` slot.
        partial_output: The cached partial output from the failed attempt.
            Rendered into the template's ``{{ agent_response }}`` slot.
        has_local_access: Whether the target inferencer can write to the local
            filesystem. Rendered into the template's ``{{ has_local_access }}``
            slot (``recovery/update.jinja2`` branches on it: edit-in-place for a
            local agent, re-emit-inline for a no-local agent). Defaults to True to
            preserve the historical edit-in-place rendering for existing callers.
        reason: The judge's concrete ``<reason>`` for a guardrail UPDATE (e.g.
            "missing the required <Response> block"), rendered into the template's
            ``{{ reason }}`` slot for a guided fix. ``None`` for non-guardrail
            recoveries (resume-from-cache), which render without it.

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
    # The recovery templates use ``agent_prompt``/``agent_response`` (see
    # resources/prompt_templates/recovery/*.jinja2); map this function's stable
    # ``prompt``/``partial_output`` API onto those template variables.
    return _RECOVERY_TM(
        template_key,
        agent_prompt=prompt,
        agent_response=partial_output,
        has_local_access=has_local_access,
        reason=reason,
    )
