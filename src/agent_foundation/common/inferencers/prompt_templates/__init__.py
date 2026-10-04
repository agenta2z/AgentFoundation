"""Recovery prompt templates for streaming inferencer fallback.

Backward-compatible re-export shim. The canonical implementation lives in
``agent_foundation.common.inferencers.recovery`` and its template directory is
resolved via ``constants.paths.DEFAULT_RECOVERY_DIR`` (which points at
``agent_foundation/resources/prompt_templates``).

This module previously carried its OWN copy of ``render_recovery_prompt`` plus a
divergent local template path (``common/inferencers/resources/prompt_templates``)
that had drifted from the canonical templates (it still shipped the old
``reference.jinja2`` and lacked ``judge.jinja2``). It now simply re-exports the
canonical symbols so there is a single source of truth.

Public API:
    render_recovery_prompt(template_key, prompt, partial_output) — Render by key.
    DEFAULT_RECOVERY_DIR — Path to the default recovery template directory.
"""

from agent_foundation.common.inferencers.constants.paths import DEFAULT_RECOVERY_DIR
from agent_foundation.common.inferencers.recovery import render_recovery_prompt

__all__ = [
    "DEFAULT_RECOVERY_DIR",
    "render_recovery_prompt",
]
