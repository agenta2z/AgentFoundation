"""Whether Claude Code ran AgentFoundation's hooks for a turn.

Claude Code drops hooks without a trace in its output when a managed policy
allows only managed hooks (``allowManagedHooksOnly``) or settings disable them
(``disableAllHooks``). The effective managed policy cannot be read up front:
the Meta launcher may compose it per run (presets, Configerator, the sensitive
launchers) and mount it over ``/etc/claude-code/managed-settings.json`` inside
the vendor's namespace. So the native backends check that the hooks actually
ran: every turn on the hook channel must reach AF's ``UserPromptSubmit`` hook
before the model starts on the turn, else the turn is stopped — also a turn
whose context is not due, since the hook runs whether or not it has context
to add. One policy governs all of AF's hooks, so this one hook also vouches
for the turn stop and the subagent guard.
"""

from __future__ import annotations

from typing import Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    L2Channel,
    TurnRequest,
)

HOOKS_INACTIVE = (
    "Claude Code did not run AgentFoundation's UserPromptSubmit hook, so the turn "
    "context, the turn stop and the subagent guard are inactive; the turn was "
    "stopped. Check for a managed Claude Code policy that only allows managed "
    "hooks (allowManagedHooksOnly) or settings that disable hooks "
    "(disableAllHooks)."
)


class PromptHookWatch:
    """One turn's expectation that Claude Code runs AF's ``UserPromptSubmit``
    hook: on every turn whose context channel is that hook, with or without
    context to add, except a passed-through local command (``local_commands``,
    e.g. ``/compact``), which Claude Code runs without the hook and without
    the model.

    Any other ``/…`` prompt needs the hook too — Claude Code runs unknown
    names, paths, user commands and skills through it to the model — but is
    checked only at the first model output: a local command the backend does
    not list (``/release-notes``) also runs without the hook and without the
    model, and only the model acting without the hook is a problem (claude
    2.1.289, CLI and SDK)."""

    def __init__(self, local_commands: tuple[str, ...] = ()) -> None:
        self._local_commands = local_commands
        self._expected = False
        self._at_model_output = False
        self._fired = False

    def start(self, request: TurnRequest, *, hooks_registered: bool = True) -> None:
        slash = request.text.startswith("/")
        self._expected = (
            hooks_registered
            and request.channel is L2Channel.HOOK
            and not (slash and _command_name(request.text) in self._local_commands)
        )
        self._at_model_output = slash
        self._fired = False

    def fired(self) -> None:
        self._fired = True

    def problem(self, *, model_output: bool = False) -> Optional[str]:
        """``HOOKS_INACTIVE`` the first time it is asked after the hook was
        due and did not run — for a ``/…`` prompt only once ``model_output``
        (the model started on the turn); ``None`` otherwise."""
        if not self._expected or self._fired:
            return None
        if self._at_model_output and not model_output:
            return None
        self._expected = False
        return HOOKS_INACTIVE


def _command_name(text: str) -> str:
    return text[1:].split(None, 1)[0] if len(text) > 1 else ""
