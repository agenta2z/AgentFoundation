"""Backslash-command system for ConversationalInferencer.

Commands are internal CI methods decorated with @command. They are distinct
from tools (external CLI executors with tool.json + executor.py):

  - Tools: LLM-invocable, dispatched by ToolDispatcher, stateless executors
  - Commands: user-invocable, dispatched by CommandRegistry, have direct
    access to CI state (messages, prior_context, tracker)

Discovery happens at CI.__attrs_post_init__ time by scanning the MRO for
methods with a __command__ attribute (set by the @command decorator).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommandMeta:
    """Metadata attached to a @command-decorated CI method."""

    name: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    requires_active_sop: bool = False


class UnknownCommand(LookupError):
    """Raised when a slash-prefixed input doesn't match any registered command."""

    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__(f"Unknown command: /{token}")


def command(
    name: str,
    description: str = "",
    *,
    aliases: tuple[str, ...] = (),
    requires_active_sop: bool = False,
):
    """Decorator that marks a CI method as a backslash command.

    Usage::

        class ConversationalInferencer:
            @command("pause", description="Pause SOP execution",
                     requires_active_sop=True)
            async def _cmd_pause(self) -> str:
                self._paused = True
                return "SOP paused. Use /resume to continue."
    """

    def _decorator(method):
        method.__command__ = CommandMeta(
            name=name,
            description=description,
            aliases=tuple(aliases),
            requires_active_sop=requires_active_sop,
        )
        return method

    return _decorator


class CommandRegistry:
    """Per-CI command registry built at __attrs_post_init__ time.

    Scans the class MRO for methods with __command__ attribute and indexes
    them by name + aliases for O(1) lookup.
    """

    def __init__(self, inferencer: Any) -> None:
        self._inferencer = inferencer
        self._by_name: dict[str, tuple[CommandMeta, str]] = {}

        for cls in type(inferencer).__mro__:
            for attr_name in vars(cls):
                desc = getattr(getattr(cls, attr_name, None), "__command__", None)
                if desc is None:
                    continue
                for key in (desc.name, *desc.aliases):
                    if key in self._by_name:
                        existing_meta, existing_attr = self._by_name[key]
                        raise ValueError(
                            f"Duplicate command '/{key}' on "
                            f"{type(inferencer).__name__}: "
                            f"{existing_attr} and {attr_name}"
                        )
                    self._by_name[key] = (desc, attr_name)

    def is_command(self, user_input: str) -> bool:
        """Check if user_input is a registered slash command."""
        if not user_input.startswith("/"):
            return False
        token = user_input[1:].split(None, 1)[0] if len(user_input) > 1 else ""
        return token in self._by_name

    async def dispatch(self, user_input: str) -> str:
        """Execute the matching command. Returns the textual response."""
        token, _, rest = user_input[1:].partition(" ")
        entry = self._by_name.get(token)
        if entry is None:
            raise UnknownCommand(token)

        meta, attr_name = entry

        if meta.requires_active_sop:
            if not getattr(self._inferencer, "sop_state", None):
                return f"/{meta.name} requires an active SOP. Use /sop <name> first."

        handler = getattr(self._inferencer, attr_name)
        return await handler()

    def list_commands(self) -> list[CommandMeta]:
        """Return deduplicated list of commands sorted by name."""
        seen: set[str] = set()
        out: list[CommandMeta] = []
        for meta, _ in self._by_name.values():
            if meta.name not in seen:
                seen.add(meta.name)
                out.append(meta)
        return sorted(out, key=lambda m: m.name)
