"""Backslash-command system for ConversationalInferencer.

Commands are internal CI methods decorated with @command. They are distinct
from tools (external CLI executors with tool.json + executor.py):

  - Tools: LLM-invocable, dispatched by ToolDispatcher, stateless executors
  - Commands: user-invocable AND LLM-invocable, dispatched by CommandRegistry,
    have direct access to CI state (messages, prior_context, sop_state)

Discovery happens at CI.__attrs_post_init__ time by scanning the MRO for
methods with a __command__ attribute (set by the @command decorator).
Commands are listed in declaration order, base classes first.

Commands are rendered in the prompt alongside tools so the LLM can invoke
them when the user's intent maps to one (e.g., "change model to sonnet"
→ LLM invokes set_model command).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class CommandMeta:
    """Metadata attached to a @command-decorated CI method."""

    name: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    requires_active_sop: bool = False
    requires_args: bool = False


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
    requires_args: bool = False,
):
    """Decorator that marks a CI method as a backslash command.

    Usage::

        class ConversationalInferencer:
            @command("model", description="Change LLM model",
                     requires_args=True)
            async def _cmd_set_model(self, model_name: str = "") -> str:
                self.prior_context["model_name"] = model_name
                return f"Model set to {model_name}."
    """

    def _decorator(method):
        method.__command__ = CommandMeta(
            name=name,
            description=description,
            aliases=tuple(aliases),
            requires_active_sop=requires_active_sop,
            requires_args=requires_args,
        )
        return method

    return _decorator


def _declared_commands(
    classes: Iterable[type],
) -> Iterator[tuple[str, CommandMeta, type]]:
    """``(method name, meta, class)`` for each ``@command`` method declared
    directly on ``classes``, in class order then declaration order."""
    for cls in classes:
        for attr_name in vars(cls):
            desc = getattr(getattr(cls, attr_name, None), "__command__", None)
            if desc is not None:
                yield attr_name, desc, cls


class CommandRegistry:
    """Per-CI command registry built at __attrs_post_init__ time.

    Scans the class MRO for methods with __command__ attribute and indexes
    them by name + aliases for O(1) lookup.
    """

    def __init__(self, inferencer: Any) -> None:
        self._inferencer = inferencer
        self._by_name: dict[str, tuple[CommandMeta, str]] = {}

        mro = type(inferencer).__mro__
        # Most-derived first. A command stays bound to its method name, so an
        # override (decorated or not) dispatches to the overriding method; a
        # command name redefined further down the MRO shadows the base one.
        # Two commands claiming one name in the SAME class are a bug.
        meta_by_attr: dict[str, tuple[CommandMeta, type]] = {}
        for attr_name, desc, cls in _declared_commands(mro):
            meta_by_attr.setdefault(attr_name, (desc, cls))
        by_name: dict[str, tuple[CommandMeta, str]] = {}
        key_owner: dict[str, type] = {}
        for attr_name, (desc, owner) in meta_by_attr.items():
            for key in (desc.name, *desc.aliases):
                if key not in by_name:
                    by_name[key] = (desc, attr_name)
                    key_owner[key] = owner
                elif key_owner[key] is owner:
                    _, existing_attr = by_name[key]
                    raise ValueError(
                        f"Duplicate command '/{key}' on "
                        f"{type(inferencer).__name__}: "
                        f"{existing_attr} and {attr_name}"
                    )

        # Listing order (help text, prompt) is declaration order, base classes
        # first; an override or shadowing command keeps the position where its
        # name was first declared. So moving a command between a host and its
        # mixins, or overriding it, never reorders the rendered prompt.
        position: dict[str, int] = {}
        for _, desc, _ in _declared_commands(reversed(mro)):
            for key in (desc.name, *desc.aliases):
                position.setdefault(key, len(position))
        self._by_name = dict(sorted(by_name.items(), key=lambda kv: position[kv[0]]))

    def is_command(self, user_input: str) -> bool:
        """Check if user_input is a registered slash command."""
        if not user_input.startswith("/"):
            return False
        token = user_input[1:].split(None, 1)[0] if len(user_input) > 1 else ""
        return token in self._by_name

    def is_command_name(self, name: str) -> bool:
        """Check if a tool name matches a registered command (for _execute_tool_call)."""
        return name in self._by_name

    async def dispatch(self, user_input: str) -> str:
        """Execute from slash-command text (e.g., '/model sonnet')."""
        token, _, rest = user_input[1:].partition(" ")
        entry = self._by_name.get(token)
        if entry is None:
            raise UnknownCommand(token)

        meta, attr_name = entry

        if meta.requires_active_sop:
            if not getattr(self._inferencer, "sop_state", None):
                return f"/{meta.name} requires an active SOP. Use /sop <name> first."

        handler = getattr(self._inferencer, attr_name)
        if meta.requires_args:
            return await handler(rest.strip())
        return await handler()

    async def dispatch_as_tool(self, tool_name: str, arguments: Any) -> str:
        """Execute from LLM tool call (e.g., name='set_model', arguments={'model_name': 'sonnet'})."""
        entry = self._by_name.get(tool_name)
        if entry is None:
            raise UnknownCommand(tool_name)

        meta, attr_name = entry
        handler = getattr(self._inferencer, attr_name)

        if meta.requires_args and arguments:
            # Pass the command's positional arg. The LLM usually emits a dict
            # (first value wins), but sometimes emits `arguments` as a bare
            # string (the whole arg line) or a single-element list — tolerate
            # all of them so a benign output-shape variance doesn't crash the turn.
            if isinstance(arguments, dict):
                first_val = next(iter(arguments.values()), "")
            elif isinstance(arguments, (list, tuple)):
                first_val = arguments[0] if arguments else ""
            else:
                first_val = arguments
            return await handler(str(first_val))
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

    def render_for_prompt(self) -> str:
        """Render commands as tool-like descriptions for the LLM prompt."""
        lines: list[str] = []
        for meta, attr_name in self._by_name.values():
            if meta.name in {
                m.name for m in self.list_commands() if m.name == meta.name
            }:
                handler = getattr(type(self._inferencer), attr_name, None)
                # Derive parameter info from method signature
                params = ""
                if handler and meta.requires_args:
                    sig = inspect.signature(handler)
                    param_names = [
                        p.name for p in sig.parameters.values() if p.name != "self"
                    ]
                    if param_names:
                        params = f" <{'> <'.join(param_names)}>"

                aliases = (
                    f" (aliases: {', '.join('/' + a for a in meta.aliases)})"
                    if meta.aliases
                    else ""
                )
                lines.append(f"- `/{meta.name}{params}`{aliases}: {meta.description}")
        # Deduplicate (aliases cause repeats)
        seen: set[str] = set()
        result: list[str] = []
        for line in lines:
            cmd_name = line.split("`")[1].split()[0]
            if cmd_name not in seen:
                seen.add(cmd_name)
                result.append(line)
        return "\n".join(result)
