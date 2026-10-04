"""Action-tool and command dispatch shared by the conversational orchestrators.

``execute`` runs one tool call of the model and describes it as a
``ToolOutcome``: a command invoked as a tool, an asynchronous tool started in
the background, or a tool run to completion. It does not apply the result to
the host; ``apply_outcome`` does (merge the tool's ``context_updates``, check
SOP phase completion). A caller can therefore decide first whether the result
still applies: the native bridge drops it when the turn ended while the tool
ran.

A background run is held by the host's ``AsyncToolTasks`` until it finishes;
then its context updates and the phase check are applied and
``on_async_done(tool_name, result)`` lets the host publish the result.

``ToolDispatchMixin`` keeps the orchestrators' method names
(``_execute_tool_call``, ``_resolve_tool_name``, ``_valid_tool_names``) as
delegators, with two seams where the orchestrators differ:

* ``_on_command_followup`` — what to do with the follow-up request a command
  (e.g. ``/sop X <request>``) seeds. The text-protocol loop surfaces it as a
  user turn; a native orchestrator returns it inside the tool result.
* ``_publish_async_completion`` — how a background tool's result reaches the
  model. The text-protocol loop appends it to its transcript; a native
  orchestrator queues it for the next turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Iterator, Mapping, Optional, Protocol

from agent_foundation.common.inferencers.agentic_inferencers.conversational.commands import (
    CommandRegistry,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.inbox import (
    ToolCompletion,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocol_text import (
    TOOL_RESULTS_PREFIX as _TOOL_RESULTS_PREFIX,
)
from agent_foundation.common.inferencers.inferencer_base import safe_slot
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    enter_run,
    exit_run,
)

logger: logging.Logger = logging.getLogger(__name__)

AsyncDoneCallback = Callable[[str, Any], None]


@dataclass(frozen=True)
class ToolOutcome:
    """What one tool call produced; not yet applied to the host."""

    # Canonical tool (or command) name.
    tool_name: str
    text: str
    context_updates: Mapping[str, Any] = field(default_factory=dict)
    is_async: bool = False
    is_command: bool = False
    # The request a command (e.g. ``sop``) seeded for the SOP it entered.
    followup_text: Optional[str] = None
    error: Optional[str] = None


class AsyncToolTasks:
    """An orchestrator's background tool runs.

    Each run is referenced here until it finishes: the event loop keeps only
    weak references to tasks, so an unreferenced run can be garbage-collected
    mid-flight. Dispatches are numbered so concurrent runs of one tool get
    distinct run contexts.
    """

    def __init__(self) -> None:
        self._dispatches: Iterator[int] = itertools.count()
        self._pending: set[asyncio.Task[None]] = set()
        # The most recently started run.
        self.latest: Optional[asyncio.Task[None]] = None

    def next_dispatch(self) -> int:
        return next(self._dispatches)

    def start(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        self.latest = task
        return task

    @property
    def pending(self) -> frozenset[asyncio.Task[None]]:
        return frozenset(self._pending)


class ToolDispatchHost(Protocol):
    """What ``execute`` and ``apply_outcome`` need from an orchestrator."""

    tool_registry: Mapping[str, Any]
    tool_executor: Any  # ToolExecutorCallable
    sop_controller: Any  # SOPController
    async_tool_tasks: AsyncToolTasks

    @property
    def commands(self) -> CommandRegistry: ...

    def update_prior_context(self, **updates: Any) -> None: ...


def resolve_tool_name(registry: Mapping[str, Any], name: str) -> str:
    """Resolve a tool name or alias to the canonical tool name.

    Strips a leading ``/`` first — the LLM sometimes emits an action name like
    ``/sop`` copying the prompt's slash-command prose, and command/registry
    keys are never slash-prefixed. Then matches against each tool's ``name``,
    ``aliases``, and ``preferred_prompt_alias``.
    """
    if name.startswith("/"):
        name = name[1:]
    if name in registry:
        return name
    normalized = name.replace("-", "_")
    for tool in registry.values():
        if (
            name in getattr(tool, "aliases", [])
            or normalized == tool.name
            or normalized == getattr(tool, "preferred_prompt_alias", "")
        ):
            return tool.name
    if normalized in registry:
        return normalized
    return name


def valid_tool_names(registry: Mapping[str, Any]) -> set[str]:
    """Tool names including aliases and preferred prompt aliases."""
    names: set[str] = set()
    for tool in registry.values():
        names.add(tool.name)
        for alias in getattr(tool, "aliases", []):
            names.add(alias)
        preferred = getattr(tool, "preferred_prompt_alias", "")
        if preferred:
            names.add(preferred)
    return names


def tool_run_context(parent: Any, tool_name: str, dispatch: Optional[int] = None):
    """The context a tool body runs under: ``tool/<name>`` below ``parent``
    (``tool/<name>/async_<n>`` for an async dispatch, which can outlive the
    turn and overlap a same-name one), keeping ``parent``'s workspace.
    ``None`` without a parent."""
    if parent is None:
        return None
    slots = ["tool", tool_name]
    if dispatch is not None:
        slots.append(f"async_{dispatch}")
    ctx = parent
    for depth, slot in enumerate(slots):
        deepest = depth == len(slots) - 1
        ctx = ctx.child(
            safe_slot(slot), workspace=parent.workspace if deepest else None
        )
    return ctx


@contextlib.contextmanager
def _bound(ctx: Any) -> Iterator[None]:
    if ctx is None:
        yield
        return
    token = enter_run(ctx)
    try:
        yield
    finally:
        exit_run(token)


async def _run_tool_executor(executor, tool_ctx, name: str, arguments: Any) -> Any:
    """``executor(name, arguments)`` with ``tool_ctx`` bound (unbound when ``None``)."""
    with _bound(tool_ctx):
        return await executor(name, arguments)


async def execute(
    host: ToolDispatchHost,
    name: str,
    arguments: Any,
    *,
    run_ctx: Any = None,
    on_async_done: Optional[AsyncDoneCallback] = None,
) -> ToolOutcome:
    """Run the model's call of ``name`` with ``arguments``.

    ``run_ctx`` is the turn's run context: a command runs under it, a tool body
    under its ``tool/<name>`` child (``None``: no context is bound). A command
    of the host's ``CommandRegistry`` runs first; a tool marked
    ``asynchronous`` is started in the background (``on_async_done`` is called
    once it finished and its result was applied); any other tool runs to
    completion. A failing tool body is reported in ``error``; a failing
    command raises.
    """
    canonical = resolve_tool_name(host.tool_registry, name)

    if host.commands.is_command_name(canonical):
        with _bound(run_ctx):
            text = await host.commands.dispatch_as_tool(canonical, arguments or {})
        return ToolOutcome(
            tool_name=canonical,
            text=text,
            is_command=True,
            followup_text=host.sop_controller.consume_pending_followup(),
        )

    if host.tool_executor is None:
        return ToolOutcome(
            tool_name=canonical,
            text=f"No tool executor configured for: {canonical}",
            error="no tool executor configured",
        )

    tool_def = host.tool_registry.get(canonical)
    if tool_def and getattr(tool_def, "asynchronous", False):
        return _start_async(host, canonical, arguments, run_ctx, on_async_done)

    try:
        result = await _run_tool_executor(
            host.tool_executor,
            tool_run_context(run_ctx, canonical),
            canonical,
            arguments,
        )
    except Exception as e:
        logger.error("Tool execution error for %s: %s", canonical, e)
        return ToolOutcome(
            tool_name=canonical, text=f"Error executing {canonical}: {e}", error=str(e)
        )
    return ToolOutcome(
        tool_name=canonical,
        text=result.result if hasattr(result, "result") else str(result),
        context_updates=_context_updates(result),
    )


def apply_outcome(host: ToolDispatchHost, outcome: ToolOutcome) -> ToolOutcome:
    """Apply a finished call's result to ``host``: merge its context updates,
    then let the SOP check phase completion under the tool's name. Starting a
    background run and a failed call change nothing. Failing to apply a tool's
    result fails the call (the returned outcome carries the error)."""
    if outcome.is_command:
        host.sop_controller.check_phase_completion(outcome.tool_name)
        return outcome
    if outcome.is_async or outcome.error is not None:
        return outcome
    try:
        if outcome.context_updates:
            host.update_prior_context(**outcome.context_updates)
        host.sop_controller.check_phase_completion(outcome.tool_name)
    except Exception as e:
        logger.error("Tool execution error for %s: %s", outcome.tool_name, e)
        return dataclasses.replace(
            outcome, text=f"Error executing {outcome.tool_name}: {e}", error=str(e)
        )
    return outcome


def _start_async(
    host: ToolDispatchHost,
    canonical: str,
    arguments: Any,
    run_ctx: Any,
    on_async_done: Optional[AsyncDoneCallback],
) -> ToolOutcome:
    tasks = host.async_tool_tasks
    tool_ctx = tool_run_context(run_ctx, canonical, tasks.next_dispatch())
    # The phase shows as running while the tool works (forward-only).
    host.sop_controller.mark_async_tool_phase_running(canonical)
    tasks.start(
        _run_async(
            host, host.tool_executor, tool_ctx, canonical, arguments, on_async_done
        )
    )
    return ToolOutcome(
        tool_name=canonical,
        text=(
            f"Tool '{canonical}' launched asynchronously. "
            f"Check the task panel for progress and results."
        ),
        is_async=True,
    )


async def _run_async(
    host: ToolDispatchHost,
    executor: Any,
    tool_ctx: Any,
    canonical: str,
    arguments: Any,
    on_async_done: Optional[AsyncDoneCallback],
) -> None:
    try:
        result = await _run_tool_executor(executor, tool_ctx, canonical, arguments)
        updates = _context_updates(result)
        if updates:
            host.update_prior_context(**updates)
        host.sop_controller.check_phase_completion(canonical)
        if on_async_done is not None:
            on_async_done(canonical, result)
    except Exception as e:
        logger.error("Async tool %s failed: %s", canonical, e)


def _context_updates(result: Any) -> dict[str, Any]:
    updates = getattr(result, "context_updates", None)
    return dict(updates) if updates else {}


class ToolDispatchMixin:
    """The orchestrators' tool-dispatch methods, as delegators to the
    functions above."""

    @property
    def commands(self) -> CommandRegistry:
        return self._commands

    async def _execute_tool_call(self, tool_call: Any) -> str:
        """Execute a tool call, apply its result, and return the result text.

        Tools marked asynchronous=True in the tool registry are launched as
        background asyncio tasks (fire-and-forget) so the conversation turn
        completes immediately. The tool sends task_status notifications to
        the frontend independently.
        """
        outcome = apply_outcome(
            self,
            await execute(
                self,
                tool_call.name,
                tool_call.arguments,
                run_ctx=active_run_context(),
                on_async_done=self._on_async_tool_done,
            ),
        )
        if outcome.is_command:
            self._on_command_followup(outcome.followup_text)
        if outcome.is_async:
            self._async_tool_dispatched = True
        return outcome.text

    def _resolve_tool_name(self, name: str) -> str:
        return resolve_tool_name(self.tool_registry, name)

    @property
    def _valid_tool_names(self) -> set[str]:
        return valid_tool_names(self.tool_registry)

    def _on_async_tool_done(self, tool_name: str, result: Any) -> None:
        """A background tool finished and its result was applied: publish it,
        then wake the inbox loop (if enabled)."""
        self._publish_async_completion(tool_name, result)
        if self._inbox is not None:
            try:
                self._inbox.put_nowait(ToolCompletion(tool_name=tool_name))
            except Exception:
                logger.warning("Inbox put failed for tool %s", tool_name)

    def _on_command_followup(self, followup: Optional[str]) -> None:
        """A command (e.g. /sop, /resume_sop) may seed an initial request for
        the just-entered SOP. Surface it as a user turn so the loop's next
        self-continuation acts on the concrete goal, not just the SOP guidance."""
        if followup:
            self.add_message("user", followup)

    def _publish_async_completion(self, canonical: str, result: Any) -> None:
        """Make a finished background tool's result visible to the model."""
        if hasattr(result, "result"):
            self.add_message(
                "user",
                f"{_TOOL_RESULTS_PREFIX}\n{canonical}: {result.result}",
            )
