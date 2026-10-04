# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""Conversation tool handler protocol — typed, registry-friendly contract.

Three-method handler interface plus a typed `HandlerResult` whose effects are
applied to the inferencer via small `InferencerEffect` classes (open/closed
for new effect kinds).

IMPORTANT — subclassing requirement: Python's Protocol default-method bodies
are inherited ONLY when a class explicitly subclasses the Protocol
(`class FooHandler(ConversationToolHandler): ...`). Pure structural conformers
that just happen to have build_input_mode/handle_response methods without
subclassing DO NOT inherit the default `enrich_before_send` — they would hit
AttributeError at dispatch time. ALL framework handlers MUST explicitly
subclass `ConversationToolHandler`.

Mutation contract: handlers MAY mutate `tool.metadata` in `enrich_before_send`;
MUST NOT add dynamic attributes to `tool`; MUST NOT mutate `ctx.prior_context`
(it's `MappingProxyType`-wrapped — top-level mutation raises `TypeError`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ClassVar,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
)

from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversation_tools import (
    ConversationTool,
    ConversationToolType,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    PromptRenderer,
    ToolExecutorCallable,
)
from agent_foundation.common.ui.input_modes import InputModeConfig
from agent_foundation.common.ui.interactive_base import InteractiveBase
from agent_foundation.resources.tools.models import ToolDefinition

if TYPE_CHECKING:
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.handler_registry import (
        ConversationToolHandlerRegistry,
    )


@dataclass(frozen=True)
class HandlerContext:
    """Read-only-ish bag of dependencies passed to every handler call.

    Constructed fresh per-tool. `prior_context` is a `MappingProxyType` LIVE
    VIEW of the inferencer's actual prior_context dict (NOT a snapshot copy).
    Top-level mutation through ctx raises TypeError; nested dicts/lists remain
    mutable. Handlers should communicate exclusively via returned effects.
    """

    prior_context: Mapping[str, Any]
    prompt_renderer: PromptRenderer | None
    tool_executor: ToolExecutorCallable | None
    interactive: InteractiveBase | None
    # Bundle context (None on standalone single-tool path):
    action_tools: list[dict[str, Any]] | None
    tool_registry: dict[str, ToolDefinition] | None
    resolve_tool_name: Callable[[str], str] | None
    # Optional (Phase A2 additions):
    # `session_root`: for composite typed inputs (`finalize_input_value` in
    # SingleChoice/Clarification handlers). None on paths without a session root.
    # `handler_registry`: for the module-level `_build_input_mode(tool, ctx)` to
    # dispatch through `ctx.handler_registry.require(tool.tool_type)`. Handlers
    # themselves never read this — only the top-level dispatcher does.
    session_root: str | None = None
    handler_registry: ConversationToolHandlerRegistry | None = None


class HandlerResultMergeConflict(Exception):
    """Raised when two handlers in a multi-tool bundle produce conflicting effects.

    Today no production bundle produces overlapping effects; raise enforces
    deterministic forward safety. Future bundles must explicitly resolve.
    """

    def __init__(
        self,
        field_name: str,
        key: str | None,
        existing: Any,
        new: Any,
    ) -> None:
        self.field_name = field_name
        self.key = key
        self.existing = existing
        self.new = new
        super().__init__(
            f"HandlerResult merge conflict on {field_name!r}"
            + (f"[{key!r}]" if key is not None else "")
            + f": existing={existing!r}, new={new!r}"
        )


@dataclass
class WidgetMailboxes:
    """What a widget answer's effects leave for the code that continues after
    the answer: parameter overrides for the answer's bundled action tools, turn
    variables, and dashboard directives. Each is ``None`` until an effect sets
    it; the continuation takes and clears them after every answer."""

    action_overrides: Optional[dict[str, Any]] = None
    turn_variables: Optional[dict[str, str]] = None
    dashboard_directives: Optional[dict[str, Any]] = None

    def clear(self) -> None:
        self.action_overrides = None
        self.turn_variables = None
        self.dashboard_directives = None


@runtime_checkable
class EffectTarget(Protocol):
    """The orchestrator surface widget effects act on. Implemented by both
    ``ConversationalInferencer`` and the native orchestrator."""

    prior_context: dict[str, Any]
    prompt_renderer: Any
    mailboxes: WidgetMailboxes

    def update_prior_context(self, **updates: Any) -> None: ...

    def set_session_variables(
        self, variables: dict[str, Any], *, tool_type: Optional[str] = None
    ) -> None: ...


def effect_target(target: Any) -> EffectTarget:
    """``target`` as an ``EffectTarget``.

    An object of the earlier effect-target shape — ``prior_context``,
    ``prompt_renderer``, ``set_session_variables`` and the mailboxes as
    ``_next_action_tool_overrides`` / ``_next_turn_variables`` /
    ``_next_dashboard_directives`` attributes — is wrapped in a view whose
    ``update_prior_context`` writes ``prior_context`` directly, as effects did
    before ``EffectTarget`` had that method.
    """
    if isinstance(target, EffectTarget):
        return target
    return _AttributeEffectTarget(target)


class _AttributeMailboxes:
    """``WidgetMailboxes`` kept as ``_next_*`` attributes of an object."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    @property
    def action_overrides(self) -> Optional[dict[str, Any]]:
        return self._owner._next_action_tool_overrides

    @action_overrides.setter
    def action_overrides(self, value: Optional[dict[str, Any]]) -> None:
        self._owner._next_action_tool_overrides = value

    @property
    def turn_variables(self) -> Optional[dict[str, str]]:
        return self._owner._next_turn_variables

    @turn_variables.setter
    def turn_variables(self, value: Optional[dict[str, str]]) -> None:
        self._owner._next_turn_variables = value

    @property
    def dashboard_directives(self) -> Optional[dict[str, Any]]:
        return self._owner._next_dashboard_directives

    @dashboard_directives.setter
    def dashboard_directives(self, value: Optional[dict[str, Any]]) -> None:
        self._owner._next_dashboard_directives = value

    def clear(self) -> None:
        self.action_overrides = None
        self.turn_variables = None
        self.dashboard_directives = None


class _AttributeEffectTarget:
    """``EffectTarget`` view of an object of the earlier effect-target shape."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self.mailboxes = _AttributeMailboxes(owner)

    @property
    def prior_context(self) -> dict[str, Any]:
        return self._owner.prior_context

    @property
    def prompt_renderer(self) -> Any:
        return self._owner.prompt_renderer

    def update_prior_context(self, **updates: Any) -> None:
        self._owner.prior_context.update(updates)

    def set_session_variables(
        self, variables: dict[str, Any], *, tool_type: Optional[str] = None
    ) -> None:
        self._owner.set_session_variables(variables, tool_type=tool_type)


@runtime_checkable
class InferencerEffect(Protocol):
    """A side-effect to apply to the inferencer after a handler returns.

    All inferencer-state mutations a handler wants to perform are expressed
    as concrete InferencerEffect instances. The dispatcher iterates
    `for effect in result.effects: await effect.apply(inferencer)` —
    it never references handler-specific concepts by name. Adding a new
    effect kind ships its own InferencerEffect subclass in `effects/`;
    the dispatcher needs zero changes.

    Effects are in-process objects; not picklable across process boundaries.
    """

    async def apply(self, inferencer: EffectTarget) -> None: ...


@dataclass
class HandlerResult:
    """Return value from `ConversationToolHandler.handle_response`.

    Multi-tool merge semantics (when `widget_core` decodes a bundle's answers):
    each tool's HandlerResult is applied sequentially in tool order.
    The dispatcher iterates `result.effects` and calls `effect.apply(inferencer)`.
    If two effects target the same inferencer state with conflicting values,
    the SECOND effect's `apply()` should raise `HandlerResultMergeConflict`
    (each effect class enforces its own semantics).

    `bindings` carries per-choice composite input values decoded from the
    response (e.g. ``{"input_field_name": "value"}`` from a
    ``ChoiceItem.input: InputFieldSpec``). The loop merges these into the
    aggregate ``collected`` dict alongside the primary answer text. This is a
    decode RESULT, not a side-effect — handlers stay pure functions of
    ``(tool, response, ctx)`` per Design Principle #13.
    """

    text: str = ""
    effects: list[InferencerEffect] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ConversationToolHandler(Protocol):
    """Three-method Protocol for conversation tool handlers.

    Default no-op `enrich_before_send` lives directly on the Protocol so
    concrete handlers override only what they need — but ONLY classes that
    explicitly subclass this Protocol inherit the default. See module
    docstring for the subclassing requirement.
    """

    tool_type: ClassVar[ConversationToolType]

    def build_input_mode(
        self,
        tool: ConversationTool,
        ctx: HandlerContext,
    ) -> InputModeConfig: ...

    async def enrich_before_send(
        self,
        tool: ConversationTool,
        ctx: HandlerContext,
    ) -> None:
        """Default no-op; concrete handlers override to mutate `tool.metadata`."""
        return None

    async def handle_response(
        self,
        tool: ConversationTool,
        response: dict[str, Any],
        ctx: HandlerContext,
    ) -> HandlerResult: ...
