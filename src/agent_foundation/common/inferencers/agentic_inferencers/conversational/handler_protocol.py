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
from typing import Any, Callable, ClassVar, Mapping, Protocol, runtime_checkable, TYPE_CHECKING

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
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
        ConversationalInferencer,
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

    async def apply(self, inferencer: ConversationalInferencer) -> None: ...


@dataclass
class HandlerResult:
    """Return value from `ConversationToolHandler.handle_response`.

    Multi-tool merge semantics (when `_handle_conversation_tools` processes a
    bundle): each tool's HandlerResult is applied sequentially in tool order.
    The dispatcher iterates `result.effects` and calls `effect.apply(inferencer)`.
    If two effects target the same inferencer state with conflicting values,
    the SECOND effect's `apply()` should raise `HandlerResultMergeConflict`
    (each effect class enforces its own semantics).
    """

    text: str = ""
    effects: list[InferencerEffect] = field(default_factory=list)


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
