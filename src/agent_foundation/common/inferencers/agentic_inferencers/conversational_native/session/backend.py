"""Backend protocol for vendor agents driven by the native orchestrator.

A backend owns one vendor session (Claude Code SDK client, a per-turn CLI
process, a remote conversation). It never parses model text for tool calls:
AgentFoundation tools reach the agent as native (MCP) tools, and the backend
reports what happened as vendor-neutral ``VendorEvent``s.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Protocol

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    NativeCapabilityError,
    NativeConversationError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    VendorEvent,
)

logger: logging.Logger = logging.getLogger(__name__)


class InterruptNotAcknowledged(NativeConversationError):
    """Raised by ``interrupt()`` when the vendor did not confirm that the turn
    stopped (the request failed, or the vendor cannot cancel it); the
    interrupted turn is then recorded ``uncertain``, not ``interrupted``."""


class Evidence(str, enum.Enum):
    VERIFIED = "verified"  # exercised by a real-backend test
    EXPERIMENTAL = "experimental"  # documented / flag present, not yet exercised
    UNSUPPORTED = "unsupported"


class L2Channel(str, enum.Enum):
    """How per-turn host context reaches the model."""

    HOOK = "hook"  # UserPromptSubmit additionalContext (not user content)
    INLINE_CONFIG = "inline_config"  # per-request agent config (Metamate)
    ENVELOPE = "envelope"  # delimited block before the raw user text


class CallerTools(str, enum.Enum):
    INPROCESS = "inprocess"  # in-process MCP server (Claude Agent SDK)
    HTTP = "http"  # localhost streamable-HTTP MCP server
    SOCKET = "socket"  # unix-domain-socket MCP server (Devmate dm)
    NONE = "none"  # backend cannot host caller tools


_L2_ROUTES = {
    L2Channel.HOOK: "a UserPromptSubmit hook's additionalContext (not user text)",
    L2Channel.INLINE_CONFIG: "the per-request agent config",
    L2Channel.ENVELOPE: (
        "a labelled <af_context> envelope at the head of the user message"
    ),
}
_L3_ROUTES = {
    CallerTools.INPROCESS: "in-process MCP server `af`",
    CallerTools.HTTP: "localhost HTTP MCP server `af`",
    CallerTools.SOCKET: "unix-socket MCP server `af`",
}
_USER_ROUTES = {
    L2Channel.ENVELOPE: "the user message after the envelope (when one is sent)",
}


@dataclass(frozen=True)
class PromptRoutes:
    """How each instruction lane reaches the model on one turn ("View
    Prompt" labels the turn's prompt manifest with these)."""

    l1: str  # session instructions
    l2: str  # turn context
    l3: str  # state updates
    user: str  # the user's text


@dataclass(frozen=True)
class BackendCapabilities:
    kind: str
    caller_tools: CallerTools
    l2_channels: tuple[L2Channel, ...]
    pinned_session_id: bool
    exact_fork: bool
    turn_stop_hook: bool
    subagent_attribution: bool
    compaction_signal: bool
    persistent_process: bool
    slash_passthrough: tuple[str, ...] = ()
    # The vendor spills oversized AF tool results itself (Claude Code honours
    # the ``maxResultSizeChars`` tool annotation); the bridge then does not
    # truncate, so exactly one layer owns sizing.
    owns_result_spill: bool = False
    # ``NativeBackendSpec.environment`` values the backend can honour.
    environments: tuple[str, ...] = ("inherit",)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    # ``evidence`` keys of the capabilities this backend uses, every one its
    # flags claim included; ``require_spec`` checks them and a configured
    # non-inherit environment.
    relies_on: tuple[str, ...] = ()
    # Vendor component -> version the ``evidence`` was gathered against.
    tested_versions: dict[str, str] = field(default_factory=dict)
    # The vendor carrier of the session instructions (L1), and wording for an
    # L2 channel whose vendor carrier differs from the generic description.
    l1_route: str = ""
    l2_routes: dict[L2Channel, str] = field(default_factory=dict)

    def prompt_routes(self, channel: L2Channel) -> PromptRoutes:
        """Each lane's route on this backend when L2 travels by ``channel``."""
        server = _L3_ROUTES.get(self.caller_tools)
        l3 = "none: this backend has no AF tools"
        if server:
            l3 = (
                "appended to the result of an AF tool call that changes the "
                f"SOP state ({server})"
            )
        user = _USER_ROUTES.get(channel, "the user message")
        return PromptRoutes(
            l1=self.l1_route or "not declared by this backend",
            l2=self.l2_routes.get(channel) or _L2_ROUTES[channel],
            l3=l3,
            user=f"{user}, verbatim (host-context tags neutralized)",
        )

    def require_environment(self, environment: str) -> None:
        """Refuse, at construction, an environment the backend would ignore."""
        if environment not in self.environments:
            raise NativeCapabilityError(
                f"The {self.kind} backend cannot run with environment "
                f"{environment!r}; it supports: {', '.join(self.environments)}."
            )

    def require(self, feature: str, *, allow_experimental: bool = False) -> None:
        """Refuse a capability whose evidence does not back its use: always
        when unsupported or untested, when experimental unless the
        configuration opts in."""
        evidence = self.evidence.get(feature)
        if evidence is Evidence.VERIFIED:
            return
        if evidence is Evidence.EXPERIMENTAL and allow_experimental:
            logger.warning(
                "The %s backend uses %r, whose evidence is experimental "
                "(allow_experimental is set).",
                self.kind,
                feature,
            )
            return
        hint = ""
        if evidence is Evidence.EXPERIMENTAL:
            hint = " Set allow_experimental: true in the backend spec to use it anyway."
        state = evidence.value if evidence is not None else "missing"
        raise NativeCapabilityError(
            f"The {self.kind} backend cannot use {feature!r}: its evidence is "
            f"{state}, not verified.{hint}"
        )

    def require_spec(self, spec: "NativeBackendSpec") -> None:
        """Refuse, at construction (before any turn reaches the vendor), a
        configuration whose capabilities are not backed by evidence: the
        environment, and every capability the backend relies on."""
        self.require_environment(spec.environment)
        features = list(self.relies_on)
        if spec.environment != "inherit":
            features.append(spec.environment)
        for feature in features:
            self.require(feature, allow_experimental=spec.allow_experimental)

    def preferred_l2_channel(self, envelope_allowed: bool) -> Optional[L2Channel]:
        for channel in self.l2_channels:
            if channel is L2Channel.ENVELOPE and not envelope_allowed:
                continue
            return channel
        return None


@dataclass
class NativeBackendSpec:
    """Declarative backend configuration (YAML-instantiable)."""

    kind: str = "claude_sdk"
    model: str = ""
    cwd: str = ""
    permission_mode: str = "bypassPermissions"
    effort: Optional[str] = None
    environment: str = "inherit"  # inherit | hermetic
    # Use capabilities whose evidence is only experimental (never unsupported
    # ones); without it such a configuration fails at construction.
    allow_experimental: bool = False
    l2_envelope_allowed: bool = False
    cli_path: Optional[str] = None
    mcp_tool_timeout_ms: int = 7_200_000
    disable_osx_sandbox: Optional[bool] = None
    # Additional MCP servers attached next to ``af`` (name -> the vendor's
    # server config); in ``inherit`` mode the user's own servers also stay.
    extra_mcp_servers: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_inferencer(cls, definition: Any, **overrides: Any) -> "NativeBackendSpec":
        """Map an existing backend inferencer definition — a config mapping
        (``_target_: ClaudeCodeCLI`` ...) or a live inferencer instance — to the
        native backend spec of the same vendor. The definition is only read."""
        if isinstance(definition, dict):
            target = str(definition.get("_target_", ""))
            read = definition.get
        else:
            target = type(definition).__name__
            read = lambda key, default=None: getattr(definition, key, default)  # noqa: E731
        kind = _INFERENCER_KINDS.get(target.rsplit(".", 1)[-1])
        if kind is None:
            raise ValueError(
                f"No native backend for inferencer {target or type(definition).__name__!r}; "
                f"known: {sorted(_INFERENCER_KINDS)}"
            )
        values: dict[str, Any] = {"kind": kind}
        for source, dest in _INFERENCER_FIELDS:
            value = read(source, None)
            if value is not None and dest not in values:
                values[dest] = value
        values.update(overrides)
        return cls(**values)


# Existing inferencer classes / config aliases -> native backend kind.
_INFERENCER_KINDS = {
    "ClaudeCodeSDK": "claude_sdk",
    "ClaudeCodeSdkInferencer": "claude_sdk",
    "ClaudeCodeCLI": "claude_cli",
    "ClaudeCodeCliInferencer": "claude_cli",
    "DevmateCLI": "devmate_dm",
    "Devmate": "devmate_dm",
    "DevmateCliInferencer": "devmate_dm",
    "CodexCLI": "codex_cli",
    "CodexCliInferencer": "codex_cli",
    "MetamateSDK": "metamate",
    "Metamate": "metamate",
    "MetamateSDKInferencer": "metamate",
}
# (inferencer attribute, spec field), first match wins per field.
_INFERENCER_FIELDS = (
    ("model_name", "model"),
    ("model_id", "model"),
    ("target_path", "cwd"),
    ("permission_mode", "permission_mode"),
    ("effort", "effort"),
    ("disable_osx_sandbox", "disable_osx_sandbox"),
)


@dataclass(frozen=True)
class BridgeToolSpec:
    """One AgentFoundation tool as the vendor sees it."""

    name: str  # unprefixed MCP tool name (``mcp__af__<name>`` to the agent)
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class SessionHooks(Protocol):
    """Callbacks a backend invokes from its vendor hooks."""

    def l2_for_turn(self) -> str: ...

    async def before_af_tool(
        self, tool_name: str, tool_use_id: str, agent_id: Optional[str]
    ) -> Optional[str]:
        """Return a deny reason, or ``None`` to allow."""
        ...

    async def after_af_tool(self, tool_name: str, tool_use_id: str) -> bool:
        """Return ``True`` when the vendor turn must stop now."""
        ...

    def on_compaction(self) -> None: ...


@dataclass
class SessionOpenRequest:
    session_id: str
    resume: bool
    l1_text: str
    l1_path: str
    tools: list[BridgeToolSpec]
    hooks: SessionHooks
    cwd: str
    model: str
    fork_from: Optional[tuple[str, str]] = None  # (session_id, up_to_message_uuid)
    # Results above this size are spilled to a file (by the vendor when it
    # owns spilling, otherwise by the bridge); 0 disables.
    result_max_chars: int = 0


@dataclass
class TurnRequest:
    text: str
    l2_text: str = ""
    channel: L2Channel = L2Channel.HOOK


class NativeSessionBackend(Protocol):
    capabilities: BackendCapabilities

    async def open(self, request: SessionOpenRequest) -> None: ...

    def run_turn(self, request: TurnRequest) -> AsyncIterator[VendorEvent]: ...

    async def interrupt(self) -> None:
        """Stop the running turn; raises ``InterruptNotAcknowledged`` when the
        vendor did not confirm the stop."""
        ...

    async def set_model(self, model: str) -> None: ...

    async def close(self) -> None: ...

    @property
    def session_id(self) -> str: ...
