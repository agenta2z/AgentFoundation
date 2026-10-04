"""Typed errors raised by the native conversational orchestrator."""

from __future__ import annotations


class NativeConversationError(RuntimeError):
    """Base class for native-orchestrator errors."""


class NativeCapabilityError(NativeConversationError):
    """A backend lacks a capability this configuration requires. Raised before
    any user turn is submitted."""


class StablePolicyChanged(NativeConversationError):
    """The session instructions changed for an existing vendor session and the
    configured drift policy is ``fail``."""


class RewindUnsupported(NativeConversationError):
    """The host asked to rewind to an earlier turn and the backend has no
    certified way to fork its session at that boundary."""


class SessionResumeRejected(NativeConversationError):
    """A persisted vendor session cannot be resumed under the current identity
    (backend, model, principal or permission policy changed)."""


class VendorTurnFailed(NativeConversationError):
    """The vendor reported an error for a turn."""

    def __init__(self, message: str, *, session_missing: bool = False) -> None:
        super().__init__(message)
        self.session_missing = session_missing


class VendorSessionMissing(NativeConversationError):
    """The vendor session the host asked to resume does not exist (expired or
    deleted). Raised by a backend's ``open``; nothing was submitted."""


class VendorTurnStalled(NativeConversationError):
    """The vendor produced no events for longer than the stall timeout while no
    AgentFoundation tool was running; the turn was interrupted."""


class BridgeRefused(NativeConversationError):
    """An AgentFoundation tool call was refused (no active turn, subagent
    caller, or the turn's gate is closed)."""
