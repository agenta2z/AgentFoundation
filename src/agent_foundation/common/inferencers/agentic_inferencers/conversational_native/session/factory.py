"""Explicit mapping from backend kind to backend implementation.

Backends are imported lazily so a host only needs the dependencies of the
backend it actually uses (e.g. the Claude Agent SDK, or Metamate's msl SDK).
"""

from __future__ import annotations

import copy
import importlib
from typing import Any, Union

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.errors import (
    NativeCapabilityError,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    NativeBackendSpec,
    NativeSessionBackend,
)

BACKEND_KINDS = ("claude_sdk", "claude_cli", "devmate_dm", "codex_cli", "metamate")


def normalize_spec(spec: Union[NativeBackendSpec, dict[str, Any]]) -> NativeBackendSpec:
    """A spec owned by one conversation: always a fresh copy, so per-session
    changes (``/model``, ``/root``) never leak into a shared definition."""
    if isinstance(spec, NativeBackendSpec):
        return copy.deepcopy(spec)
    if isinstance(spec, dict):
        known = set(NativeBackendSpec.__dataclass_fields__)
        unknown = set(spec) - known - {"_target_"}
        if unknown:
            raise ValueError(f"Unknown native backend option(s): {sorted(unknown)}")
        return NativeBackendSpec(
            **copy.deepcopy({k: v for k, v in spec.items() if k in known})
        )
    raise TypeError(f"Unsupported native backend spec: {type(spec).__name__}")


_BACKEND_MODULES = {
    "claude_sdk": ("claude_sdk", "ClaudeSdkBackend"),
    "claude_cli": ("claude_cli", "ClaudeCliBackend"),
    "devmate_dm": ("devmate_dm", "DevmateDmBackend"),
    "codex_cli": ("codex_cli", "CodexCliBackend"),
    "metamate": ("metamate", "MetamateBackend"),
}


def backend_class(kind: str) -> type:
    """The backend class for ``kind`` (its ``capabilities`` is a class attribute)."""
    entry = _BACKEND_MODULES.get(kind)
    if entry is None:
        raise NativeCapabilityError(
            f"No native backend for kind {kind!r}; supported: {', '.join(BACKEND_KINDS)}"
        )
    module_name, class_name = entry
    path = (
        "agent_foundation.common.inferencers.agentic_inferencers."
        f"conversational_native.session.{module_name}"
    )
    try:
        module = importlib.import_module(path)
    except ModuleNotFoundError as exc:
        if exc.name != path:
            raise
        # A backend with its own Buck target (Metamate) the host did not add.
        raise NativeCapabilityError(
            f"The {kind} backend is not part of this binary: add the Buck target "
            f"that provides {path} to the host."
        ) from exc
    return getattr(module, class_name)


def make_backend(spec: NativeBackendSpec, **runtime: Any) -> NativeSessionBackend:
    """Instantiate the backend for ``spec.kind``. ``runtime`` carries host-owned
    resources a backend may use (e.g. the runtime manager for the shared HTTP
    MCP server); backends ignore what they do not need."""
    return backend_class(spec.kind)(spec, **runtime)
