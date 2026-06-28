
# pyre-strict

"""Common utilities for Claude Code inferencers.

Claude Code CLI expects dash-separated version numbers in model tags
(e.g. ``claude-opus-4-6``), while other systems like Devmate use
dot-separated versions (e.g. ``claude-opus-4.6``) and the Anthropic
API uses full date-qualified identifiers (e.g. ``claude-opus-4-6-20260204``
from ``ClaudeModels``).

This module provides ``resolve_model_tag`` to normalize any model tag
string into the dash-separated format expected by Claude Code, plus
shared Literal type aliases for ``--effort`` and ``--permission-mode``.
"""

import os
import re
from typing import Literal, Optional

# Reasoning-effort levels accepted by ``claude --effort <level>`` (CLI
# v2.1.119+). Higher levels allocate more thinking budget to the model
# at the cost of latency and tokens.
EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]

# Permission modes accepted by ``claude --permission-mode <mode>`` (CLI
# v2.1.119+) and the equivalent ``ClaudeAgentOptions.permission_mode``
# field. Note: the SDK's own Literal is narrower (no ``auto`` / ``dontAsk``)
# but the SDK transport just stringifies and appends, so the wider CLI set
# works at runtime — we route the extras through ``extra_args`` for type
# cleanliness when calling the SDK.
PermissionModeLiteral = Literal[
    "default", "acceptEdits", "plan", "auto", "dontAsk", "bypassPermissions"
]

# Subset of ``EffortLevel`` that the installed claude-agent-sdk's typed
# ``effort`` field accepts. Values outside this set must be passed via
# ``extra_args`` instead of the typed field.
SDK_NATIVE_EFFORT_LEVELS: frozenset[str] = frozenset(
    {"low", "medium", "high", "max"}
)

# Subset of ``PermissionModeLiteral`` that the installed claude-agent-sdk's
# typed ``permission_mode`` field accepts.
SDK_NATIVE_PERMISSION_MODES: frozenset[str] = frozenset(
    {"default", "acceptEdits", "plan", "bypassPermissions"}
)

# --- macOS seatbelt sandbox toggle -----------------------------------------
# Bare flag name (no leading dashes) for the Meta ``claude`` launcher option
# that disables the macOS seatbelt sandbox. The CLI inferencer prepends
# ``--`` to build the CLI token; the SDK inferencer passes the bare name as an
# ``extra_args`` key (value ``None`` ⇒ a value-less boolean flag).
DANGEROUSLY_DISABLE_OSX_SANDBOX = "dangerously-disable-osx-sandbox"

# Environment variable that toggles the above, globally, for ALL Claude Code
# inferencers (CLI + SDK) that don't set ``disable_osx_sandbox`` explicitly.
# (Name spelling is intentional — matches the operator-set variable.)
ENV_DISABLE_OSX_SANDBOX = "CLAUDE_CODE_INFERANCER_NO_SANDBOX"

# Values (case-insensitive, surrounding whitespace stripped) that count as
# "on" for a boolean environment flag.
_TRUTHY_ENV_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})


def env_flag_enabled(name: str) -> bool:
    """Return ``True`` iff env var ``name`` is set to a truthy value.

    Truthy = one of ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive,
    whitespace-stripped). Unset, empty, or any other value ⇒ ``False``.
    """
    val = os.environ.get(name)
    return val is not None and val.strip().lower() in _TRUTHY_ENV_VALUES


def resolve_disable_osx_sandbox(explicit: Optional[bool]) -> bool:
    """Resolve whether to pass ``--dangerously-disable-osx-sandbox`` to ``claude``.

    Precedence: an explicit constructor value (``True`` / ``False``) always
    wins; ``None`` (the default) falls back to the
    ``CLAUDE_CODE_INFERANCER_NO_SANDBOX`` environment variable.

    Why this exists: the Meta ``claude`` launcher wraps each agent in a macOS
    seatbelt sandbox. macOS forbids *nesting* seatbelt sandboxes, so when
    ``claude`` runs inside an already-sandboxed process its ``sandbox_apply``
    fails and the process exits 71 ("Operation not permitted") with no output.
    Passing this flag skips the (failing) nested sandbox apply.

    SECURITY: enabling this removes the OS-level confinement of the agent's
    file access. Only enable it in trusted contexts where an outer sandbox (or
    other isolation) is already in force.
    """
    if explicit is not None:
        return bool(explicit)
    return env_flag_enabled(ENV_DISABLE_OSX_SANDBOX)

# Explicit mapping for Anthropic API / ClaudeModels values whose short
# alias cannot be derived by simple date-stripping + regex conversion.
# Covers legacy 3.x naming where family and version are ordered differently.
_KNOWN_ALIASES: dict[str, str] = {
    # Legacy 3.x (API: claude-{major}-{family}-{date})
    "claude-3-opus-20240229": "claude-3-opus",
    "claude-3-haiku-20240307": "claude-3-haiku",
    # Legacy 3.5 / 3.7 (API: claude-{major}-{minor}-{family}-{date})
    "claude-3-5-sonnet-20241022": "claude-3-5-sonnet",
    "claude-3-7-sonnet-20250219": "claude-3-7-sonnet",
    # Devmate plugboard names (dot → dash)
    "claude3.5-sonnet": "claude-3-5-sonnet",
    "claude3.7-sonnet": "claude-3-7-sonnet",
    "claude3.5-haiku": "claude-3-5-haiku",
    "claude3-haiku": "claude-3-haiku",
    "claude4-sonnet": "claude-sonnet-4",
}

# Pattern: trailing -YYYYMMDD (8-digit date suffix)
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")

# Pattern: digit.digit NOT followed by 3+ more digits (version separator)
_DOT_VERSION_RE = re.compile(r"(\d)\.(\d)(?!\d{3,})")


def resolve_model_tag(model_tag: str) -> str:
    """Normalize a model tag for Claude Code CLI / SDK.

    Handles three input formats:

    1. **Anthropic API / ClaudeModels** (full date-qualified)::

        claude-opus-4-6-20260204   → claude-opus-4-6
        claude-sonnet-4-5-20250929 → claude-sonnet-4-5
        claude-3-5-sonnet-20241022 → claude-3-5-sonnet

    2. **Devmate / dot-separated versions**::

        claude-opus-4.6            → claude-opus-4-6
        claude-sonnet-4.5          → claude-sonnet-4-5
        claude3.5-sonnet           → claude-3-5-sonnet

    3. **Already-correct Claude Code format** (no-op)::

        claude-opus-4-6            → claude-opus-4-6
        sonnet                     → sonnet

    Args:
        model_tag: Model tag string in any format.

    Returns:
        Model tag normalized for Claude Code CLI (dash-separated versions,
        no date suffix).
    """
    # 1. Check explicit alias table first
    if model_tag in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[model_tag]

    # 2. Strip trailing date suffix (-YYYYMMDD)
    result = _DATE_SUFFIX_RE.sub("", model_tag)

    # 3. Convert dot-separated version digits to dashes
    result = _DOT_VERSION_RE.sub(r"\1-\2", result)

    return result
