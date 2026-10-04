"""Claude Code stream facts shared by the SDK and CLI backends (same binary)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

AF_PREFIX = "mcp__af__"

# Claude Code's local commands that do something under `-p`: it runs them
# without UserPromptSubmit and without a model call (`num_turns` 0) and reports
# their output as a `<synthetic>` assistant message (claude 2.1.289, CLI and
# SDK; `/cost` is an alias of `/usage`). The backends pass them through
# verbatim without an L2. `/help` never reaches the vendor: AF answers it.
LOCAL_COMMANDS = ("compact", "context", "cost", "usage")

# The ``model`` of the assistant message that carries a local command's output.
SYNTHETIC_MODEL = "<synthetic>"


def af_unusable(status: Any, detail: Any = None) -> str:
    """The actionable error for an AF tool server Claude Code cannot use."""
    return (
        f"The AgentFoundation MCP server 'af' is {status!r}"
        f"{': ' + str(detail) if detail else ''}, so AgentFoundation tools are "
        "unavailable. Check for a managed Claude Code policy that only allows "
        "managed MCP servers (allowManagedMcpServersOnly) or denies mcp__af__ tools."
    )


def af_status_problem(server: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Why ``af`` is unusable according to its ``get_mcp_status`` entry;
    ``None`` when it is connected and lists its tools."""
    if server is None:
        return af_unusable("not listed")
    if server.get("status") != "connected":
        return af_unusable(server.get("status"), server.get("error"))
    if not server.get("tools"):
        return af_unusable("connected", "it lists no tools")
    return None


def af_init_problem(init: Mapping[str, Any]) -> Optional[str]:
    """Why ``af`` is unusable according to a turn's ``system``/``init``
    message; ``None`` when it is connected and its tools are offered.

    Verified with claude 2.1.288: init lists ``af`` with ``connected`` (HTTP
    from the CLI, ``source: sdk`` from the Agent SDK) or ``failed`` (refused,
    bad bearer, or no answer within ``MCP_TIMEOUT``), and ``tools`` names every
    ``mcp__af__*`` tool, deferred ones included. Init comes after the
    ``UserPromptSubmit`` hooks and before the first model request."""
    server = next(
        (s for s in init.get("mcp_servers") or () if s.get("name") == "af"), None
    )
    if server is None:
        return af_unusable("not listed")
    if server.get("status") != "connected":
        return af_unusable(server.get("status"))
    if not any(str(name).startswith(AF_PREFIX) for name in init.get("tools") or ()):
        return af_unusable("connected", "it offers the model no mcp__af__ tools")
    return None


def blocked_turn_reason(system_message: Mapping[str, Any]) -> Optional[str]:
    """Claude Code's notice that a hook stopped the turn (a ``system`` /
    ``informational`` message with ``prevent_continuation``), without the
    prompt it echoes; ``None`` for any other system message.

    Verified with claude 2.1.288: a ``UserPromptSubmit`` hook that blocks
    emits this notice and then a successful ``result`` with ``num_turns == 0``
    whose text is the notice — the model never saw the prompt."""
    if system_message.get("subtype") != "informational":
        return None
    if not system_message.get("prevent_continuation"):
        return None
    content = str(system_message.get("content") or "a hook stopped the turn")
    return content.split("\n\nOriginal prompt:", 1)[0].strip()
