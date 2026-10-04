# pyre-strict
"""MetaMate Inferencer — shared constants, enums, and parsing utilities.

Provides:
- API defaults (key, surface, mode, stream type)
- ``MetamateAgent`` enum for agent selection
- ``parse_assistant_text()`` — extract text from BridgeOutput list
- ``get_assistant_message_status()`` — extract terminal status
- ``needs_continuation()`` — detect clarification questions
- ``resolve_conversation_fbid()`` — multi-turn FBID lookup
- ``summarize_tool_activity()`` / ``http_error_details()`` — the diagnostics a
  stream call records when it ends (``MetamateConversationSummary``)
- ``tool_call_budget_directive()`` — the per-turn work budget appended to a task

Uses ``getattr()`` duck-typing throughout, consistent with
``query_metamate.py`` and avoiding direct ``sdk_types`` imports.
"""

import enum
import logging
import os
from typing import Any, List, Optional, Type

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client-class resolver (upstream vs metamate_standalone)
# ---------------------------------------------------------------------------
# The MetaMate GraphQL client can be sourced from either:
#   - Upstream: ``msl.metamate.cli.metamate_graphql.MetamateGraphQLClient``
#     (the canonical //msl/metamate/cli:metamate_graphql library)
#   - Standalone: ``metamate_standalone.cli.metamate_graphql.MetamateGraphQLClient``
#     (the extracted //metamate_standalone/cli:metamate_graphql library)
#
# The two are API-compatible (the standalone is a verbatim copy with the
# thrift namespace re-anchored to ``metamate_standalone.sdk``). Selection
# is controlled by, in priority order:
#   1. ``use_standalone=True`` passed to ``MetamateSDKInferencer.__init__``
#   2. ``METAMATE_USE_STANDALONE=1`` environment variable
#   3. Default: upstream
#
# Binaries that opt into the standalone MUST add
# ``//metamate_standalone:metamate_standalone`` (or directly
# ``//metamate_standalone/cli:metamate_graphql``) to their Buck deps —
# the lazy import inside ``resolve_metamate_client_cls`` will otherwise
# raise ``ImportError`` at call time, not build time.

ENV_USE_STANDALONE: str = "METAMATE_USE_STANDALONE"


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def resolve_metamate_client_cls(use_standalone: Optional[bool] = None) -> Type[Any]:
    """Return the active ``MetamateGraphQLClient`` class.

    Args:
        use_standalone: When True, force the standalone import. When False,
            force the upstream import. When None (default), check the
            ``METAMATE_USE_STANDALONE`` env var; absent/falsy → upstream.

    Returns:
        The ``MetamateGraphQLClient`` class object (NOT an instance).

    Raises:
        RuntimeError: If the requested module isn't available (typical
            cause: the Buck dep wasn't declared on the consuming binary).
    """
    if use_standalone is None:
        use_standalone = _env_truthy(ENV_USE_STANDALONE)
    module_path = (
        "metamate_standalone.cli.metamate_graphql"
        if use_standalone
        else "msl.metamate.cli.metamate_graphql"
    )
    try:
        from importlib import import_module

        mod = import_module(module_path)
    except ImportError as e:
        # U2d: raise a TYPED dependency error (not a generic RuntimeError) so the
        # retry runner classifies it non-retryable. Call-time import avoids any
        # module-load cycle (inferencer_base is always loaded by this point).
        from agent_foundation.common.inferencers.inferencer_base import (
            MissingDependencyError,
        )

        flavor = "standalone" if use_standalone else "upstream"
        raise MissingDependencyError(
            f"MetaMate {flavor} client not available: {e}. "
            f"Ensure the binary's Buck deps include "
            f"{'//metamate_standalone:metamate_standalone' if use_standalone else '//msl/metamate/cli:metamate_graphql'}."
        ) from e
    return mod.MetamateGraphQLClient


# ---------------------------------------------------------------------------
# API defaults (mirror query_metamate.py)
# ---------------------------------------------------------------------------
DEFAULT_API_KEY: str = "m8-api-86761d6a0b64"
DEFAULT_SURFACE: str = "VS_CODE"
DEFAULT_MODE: str = "AUTO"
DEFAULT_STREAM_TYPE: str = "SKYWALKER_PER_REQUEST"
DEFAULT_POLL_INTERVAL: float = 3.0
DEFAULT_TIMEOUT: int = 600
MAX_CONTINUATIONS: int = 5

AUTO_CONTINUE_REPLY: str = (
    "Use your own judgment, please proceed with the research. "
    "Provide a comprehensive, detailed answer with all findings."
)

# Terminal statuses from the MessageStatus Thrift enum
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "COMPLETED",
        "STOPPED",
        "TRUNCATED",
        "ERROR",
        "CANCELLED",
        "CANCELED",
        "FAILED",
        "TIMEOUT",
    }
)

# Phrases that indicate the agent is asking the user to confirm/proceed
_CONTINUATION_PHRASES: List[str] = [
    "should i proceed",
    "shall i proceed",
    "would you like me to",
    "do you want me to",
    "use your own judgment",
    "please proceed",
    "let me know if",
    "want me to research",
    "want me to look into",
    "narrow this",
    "tell me your",
    "tell me whether",
    "tell me which",
    "which one are you",
    "are you most interested in",
    "i can tailor",
    "more actionable",
    # Deep-Research / "think longer" gatekeeper stubs. When a deep-research
    # template is sent to ``engine_start_v2`` (which has no ``force_async``
    # plumbing), the Metamate server replies with a button-stub like
    # "If you click Deep Research, ...". These phrases let
    # ``auto_continue`` send the canonical "please proceed" reply and
    # squeeze a real synthesis out of the same conversation instead of
    # giving up after the first turn.
    "deep research button",
    "click deep research",
    "click the deep research",
    "think longer for a better",
    "if you proceed without deep research",
    "if you click",
    "if you want me to run it",
    "if you proceed",
]


class MetamateAgent(str, enum.Enum):
    """Well-known MetaMate agent names."""

    DEFAULT = "DEFAULT"
    DEEP_RESEARCH = "SPACES_DEEP_RESEARCH_AGENT"
    METAMATE_MDR = "METAMATE_MDR"


# ---------------------------------------------------------------------------
# BridgeOutput parsing helpers (getattr duck-typing)
# ---------------------------------------------------------------------------


def _get_assistant_block_uuids(bridge_outputs: Any) -> set[str]:
    """Return the set of block UUIDs belonging to ASSISTANT messages."""
    uuids: set[str] = set()
    for output in bridge_outputs:
        msg = getattr(output, "message", None)
        if msg is None:
            continue
        role = str(getattr(msg, "role", "")).upper()
        if role == "ASSISTANT":
            for bu in getattr(msg, "block_uuids", []):
                uuids.add(bu)
    return uuids


def parse_assistant_text(bridge_outputs: Any) -> str:
    """Extract text ONLY from blocks belonging to ASSISTANT messages.

    Handles the following content types via ``getattr()`` duck-typing:
    - ``markdown.value``
    - ``agent_message.markdown``
    - ``agent_message_summary.markdown`` (field is optional)
    - ``text_string.value``
    - ``inline_reasoning.content`` (per Thrift ``BlockContentInlineReasoning``)
    - ``code_interpreter`` (code + output + summary)

    Args:
        bridge_outputs: List of BridgeOutput objects from
            ``get_conversation_for_stream()``.

    Returns:
        Concatenated assistant text.
    """
    assistant_buuids = _get_assistant_block_uuids(bridge_outputs)
    parts: List[str] = []

    for output in bridge_outputs:
        block = getattr(output, "block", None)
        if block is None:
            continue
        block_uuid = getattr(block, "uuid", None)
        if block_uuid and block_uuid not in assistant_buuids:
            continue
        content = getattr(block, "content", None)
        if content is None:
            continue

        # markdown.value
        md = getattr(content, "markdown", None)
        if md and getattr(md, "value", None):
            parts.append(md.value)
            continue

        # agent_message.markdown
        am = getattr(content, "agent_message", None)
        if am and getattr(am, "markdown", None):
            parts.append(am.markdown)
            continue

        # agent_message_summary.markdown (optional field)
        ams = getattr(content, "agent_message_summary", None)
        if ams and getattr(ams, "markdown", None):
            parts.append(ams.markdown)
            continue

        # text_string.value
        ts = getattr(content, "text_string", None)
        if ts and getattr(ts, "value", None):
            parts.append(ts.value)
            continue

        # inline_reasoning.content (Thrift: BlockContentInlineReasoning.content)
        ir = getattr(content, "inline_reasoning", None)
        if ir and getattr(ir, "content", None):
            parts.append(ir.content)
            continue

        # code_interpreter (code + output + summary)
        ci = getattr(content, "code_interpreter", None)
        if ci:
            code_parts: List[str] = []
            if getattr(ci, "code", None):
                lang = getattr(ci, "language", "text")
                code_parts.append(f"```{lang}\n{ci.code}\n```")
            if getattr(ci, "output", None):
                code_parts.append(ci.output)
            if getattr(ci, "summary", None):
                code_parts.append(ci.summary)
            if code_parts:
                parts.append("\n".join(code_parts))
                continue

    return "\n".join(parts)


def get_assistant_message_status(bridge_outputs: Any) -> Optional[str]:
    """Return the terminal status string of the ASSISTANT message, or None.

    Args:
        bridge_outputs: List of BridgeOutput objects.

    Returns:
        Upper-case status string (e.g. ``"COMPLETED"``) or ``None``.
    """
    for output in bridge_outputs:
        msg = getattr(output, "message", None)
        if msg is None:
            continue
        role = str(getattr(msg, "role", "")).upper()
        if role == "ASSISTANT":
            status = getattr(msg, "status", None)
            if status is not None:
                return str(status).split(".")[-1].upper()
    return None


def needs_continuation(text: str) -> bool:
    """Return True when the assistant's response is a clarification question.

    Heuristics:
    1. Short text (< 2000 chars) containing a known continuation phrase.
    2. Short text (< 800 chars) ending with a question mark.

    Args:
        text: The assistant response text.

    Returns:
        Whether a continuation reply should be sent.
    """
    stripped = text.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    for phrase in _CONTINUATION_PHRASES:
        if phrase in lower and len(stripped) < 2000:
            return True
    if len(stripped) < 800 and stripped.endswith("?"):
        return True
    return False


DEFAULT_TOOL_CALL_BUDGET: int = 6


def tool_call_budget_directive(budget: int) -> str:
    """The work budget appended to a task: at most ``budget`` tool calls, then an
    answer from what was verified.

    MetaMate runs a whole agent turn inside one web request, which the server cuts
    at 120 s, and a turn also stops at its per-request memory budget; both are
    spent on tool calls. A wall-clock instruction is not actionable (the agent sees
    no clock), a count of tool calls is. On the 24 shards of the scoped fan-out
    gate this text (with 6) turned 8/24 substantive worker answers, 10 memory
    give-ups and 6 killed requests into 23/24, 1 and 0; on flow_03's full
    12.7 KB task, 0/8 substantive answers into 5/8, where the remaining kills come
    from answers so long that writing them nears 120 s
    (``scripts/metamate_prompt_ab.py``, variant B).
    """
    return (
        "\n\n## Working budget (strict)\n"
        "MetaMate cuts off any single turn after about two minutes, and anything "
        "unfinished is lost. Plan for that:\n"
        f"- Use at most {budget} tool calls in total (searches and file reads), then "
        "stop using tools.\n"
        "- Prefer one targeted search over several broad ones; read only the line "
        "ranges you need.\n"
        "- Then write your answer from what you verified, and list anything you "
        'could not check under "Not verified".\n'
        f"A complete answer built on {budget} tool calls is better than an "
        "unfinished one."
    )


def answer_format_directive(max_findings: int) -> str:
    """The answer-format cap appended after the work budget: at most
    ``max_findings`` findings of a few bullets each.

    Writing the answer counts against the same 120 s as the tool calls. MetaMate
    ignores a word count (asked for under 2,000 words, it wrote 3.4–4.5 K) but
    keeps to a structure. On the 24 shards of the scoped fan-out gate, after the
    6-call budget, this text (with 8) gave 24/24 substantive worker answers
    against 19/24, at a median 40 s against 70 s and with more citations; on
    flow_03's full task it turned killed requests into early give-ups instead
    (``scripts/metamate_prompt_ab.py``, variant E).
    """
    return (
        "\nAnswer format limit, since writing the answer counts against the same "
        f"two minutes: at most {max_findings} findings, each a one-line heading plus "
        "at most 4 short bullets; no tables, no code blocks, and cite each source "
        "once."
    )


_DIAGNOSTIC_TEXT_CHARS: int = 300
_HTTP_BODY_CHARS: int = 2000
_DIAGNOSTIC_HEADERS: frozenset[str] = frozenset(
    {"content-type", "content-length", "retry-after"}
)


def _clip(value: Any, limit: int = _DIAGNOSTIC_TEXT_CHARS) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def summarize_tool_activity(bridge_outputs: Any, recent: int = 8) -> dict[str, Any]:
    """The agent's tool calls in a polled conversation, read from its
    thinking-panel entries: how many blocks the conversation has, a count per
    tool, and the most recent entries with their status, summary and warning."""
    counts: dict[str, int] = {}
    entries: List[dict[str, Any]] = []
    blocks = 0
    for output in bridge_outputs or ():
        block = getattr(output, "block", None)
        if block is None:
            continue
        blocks += 1
        content = getattr(block, "content", None)
        entry = getattr(content, "thinking_panel_entry", None)
        if entry is None:
            continue
        name = getattr(entry, "tool_call_display_name", None) or "<unnamed>"
        counts[name] = counts.get(name, 0) + 1
        entries.append(
            {
                "tool": name,
                "status": str(getattr(entry, "status", "")).split(".")[-1],
                "summary": _clip(getattr(entry, "summary", None)),
                "warning": _clip(getattr(entry, "warning", None)),
            }
        )
    return {
        "blocks": blocks,
        "tool_calls": counts,
        "recent_tool_calls": entries[-recent:],
    }


def http_error_details(exc: BaseException) -> Optional[dict[str, Any]]:
    """What an HTTP error's response says (status, the ``x-fb-*`` debug headers,
    the body), or ``None`` for an error without a response. The URL drops its
    query string."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None) or {}
    elapsed = getattr(response, "elapsed", None)
    return {
        "status_code": getattr(response, "status_code", None),
        "reason": getattr(response, "reason", None),
        "url": str(getattr(response, "url", "") or "").split("?", 1)[0],
        "headers": {
            key: value
            for key, value in headers.items()
            if key.lower().startswith("x-fb") or key.lower() in _DIAGNOSTIC_HEADERS
        },
        "body": _clip(getattr(response, "text", None), _HTTP_BODY_CHARS),
        "elapsed_s": (
            round(elapsed.total_seconds(), 3)
            if hasattr(elapsed, "total_seconds")
            else None
        ),
    }


def resolve_conversation_fbid(client: Any, conversation_uuid: str) -> Optional[str]:
    """Look up conversation FBID from a conversation UUID.

    Calls ``client.get_conversation_for_stream(uuid)`` and iterates
    outputs looking for a ``conversation`` attribute with an ``fbid`` field.

    Args:
        client: ``MetamateGraphQLClient`` instance.
        conversation_uuid: The conversation UUID to look up.

    Returns:
        The conversation FBID string, or ``None`` if not found.
    """
    try:
        outputs = client.get_conversation_for_stream(conversation_uuid)
    except Exception:
        logger.warning(
            "Failed to look up conversation FBID for uuid=%s",
            conversation_uuid,
            exc_info=True,
        )
        return None

    for output in outputs:
        conv = getattr(output, "conversation", None)
        if conv is not None:
            fbid = getattr(conv, "fbid", None)
            if fbid:
                return str(fbid)
    return None
