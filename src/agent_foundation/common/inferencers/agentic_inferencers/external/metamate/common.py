# pyre-strict
"""MetaMate Inferencer — shared constants, enums, and parsing utilities.

Provides:
- API defaults (key, surface, mode, stream type)
- ``MetamateAgent`` enum for agent selection
- ``parse_assistant_text()`` — extract text from BridgeOutput list
- ``get_assistant_message_status()`` — extract terminal status
- ``needs_continuation()`` — detect clarification questions
- ``resolve_conversation_fbid()`` — multi-turn FBID lookup

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
        flavor = "standalone" if use_standalone else "upstream"
        raise RuntimeError(
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
