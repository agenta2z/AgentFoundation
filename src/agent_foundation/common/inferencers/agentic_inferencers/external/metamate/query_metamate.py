# pyre-strict
"""
Query Meta Mate — a simple CLI to send a query to the Metamate AI assistant.

Uses the production MetamateGraphQLClient from msl.metamate.cli to call the
real Metamate API via the engine_start_v2 GraphQL mutation.

Usage (via buck run):
    # Default query
    buck run fbcode//agent_foundation.common.inferencers.agentic_inferencers.external.metamate:query_metamate

    # Custom query
    buck run fbcode//agent_foundation.common.inferencers.agentic_inferencers.external.metamate:query_metamate -- \\
        --query "How does the ranking model work?"

    # Deep Research mode (longer, more thorough)
    buck run fbcode//agent_foundation.common.inferencers.agentic_inferencers.external.metamate:query_metamate -- \\
        --query "Research how ranking models are optimized at Meta" \\
        --deep-research --timeout 600

    # With specific agent (UPPER_SNAKE_CASE enum value)
    buck run fbcode//agent_foundation.common.inferencers.agentic_inferencers.external.metamate:query_metamate -- \\
        --query "Search for auth docs" \\
        --agent-name "METAMATE_GENERAL_AGENT"
"""

import argparse
import logging
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from msl.metamate.cli.metamate_graphql import MetamateGraphQLClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_API_KEY: str = "m8-api-86761d6a0b64"
DEFAULT_SURFACE: str = "VS_CODE"
DEFAULT_MODE: str = "AUTO"
DEFAULT_STREAM_TYPE: str = "SKYWALKER_PER_REQUEST"
DEEP_RESEARCH_AGENT: str = "SPACES_DEEP_RESEARCH_AGENT"

# Terminal statuses from the MessageStatus Thrift enum (sdk.thrift)
_TERMINAL_STATUSES: frozenset[str] = frozenset({
    "COMPLETED", "STOPPED", "TRUNCATED", "ERROR",
    "CANCELLED", "CANCELED", "FAILED", "TIMEOUT",
})

# Phrases indicating the agent is asking to confirm rather than delivering content
_CONTINUATION_PHRASES: list[str] = [
    "should i proceed", "shall i proceed", "would you like me to",
    "do you want me to", "use your own judgment", "please proceed",
    "let me know if", "want me to research", "want me to look into",
    "narrow this", "tell me your", "tell me whether", "tell me which",
    "which one are you", "are you most interested in", "i can tailor",
    "more actionable",
]

AUTO_CONTINUE_REPLY: str = (
    "Use your own judgment, please proceed with the research. "
    "Provide a comprehensive, detailed answer with all findings."
)


@dataclass
class QueryResult:
    """Parsed result from a Metamate query."""

    text: str
    code_blocks: list[dict[str, str]] = field(default_factory=list)
    conversation_uuid: str = ""
    conversation_fbid: str = ""
    elapsed_seconds: float = 0.0


def _extract_code_blocks(markdown: str) -> list[dict[str, str]]:
    """Extract fenced code blocks from markdown text."""
    blocks: list[dict[str, str]] = []
    for match in re.finditer(r"```(\w+)?\n(.*?)\n```", markdown, re.DOTALL):
        blocks.append({"language": match.group(1) or "text", "code": match.group(2)})
    return blocks


# ---------------------------------------------------------------------------
# Bridge output parsing helpers (self-contained for standalone binary)
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


def _parse_assistant_text(bridge_outputs: Any) -> str:
    """Extract text ONLY from blocks belonging to ASSISTANT messages."""
    assistant_buuids = _get_assistant_block_uuids(bridge_outputs)
    parts: list[str] = []
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
        md = getattr(content, "markdown", None)
        if md and getattr(md, "value", None):
            parts.append(md.value)
            continue
        am = getattr(content, "agent_message", None)
        if am and getattr(am, "markdown", None):
            parts.append(am.markdown)
            continue
        ams = getattr(content, "agent_message_summary", None)
        if ams and getattr(ams, "markdown", None):
            parts.append(ams.markdown)
            continue
        ts = getattr(content, "text_string", None)
        if ts and getattr(ts, "value", None):
            parts.append(ts.value)
            continue
        ir = getattr(content, "inline_reasoning", None)
        if ir and getattr(ir, "markdown_content", None):
            parts.append(ir.markdown_content)
            continue
        ci = getattr(content, "code_interpreter", None)
        if ci:
            code_parts: list[str] = []
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
        gen = getattr(content, "generic", None)
        if gen and getattr(gen, "raw_json", None):
            parts.append(f"[{getattr(gen, 'typename', 'unknown')}]")
    return "\n".join(parts)


def _assistant_message_status(bridge_outputs: Any) -> str | None:
    """Return the status string of the ASSISTANT message, or None."""
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


def _needs_continuation(text: str) -> bool:
    """Return True when the assistant asks for clarification rather than delivering content."""
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


# ---------------------------------------------------------------------------
# Polling helpers
# ---------------------------------------------------------------------------


def _poll_for_response(
    client: Any,
    conv_uuid: str,
    poll_interval: float = 3.0,
    max_wait: float = 600.0,
) -> tuple[Any, str]:
    """
    Poll ``get_conversation_for_stream`` until the assistant response is ready.

    Deep research agents may still be generating when we first call
    ``get_conversation_for_stream``.  This helper re-fetches every
    *poll_interval* seconds until the assistant message reaches a terminal
    status (COMPLETED, ERROR, …) or until *max_wait* seconds have elapsed.

    Returns:
        (bridge_outputs, parsed_text)  — text from ASSISTANT blocks only.
    """
    deadline = time.time() + max_wait
    start_time = time.time()
    attempt = 0

    while True:
        attempt += 1
        bridge_outputs = client.get_conversation_for_stream(conv_uuid)
        text = _parse_assistant_text(bridge_outputs)
        status = _assistant_message_status(bridge_outputs)

        if status and status in _TERMINAL_STATUSES and len(text.strip()) > 0:
            return bridge_outputs, text

        # Fallback: substantial assistant text means response is ready
        if len(text.strip()) > 200:
            return bridge_outputs, text

        if time.time() >= deadline:
            logger.warning(
                "Polling timed out after %.0f s (%d attempts). "
                "Assistant status=%s, content=%d chars.",
                max_wait,
                attempt,
                status,
                len(text),
            )
            return bridge_outputs, text

        logger.info(
            "Response not ready yet (assistant status=%s, %d chars, attempt %d). "
            "Retrying in %.0f s …",
            status,
            len(text),
            attempt,
            poll_interval,
        )
        # Progress print to stdout — keeps the CLI inferencer's idle timer alive.
        # MetamateCliInferencer.parse_output() ignores lines outside RESPONSE
        # delimiters, so these are harmless to the parsed result.
        print(
            f"[polling] attempt={attempt}, status={status}, "
            f"chars={len(text)}, elapsed={time.time() - start_time:.0f}s",
            flush=True,
        )
        time.sleep(poll_interval)


def query_metamate(
    query: str,
    api_key: str = DEFAULT_API_KEY,
    agent_name: Optional[str] = None,
    cat_token: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    auto_continue: bool = True,
    max_continuations: int = 5,
) -> QueryResult:
    """
    Send a query to Metamate and return the parsed response.

    When *auto_continue* is True (default) and the agent responds with a
    clarification question instead of research content, the function will
    automatically reply "Use your own judgment, please proceed with the
    research" in the same conversation and keep polling until actual content
    arrives.  This is critical for deep-research agents that sometimes ask
    for follow-up before producing their report.

    Args:
        query: The question or prompt to send.
        api_key: Metamate API key.
        agent_name: Optional specific agent name.
        cat_token: Optional CAT token (uses current user if None).
        timeout_seconds: Optional timeout for the request.
        auto_continue: Automatically reply when the agent asks for
            confirmation instead of delivering results.
        max_continuations: Max number of auto-continue follow-ups to send.

    Returns:
        QueryResult with text, code_blocks, and conversation ids.
    """
    client = MetamateGraphQLClient(cat=cat_token)
    request_id = str(uuid.uuid4())

    logger.info("Sending query to Metamate …")
    start = time.time()

    result = client.engine_start_v2(
        prompt=query,
        request_id=request_id,
        api_key=api_key,
        surface=DEFAULT_SURFACE,
        mode=DEFAULT_MODE,
        stream_type=DEFAULT_STREAM_TYPE,
        agent_name=agent_name,
        timeout_seconds=timeout_seconds,
    )

    conv_uuid: str = result.conversation.uuid
    conv_fbid: str = result.conversation.fbid

    logger.info(
        "Conversation started — uuid=%s, fbid=%s. Fetching response …",
        conv_uuid,
        conv_fbid,
    )
    # Progress print to keep CLI inferencer's idle timer alive
    print(f"[polling] Conversation started, waiting for response...", flush=True)

    # For deep-research (or any agent), poll until the response is actually
    # ready.  The first fetch may return only the echoed user query if the
    # agent is still generating.
    max_poll = float(timeout_seconds) if timeout_seconds else 600.0
    bridge_outputs, text = _poll_for_response(
        client, conv_uuid, poll_interval=3.0, max_wait=max_poll,
    )

    # ---------------------------------------------------------------
    # Auto-continue: if the agent asked a clarification question
    # rather than delivering research, reply and re-poll.
    # ---------------------------------------------------------------
    continuations_sent = 0
    while (
        auto_continue
        and _needs_continuation(text)
        and continuations_sent < max_continuations
    ):
        continuations_sent += 1
        logger.info(
            "Agent asked for clarification (turn %d/%d). "
            "Auto-replying: \"%s\"",
            continuations_sent,
            max_continuations,
            AUTO_CONTINUE_REPLY[:60] + "…",
        )
        print(
            f"[polling] Auto-continuing (turn {continuations_sent}/{max_continuations})...",
            flush=True,
        )

        follow_up_request_id = str(uuid.uuid4())
        remaining = max(30.0, max_poll - (time.time() - start))

        result = client.engine_start_v2(
            prompt=AUTO_CONTINUE_REPLY,
            request_id=follow_up_request_id,
            api_key=api_key,
            surface=DEFAULT_SURFACE,
            mode=DEFAULT_MODE,
            stream_type=DEFAULT_STREAM_TYPE,
            agent_name=agent_name,
            timeout_seconds=int(remaining) if timeout_seconds else None,
            conversation_uuid=conv_uuid,
            conversation_fbid=conv_fbid,
        )

        bridge_outputs, text = _poll_for_response(
            client, conv_uuid, poll_interval=3.0, max_wait=remaining,
        )

    elapsed = time.time() - start

    if continuations_sent > 0:
        logger.info(
            "Auto-continued %d time(s). Final response: %d chars.",
            continuations_sent,
            len(text),
        )

    code_blocks = _extract_code_blocks(text)

    logger.info("Response received in %.1f s (%d chars).", elapsed, len(text))

    return QueryResult(
        text=text,
        code_blocks=code_blocks,
        conversation_uuid=conv_uuid,
        conversation_fbid=conv_fbid,
        elapsed_seconds=elapsed,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a query to Meta Mate and print the response.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are latest Meta Devmate features?",
        help="The question to ask Meta Mate.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="Metamate API key.",
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default=None,
        help=(
            "Optional agent name from XFBMetamateEngineAgentName enum "
            "(e.g. METAMATE_GENERAL_AGENT, ANALYTICS_AGENT)."
        ),
    )
    parser.add_argument(
        "--deep-research",
        action="store_true",
        default=False,
        help=(
            "Enable Deep Research mode (uses SPACES_DEEP_RESEARCH_AGENT). "
            "Produces longer, more thorough responses with citations. "
            "Consider using --timeout 600 with this flag."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for the request.",
    )
    return parser


def main() -> None:
    """CLI entry-point."""
    args = _build_parser().parse_args()

    agent_name: Optional[str] = args.agent_name
    if args.deep_research and agent_name is None:
        agent_name = DEEP_RESEARCH_AGENT

    print("=" * 72)
    print(f"  Query : ({len(args.query)} chars)")
    if agent_name:
        print(f"  Agent : {agent_name}")
    if args.deep_research:
        print("  Mode  : Deep Research")
    print("=" * 72)
    print()

    try:
        result = query_metamate(
            query=args.query,
            api_key=args.api_key,
            agent_name=agent_name,
            timeout_seconds=args.timeout,
        )
    except Exception as exc:
        logger.error("Metamate query failed: %s", exc)
        sys.exit(1)

    print("-" * 72)
    print("RESPONSE")
    print("-" * 72)
    print(result.text if result.text else "(empty response)")
    print()

    if result.code_blocks:
        print(f"Code blocks extracted: {len(result.code_blocks)}")
        for i, block in enumerate(result.code_blocks, 1):
            print(f"  [{i}] {block['language']} — {len(block['code'])} chars")
        print()

    print("-" * 72)
    print(f"  Conversation UUID : {result.conversation_uuid}")
    print(f"  Conversation FBID : {result.conversation_fbid}")
    print(f"  Elapsed           : {result.elapsed_seconds:.1f} s")
    print("-" * 72)


if __name__ == "__main__":
    main()
