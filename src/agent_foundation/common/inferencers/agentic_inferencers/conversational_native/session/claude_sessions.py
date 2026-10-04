"""Claude Code transcript operations shared by the SDK and CLI backends.

Both run the same ``claude`` binary, which writes each session's transcript
under ``~/.claude/projects``; the Agent SDK's session helpers operate on those
files directly, so a session started by either backend can be forked by both.
"""

from __future__ import annotations

import asyncio


async def fork_claude_session(source: str, up_to_message_id: str) -> str:
    """A new session holding ``source``'s transcript up to (and including)
    ``up_to_message_id``; returns its id."""
    from claude_agent_sdk import fork_session

    result = await asyncio.to_thread(fork_session, source, None, up_to_message_id)
    return result.session_id


def claude_fork_message_map(source: str, forked: str) -> dict[str, str]:
    """Old -> new message ids after a fork: the forked transcript is the
    source's prefix in the same order, with fresh ids."""
    from claude_agent_sdk import get_session_messages

    old = [m.uuid for m in get_session_messages(source)]
    new = [m.uuid for m in get_session_messages(forked)]
    return dict(zip(old, new))
