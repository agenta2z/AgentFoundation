"""Claude Code command-hook relay (stdlib only), run by ``claude -p`` hooks.

Claude Code runs a command hook with the hook input as JSON on stdin and
reads the decision as JSON from stdout. This relay forwards the input to the
AgentFoundation process that owns the turn (``AF_HOOK_URL``, authenticated by
``AF_HOOK_TOKEN`` — both inherited from the ``claude`` process environment, so
no secret is on any command line) and prints the decision it returns.

It fails closed: when the host cannot be reached or does not answer with a
JSON object, the relay prints the decision that keeps AF in control of the
turn instead of letting the hook pass silently — an AF tool call is denied
(``PreToolUse``), the turn stops after an AF tool ran (``PostToolUse``), and
the prompt is blocked rather than sent without its turn context
(``UserPromptSubmit``). Other events only report the failure on stderr.

Usage (configured by the Claude CLI backend):  python3 af_hook.py
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Optional

# Below the hook timeout the Claude CLI backend configures, so the relay
# answers (fail-closed) before Claude Code gives up on the hook, which it
# would treat as a non-blocking error.
TIMEOUT_S = 30.0
_AF_PREFIX = "mcp__af__"


def fail_closed_output(payload: dict[str, Any], problem: str) -> Optional[dict]:
    """The hook decision that keeps AF in control when the host is not reachable."""
    event = payload.get("hook_event_name", "")
    is_af_tool = str(payload.get("tool_name", "")).startswith(_AF_PREFIX)
    detail = f"The AgentFoundation host did not answer the {event} hook ({problem})."
    if event == "PreToolUse" and is_af_tool:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"{detail} AgentFoundation tools are "
                "unavailable; do not retry them in this turn.",
            }
        }
    if event == "PostToolUse" and is_af_tool:
        return {"continue": False, "stopReason": f"{detail} The turn was stopped."}
    if event == "UserPromptSubmit":
        return {
            "decision": "block",
            "reason": f"{detail} The message was not sent without its context.",
        }
    return None


def _parse(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw or b"{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _ask_host(url: str, token: str, payload: bytes) -> dict[str, Any]:
    """The host's decision; raises when there is none."""
    request = urllib.request.Request(
        url,
        data=payload or b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    # The host is on loopback: never route through an environment proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=TIMEOUT_S) as response:
        body = response.read()
    decision = json.loads(body or b"{}")
    if not isinstance(decision, dict):
        raise ValueError("host reply is not a JSON object")
    return decision


def main() -> int:
    url = os.environ.get("AF_HOOK_URL", "")
    token = os.environ.get("AF_HOOK_TOKEN", "")
    raw = sys.stdin.buffer.read()
    try:
        if not url or not token:
            raise ValueError("AF_HOOK_URL / AF_HOOK_TOKEN are not set")
        decision = _ask_host(url, token, raw)
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        OSError,
        ValueError,
    ) as exc:
        problem = str(getattr(exc, "reason", None) or exc)
        sys.stderr.write(f"af_hook: {problem}\n")
        decision = fail_closed_output(_parse(raw), problem) or {}
    if decision:
        sys.stdout.write(json.dumps(decision))
    return 0


if __name__ == "__main__":
    sys.exit(main())
