"""Slash-command argument parsing for the task tool.

Extracted from OpenStartup's manager_websocket_routes.py so the task
tool (now in AgentFoundation) doesn't depend on the server routes.
"""

from __future__ import annotations

import shlex as _shlex
from typing import Any

_REPEATABLE_KEYS = {"override"}

TASK_BOOL_FLAGS = {
    "plan", "execute", "full", "confirm",
    "no_dual", "no_aggregate", "analysis", "multi_iter",
    "in_place", "copy_workspace",
}

TASK_MODE_ALIASES = {
    "task_plan": "plan",
    "task_execute": "execute",
    "task_full": "full",
    "task_confirm": "confirm",
}


def parse_slash_args(
    args_str: str, bool_flags: set[str] = frozenset()
) -> dict[str, Any]:
    """Parse ``--key value`` pairs + bare ``--flag`` + positional ``request``.

    Rules:
      - known bool-flag → {key: True}, advance 1
      - next token starts with ``--`` → bare flag, advance 1
      - otherwise consume next token as value
    Repeated keys in ``_REPEATABLE_KEYS`` accumulate into a list.
    All unconsumed non-flag tokens become the positional ``request``.
    """
    result: dict[str, Any] = {}
    try:
        parts = _shlex.split(args_str, posix=True)
    except ValueError:
        parts = args_str.split()
    consumed: set[int] = set()
    i = 0
    while i < len(parts):
        if parts[i].startswith("--"):
            key = parts[i].lstrip("-").replace("-", "_")
            if key in bool_flags or i + 1 >= len(parts) or parts[i + 1].startswith("--"):
                result[key] = True
                consumed.add(i)
                i += 1
            else:
                val = parts[i + 1]
                if key in _REPEATABLE_KEYS:
                    result.setdefault(key, []).append(val)
                else:
                    result[key] = val
                consumed.update({i, i + 1})
                i += 2
        else:
            i += 1
    positional = [
        parts[j] for j in range(len(parts))
        if j not in consumed and not parts[j].startswith("--")
    ]
    if positional:
        result.setdefault("request", " ".join(positional))
    return result
