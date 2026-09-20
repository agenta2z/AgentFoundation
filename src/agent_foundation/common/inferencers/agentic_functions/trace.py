# pyre-strict

"""ContextVar-based observability for agentic-function calls."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, Optional

import attr


@attr.s(auto_attribs=True)
class AgenticFunctionTrace:
    """Per-call observability record.

    Published to a per-function ``ContextVar`` at the end of each call rather
    than stored on the wrapper instance: a shared instance attribute would be
    clobbered by concurrent asyncio tasks / threads, whereas a ContextVar is
    copied per task, so ``fn.last_call`` returns that function's most recent call
    in the current context.
    """

    function: str = ""
    # "precheck" = returned deterministically; "agentic" = inference ran.
    path: str = ""
    prompt: Optional[str] = None
    raw_text: Optional[str] = None
    stage1_result: Any = None
    # Final typed return value (post-body). This is a *return* value, never a
    # call argument, so it does not weaken the no-argument-fields invariant that
    # ``test_trace_has_no_argument_fields`` guards.
    result: Any = None
    parsed: bool = False
    stages_used: tuple[str, ...] = ()
    attempts: int = 0
    errors: tuple[str, ...] = ()
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view for JSON persistence (nested attrs expand recursively)."""
        return attr.asdict(self)


def new_trace_var(name: str) -> "ContextVar[Optional[AgenticFunctionTrace]]":
    """Create a per-decorated-function ContextVar holding its last call's trace.

    One per function (created at decoration): concurrency-safe (copied per
    asyncio task / thread) and isolated per function.
    """
    return ContextVar(name, default=None)
