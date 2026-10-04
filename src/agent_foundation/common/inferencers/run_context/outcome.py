"""Typed node outcomes, published at an invocation's successful close (plan v8 §5.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import RunContext
    from .state import NodeOutcomeState
    from .store import CreatorKey


def publish_outcome(
    ctx: RunContext, creator: CreatorKey | None, outcome: NodeOutcomeState
) -> None:
    ctx.store.publish_outcome(ctx.path, creator, outcome)


def read_outcome(ctx: RunContext) -> NodeOutcomeState | None:
    """The outcome at ``ctx.path``; never creates or claims the node."""
    node = ctx.store.peek(ctx.path)
    return None if node is None else node.outcome
