"""Example (M3/M6 — diamond fan-out isolation): N concurrent branches on ONE
shared instance keep isolated per-branch state + live handles (V8).

This is the property that makes a shared orchestrator instance safe under
parallel fan-out: each ``ctx.child(f"worker_{i}")`` branch owns its own context
node + Tier-3 handles, so concurrent workers never collide. No credentials.
"""

from __future__ import annotations

import asyncio

from agent_foundation.common.inferencers.run_context import (
    RunContext,
    active_run_context,
    enter_run,
    exit_run,
)


def main() -> None:
    root = RunContext.root(workspace=None)
    n = 8

    async def worker(i: int) -> tuple[str, str]:
        # Each gathered coroutine binds its OWN per-input child context.
        tok = enter_run(root.child(f"worker_{i}"))
        try:
            await asyncio.sleep(0)  # yield so tasks interleave
            ctx = active_run_context()
            ctx.handles.live_session_id = f"sess-{i}"
            ctx.node().call = {"i": i}
            await asyncio.sleep(0)
            # After interleaving, this task STILL sees its own branch — no clobber.
            return ctx.path, ctx.handles.live_session_id
        finally:
            exit_run(tok)

    async def _run_all():
        return await asyncio.gather(*(worker(i) for i in range(n)))

    results = asyncio.run(_run_all())
    for i, (path, sid) in enumerate(results):
        assert path == f"/worker_{i}", (i, path)
        assert sid == f"sess-{i}", (i, sid)
    # Every branch's state persisted independently in the shared store.
    for i in range(n):
        assert root._store.node(f"/worker_{i}").call == {"i": i}
    assert active_run_context() is None  # nothing leaked
    print(f"parallel-isolation example OK: {n} branches, no cross-talk")


if __name__ == "__main__":
    main()
