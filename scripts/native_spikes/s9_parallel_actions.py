"""S9 — two AF action calls in one assistant message.

The model is asked for two ``lookup_record`` calls in one message (each call
takes ``--work`` seconds). Asserted: both tool_use blocks are in the same API
message, both handlers run exactly once with their own arguments, and the
turn finishes with both results. Recorded (``[INFO]``): when the vendor
delivers each ``AssistantMessage`` (one per content block) relative to that
tool's ``PreToolUse`` hook, handler start/end and ``PostToolUse`` hook — the
ordering the native turn loop sees when it decides "stop after the last AF
call of this message" — and whether the two handlers overlap.

Agent SDK (in-process ``af`` server, Python hooks, partial messages) and
``claude -p`` (HTTP ``af`` server, command hooks, stream-json).

    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/s9_parallel_actions.py [--kinds sdk,cli] [--model sonnet]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
import uuid
from typing import Any, Optional

from _spike_common import (
    CallLog,
    Checks,
    claude_argv,
    CommandHooks,
    hook_matcher,
    HttpMcpServer,
    run_claude,
    run_sdk,
    sdk_mcp_server,
    sdk_options,
    section,
    SpikeTool,
)

PROMPT = (
    "Look up the records with keys A-1 and B-2 using the lookup_record tool. Issue both "
    "lookup_record calls together in one single message (parallel tool calls), then report "
    "both statuses."
)


def _tool(seconds: float) -> SpikeTool:
    async def _lookup(args: dict[str, Any]) -> str:
        await asyncio.sleep(seconds)
        return f"record {args.get('key')}: status=green"

    return SpikeTool(
        name="lookup_record",
        description="Look up a record in the archive by key.",
        properties={"key": {"type": "string"}},
        required=("key",),
        handler=_lookup,
    )


class Timeline:
    def __init__(self) -> None:
        self.rows: list[tuple[float, str]] = []

    def add(self, t: float, label: str) -> None:
        self.rows.append((t, label))

    def first(self, prefix: str) -> Optional[float]:
        return min(
            (t for t, label in self.rows if label.startswith(prefix)), default=None
        )

    def dump(self, c: Checks, k: str) -> None:
        if not self.rows:
            return
        t0 = min(t for t, _ in self.rows)
        for t, label in sorted(self.rows):
            c.info(f"{k} t+{t - t0:6.2f}s", label)


def _check(
    c: Checks,
    k: str,
    calls: CallLog,
    uses: list[tuple[str, str, str]],
    text: str,
    tl: Timeline,
) -> None:
    """``uses``: (message_id, tool_use_id, key) for each lookup_record tool_use."""
    keys = sorted(x.arguments.get("key", "") for x in calls.named("lookup_record"))
    c.check(f"{k} both handlers ran exactly once", keys == ["A-1", "B-2"], keys)
    messages = {m for m, _, _ in uses}
    c.check(
        f"{k} both tool_use blocks in one API message",
        len(uses) == 2 and len(messages) == 1,
        uses,
    )
    c.check(
        f"{k} the turn reports both results",
        "A-1" in text and "B-2" in text,
        text[:200],
    )
    runs = calls.named("lookup_record")
    if len(runs) == 2:
        a, b = sorted(runs, key=lambda x: x.started)
        overlap = (a.ended or a.started) > b.started
        c.info(f"{k} handlers overlap (run concurrently)", overlap)
    for _message_id, _tool_use_id, key in uses:
        msg = tl.first(f"AssistantMessage tool_use {key}")
        pre = tl.first(f"PreToolUse {key}")
        start = tl.first(f"handler start {key}")
        if msg is not None and pre is not None and start is not None:
            order = sorted(
                [(msg, "AssistantMessage"), (pre, "PreToolUse"), (start, "handler")]
            )
            c.info(f"{k} order for {key}", " < ".join(name for _, name in order))
    tl.dump(c, k)


async def sdk_spike(model: str, work_s: float, c: Checks) -> None:
    from claude_agent_sdk import (
        AssistantMessage,
        StreamEvent,
        ToolUseBlock,
        UserMessage,
    )

    k = "S9[sdk]"
    section(k)
    calls, tl = CallLog(), Timeline()
    tool = _tool(work_s)
    inner = tool.handler

    async def _traced(args: dict[str, Any]) -> str:
        tl.add(time.time(), f"handler start {args.get('key')}")
        out = await inner(args)
        tl.add(time.time(), f"handler end {args.get('key')}")
        return out

    tool.handler = _traced

    async def _hook(data: dict[str, Any], tool_use_id: Any, context: Any) -> dict:
        key = (data.get("tool_input") or {}).get("key")
        tl.add(time.time(), f"{data.get('hook_event_name')} {key}")
        return {}

    uses: list[tuple[str, str, str]] = []

    def _on(message: Any) -> None:
        now = getattr(message, "_recv", time.time())
        if isinstance(message, StreamEvent) and not message.parent_tool_use_id:
            event = message.event or {}
            block = event.get("content_block") or {}
            if (
                event.get("type") == "content_block_start"
                and block.get("type") == "tool_use"
            ):
                tl.add(
                    now,
                    f"stream content_block_start tool_use {block.get('id', '')[-6:]}",
                )
        elif isinstance(message, AssistantMessage) and not message.parent_tool_use_id:
            for block in message.content:
                if isinstance(block, ToolUseBlock) and block.name.endswith(
                    "lookup_record"
                ):
                    key = block.input.get("key", "")
                    uses.append((message.message_id or "", block.id, key))
                    tl.add(
                        now,
                        f"AssistantMessage tool_use {key} (msg {str(message.message_id)[-6:]})",
                    )
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for part in message.content:
                if getattr(part, "tool_use_id", None):
                    tl.add(now, f"UserMessage tool_result {str(part.tool_use_id)[-6:]}")

    run = await run_sdk(
        sdk_options(
            cwd=tempfile.mkdtemp(prefix="s9_sdk_"),
            model=model,
            session_id=str(uuid.uuid4()),
            mcp_servers={"af": sdk_mcp_server([tool], calls)},
            hooks={
                "PreToolUse": [hook_matcher(_hook, "mcp__af__.*")],
                "PostToolUse": [hook_matcher(_hook, "mcp__af__.*")],
            },
            partial=True,
        ),
        PROMPT,
        on_message=_on,
    )
    c.check(f"{k} turn ran", not run.error and run.result is not None, run.error)
    _check(c, k, calls, uses, run.text, tl)


async def cli_spike(model: str, work_s: float, c: Checks) -> None:
    k = "S9[cli]"
    section(k)
    work = tempfile.mkdtemp(prefix="s9_cli_")
    calls, tl = CallLog(), Timeline()
    tool = _tool(work_s)
    inner = tool.handler

    async def _traced(args: dict[str, Any]) -> str:
        tl.add(time.time(), f"handler start {args.get('key')}")
        out = await inner(args)
        tl.add(time.time(), f"handler end {args.get('key')}")
        return out

    tool.handler = _traced
    hooks = CommandHooks(os.path.join(work, ".hooks"))
    server = await HttpMcpServer([tool], calls).start()
    try:
        run = await run_claude(
            claude_argv(
                PROMPT,
                model=model,
                session_id=str(uuid.uuid4()),
                mcp_config=server.write_config(work),
                settings=hooks.settings(
                    {"PreToolUse": "mcp__af__.*", "PostToolUse": "mcp__af__.*"}
                ),
                allowed_tools=["mcp__af"],
                partial=True,
            ),
            cwd=work,
        )
    finally:
        await server.stop()
    uses: list[tuple[str, str, str]] = []
    for e in run.events:
        if e.get("parent_tool_use_id"):
            continue
        if e.get("type") == "stream_event":
            event = e.get("event") or {}
            block = event.get("content_block") or {}
            if (
                event.get("type") == "content_block_start"
                and block.get("type") == "tool_use"
            ):
                tl.add(
                    e["_recv"],
                    f"stream content_block_start tool_use {block.get('id', '')[-6:]}",
                )
        elif e.get("type") == "assistant":
            message = e.get("message") or {}
            for block in message.get("content") or []:
                if block.get("type") == "tool_use" and str(block.get("name")).endswith(
                    "lookup_record"
                ):
                    key = (block.get("input") or {}).get("key", "")
                    uses.append((message.get("id", ""), block.get("id", ""), key))
                    tl.add(
                        e["_recv"],
                        f"AssistantMessage tool_use {key} (msg {str(message.get('id'))[-6:]})",
                    )
    for payload in hooks.log():
        key = (payload.get("tool_input") or {}).get("key")
        tl.add(payload["_ts"], f"{payload.get('hook_event_name')} {key}")
    c.check(
        f"{k} turn ran",
        bool(run.result) and not run.result.get("is_error"),
        run.result.get("subtype") or run.stderr[-200:],
    )
    _check(c, k, calls, uses, run.text, tl)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--kinds", default="sdk,cli")
    parser.add_argument("--work", type=float, default=3.0)
    args = parser.parse_args()
    c = Checks("S9")
    for kind in args.kinds.split(","):
        if kind == "sdk":
            await sdk_spike(args.model, args.work, c)
        else:
            await cli_spike(args.model, args.work, c)
    return c.exit_code()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
