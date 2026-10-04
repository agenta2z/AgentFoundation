"""S5 (timeout half) — what ``MCP_TOOL_TIMEOUT`` does to a slow AF tool.

An AF tool that takes ``--slow`` seconds is called once per run:

  * with a small ``MCP_TOOL_TIMEOUT`` (``--small-ms``, below the tool's time):
    does Claude Code abandon the call, what does the model get, and does the
    tool's handler keep running (side effects after the vendor gave up)?
  * with the native backends' configured value (7,200,000 ms): the call
    completes and the model gets the tool's output.

Both AF transports: the Agent SDK's in-process MCP server (``claude_sdk``)
and a localhost streamable-HTTP MCP server (``claude_cli``). Evidence: the
transcript's tool_use / tool_result pair (timestamps, ``is_error``, text) and
the handler's own start / end / cancellation.

The interrupt half of S5 (cancel during an AF tool call through the native
orchestrator) is in ``s1_s3_s5_s8_claude.py``.

    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/s5_mcp_tool_timeout.py [--kinds sdk,cli] [--model haiku]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import uuid
from datetime import datetime
from typing import Any

from _spike_common import (
    CallLog,
    Checks,
    claude_argv,
    HttpMcpServer,
    run_claude,
    run_sdk,
    sdk_mcp_server,
    sdk_options,
    section,
    session_entries,
    SpikeTool,
    tool_result_text,
)

CONFIGURED_MS = 7_200_000  # NativeBackendSpec.mcp_tool_timeout_ms


def _slow_tool(seconds: float) -> SpikeTool:
    async def _lookup(args: dict[str, Any]) -> str:
        await asyncio.sleep(seconds)
        return f"record {args.get('key')}: status=green"

    return SpikeTool(
        name="slow_lookup",
        description="Look up a record in the slow archive by key.",
        properties={"key": {"type": "string"}},
        required=("key",),
        handler=_lookup,
    )


def _ts(entry: dict[str, Any]) -> float:
    return datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00")).timestamp()


def _pair(entries: list[dict[str, Any]]) -> tuple[Any, Any, float]:
    """(tool_use entry, tool_result part, seconds between them) for slow_lookup."""
    use_entry, use_id = None, ""
    for e in entries:
        if e.get("type") != "assistant":
            continue
        for part in (e.get("message") or {}).get("content") or []:
            if part.get("type") == "tool_use" and part.get("name", "").endswith(
                "slow_lookup"
            ):
                use_entry, use_id = e, part.get("id")
                break
        if use_entry:
            break
    for e in entries:
        if e.get("type") != "user":
            continue
        for part in (e.get("message") or {}).get("content") or []:
            if isinstance(part, dict) and part.get("tool_use_id") == use_id and use_id:
                return use_entry, part, _ts(e) - _ts(use_entry)
    return use_entry, None, 0.0


async def _run(
    kind: str, model: str, timeout_ms: int, slow: float, key: str
) -> tuple[CallLog, list, str, float]:
    """Returns the calls, the transcript, the reply and when the turn ended."""
    calls = CallLog()
    tools = [_slow_tool(slow)]
    work = tempfile.mkdtemp(prefix=f"s5to_{kind}_")
    sid = str(uuid.uuid4())
    prompt = (
        f"Call the slow_lookup tool exactly once with key '{key}' and report the result. "
        "If the call fails, do not retry: report the exact error text instead."
    )
    env = {"MCP_TOOL_TIMEOUT": str(timeout_ms)}
    if kind == "sdk":
        # Keep the client (and its in-process server) up until an abandoned
        # handler would have finished, so a cancellation is the vendor's doing.
        run = await run_sdk(
            sdk_options(
                cwd=work,
                model=model,
                session_id=sid,
                mcp_servers={"af": sdk_mcp_server(tools, calls)},
                env=env,
            ),
            prompt,
            timeout=slow * 4 + 240,
            linger_s=slow + 2 if timeout_ms < slow * 1000 else 0.0,
        )
        text = run.text or run.error
        ended = getattr(run.result, "_recv", 0.0)
    else:
        server = await HttpMcpServer(tools, calls).start()
        try:
            run = await run_claude(
                claude_argv(
                    prompt,
                    model=model,
                    session_id=sid,
                    mcp_config=server.write_config(work),
                    allowed_tools=["mcp__af"],
                ),
                cwd=work,
                env=env,
                timeout=slow * 4 + 240,
            )
            # Let an abandoned handler finish (or not) before inspecting it.
            await _settle(calls, slow)
        finally:
            await server.stop()
        text = run.text or run.stderr[-300:]
        ended = run.result.get("_recv", 0.0)
    await _settle(calls, slow)
    return calls, session_entries(sid), text, ended


async def _settle(calls: CallLog, slow: float) -> None:
    for _ in range(int(slow * 2) + 10):
        if all(c.ended is not None for c in calls.calls):
            return
        await asyncio.sleep(0.5)


async def spike(kind: str, model: str, small_ms: int, slow: float, c: Checks) -> None:
    k = f"S5[{kind}]"
    section(f"{k} MCP_TOOL_TIMEOUT={small_ms} ms vs a {slow:.0f} s AF tool")
    calls, entries, text, ended = await _run(kind, model, small_ms, slow, "T-1")
    use, result, waited = _pair(entries)
    result_text = tool_result_text(result) if result else ""
    c.check(
        f"{k} small timeout: the tool was called", bool(calls.calls), len(calls.calls)
    )
    c.check(
        f"{k} small timeout: Claude Code abandons the call before the tool finishes",
        result is not None and waited < slow - 2 and "status=green" not in result_text,
        f"tool_result after {waited:.1f}s, is_error={result.get('is_error') if result else None}: {result_text[:160]}",
    )
    first = calls.calls[0] if calls.calls else None
    if first is not None and first.ended is not None:
        ran = first.ended - first.started
        after_turn = first.ended - ended if ended else float("nan")
        fate = "cancelled" if first.cancelled else "ran to completion"
        c.info(
            f"{k} small timeout: the abandoned handler",
            f"{fate} {ran:.1f}s after it started ({after_turn:+.1f}s vs the turn's result)",
        )
    c.info(
        f"{k} small timeout: calls made / model reply",
        f"{len(calls.calls)} / {text[:200]!r}",
    )

    section(f"{k} MCP_TOOL_TIMEOUT={CONFIGURED_MS} ms (native default)")
    calls, entries, text, _ = await _run(kind, model, CONFIGURED_MS, slow, "T-2")
    use, result, waited = _pair(entries)
    result_text = tool_result_text(result) if result else ""
    c.check(
        f"{k} configured timeout: the call completes with the tool's output",
        result is not None
        and "record T-2: status=green" in result_text
        and not result.get("is_error"),
        f"tool_result after {waited:.1f}s: {result_text[:120]}",
    )
    c.check(
        f"{k} configured timeout: exactly one call",
        len(calls.calls) == 1,
        len(calls.calls),
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--kinds", default="sdk,cli")
    parser.add_argument("--small-ms", type=int, default=3000)
    parser.add_argument("--slow", type=float, default=20.0)
    args = parser.parse_args()
    c = Checks("S5-timeout")
    for kind in args.kinds.split(","):
        await spike(kind.strip(), args.model, args.small_ms, args.slow, c)
    return c.exit_code()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
