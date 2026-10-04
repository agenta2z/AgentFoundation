"""S2 / S4 / S12 (+ H) — Claude Code hooks the native backends rely on, on both
transports: the Agent SDK (Python hook callbacks, in-process MCP server ``af``;
the ``claude_sdk`` backend) and ``claude -p`` (``--settings`` command hooks,
localhost HTTP MCP server ``af``; the ``claude_cli`` backend).

* S2  ``UserPromptSubmit`` ``additionalContext``: visible to the model; recorded
      in the transcript as a hook attachment, not in the user message; still
      known on a later resumed turn that sends none (persisted, not per-turn);
      not attributed to the user (asked); size: 10,000 chars stay inline
      (head + tail codewords reach the model), 10,001 and 100,000 chars are
      spilled to a file (the transcript attachment is a preview + path).
* S4  ``PostToolUse`` ``continue:false`` after an AF tool ends the turn: the tool
      runs once, no text or tool call follows, the result is not an error, the
      tool_use has its tool_result, and the next resume works and remembers it.
* S12 ``PreToolUse`` input carries ``agent_id`` for a Task-subagent's AF call and
      none for the main thread's.
* H   the hooks still fire under hermetic ``--setting-sources "" --strict-mcp-config``.

The hook side is the spike's own (file-backed command hooks / SDK callbacks), so
the evidence is the vendor's behavior, independent of AgentFoundation's relay.

    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/s2_s4_s12_claude_cli_hooks.py \\
        [--kinds sdk,cli] [--only S2,S4,S12,H] [--model sonnet]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from _spike_common import (
    CallLog,
    Checks,
    claude_argv,
    CommandHooks,
    echo_tool,
    entry_text,
    hook_contexts,
    hook_matcher,
    HttpMcpServer,
    run_claude,
    run_sdk,
    sdk_mcp_server,
    sdk_options,
    section,
    session_entries,
    tool_result_text,
    tool_results,
    tool_uses,
    user_prompts,
)

AF = "mcp__af__"
# Claude Code 2.1.288 keeps hook additionalContext of up to 10,000 characters
# inline; longer context is saved to a file and the model gets a ~2 KB preview
# plus the file path (the same spill as oversized tool results).
INLINE_LIMIT = 10_000


@dataclass
class TurnOut:
    text: str
    error: str
    is_error: bool
    subtype: str
    stop_reason: str
    blocks: list[list[dict[str, Any]]] = field(default_factory=list)  # per message

    def blocks_after_first_af_call(self) -> list[dict[str, Any]]:
        flat = [b for message in self.blocks for b in message]
        for i, block in enumerate(flat):
            if block.get("type") == "tool_use" and str(block.get("name")).startswith(
                AF
            ):
                return flat[i + 1 :]
        return []


def _sdk_blocks(message: Any) -> list[dict[str, Any]]:
    from claude_agent_sdk import TextBlock, ToolUseBlock

    out = []
    for block in message.content:
        if isinstance(block, TextBlock):
            out.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            out.append(
                {
                    "type": "tool_use",
                    "name": block.name,
                    "id": block.id,
                    "input": block.input,
                }
            )
    return out


class Transport:
    """Hooks + AF tool server + one turn, for one backend shape."""

    kind = ""

    def __init__(self, model: str) -> None:
        self.model = model
        self.work = tempfile.mkdtemp(prefix=f"s2s4_{self.kind}_")
        self.calls = CallLog()
        self.tools = [echo_tool()]

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def respond(self, event: str, payload: Optional[dict[str, Any]]) -> None: ...

    def hook_log(self) -> list[dict[str, Any]]: ...

    async def turn(
        self,
        prompt: str,
        *,
        session_id: str = "",
        resume: str = "",
        hermetic: bool = False,
    ) -> TurnOut: ...


class CliTransport(Transport):
    kind = "cli"

    async def start(self) -> None:
        self.hooks = CommandHooks(os.path.join(self.work, ".hooks"))
        self.server = await HttpMcpServer(self.tools, self.calls).start()
        self.mcp_config = self.server.write_config(self.work)
        self.settings = self.hooks.settings(
            {
                "UserPromptSubmit": None,
                "PreToolUse": None,
                "PostToolUse": f"{AF}.*",
            }
        )

    async def stop(self) -> None:
        await self.server.stop()

    def respond(self, event: str, payload: Optional[dict[str, Any]]) -> None:
        self.hooks.respond(event, payload)

    def hook_log(self) -> list[dict[str, Any]]:
        return self.hooks.log()

    async def turn(
        self,
        prompt: str,
        *,
        session_id: str = "",
        resume: str = "",
        hermetic: bool = False,
    ) -> TurnOut:
        run = await run_claude(
            claude_argv(
                prompt,
                model=self.model,
                session_id=session_id,
                resume=resume,
                mcp_config=self.mcp_config,
                settings=self.settings,
                allowed_tools=["mcp__af"],
                hermetic=hermetic,
            ),
            cwd=self.work,
        )
        result = run.result
        return TurnOut(
            text=run.text,
            error="" if result else f"rc={run.rc} {run.stderr[-300:]}",
            is_error=bool(result.get("is_error")),
            subtype=str(result.get("subtype", "")),
            stop_reason=f"{result.get('stop_reason')}/{result.get('terminal_reason')}",
            blocks=[
                list((e.get("message") or {}).get("content") or [])
                for e in run.assistant()
            ],
        )


class SdkTransport(Transport):
    kind = "sdk"

    async def start(self) -> None:
        self.responses: dict[str, dict[str, Any]] = {}
        self.log: list[dict[str, Any]] = []
        self.server = sdk_mcp_server(self.tools, self.calls)

    def respond(self, event: str, payload: Optional[dict[str, Any]]) -> None:
        if payload is None:
            self.responses.pop(event, None)
            return
        # The SDK spells the JSON key ``continue`` as ``continue_``.
        self.responses[event] = {
            ("continue_" if k == "continue" else k): v for k, v in payload.items()
        }

    def hook_log(self) -> list[dict[str, Any]]:
        return list(self.log)

    async def _hook(
        self, input_data: dict[str, Any], tool_use_id: Optional[str], context: Any
    ) -> dict[str, Any]:
        self.log.append(dict(input_data, _ts=time.time()))
        return dict(self.responses.get(str(input_data.get("hook_event_name")), {}))

    async def turn(
        self,
        prompt: str,
        *,
        session_id: str = "",
        resume: str = "",
        hermetic: bool = False,
    ) -> TurnOut:
        hooks = {
            "UserPromptSubmit": [hook_matcher(self._hook)],
            "PreToolUse": [hook_matcher(self._hook)],
            "PostToolUse": [hook_matcher(self._hook, f"{AF}.*")],
        }
        run = await run_sdk(
            sdk_options(
                cwd=self.work,
                model=self.model,
                session_id=session_id,
                resume=resume,
                hooks=hooks,
                mcp_servers={"af": self.server},
                hermetic=hermetic,
            ),
            prompt,
        )
        result = run.result
        return TurnOut(
            text=run.text,
            error=run.error if run.error or result else "no ResultMessage",
            is_error=bool(getattr(result, "is_error", False)),
            subtype=str(getattr(result, "subtype", "")),
            stop_reason=str(getattr(result, "stop_reason", "")),
            blocks=[_sdk_blocks(m) for m in run.assistant()],
        )


def _context(text: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": text,
        }
    }


def _filler(chars: int) -> str:
    lines, n = [], 0
    while sum(map(len, lines)) < chars:
        lines.append(f"Reference note {n:05d}: background material, not a codeword. ")
        n += 1
    return "".join(lines)[:chars]


async def s2(t: Transport, c: Checks) -> None:
    section(f"S2 [{t.kind}] UserPromptSubmit additionalContext")
    k = f"S2[{t.kind}]"
    sid = str(uuid.uuid4())
    word = f"MAGENTA-{uuid.uuid4().hex[:5].upper()}"
    t.respond(
        "UserPromptSubmit",
        _context(f"<af_context>The session codeword is {word}.</af_context>"),
    )
    out = await t.turn("Reply with exactly the word READY.", session_id=sid)
    t.respond("UserPromptSubmit", None)
    c.check(
        f"{k} turn with hook context ran",
        not out.error and "READY" in out.text.upper(),
        out.error or out.text,
    )
    entries = session_entries(sid)
    recorded = [e for e in hook_contexts(entries) if word in entry_text(e)]
    c.check(
        f"{k} context recorded in the transcript as a hook attachment",
        bool(recorded),
        f"{len(recorded)} attachment(s)",
    )
    if recorded:
        rendered = json.dumps(recorded[0].get("rendered"))[:160]
        c.info(
            f"{k} attachment renderedRole={recorded[0].get('renderedRole')!r}", rendered
        )
    c.check(
        f"{k} context is not inside the user message",
        not any(word in p for p in user_prompts(entries)),
        user_prompts(entries)[-1:],
    )

    out = await t.turn(
        "What is the session codeword? Reply with only the codeword, or NONE if you were never given one.",
        resume=sid,
    )
    c.check(
        f"{k} visible + persisted: a later resumed turn without context still knows it",
        word in out.text,
        out.text,
    )

    out = await t.turn(
        "Did I, the user, type the session codeword in one of my own messages? Start your reply with "
        "exactly one word, USER if I typed it in my message or ELSEWHERE if it reached you another way, "
        "then one sentence saying where it appeared.",
        resume=sid,
    )
    first = (
        out.text.strip().split()[0].strip("*.,:").upper() if out.text.strip() else ""
    )
    c.check(f"{k} not attributed to the user", first == "ELSEWHERE", out.text)

    for size in (INLINE_LIMIT, INLINE_LIMIT + 1, 100_000):
        await _size(t, c, size)


async def _size(t: Transport, c: Checks, size: int) -> None:
    k = f"S2[{t.kind}] {size} chars"
    sid = str(uuid.uuid4())
    head = f"HEAD-{uuid.uuid4().hex[:5].upper()}"
    tail = f"TAIL-{uuid.uuid4().hex[:5].upper()}"
    prefix = f"<af_context>The head codeword is {head}. "
    suffix = f" The tail codeword is {tail}.</af_context>"
    body = prefix + _filler(size - len(prefix) - len(suffix)) + suffix
    t.respond("UserPromptSubmit", _context(body))
    out = await t.turn(
        "Reply with the head codeword and the tail codeword from your context, separated by one space.",
        session_id=sid,
    )
    t.respond("UserPromptSubmit", None)
    stored = [e for e in hook_contexts(session_entries(sid)) if head in entry_text(e)]
    text = entry_text(stored[0]) if stored else ""
    c.check(
        f"{k} turn ran", not out.error and not out.is_error, out.error or out.subtype
    )
    if size <= INLINE_LIMIT:
        c.check(
            f"{k} stored inline and whole",
            tail in text and "<persisted-output>" not in text,
            f"attachment {len(text)} chars",
        )
        c.check(
            f"{k} head + tail codewords reach the model",
            head in out.text and tail in out.text,
            out.text,
        )
        return
    c.check(
        f"{k} spilled to a file: the model gets a preview + path, not the text",
        "<persisted-output>" in text
        and "saved to" in text
        and head in text
        and tail not in text,
        f"attachment {len(text)} chars",
    )
    reads = [
        b.get("name") for m in out.blocks for b in m if b.get("type") == "tool_use"
    ]
    c.info(f"{k} tail in the reply: {tail in out.text}; the model's tool calls", reads)


async def s4(t: Transport, c: Checks) -> None:
    section(f"S4 [{t.kind}] PostToolUse continue:false after an AF tool")
    k = f"S4[{t.kind}]"
    sid = str(uuid.uuid4())
    nonce = uuid.uuid4().hex[:6]
    t.respond("PostToolUse", {"continue": False, "stopReason": "AF ended the turn"})
    before = len(t.calls.calls)
    out = await t.turn(
        f"Call the echo tool with text {nonce}, then call it again with text second-{nonce}, "
        "then write a long poem about the sea.",
        session_id=sid,
    )
    t.respond("PostToolUse", None)
    calls = [call.arguments.get("text") for call in t.calls.calls[before:]]
    after = out.blocks_after_first_af_call()
    c.check(f"{k} the AF tool ran exactly once", calls == [nonce], calls)
    c.check(
        f"{k} nothing follows the AF call (no text, no tool call)", not after, after[:2]
    )
    c.check(
        f"{k} the turn ended without error",
        not out.error and not out.is_error,
        f"{out.subtype} {out.stop_reason} {out.error}",
    )
    entries = session_entries(sid)
    use_ids = {
        u.get("id") for u in tool_uses(entries) if str(u.get("name")).startswith(AF)
    }
    result_ids = {r.get("tool_use_id") for r in tool_results(entries)}
    c.check(
        f"{k} the AF tool_use has its tool_result (no orphan)",
        bool(use_ids) and use_ids <= result_ids,
        f"{len(use_ids)} use(s)",
    )
    results = [tool_result_text(r)[:80] for r in tool_results(entries)]
    c.info(
        f"{k} tool results in the transcript (ToolSearch loads the deferred AF tool first)",
        results,
    )

    out = await t.turn("Reply with exactly: RESUMED", resume=sid)
    c.check(
        f"{k} next resume works",
        not out.error and "RESUMED" in out.text,
        out.text or out.error,
    )
    out = await t.turn(
        "What exact text did the echo tool return earlier? Reply with only that text.",
        resume=sid,
    )
    c.check(
        f"{k} resumed session holds the tool result",
        f"ECHO::{nonce}" in out.text,
        out.text,
    )


async def s12(t: Transport, c: Checks) -> None:
    section(f"S12 [{t.kind}] agent_id in PreToolUse for a subagent's AF call")
    k = f"S12[{t.kind}]"
    sid = str(uuid.uuid4())
    mark = len(t.hook_log())
    out = await t.turn(
        "Do two things in order. First, call the mcp__af__echo tool yourself with text from-main. "
        "Second, use the Agent tool (the Task tool) to launch a general-purpose subagent whose only job "
        "is to call the mcp__af__echo tool with text from-subagent and report what it returned. "
        "Do not call echo with from-subagent yourself.",
        session_id=sid,
    )
    pre = [p for p in t.hook_log()[mark:] if p.get("hook_event_name") == "PreToolUse"]
    echo = {
        str((p.get("tool_input") or {}).get("text")): p
        for p in pre
        if p.get("tool_name") == f"{AF}echo"
    }
    main, sub = echo.get("from-main"), echo.get("from-subagent")
    c.check(f"{k} turn ran", not out.error, out.error or out.subtype)
    c.check(
        f"{k} main-thread AF call seen by PreToolUse", main is not None, sorted(echo)
    )
    c.check(f"{k} subagent AF call seen by PreToolUse", sub is not None, sorted(echo))
    if main is not None:
        c.check(
            f"{k} main-thread call has no agent_id",
            not main.get("agent_id"),
            main.get("agent_id"),
        )
    if sub is not None:
        c.check(
            f"{k} subagent call carries agent_id",
            bool(sub.get("agent_id")),
            f"agent_id={'set' if sub.get('agent_id') else None} agent_type={sub.get('agent_type')!r}",
        )
    spawn = [p for p in pre if p.get("tool_name") in ("Agent", "Task")]
    c.info(
        f"{k} PreToolUse tools",
        [(p.get("tool_name"), bool(p.get("agent_id"))) for p in pre],
    )
    if spawn:
        c.info(
            f"{k} spawn call keys",
            sorted(k2 for k2 in spawn[0] if not k2.startswith("_")),
        )


async def hermetic(t: Transport, c: Checks) -> None:
    section(
        f"H [{t.kind}] hooks under hermetic --setting-sources '' --strict-mcp-config"
    )
    k = f"H[{t.kind}]"
    word = f"TEAL-{uuid.uuid4().hex[:5].upper()}"
    t.respond(
        "UserPromptSubmit",
        _context(f"<af_context>The session codeword is {word}.</af_context>"),
    )
    mark = len(t.hook_log())
    out = await t.turn(
        "Reply with only the session codeword from your context.",
        session_id=str(uuid.uuid4()),
        hermetic=True,
    )
    t.respond("UserPromptSubmit", None)
    fired = any(
        p.get("hook_event_name") == "UserPromptSubmit" for p in t.hook_log()[mark:]
    )
    c.check(f"{k} UserPromptSubmit hook fired", fired)
    c.check(
        f"{k} its context reached the model", word in out.text, out.text or out.error
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--kinds", default="sdk,cli")
    parser.add_argument("--only", default="S2,S4,S12,H")
    args = parser.parse_args()
    only = set(args.only.split(","))
    c = Checks("S2/S4/S12/H")
    spikes = (("S2", s2), ("S4", s4), ("S12", s12), ("H", hermetic))
    for kind in args.kinds.split(","):
        t: Transport = (
            SdkTransport(args.model) if kind == "sdk" else CliTransport(args.model)
        )
        await t.start()
        try:
            for name, fn in spikes:
                if name in only:
                    await fn(t, c)
        finally:
            await t.stop()
    return c.exit_code()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
