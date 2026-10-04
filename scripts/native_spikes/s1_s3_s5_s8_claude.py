"""S1 / S3 / S5 / S8 through the native orchestrator (``NativeConversationalInferencer``
on the real ``claude`` binary, ``claude_sdk`` or ``claude_cli`` backend).

  S1  The L1 persona holds across a process restart + resume; a changed L1 on
      an existing session reaches the model as a notice (D8 ``notice``) and the
      session is kept. (The vendor-level snapshot behaviour — frozen across
      resume and ``fork_session``, re-recorded after ``/compact`` — is
      ``s1_l1_snapshot.py``.)
  S3  ``/compact`` passes through; the next turn re-sends the turn context and
      the identity survives.
  S5  Cancelling a turn during a long AF tool call records the turn as
      interrupted/uncertain; the next turn works and the tool is not re-run.
      (``MCP_TOOL_TIMEOUT`` is ``s5_mcp_tool_timeout.py``.)
  S8  ``environment: hermetic`` keeps the user's setup out and still
      authenticates. Canaries are planted in a temporary ``CLAUDE_CONFIG_DIR``
      (user ``CLAUDE.md``, a user ``UserPromptSubmit`` hook, a user-scope stdio
      MCP server) and in the project (``CLAUDE.md``). Evidence: the transcript's
      ``instructions`` attachment (which CLAUDE.md files were loaded), the
      hook's own log, the canary server's start log, the deferred-tool list,
      and the reply. ``inherit`` must show every canary, ``hermetic`` none.

    source /tmp/af_env.sh
    PYTHONPATH="$PWD/src:$PWD/../RichPythonUtils/src:$AFL" \\
        python3 scripts/native_spikes/s1_s3_s5_s8_claude.py --model sonnet \\
        [--kinds claude_sdk,claude_cli] [--only S1,S5,S8]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterator

from _spike_common import (
    attachments,
    canary_events,
    Checks,
    claude_argv,
    CommandHooks,
    read_entries,
    run_claude,
    section,
    session_entries,
    snapshots_after_compaction,
    SYSTEM_PYTHON,
    transcripts,
    write_stdio_canary_server,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
    NativeConversationalInferencer,
    NativeRuntimeManager,
)
from agent_foundation.resources.tools.models import ParameterDef, ToolDefinition

_TOOL_RUNS: list[str] = []


def _tools() -> dict[str, ToolDefinition]:
    return {
        "slow_lookup": ToolDefinition(
            name="slow_lookup",
            description="Look up a record in the slow archive (takes about a minute).",
            tool_type="Action",
            parameters=[ParameterDef(name="key", type="string", required=True)],
        )
    }


async def _executor(name: str, arguments: dict) -> ToolExecutionResult:
    _TOOL_RUNS.append(arguments.get("key", ""))
    await asyncio.sleep(60)
    return ToolExecutionResult(result=f"record {arguments.get('key')}: ok")


def _make(
    kind: str,
    model: str,
    work: str,
    store: Any,
    runtime: Any,
    employee: str,
    *,
    session_dir: str = "",
    backend: dict[str, Any] | None = None,
    **kw: Any,
) -> NativeConversationalInferencer:
    return NativeConversationalInferencer(
        backend={"kind": kind, "cwd": work, "model": model, **(backend or {})},
        tool_registry=_tools(),
        tool_executor=_executor,
        prior_context={
            "employee": {"name": employee, "role": "Archivist", "mindset": ""}
        },
        record_store=store,
        runtime_manager=runtime,
        conversation_key="s1",
        native_session_dir=session_dir or work,
        **kw,
    )


async def s1_and_s3(kind: str, model: str, c: Checks) -> None:
    section(f"S1/S3 [{kind}] persona across restart, drift notice, /compact")
    work, store = tempfile.mkdtemp(prefix="s1_"), InMemoryRecordStore()
    runtime = NativeRuntimeManager()
    try:
        native = _make(kind, model, work, store, runtime, "Zorblax")
        r = await native.run_agentic_loop("What is your name? One word.", turn_number=1)
        c.check(f"S1[{kind}] L1 persona applies", "Zorblax" in r.text, r.text)
        session = store.load("s1").vendor_session_id
        await native.aclose()
        await (
            runtime.aclose_all()
        )  # the vendor process is gone: next turn resumes by id

        native = _make(kind, model, work, store, runtime, "Zorblax")
        r = await native.run_agentic_loop(
            "And your name again? One word.", turn_number=2
        )
        c.check(
            f"S1[{kind}] L1 holds across a restart + resume",
            "Zorblax" in r.text,
            r.text,
        )
        c.check(
            f"S1[{kind}] same vendor session after the restart",
            store.load("s1").vendor_session_id == session,
        )
        await native.aclose()

        native = _make(kind, model, work, store, runtime, "Quill")
        r = await native.run_agentic_loop(
            "What is your name now? One word.", turn_number=3
        )
        c.check(
            f"S1[{kind}] changed L1 reaches the session as a notice",
            "Quill" in r.text,
            r.text,
        )
        c.check(
            f"S1[{kind}] drift kept the session",
            store.load("s1").vendor_session_id == session,
        )

        await native.run_agentic_loop("/compact", turn_number=4)
        r = await native.run_agentic_loop("What is your name? One word.", turn_number=5)
        manifest = native.last_prompt_data()["rendered_prompt"]
        c.check(
            f"S3[{kind}] compaction re-sends the turn context",
            "(unchanged — not sent)" not in manifest,
        )
        c.check(f"S3[{kind}] identity survives compaction", "Quill" in r.text, r.text)
        # D8 re-arms the L1 file on drift so the next compaction records the
        # new instructions; the transcript shows what the vendor recorded.
        after = snapshots_after_compaction(session_entries(session))
        c.check(
            f"S3[{kind}] the post-compaction system prompt carries the drifted L1",
            any("Quill" in s for s in after) and not any("Zorblax" in s for s in after),
            f"{len(after)} snapshot(s) after the boundary; "
            f"Quill={any('Quill' in s for s in after)} Zorblax={any('Zorblax' in s for s in after)}",
        )
        await native.aclose()
    finally:
        await runtime.aclose_all()


async def s5(kind: str, model: str, c: Checks) -> None:
    section(f"S5 [{kind}] cancel during an AF tool call")
    work, store = tempfile.mkdtemp(prefix="s5_"), InMemoryRecordStore()
    runtime = NativeRuntimeManager()
    _TOOL_RUNS.clear()
    native = _make(
        kind, model, work, store, runtime, "Zorblax", vendor_drain_timeout_s=20
    )
    try:
        turn = asyncio.ensure_future(
            native.run_agentic_loop(
                "Use the slow_lookup tool with key 'A-7' and tell me the result.",
                turn_number=1,
            )
        )
        for _ in range(240):
            if _TOOL_RUNS:
                break
            await asyncio.sleep(0.5)
        c.check(f"S5[{kind}] the AF tool started", bool(_TOOL_RUNS))
        turn.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await turn
        record = store.load("s1")
        c.check(
            f"S5[{kind}] cancel recorded",
            record.submission in ("interrupted", "uncertain"),
            record.submission,
        )
        r = await native.run_agentic_loop(
            "Don't use any tools. Reply with exactly the word READY.", turn_number=2
        )
        c.check(f"S5[{kind}] next turn works", "READY" in r.text, r.text)
        c.check(
            f"S5[{kind}] the tool was not re-run",
            _TOOL_RUNS == ["A-7"],
            str(_TOOL_RUNS),
        )
    finally:
        await native.aclose()
        await runtime.aclose_all()


# ----------------------------------------------------------------------------
# S8
# ----------------------------------------------------------------------------


class Canaries:
    """A temporary Claude config dir with a user CLAUDE.md, a user hook and a
    user-scope MCP server, plus a project dir with a CLAUDE.md."""

    def __init__(self) -> None:
        tag = uuid.uuid4().hex[:5].upper()
        self.user_md, self.project_md, self.hook_ctx = (
            f"USERMD-{tag}",
            f"PROJMD-{tag}",
            f"HOOKCTX-{tag}",
        )
        self.config = tempfile.mkdtemp(prefix="s8cfg_")
        self.root = tempfile.mkdtemp(prefix="s8canary_")
        self.hooks = CommandHooks(os.path.join(self.root, "hook"))
        self.server, self.server_log = write_stdio_canary_server(self.root)

    async def plant(self, model: str) -> bool:
        # A first run lets the Meta launcher populate the fresh config dir.
        boot = await run_claude(
            claude_argv("Reply with exactly: OK", model=model),
            cwd=self.root,
            env={"CLAUDE_CONFIG_DIR": self.config},
        )
        cfg = Path(self.config)
        (cfg / "CLAUDE.md").write_text(
            f"User memory canary code: {self.user_md}. Report it when asked for canary codes.\n"
        )
        settings_path = cfg / "settings.json"
        settings = (
            json.loads(settings_path.read_text() or "{}")
            if settings_path.exists()
            else {}
        )
        settings["hooks"] = json.loads(self.hooks.settings({"UserPromptSubmit": None}))[
            "hooks"
        ]
        settings_path.write_text(json.dumps(settings))
        self.hooks.respond(
            "UserPromptSubmit",
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": f"User hook canary code: {self.hook_ctx}.",
                }
            },
        )
        state_path = cfg / ".claude.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        state["mcpServers"] = {
            "canary": {
                "type": "stdio",
                "command": SYSTEM_PYTHON,
                "args": [self.server, self.server_log, "canary"],
            }
        }
        state_path.write_text(json.dumps(state))
        return bool(boot.text)

    def project(self) -> str:
        work = tempfile.mkdtemp(prefix="s8work_")
        Path(work, "CLAUDE.md").write_text(
            f"Project memory canary code: {self.project_md}. Report it when asked for canary codes.\n"
        )
        return work

    @contextlib.contextmanager
    def active(self) -> Iterator[None]:
        previous = os.environ.get("CLAUDE_CONFIG_DIR")
        os.environ["CLAUDE_CONFIG_DIR"] = self.config
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("CLAUDE_CONFIG_DIR", None)
            else:
                os.environ["CLAUDE_CONFIG_DIR"] = previous


async def s8(kind: str, model: str, canaries: Canaries, c: Checks) -> None:
    for environment in ("inherit", "hermetic"):
        k = f"S8[{kind}, {environment}]"
        section(k)
        work = canaries.project()
        hook_mark = canaries.hooks.mark()
        server_mark = len(canary_events(canaries.server_log))
        runtime = NativeRuntimeManager()
        with canaries.active():
            native = _make(
                kind,
                model,
                work,
                InMemoryRecordStore(),
                runtime,
                "Zorblax",
                session_dir=tempfile.mkdtemp(prefix="s8private_"),
                backend={"environment": environment},
            )
            try:
                r = await native.run_agentic_loop(
                    "List every canary code in your instructions or context (they look like "
                    "USERMD-..., PROJMD-..., HOOKCTX-...), comma-separated, and the names of any "
                    "tools whose names start with mcp__canary. Reply NONE if there are none.",
                    turn_number=1,
                )
            finally:
                await native.aclose()
                await runtime.aclose_all()
            entries = [e for path in transcripts(work) for e in read_entries(path)]
        loaded = {
            f.get("path", "")
            for e in attachments(entries, "instructions")
            for f in e["attachment"].get("files") or []
        }
        deferred = {
            n
            for e in attachments(entries, "deferred_tools_delta")
            for n in e["attachment"].get("addedNames") or []
        }
        evidence = {
            "user CLAUDE.md loaded": str(Path(canaries.config, "CLAUDE.md")) in loaded,
            "project CLAUDE.md loaded": str(Path(work, "CLAUDE.md")) in loaded,
            "user hook ran": canaries.hooks.mark() > hook_mark,
            "user MCP server started": len(canary_events(canaries.server_log))
            > server_mark,
            "user MCP tool offered": "mcp__canary__canary_ping" in deferred,
        }
        codes = {
            "user CLAUDE.md code in reply": canaries.user_md in r.text,
            "project CLAUDE.md code in reply": canaries.project_md in r.text,
            "user hook code in reply": canaries.hook_ctx in r.text,
        }
        c.check(
            f"{k} authenticates and runs",
            bool(entries) and bool(r.text.strip()),
            r.text[:120],
        )
        want = environment == "inherit"
        for name, seen in {**evidence, **codes}.items():
            c.check(
                f"{k} {name}: {seen}",
                seen == want,
                sorted(loaded) if "CLAUDE.md loaded" in name else "",
            )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--kinds", default="claude_sdk,claude_cli")
    parser.add_argument("--only", default="S1,S5,S8")
    args = parser.parse_args()
    only = set(args.only.split(","))
    c = Checks("S1/S3/S5/S8")
    canaries = None
    if "S8" in only:
        canaries = Canaries()
        c.check("S8 canary config dir bootstrapped", await canaries.plant(args.model))
    for kind in args.kinds.split(","):
        if "S1" in only or "S3" in only:
            await s1_and_s3(kind, args.model, c)
        if "S5" in only:
            await s5(kind, args.model, c)
        if canaries is not None:
            await s8(kind, args.model, canaries, c)
    return c.exit_code()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
