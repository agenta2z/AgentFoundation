"""Real-vendor qualification of a native conversational backend (plan §12.2).

One conversation runs the qualification sequence against the real vendor:

  1. L1 sentinel + history canary
  2. a built-in vendor tool reads a file
  3. an L2 (turn context) generation change reaches the model
  4. one vendor session across the turns
  5. host restart (new inferencer + runtime) resumes it, canary kept
  6. a turn cancelled during a slow AF tool: no duplicate side effects after
  7. ``/new`` starts another vendor session, without the canary

Checks rest on transport and session evidence: the vendor's own session store
(Claude Code transcript JSONL, Codex rollout JSONL), the argv each vendor
process received (recorded by an exec wrapper, which adds no process), the
vendor processes left after a cancel, the AF tool calls the host executed,
and the native session record. Model answers count only through unguessable tokens,
which the model can produce only if it was given them.

Opt-in: ``AF_REAL_NATIVE=1`` (spawns real vendors, costs money). Optional:
``AF_REAL_NATIVE_MODEL``, ``AF_REAL_NATIVE_ENVIRONMENT`` (inherit | hermetic),
``AF_REAL_NATIVE_TURN_TIMEOUT`` (s). Output: ``[PASS]/[FAIL] <check>`` lines;
session ids are shortened.
"""

from __future__ import annotations

import asyncio
import glob
import importlib
import json
import os
import re
import shlex
import shutil
import signal
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational.protocols import (
    ToolExecutionResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native import (
    InMemoryRecordStore,
    NativeConversationalInferencer,
    NativeRuntimeManager,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.devmate_dm import (
    resolve_cats_file,
)
from agent_foundation.resources.tools.models import ParameterDef, ToolDefinition
from agent_foundation.resources.tools.registry import load_all_tools

ENV_GATE = "AF_REAL_NATIVE"
_BINARIES = {
    "claude_sdk": "claude",
    "claude_cli": "claude",
    "codex_cli": "codex",
    "devmate_dm": "dm",
}
_DEFAULT_MODELS = {"claude_sdk": "sonnet", "claude_cli": "sonnet"}
_SESSION_FLAGS = ("--session-id", "--resume")
_GENERATION = re.compile(r'<af_context[^>]*\bgeneration="(\d+)"')
_AF_CONTEXT = re.compile(r"<af_context\b.*?</af_context>", re.DOTALL)
SLOW_TOOL = "slow_task"
SOP_NAME = "quali_probe"


def skip_reason(kind: str) -> Optional[str]:
    """Why the real qualification of ``kind`` cannot run here, else None."""
    if os.environ.get(ENV_GATE) != "1":
        return (
            f"real vendor test: set {ENV_GATE}=1 to run it (spawns the real "
            f"{kind} vendor and costs money)"
        )
    binary = _BINARIES.get(kind)
    if binary and not shutil.which(binary):
        return f"`{binary}` is not on PATH"
    if kind == "devmate_dm" and resolve_cats_file({}) is None:
        return (
            "dm real-model turns with AF tools need dm's workflow mode "
            "(DM_CORE_WORKFLOW=1), which requires CAT credentials (--cats-file), "
            "and none exist here (backend option cats_file, $DM_CATS_FILE, "
            "$PREMINTED_CATS_FILE, /tmp/devinfra/user_cats/cats); without them "
            "dm-core's MCP warmup stalls the turn"
        )
    if kind == "metamate":
        try:
            importlib.import_module("msl.metamate.sdk.client")
        except ImportError:
            return (
                "the Metamate SDK (//msl/metamate/sdk) is Buck-only and not "
                "importable here: run under a Buck binary that includes "
                "conversational_native/session/metamate:metamate"
            )
    return None


def _short(session_id: Optional[str]) -> str:
    return (session_id or "")[:8]


def _token(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------


class Checks:
    def __init__(self, label: str) -> None:
        self.label = label
        self.passed: list[str] = []
        self.failures: list[str] = []

    def check(self, name: str, ok: bool, detail: Any = "") -> bool:
        text = detail if isinstance(detail, str) else repr(detail)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {text[:400]!r}", flush=True)
        (self.passed if ok else self.failures).append(name)
        return ok

    def info(self, name: str, detail: Any = "") -> None:
        text = detail if isinstance(detail, str) else repr(detail)
        print(f"[INFO] {name} {text[:600]}", flush=True)

    def summary(self) -> str:
        total = len(self.passed) + len(self.failures)
        return (
            f"{self.label}: {len(self.passed)}/{total} checks passed; "
            f"failed: {self.failures}"
        )


# ----------------------------------------------------------------------------
# Vendor processes and session stores
# ----------------------------------------------------------------------------


class ArgvLog:
    """``cli_path`` for the backend: records each invocation's argv, then
    ``exec``s the installed vendor binary (same pid: the backend signals the
    process it would signal without the wrapper)."""

    def __init__(self, binary: str) -> None:
        self.directory = tempfile.mkdtemp(prefix="qual_argv_")
        self.path = os.path.join(self.directory, "vendor")
        real = shutil.which(binary) or binary
        script = (
            "#!/bin/sh\n"
            f"d={shlex.quote(self.directory)}\n"
            'n=$(( $(cat "$d/count" 2>/dev/null || echo 0) + 1 ))\n'
            'echo "$n" > "$d/count"\n'
            'printf \'%s\\0\' "$@" > "$d/argv$n"\n'
            f'exec {shlex.quote(real)} "$@"\n'
        )
        with open(self.path, "w") as f:
            f.write(script)
        os.chmod(self.path, 0o700)

    def invocations(self) -> list[list[str]]:
        """Argv (without the binary) of every session process, version
        probes excluded."""
        out = []
        for name in sorted(
            (n for n in os.listdir(self.directory) if n.startswith("argv")),
            key=lambda n: int(n[4:]),
        ):
            with open(os.path.join(self.directory, name), "rb") as f:
                argv = f.read().decode(errors="replace").split("\0")[:-1]
            if argv not in (["-v"], ["--version"]):
                out.append(argv)
        return out


def processes_in(directory: str) -> dict[int, str]:
    """Live processes (of this user) whose working directory is
    ``directory``: pid -> command line."""
    target = os.path.realpath(directory)
    found = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            if os.path.realpath(f"/proc/{entry}/cwd") != target:
                continue
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                command = f.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        found[int(entry)] = command[:160]
    return found


def session_arg(argv: list[str]) -> tuple[str, str]:
    """The session a vendor process was told to start or continue."""
    for flag in _SESSION_FLAGS:
        if flag in argv:
            return flag, argv[argv.index(flag) + 1]
    if argv[:2] == ["exec", "resume"]:
        return "exec resume", argv[2]
    return "new", ""


@dataclass
class ToolRun:
    name: str
    call: str  # the call's input, serialized
    result: str


class SessionStore:
    """A vendor's on-disk record of one session (None: not inspectable)."""

    def entries(self, session_id: str) -> list[dict[str, Any]]:
        return []

    def exists(self, session_id: str) -> bool:
        return False

    def instructions(self, entries: list[dict]) -> list[str]:
        return []

    def user_texts(self, entries: list[dict]) -> list[str]:
        return []

    def turn_contexts(self, entries: list[dict]) -> list[str]:
        return []

    def tool_runs(self, entries: list[dict]) -> list[ToolRun]:
        return []


def _read_jsonl(path: Optional[str]) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _texts(content: Any, kinds: tuple[str, ...]) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "")
        for part in content or []
        if isinstance(part, dict) and part.get("type") in kinds
    )


class ClaudeTranscripts(SessionStore):
    """``$CLAUDE_CONFIG_DIR/projects/<cwd slug>/<session id>.jsonl``."""

    def _path(self, session_id: str) -> Optional[str]:
        root = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
        hits = glob.glob(os.path.join(root, "projects", "*", f"{session_id}.jsonl"))
        return hits[0] if hits else None

    def exists(self, session_id: str) -> bool:
        return bool(session_id) and self._path(session_id) is not None

    def entries(self, session_id: str) -> list[dict[str, Any]]:
        return _read_jsonl(self._path(session_id)) if session_id else []

    @staticmethod
    def _attachments(entries: list[dict], kind: str) -> list[dict]:
        return [
            e["attachment"]
            for e in entries
            if e.get("type") == "attachment"
            and isinstance(e.get("attachment"), dict)
            and e["attachment"].get("type") == kind
        ]

    def instructions(self, entries: list[dict]) -> list[str]:
        out = []
        for snapshot in self._attachments(entries, "prompt_snapshot"):
            prompt = snapshot.get("systemPrompt")
            out.append("\n".join(prompt) if isinstance(prompt, list) else str(prompt))
        return out

    def user_texts(self, entries: list[dict]) -> list[str]:
        out = []
        for e in entries:
            if e.get("type") != "user" or e.get("isSidechain"):
                continue
            text = _texts((e.get("message") or {}).get("content"), ("text",))
            if text:
                out.append(text)
        return out

    def turn_contexts(self, entries: list[dict]) -> list[str]:
        out = []
        for hook in self._attachments(entries, "hook_additional_context"):
            if hook.get("hookEvent") != "UserPromptSubmit":
                continue
            content = hook.get("content")
            out.append(
                "\n".join(content) if isinstance(content, list) else str(content)
            )
        return out

    def tool_runs(self, entries: list[dict]) -> list[ToolRun]:
        calls: dict[str, ToolRun] = {}
        for e in entries:
            content = (e.get("message") or {}).get("content")
            if not isinstance(content, list) or e.get("isSidechain"):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if e.get("type") == "assistant" and part.get("type") == "tool_use":
                    calls[part.get("id", "")] = ToolRun(
                        part.get("name", ""), json.dumps(part.get("input")), ""
                    )
                elif e.get("type") == "user" and part.get("type") == "tool_result":
                    run = calls.get(part.get("tool_use_id", ""))
                    if run is not None:
                        run.result += _texts(part.get("content"), ("text",))
        return list(calls.values())


class CodexRollouts(SessionStore):
    """``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<thread id>.jsonl``."""

    def _path(self, thread_id: str) -> Optional[str]:
        root = os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
        pattern = os.path.join(root, "sessions", "**", f"rollout-*-{thread_id}.jsonl")
        hits = glob.glob(pattern, recursive=True)
        return hits[0] if hits else None

    def exists(self, session_id: str) -> bool:
        return bool(session_id) and self._path(session_id) is not None

    def entries(self, session_id: str) -> list[dict[str, Any]]:
        return _read_jsonl(self._path(session_id)) if session_id else []

    @staticmethod
    def _items(entries: list[dict]) -> list[dict]:
        return [
            e["payload"]
            for e in entries
            if e.get("type") == "response_item" and isinstance(e.get("payload"), dict)
        ]

    def _messages(self, entries: list[dict], role: str) -> list[str]:
        return [
            _texts(item.get("content"), ("input_text", "output_text"))
            for item in self._items(entries)
            if item.get("type") == "message" and item.get("role") == role
        ]

    def instructions(self, entries: list[dict]) -> list[str]:
        return self._messages(entries, "developer")

    def user_texts(self, entries: list[dict]) -> list[str]:
        return self._messages(entries, "user")

    def turn_contexts(self, entries: list[dict]) -> list[str]:
        return [
            block
            for text in self.user_texts(entries)
            for block in _AF_CONTEXT.findall(text)
        ]

    def tool_runs(self, entries: list[dict]) -> list[ToolRun]:
        calls: dict[str, ToolRun] = {}
        for item in self._items(entries):
            kind = item.get("type", "")
            if kind in ("function_call", "custom_tool_call", "local_shell_call"):
                call = item.get("arguments") or item.get("input") or item.get("action")
                calls[item.get("call_id", "")] = ToolRun(
                    item.get("name", kind), json.dumps(call), ""
                )
            elif kind.endswith("_call_output"):
                run = calls.get(item.get("call_id", ""))
                if run is not None:
                    output = item.get("output")
                    run.result += (
                        output
                        if isinstance(output, str)
                        else _texts(output, ("input_text", "output_text"))
                        or json.dumps(output)
                    )
        return list(calls.values())


_STORES = {
    "claude_sdk": ClaudeTranscripts,
    "claude_cli": ClaudeTranscripts,
    "codex_cli": CodexRollouts,
}


# ----------------------------------------------------------------------------
# Host side: tools, SOP, inferencer
# ----------------------------------------------------------------------------


class SlowTask:
    """AF tool executor: ``slow_task`` records each start and runs for
    ``seconds`` (long enough to be cancelled mid-call)."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.started: list[str] = []
        self.finished: list[str] = []
        self.running = asyncio.Event()

    async def __call__(self, name: str, arguments: dict) -> ToolExecutionResult:
        if name != SLOW_TOOL:
            return ToolExecutionResult(result=f"{name} done")
        label = str((arguments or {}).get("label", ""))
        self.started.append(label)
        self.running.set()
        await asyncio.sleep(self.seconds)
        self.finished.append(label)
        return ToolExecutionResult(result=f"{SLOW_TOOL} finished for {label}")


def _tools() -> dict[str, ToolDefinition]:
    tools = {k: v for k, v in load_all_tools().items() if v.tool_type == "Conversation"}
    tools[SLOW_TOOL] = ToolDefinition(
        name=SLOW_TOOL,
        description="Run the slow qualification task for a label and return when it finished.",
        tool_type="Action",
        parameters=[
            ParameterDef(name="label", type="string", required=True, positional=True)
        ],
    )
    return tools


def _write_probe_sop(token: str) -> str:
    root = tempfile.mkdtemp(prefix="qual_sops_")
    directory = Path(root, SOP_NAME)
    directory.mkdir()
    (directory / "SOP.md").write_text(
        "# Qualification Probe\n\n"
        "A one-phase probe procedure for the native qualification tests.\n\n"
        "## Phase 0 -- Probe\n[__initial__]\n\n"
        f"The probe token of this phase is {token}. When the user asks for the "
        "probe token, reply with it exactly; no tools are needed.\n"
    )
    (directory / "sop.config.json").write_text(
        json.dumps(
            {
                "name": SOP_NAME,
                "display_name": "Qualification Probe",
                "version": "1.0.0",
                "description": "A one-phase probe procedure for tests.",
                "available_modes": ["default", "yolo"],
                "requires_tools": [],
                "labels": ["test"],
                "linear_only": True,
                "max_goto_iterations": 1,
                "max_total_nodes": 5,
            }
        )
    )
    return root


@dataclass
class _Tokens:
    l1: str = field(default_factory=lambda: _token("QL1"))
    canary: str = field(default_factory=lambda: _token("CANARY"))
    file: str = field(default_factory=lambda: _token("FILE"))
    l2: str = field(default_factory=lambda: _token("QL2"))


class Qualification:
    """Runs the §12.2 sequence on one backend kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.checks = Checks(f"real_native_{kind}")
        self.tokens = _Tokens()
        self.store: SessionStore = _STORES.get(kind, SessionStore)()
        binary = _BINARIES.get(kind)
        self.argv_log = ArgvLog(binary) if binary else None
        self.work = tempfile.mkdtemp(prefix=f"qual_{kind}_work_")
        self.session_dir = tempfile.mkdtemp(prefix=f"qual_{kind}_session_")
        self.note = Path(self.work, "probe_note.txt")
        self.note.write_text(f"The file token is {self.tokens.file}.\n")
        self.sop_dir = _write_probe_sop(self.tokens.l2)
        self.records = InMemoryRecordStore()
        self.executor = SlowTask(float(os.environ.get("AF_REAL_NATIVE_SLOW_S", "90")))
        self.turn_timeout = float(os.environ.get("AF_REAL_NATIVE_TURN_TIMEOUT", "420"))
        self.conversation_key = f"qual-{kind}-{uuid.uuid4().hex[:6]}"
        # Metamate has no AF tools and no file tools.
        self.tool_capable = kind != "metamate"
        self.turn_number = 0
        self.runtime: Optional[NativeRuntimeManager] = None
        self.native: Optional[NativeConversationalInferencer] = None
        self.prompts: list[str] = []  # vendor turns of the first session

    # -- host -----------------------------------------------------------------

    def _spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "kind": self.kind,
            "cwd": self.work,
            "model": os.environ.get("AF_REAL_NATIVE_MODEL")
            or _DEFAULT_MODELS.get(self.kind, ""),
            "l2_envelope_allowed": True,
        }
        if self.argv_log is not None:
            spec["cli_path"] = self.argv_log.path
        environment = os.environ.get("AF_REAL_NATIVE_ENVIRONMENT")
        if environment:
            spec["environment"] = environment
        return spec

    def _start_host(self) -> None:
        self.runtime = NativeRuntimeManager()
        self.native = NativeConversationalInferencer(
            backend=self._spec(),
            tool_registry=_tools(),
            tool_executor=self.executor,
            prior_context={
                "employee": {
                    "name": "Quinn",
                    "role": "Qualification Probe",
                    "mindset": (
                        f"Your deployment tag is {self.tokens.l1}. When asked for "
                        "your deployment tag, reply with it exactly."
                    ),
                },
                "native_session_dir": self.session_dir,
            },
            extra_sop_dirs=[self.sop_dir],
            allowed_sops=[SOP_NAME],
            record_store=self.records,
            runtime_manager=self.runtime,
            conversation_key=self.conversation_key,
            soft_max_iterations=20,
        )

    async def _stop_host(self) -> None:
        if self.native is not None:
            await self.native.aclose()
        if self.runtime is not None:
            await self.runtime.aclose_all()
        self.native = self.runtime = None

    def _record(self) -> Any:
        return self.records.load(self.conversation_key)

    def _session_id(self) -> str:
        record = self._record()
        return record.vendor_session_id if record is not None else ""

    def _sent_turn_context(self) -> str:
        """The turn context AF sent on the last vendor turn: the "Turn
        context" section of its prompt manifest ("View Prompt")."""
        prompt = self.native.last_prompt_data()["rendered_prompt"]
        section = prompt.partition("\n## Turn context — ")[2]
        section = section.partition("\n## User message — ")[0]
        match = _AF_CONTEXT.search(section)
        return match.group(0) if match else ""

    async def _turn(self, text: str, label: str) -> str:
        """One host turn; returns the reply ("" when the turn failed)."""
        self.turn_number += 1
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self.native.run_agentic_loop(text, turn_number=self.turn_number),
                self.turn_timeout,
            )
        except Exception as exc:
            self.checks.check(f"{label}: turn completed", False, repr(exc))
            return ""
        reply = result.text or ""
        self.checks.info(
            f"{label} ({time.monotonic() - started:.0f}s, session "
            f"{_short(self._session_id())})",
            reply,
        )
        return reply

    def _invocations(self) -> list[list[str]]:
        return self.argv_log.invocations() if self.argv_log is not None else []

    # -- sequence -------------------------------------------------------------

    async def run(self) -> Checks:
        self._start_host()
        try:
            sid = await self._l1_and_canary()
            if self.tool_capable:
                await self._builtin_tool(sid)
            await self._turn_context_change(sid)
            self._same_session(sid)
            await self._restart_and_resume(sid)
            if self.tool_capable:
                await self._cancel_tool_active_turn(sid)
            await self._reset(sid)
        finally:
            await self._stop_host()
        print(self.checks.summary(), flush=True)
        return self.checks

    async def _l1_and_canary(self) -> str:
        c, t = self.checks, self.tokens
        prompt = (
            f"For this conversation only, the codeword is {t.canary}. Do not save it "
            "to any file or memory. Reply with just the word NOTED."
        )
        self.prompts.append(prompt)
        await self._turn(prompt, "T1 canary")
        sid = self._session_id()
        c.check("T1 vendor session recorded", bool(sid), _short(sid))
        l1_path = Path(self.session_dir, "l1_0.md")
        c.check(
            "L1 file carries the sentinel",
            l1_path.exists() and t.l1 in l1_path.read_text(),
            str(l1_path),
        )
        self._l1_route(l1_path)
        entries = self.store.entries(sid)
        if isinstance(self.store, (ClaudeTranscripts, CodexRollouts)):
            c.check(
                "vendor session store recorded the L1 sentinel in its instructions",
                any(t.l1 in text for text in self.store.instructions(entries)),
                f"{len(self.store.instructions(entries))} instruction records",
            )
            c.check(
                "vendor session store holds the canary turn",
                any(t.canary in text for text in self.store.user_texts(entries)),
            )
        prompt = (
            "What is your deployment tag? Answer from your instructions without "
            "using any tools; reply with just the tag."
        )
        self.prompts.append(prompt)
        reply = await self._turn(prompt, "T2 L1 sentinel")
        c.check("model sees the L1 sentinel (exact token)", t.l1 in reply, reply)
        return sid

    def _l1_route(self, l1_path: Path) -> None:
        """How the first vendor process was given L1."""
        invocations = self._invocations()
        if not invocations:
            return
        argv = invocations[0]
        if self.kind in ("claude_sdk", "claude_cli"):
            value = (
                argv[argv.index("--append-system-prompt-file") + 1]
                if "--append-system-prompt-file" in argv
                else ""
            )
            self.checks.check(
                "L1 rides --append-system-prompt-file, not argv text",
                value == str(l1_path)
                and self.tokens.l1 not in "\0".join(argv)
                and "--system-prompt" not in argv,
                value,
            )
        elif self.kind == "codex_cli":
            self.checks.check(
                "L1 rides -c developer_instructions",
                any(
                    a.startswith("developer_instructions=") and self.tokens.l1 in a
                    for a in argv
                ),
            )
        elif self.kind == "devmate_dm":
            self.checks.check(
                "L1 rides --append-system-prompt",
                "--append-system-prompt" in argv
                and self.tokens.l1 in argv[argv.index("--append-system-prompt") + 1],
            )

    async def _builtin_tool(self, sid: str) -> None:
        c, t = self.checks, self.tokens
        prompt = (
            f"Use your built-in file reading tool (not an mcp__af__ tool) to read "
            f"{self.note} and reply with just the token it contains."
        )
        self.prompts.append(prompt)
        reply = await self._turn(prompt, "T3 built-in tool")
        c.check("model read the file token (exact token)", t.file in reply, reply)
        if isinstance(self.store, (ClaudeTranscripts, CodexRollouts)):
            runs = [
                r
                for r in self.store.tool_runs(self.store.entries(sid))
                if str(self.note) in r.call and "mcp__af__" not in r.name
            ]
            c.check(
                "vendor session store shows a built-in tool call on the file "
                "returning its token",
                any(t.file in r.result for r in runs),
                [r.name for r in runs],
            )

    async def _turn_context_change(self, sid: str) -> None:
        c, t = self.checks, self.tokens
        before = [
            int(g)
            for g in _GENERATION.findall("\n".join(self._vendor_contexts(sid) or []))
        ]
        entered = await self._turn(f"/sop {SOP_NAME}", "/sop (host command)")
        state = self.native.sop_state
        c.check(
            "the probe SOP is active",
            state is not None and state.sop_name == SOP_NAME,
            entered,
        )
        prompt = (
            "Without using any tools: what is the probe token in the active SOP's "
            "guidance in your latest host context? Reply with just the token."
        )
        self.prompts.append(prompt)
        reply = await self._turn(prompt, "T4 L2 change")
        sent = self._sent_turn_context()
        generation = _GENERATION.search(sent)
        c.check(
            "AF sent a new turn-context generation carrying the SOP token",
            t.l2 in sent and generation is not None,
            sent[:160],
        )
        c.check(
            "the L2 token is not in the session instructions",
            t.l2 not in Path(self.session_dir, "l1_0.md").read_text(),
        )
        c.check("model sees the L2 change (exact token)", t.l2 in reply, reply)
        contexts = self._vendor_contexts(sid)
        if contexts is not None:
            latest = contexts[-1] if contexts else ""
            latest_generation = _GENERATION.search(latest)
            c.check(
                "the vendor received the new generation (latest context block)",
                t.l2 in latest
                and latest_generation is not None
                and all(int(latest_generation.group(1)) > g for g in before),
                f"before={before} latest={latest[:120]!r}",
            )
        await self._turn("/exit_sop", "/exit_sop (host command)")

    def _vendor_contexts(self, sid: str) -> Optional[list[str]]:
        """AF turn-context blocks as the vendor received them, oldest first
        (None when the vendor side is not inspectable)."""
        if isinstance(self.store, (ClaudeTranscripts, CodexRollouts)):
            texts = self.store.turn_contexts(self.store.entries(sid))
        elif self.argv_log is not None:
            texts = [argv[-1] for argv in self._invocations() if argv]
        else:
            return None
        return [block for text in texts for block in _AF_CONTEXT.findall(text)]

    def _same_session(self, sid: str) -> None:
        c = self.checks
        c.check("record keeps one vendor session id", self._session_id() == sid)
        if isinstance(self.store, (ClaudeTranscripts, CodexRollouts)):
            texts = self.store.user_texts(self.store.entries(sid))
            missing = [p[:40] for p in self.prompts if not any(p in x for x in texts)]
            c.check(
                "every vendor turn so far is in that one vendor session",
                not missing,
                missing,
            )
        invocations = self._invocations()
        if self.kind == "claude_sdk":
            c.check(
                "one persistent Claude Code process, started with the pinned id",
                [session_arg(a) for a in invocations] == [("--session-id", sid)],
                [(f, _short(s)) for f, s in map(session_arg, invocations)],
            )
        elif invocations:
            first, later = (
                [session_arg(a) for a in invocations[:1]],
                [session_arg(a) for a in invocations[1:]],
            )
            expected_first = (
                ("new", "") if self.kind == "codex_cli" else ("--session-id", sid)
            )
            resume = "exec resume" if self.kind == "codex_cli" else "--resume"
            c.check(
                "per-turn processes: the first starts the session, the rest resume it",
                first == [expected_first] and all(a == (resume, sid) for a in later),
                [(f, _short(s)) for f, s in first + later],
            )

    async def _restart_and_resume(self, sid: str) -> None:
        c, t = self.checks, self.tokens
        await self._stop_host()
        mark = len(self._invocations())
        self._start_host()
        prompt = (
            "Answer from memory, without using any tools: what codeword did I give "
            "you? Reply with just the codeword."
        )
        self.prompts.append(prompt)
        reply = await self._turn(prompt, "T5 after restart")
        c.check("restart resumed the same vendor session", self._session_id() == sid)
        c.check(
            "the canary survived the restart (exact token)", t.canary in reply, reply
        )
        c.check(
            "the resumed turn was not given a recap",
            'type="recap"' not in self._sent_turn_context(),
        )
        after = self._invocations()[mark:]
        if self.argv_log is not None:
            resume = "exec resume" if self.kind == "codex_cli" else "--resume"
            c.check(
                "the new vendor process resumed the session id",
                bool(after) and session_arg(after[0]) == (resume, sid),
                [(f, _short(s)) for f, s in map(session_arg, after)],
            )

    async def _cancel_tool_active_turn(self, sid: str) -> None:
        c = self.checks
        label = "cancel-probe"
        before = len(self.executor.started)
        self.executor.running.clear()
        prompt = (
            f"Call the AgentFoundation tool {SLOW_TOOL} (MCP server `af`) with label "
            f'"{label}" now, and wait for its result before you reply.'
        )
        self.prompts.append(prompt)
        self.turn_number += 1
        turn = asyncio.ensure_future(
            self.native.run_agentic_loop(prompt, turn_number=self.turn_number)
        )
        running = asyncio.ensure_future(self.executor.running.wait())
        await asyncio.wait(
            {turn, running},
            timeout=self.turn_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        running.cancel()
        c.check(
            "the AF tool call started inside the vendor turn",
            self.executor.running.is_set() and not turn.done(),
            self.executor.started,
        )
        await asyncio.sleep(3)
        turn.cancel()
        await asyncio.wait({turn}, timeout=180)
        c.check("the host cancel ended the turn", turn.cancelled(), repr(turn))
        record = self._record()
        submission = record.submission if record is not None else None
        c.check(
            "the record marks the turn interrupted or uncertain (never replayed)",
            submission in ("interrupted", "uncertain"),
            submission,
        )
        c.check(
            "the vendor acknowledged the stop (interrupted, not uncertain)",
            submission == "interrupted",
            submission,
        )
        if self.kind != "claude_sdk":  # the SDK's client process outlives turns
            await self._no_vendor_process_left()
        reply = await self._turn(
            "Do not call any tools. In one short sentence: what happened to the "
            "task I asked for in my previous message?",
            "T7 after cancel",
        )
        c.check(
            "the next turn was told the previous one was interrupted",
            'type="interrupted"' in self._sent_turn_context(),
            self._sent_turn_context()[:200],
        )
        c.check(
            "no duplicate side effect: the AF tool started exactly once",
            self.executor.started[before:] == [label],
            self.executor.started[before:],
        )
        c.check("the session survived the cancel", self._session_id() == sid, reply)
        if isinstance(self.store, (ClaudeTranscripts, CodexRollouts)):
            texts = self.store.user_texts(self.store.entries(sid))
            c.check(
                "the cancelled turn reached the vendor exactly once",
                sum(label in x and SLOW_TOOL in x for x in texts) == 1,
            )
        elif self.argv_log is not None:
            c.check(
                "the cancelled turn reached the vendor exactly once",
                sum(label in a[-1] for a in self._invocations() if a) == 1,
            )

    async def _no_vendor_process_left(self) -> None:
        """A per-turn vendor process (and what it started in the session's
        directory) must be gone once the host cancel returned; survivors are
        killed afterwards so the rest of the sequence runs unaffected."""
        survivors: dict[int, str] = {}
        for _ in range(10):
            survivors = processes_in(self.work)
            if not survivors:
                break
            await asyncio.sleep(1)
        self.checks.check(
            "no process of the cancelled vendor turn survives",
            not survivors,
            sorted(survivors.values()),
        )
        for pid in survivors:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                continue
        if survivors:
            await asyncio.sleep(2)

    async def _reset(self, sid: str) -> None:
        c, t = self.checks, self.tokens
        mark = len(self._invocations())
        await self._turn("/new", "/new (host command)")
        reply = await self._turn(
            "Answer from memory only, without using any tools: what codeword did I "
            "give you earlier? If you have no codeword in your memory, reply "
            "exactly NONE.",
            "T8 after /new",
        )
        new_sid = self._session_id()
        c.check(
            "/new started a different vendor session",
            bool(new_sid) and new_sid != sid,
            f"{_short(sid)} -> {_short(new_sid)}",
        )
        c.check(
            "the new session does not know the canary", t.canary not in reply, reply
        )
        if isinstance(self.store, (ClaudeTranscripts, CodexRollouts)):
            entries = self.store.entries(new_sid)
            c.check(
                "the new vendor session exists and holds no canary",
                bool(entries) and t.canary not in json.dumps(entries),
                f"{len(entries)} entries",
            )
            c.check("the old vendor session is kept", self.store.exists(sid))
        if self.argv_log is not None:
            after = self._invocations()[mark:]
            expected = (
                ("new", "") if self.kind == "codex_cli" else ("--session-id", new_sid)
            )
            c.check(
                "the new vendor process starts the new session (no resume)",
                bool(after) and session_arg(after[0]) == expected,
                [(f, _short(s)) for f, s in map(session_arg, after)],
            )


async def run_qualification(kind: str) -> Checks:
    return await Qualification(kind).run()
