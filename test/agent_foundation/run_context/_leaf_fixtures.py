"""Provider leaves over fake transports, for the host-purity ratchet (plan v8 P10).

Each builder returns a real leaf instance, never a test subclass, so the ratchet
charges any per-call write to the leaf class itself. Only the external boundary is
faked:

* terminal leaves and the tool: a fake CLI run as a real subprocess;
* OpenClaw: a scripted gateway socket;
* SDK leaves: a fake of the SDK client the leaf imports (``claude_agent_sdk``,
  ``openai_codex``, the devmate and metamate clients), emitting the message shapes
  the leaf parses;
* RovoChat and RovoDev serve: an ``httpx.MockTransport`` behind the leaf's real HTTP
  client (RovoDev serve's server process is a fake process object).

A builder takes the fixture directory and a ``MonkeyPatch`` that stays active while
the ratchet measures. Every fake records what it saw in a transport log, which the
behaviour tests read. The fake CLIs and the OpenClaw gateway also drive the goldens.

This is a library, not a test module: it never imports one.
"""

import asyncio
import enum
import functools
import itertools
import json
import os
import re
import shlex
import stat
import sys
import tempfile
import types
from types import SimpleNamespace as NS
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_sdk_inferencer import (
    ClaudeCodeSdkInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_cli_inferencer import (
    CodexCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_sdk_inferencer import (
    CodexSdkInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
    DevmateCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_sdk_inferencer import (
    DevmateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.kiro.kiro_cli_inferencer import (
    KiroCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_cli_inferencer import (
    MetamateCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_sdk_inferencer import (
    MetamateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw import (
    openclaw_inferencer as openclaw_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer import (
    OpenClawInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.rovochat_inferencer import (
    RovoChatInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev import (
    common as rovodev_common,
    rovodev_cli_inferencer as rovodev_module,
    rovodev_serve_inferencer as serve_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
    RovoDevCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_serve_inferencer import (
    RovoDevServeInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.tool_inferencers.tool_as_inferencer import (
    ToolAsInferencer,
)
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    RunContext,
)

# --- Fakes shared with the goldens -------------------------------------------------

# ``codex exec --json``: reports a thread id derived from the prompt, then a reply.
FAKE_CODEX = r"""
import json, sys

prompt = sys.stdin.read().strip()
thread = "thread-" + prompt.split()[-1]
for event in (
    {"type": "thread.started", "thread_id": thread},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}},
    {"type": "turn.completed", "usage": {"output_tokens": 1}},
):
    print(json.dumps(event), flush=True)
"""

FAKE_DM = r"""
import json, os, select, sys, uuid

LOG = __LOG__
argv = sys.argv[1:]
stdin = ""
if not sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
    stdin = sys.stdin.read()
with open(LOG, "a") as f:
    record = {"argv": argv, "cwd": os.getcwd(), "stdin": stdin,
              "TMPDIR": os.environ.get("TMPDIR")}
    f.write(json.dumps(record) + "\n")
sid = argv[argv.index("--resume") + 1] if "--resume" in argv else str(uuid.uuid4())
prompt = argv[-1] if argv else ""
print("Running non-interactively (dm -p)")
print("Powered by dm-core")
print("❯ " + prompt)
print("")
print("● ANSWER(" + prompt + ")")
print("  continuation line")
print("")
print("Logs available at: " + os.getcwd() + "/.dm/" + sid + ".log")
print("Resume with: dm --resume " + sid)
if "FAIL" in prompt:
    print("fatal: backend exploded", file=sys.stderr)
    sys.exit(3)
"""

FAKE_ACLI = r"""
import json, os, sys, time, uuid

LOG = __LOG__
SESSIONS = __SESSIONS__
argv = sys.argv[1:]


def flag(name):
    if name not in argv:
        return None
    nxt = argv[argv.index(name) + 1] if argv.index(name) + 1 < len(argv) else ""
    return "" if nxt.startswith("--") else nxt


sid = flag("--restore") or str(uuid.uuid4())
os.makedirs(os.path.join(SESSIONS, sid), exist_ok=True)
with open(os.path.join(SESSIONS, sid, "session_context.json"), "w") as f:
    json.dump({"workspace_path": os.getcwd()}, f)
with open(LOG, "a") as f:
    f.write(json.dumps({"argv": argv, "cwd": os.getcwd()}) + "\n")
out = flag("--output-file")
if out:
    with open(out, "w") as f:
        f.write("clean answer to %r in session %s\n" % (argv[2], sid))
sys.stderr.write("acli stderr noise\n")
sys.stdout.write("\n\x1b[1mTUI\x1b[0m thinking\n   \nTUI done\n")
sys.stdout.flush()
time.sleep(0.2)
"""


def ctx_view(ctx: Optional[RunContext]) -> Optional[Dict[str, Any]]:
    if ctx is None:
        return None
    return {"path": ctx.path, "legacy_mint": ctx.legacy_mint}


def _res(req_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "res", "id": req_id, "payload": payload}


def _delta(run_id: str, text: str) -> Dict[str, Any]:
    payload = {"runId": run_id, "stream": "assistant", "data": {"delta": text}}
    return {"type": "event", "event": "agent", "payload": payload}


def _frames(req_id: str, run_id: str, error: Optional[str], text: str) -> List[Any]:
    ack = _res(req_id, {"status": "accepted", "runId": run_id})
    if error is not None:
        return [ack, _res(req_id, {"status": "error", "error": {"message": error}})]
    half = len(text) // 2
    final = _res(req_id, {"status": "ok", "result": {"payloads": [{"text": text}]}})
    return [ack, _delta(run_id, text[:half]), _delta(run_id, text[half:]), final]


class _FakeSocket:
    def __init__(self, gateway: "FakeGateway") -> None:
        self._gateway = gateway
        self._frames: List[Any] = []

    async def send(self, raw: str) -> None:
        self._frames.extend(self._gateway.serve(json.loads(raw)))

    async def recv(self) -> str:
        if not self._frames:
            raise RuntimeError("fake gateway socket drained")
        return json.dumps(self._frames.pop(0))

    async def close(self) -> None:
        pass


class FakeGateway:
    """Scripted gateway + pod. ``script`` holds one outcome per agent request
    (an error message; once exhausted every request is answered)."""

    def __init__(self, script: Sequence[str] = (), transcripts: Sequence[str] = ()):
        self.script = list(script)
        self.transcripts = set(transcripts)
        self.log: List[Dict[str, Any]] = []

    def connect(self) -> _FakeSocket:
        return _FakeSocket(self)

    def run_subprocess(self, cmd: str, timeout: int = 0) -> Tuple[str, str, int]:
        match = re.search(r"sessions/(.+?)\.jsonl", cmd)
        if match is None:
            raise ValueError(f"not a transcript probe: {cmd!r}")
        sid = match.group(1)
        result = "EXISTS" if sid in self.transcripts else "NEW"
        self.log.append(
            {"method": "transcript_probe", "session_id": sid, "result": result}
        )
        return result, "", 0

    def serve(self, frame: Dict[str, Any]) -> List[Any]:
        params = dict(frame["params"])
        prompt, sid = params.pop("message"), params.pop("sessionId")
        n = 1 + sum(1 for c in self.log if c["method"] == frame["method"])
        self.log.append(
            {
                "method": frame["method"],
                "prompt": prompt,
                "session_id": sid,
                "params": params,
                "active_run_context": ctx_view(active_run_context()),
            }
        )
        self.transcripts.add(sid)
        error = self.script.pop(0) if self.script else None
        return _frames(frame["id"], f"run-{n}", error, f"answer {n} to: {prompt}")


FAKE_TOOL = "import sys\nprint('tool says', ' '.join(sys.argv[1:]))\n"

# claude -p, prompt on stdin: one JSON result (sync), or stream events then the
# result (async); the session is ``sid-<last word of the prompt>``.
FAKE_CLAUDE = r"""
import json, sys
prompt = sys.stdin.read().strip()
result = {"type": "result", "subtype": "success", "result": "reply to " + prompt,
          "session_id": "sid-" + (prompt.split() or ["empty"])[-1]}
if "stream-json" in sys.argv:
    delta = {"type": "content_block_delta",
             "delta": {"type": "text_delta", "text": result["result"]}}
    print(json.dumps({"type": "stream_event", "event": delta}), flush=True)
print(json.dumps(result), flush=True)
"""

# kiro-cli: answers the prompt (stdin, else the last positional argument).
FAKE_KIRO = r"""
import select, sys
argv = sys.argv[1:]
stdin = ""
if not sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
    stdin = sys.stdin.read()
positional = [a for a in argv[2:] if not a.startswith("--")]
prompt = stdin.strip() or (positional[-1] if positional else "")
print("> " + prompt)
print("kiro answer to " + prompt)
"""

# buck, as the metamate CLI runs it: query_metamate's delimited stdout.
FAKE_BUCK = r"""
import sys
argv = sys.argv[1:]
query = argv[argv.index("--query") + 1] if "--query" in argv else ""
rule = "-" * 72
print("Building... (fake buck)")
print(rule)
print("RESPONSE")
print(rule)
print("answer to: " + query)
print(rule)
"""


def _dir(tmp, *parts):
    path = os.path.join(tmp, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def _script(tmp, name, text):
    path = os.path.join(_dir(tmp), name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _executable(tmp, name, text):
    """An executable ``name`` running the Python ``text``: a ``sh`` wrapper, since a
    shebang naming the interpreter can exceed the kernel's length limit (buck)."""
    script = _script(tmp, f"{name}.py", text)
    wrapper = (
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(script)} "$@"\n'
    )
    path = _script(tmp, name, wrapper)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


def _last_word(prompt):
    words = str(prompt).strip().split()
    return words[-1] if words else "empty"


# --- Terminal leaves and the tool --------------------------------------------------


def claude_cli(tmp, mp):
    mp.delenv("CLAUDE_CODE_COMMAND", raising=False)
    mp.delenv("CLAUDE_CODE_MAX_CONCURRENCY", raising=False)
    script = _script(tmp, "fake_claude.py", FAKE_CLAUDE)
    return ClaudeCodeCliInferencer(
        claude_command=f"{sys.executable} {script}", target_path=_dir(tmp, "work")
    )


def codex_cli(tmp, mp):
    script = _script(tmp, "fake_codex.py", FAKE_CODEX)
    return CodexCliInferencer(
        codex_command=f"{sys.executable} {script}", target_path=_dir(tmp, "work")
    )


def devmate_cli(tmp, mp):
    log = repr(os.path.join(tmp, "dm.log"))
    script = _script(tmp, "fake_dm.py", FAKE_DM.replace("__LOG__", log))
    repo = _dir(tmp, "repo")
    _dir(repo, ".sl")
    return DevmateCliInferencer(
        target_path=repo,
        cli_binary=f"{shlex.quote(sys.executable)} -I {shlex.quote(script)}",
        cache_folder=_dir(tmp, "cache"),
        dump_output=True,
    )


def rovodev_cli(tmp, mp):
    sessions = _dir(tmp, "sessions")
    script = FAKE_ACLI.replace("__LOG__", repr(os.path.join(tmp, "acli.log")))
    script = _script(
        tmp, "fake_acli.py", script.replace("__SESSIONS__", repr(sessions))
    )
    mp.setattr(tempfile, "tempdir", _dir(tmp, "tmp"))
    for name in ("find_latest_session_id", "ensure_session_metadata"):
        real = getattr(rovodev_common, name)
        mp.setattr(rovodev_module, name, functools.partial(real, sessions_dir=sessions))
    return RovoDevCliInferencer(
        acli_path=f"{sys.executable} {script}", target_path=_dir(tmp, "ws")
    )


def kiro_cli(tmp, mp):
    bindir = _dir(tmp, "bin")
    _executable(bindir, "kiro-cli", FAKE_KIRO)
    mp.setenv("PATH", bindir + os.pathsep + os.environ.get("PATH", ""))
    return KiroCliInferencer(target_path=_dir(tmp, "work"))


def metamate_cli(tmp, mp):
    bindir = _dir(tmp, "bin")
    _executable(bindir, "buck", FAKE_BUCK)
    mp.setenv("PATH", bindir + os.pathsep + os.environ.get("PATH", ""))
    return MetamateCliInferencer(target_path=_dir(tmp, "work"))


def tool_as(tmp, mp):
    script = _script(tmp, "tool.py", FAKE_TOOL)
    return ToolAsInferencer(
        tool_name="echoer",
        command=["python3", script],
        args_template=["fixed"],
        env={"PATH": sys.exec_prefix + "/bin:/usr/bin:/bin"},
    )


# --- OpenClaw ----------------------------------------------------------------------


def openclaw(tmp, mp):
    gateway = FakeGateway()
    mp.setattr(openclaw_module, "run_subprocess", gateway.run_subprocess)

    async def connect(self):
        return gateway.connect()

    mp.setattr(OpenClawInferencer, "_ws_connect", connect)
    return OpenClawInferencer(auth_token="tok")


# --- claude_agent_sdk --------------------------------------------------------------


def install_fake_claude_sdk(mp, log):
    """``ClaudeSDKClient``: connect / query / receive_response / disconnect, with
    the SDK's own message types; the session is ``sid-<last word of the prompt>``."""
    import claude_agent_sdk
    from claude_agent_sdk.types import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
    )

    class FakeClaudeSDKClient:
        def __init__(self, options=None, transport=None):
            self.options = options
            self.prompt = None

        async def connect(self, prompt=None):
            log.append(("connect", getattr(self.options, "resume", None)))

        async def query(self, prompt, session_id="default"):
            self.prompt = prompt

        async def receive_response(self):
            word = _last_word(self.prompt)
            yield AssistantMessage(
                content=[
                    TextBlock(text=f"reply to {word}"),
                    ToolUseBlock(id=f"tu-{word}", name="Read", input={"path": "x"}),
                ],
                model="fake-claude",
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id=f"sid-{word}",
                result=f"reply to {word}",
            )

        async def disconnect(self):
            log.append(("disconnect", None))

    mp.setattr(claude_agent_sdk, "ClaudeSDKClient", FakeClaudeSDKClient)


def claude_sdk(tmp, mp):
    install_fake_claude_sdk(mp, [])
    return ClaudeCodeSdkInferencer(target_path=_dir(tmp, "work"))


# --- openai_codex ------------------------------------------------------------------


class _CodexSandbox(enum.Enum):
    read_only = "read-only"
    workspace_write = "workspace-write"
    full_access = "danger-full-access"


class _CodexApprovalMode(enum.Enum):
    auto_review = "auto_review"
    deny_all = "deny_all"


def _codex_note(method, **payload):
    return NS(method=method, payload=NS(**payload))


class _CodexTurn:
    def __init__(self, thread_id, prompt):
        self.thread_id, self.prompt = thread_id, prompt

    async def stream(self):
        usage = NS(output_tokens=7, total_tokens=11)
        yield _codex_note("turn/started", thread_id=self.thread_id)
        yield _codex_note(
            "item/agentMessage/delta", delta=f"codex reply to {self.prompt}"
        )
        yield _codex_note("item/completed", item=NS(root=NS(type="commandExecution")))
        yield _codex_note("turn/completed", turn=NS(usage=usage), token_usage=None)


class _CodexThread:
    def __init__(self, thread_id):
        self.id = thread_id

    async def turn(self, prompt):
        return _CodexTurn(self.id, prompt)


class _FakeAsyncCodex:
    """``AsyncCodex``: an async context manager that starts or resumes threads."""

    def __init__(self, config=None, *, log, ids):
        self._log, self._ids = log, ids

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def thread_start(self, **kwargs):
        thread = _CodexThread(f"thread-{next(self._ids)}")
        self._log.append(("thread_start", thread.id))
        return thread

    async def thread_resume(self, thread_id, **kwargs):
        self._log.append(("thread_resume", thread_id))
        return _CodexThread(thread_id)

    async def close(self):
        self._log.append(("close", None))


def install_fake_codex_sdk(mp, log):
    """The ``openai_codex`` module: ``AsyncCodex`` starts or resumes threads whose
    turns stream the notifications the leaf reads."""
    module = types.ModuleType("openai_codex")
    module.Sandbox = _CodexSandbox
    module.ApprovalMode = _CodexApprovalMode
    module.CodexConfig = NS
    module.AsyncCodex = functools.partial(
        _FakeAsyncCodex, log=log, ids=itertools.count(1)
    )
    module.AsyncThread = _CodexThread
    mp.setitem(sys.modules, "openai_codex", module)


def codex_sdk(tmp, mp):
    install_fake_codex_sdk(mp, [])
    return CodexSdkInferencer(target_path=_dir(tmp, "work"))


# --- devmate SDK -------------------------------------------------------------------


class _Signal:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    async def emit(self, *args, **kwargs):
        for handler in self.handlers:
            await handler(*args, **kwargs)


class _DevmateStatus(enum.Enum):
    RUNNING = 2
    COMPLETED = 3


def install_fake_devmate_sdk(mp, log):
    """``DevmateSDKClient``: ``start_session`` resumes ``previous_session_id`` or
    starts ``dm-<n>``, emitting session, step and growing action events."""
    ids = itertools.count(1)

    class FakeDevmateSDKClient:
        def __init__(self, config_file_path, usecase, config_vars=None, **kwargs):
            self.config_vars = dict(config_vars or {})
            self.events = NS(
                on_session=_Signal(),
                on_step=_Signal(),
                on_action=_Signal(),
                on_error=_Signal(),
            )

        async def start_session(self, session_id=None, previous_session_id=None, **kw):
            log.append(("start_session", previous_session_id))
            sid = previous_session_id or f"dm-{next(ids)}"
            answer = f"ANSWER({self.config_vars.get('prompt', '')})"
            await self.events.on_session.emit(NS(id=sid, status=_DevmateStatus.RUNNING))
            await self.events.on_step.emit(NS(id="step-1", number=1))
            for text in (answer[: len(answer) // 2], answer):
                await self.events.on_action.emit(NS(id="act-1", output=NS(info=text)))
            await self.events.on_session.emit(
                NS(id=sid, status=_DevmateStatus.COMPLETED)
            )

        async def stop_session(self, timeout=35.0):
            pass

    module = types.ModuleType("devai.devmate_sdk.python.devmate_client")
    module.DevmateSDKClient = FakeDevmateSDKClient
    mp.setitem(sys.modules, module.__name__, module)


def devmate_sdk(tmp, mp):
    install_fake_devmate_sdk(mp, [])
    return DevmateSDKInferencer(target_path=_dir(tmp, "work"))


# --- metamate SDK ------------------------------------------------------------------


def install_fake_metamate_sdk(mp, log):
    """``MetamateGraphQLClient``: ``engine_start_v2`` starts or continues a
    conversation; each poll reveals more of the answer, then completes."""
    conversations = {}
    ids = itertools.count(1)

    class FakeMetamateGraphQLClient:
        def __init__(self, cat=None):
            pass

        def engine_start_v2(
            self,
            prompt,
            request_id,
            conversation_uuid=None,
            conversation_fbid=None,
            **kw,
        ):
            log.append(("engine_start_v2", conversation_uuid))
            cuuid = conversation_uuid or f"conv-{next(ids)}"
            answer = f"answer to: {prompt.splitlines()[-1]}"
            conversations[cuuid] = {"answer": answer, "polls": 0}
            return NS(conversation=NS(uuid=cuuid, fbid=f"fbid-{cuuid}"))

        def get_conversation_for_stream(self, conversation_uuid):
            conversation = conversations[conversation_uuid]
            conversation["polls"] += 1
            done = conversation["polls"] >= 2
            answer = conversation["answer"]
            text = answer if done else answer[: len(answer) // 2]
            status = "MessageStatus.COMPLETED" if done else "MessageStatus.IN_PROGRESS"
            return [
                NS(
                    message=NS(role="ASSISTANT", status=status, block_uuids=["b1"]),
                    block=None,
                    conversation=NS(fbid=f"fbid-{conversation_uuid}"),
                ),
                NS(
                    message=None,
                    block=NS(uuid="b1", content=NS(markdown=NS(value=text))),
                ),
            ]

    module = types.ModuleType("msl.metamate.cli.metamate_graphql")
    module.MetamateGraphQLClient = FakeMetamateGraphQLClient
    mp.setitem(sys.modules, module.__name__, module)
    mp.delenv("METAMATE_USE_STANDALONE", raising=False)


class _Scope:
    def to_directive(self):
        return "[scope: fbsource]"


async def fixed_code_scope(task):
    """A code-scope judge that always picks fbsource, without a model call."""
    return _Scope()


def metamate_sdk(tmp, mp):
    install_fake_metamate_sdk(mp, [])
    return MetamateSDKInferencer(
        poll_interval_seconds=0.01, code_scope_judge=fixed_code_scope
    )


# --- RovoChat ----------------------------------------------------------------------


class _RecordedStream(httpx.AsyncByteStream):
    """An NDJSON response body that records the run context it is closed under."""

    def __init__(self, lines, log):
        self._body = "".join(json.dumps(line) + "\n" for line in lines).encode()
        self._log = log

    async def __aiter__(self):
        yield self._body

    async def aclose(self):
        ctx = active_run_context()
        self._log.append(("stream_closed", None if ctx is None else ctx.path))


def install_fake_rovochat(mp, log):
    """RovoChat's HTTP API: conversation creation, then an NDJSON message stream
    (handshake, two answer parts, the final response); each message is logged
    with its conversation."""
    conversations = itertools.count(1)

    def handler(request):
        if not request.url.path.rstrip("/").endswith("/stream"):
            cid = f"conv-{next(conversations)}"
            log.append(("create_conversation", cid))
            return httpx.Response(200, json={"id": cid, "agent": {"id": "agent-1"}})
        # .../conversation/<id>/message/stream
        log.append(("send_message", request.url.path.split("/")[-3]))
        answer = f"answer to {_message_text(json.loads(request.content or b'{}'))}."
        lines = [
            {"type": "RECONNECT_SUPPORTED"},
            {"type": "ANSWER_PART", "message": {"content": answer[: len(answer) // 2]}},
            {"type": "ANSWER_PART", "message": {"content": answer}},
            {"type": "FINAL_RESPONSE", "message": {"content": answer}},
        ]
        return httpx.Response(
            200,
            stream=_RecordedStream(lines, log),
            headers={"Content-Type": "application/x-ndjson"},
        )

    _mock_httpx(mp, handler)
    for var in ("ROVOCHAT_BASE_URL", "ROVOCHAT_CLOUD_ID", "JIRA_URL"):
        mp.delenv(var, raising=False)


def _message_text(body):
    if isinstance(body, dict):
        for key, value in body.items():
            if key in ("text", "content") and isinstance(value, str):
                return value
            found = _message_text(value)
            if found:
                return found
    if isinstance(body, list):
        for value in body:
            found = _message_text(value)
            if found:
                return found
    return None


def _mock_httpx(mp, handler):
    real = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    mp.setattr(httpx, "AsyncClient", client)


def rovochat(tmp, mp):
    install_fake_rovochat(mp, [])
    return RovoChatInferencer(
        base_url="https://rovo.example.test", cloud_id="cloud-1", uct_token="uct"
    )


# --- RovoDev serve -----------------------------------------------------------------


class _FakeServer:
    """The ``acli rovodev serve`` process: running until signalled."""

    pid = 4242
    stderr = None

    def __init__(self, log):
        self.returncode = None
        self._log = log

    def send_signal(self, sig):
        self._log.append(("signal", sig))
        self.returncode = -sig

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


def install_fake_rovodev_serve(mp, log):
    """The serve process and its HTTP API: healthcheck, set the chat message, an
    SSE chat stream (a tool call, the answer, the end of the run), reset (logged
    with the server's port)."""
    message = {"text": ""}

    async def spawn(*cmd, **kwargs):
        log.append(("spawn", cmd[1:3]))
        return _FakeServer(log)

    def handler(request):
        path = request.url.path
        if path == "/v3/set_chat_message":
            message["text"] = json.loads(request.content)["message"]
        if path == "/v3/reset":
            log.append(("reset", request.url.port))
        if path != "/v3/stream_chat":
            return httpx.Response(200, json={"status": "ok"})
        answer = json.dumps({"delta": f"answer to {message['text']}"})
        sse = (
            "event: tool_call_start\ndata: {}\n\n"
            f"event: text_delta\ndata: {answer}\n\n"
            "event: agent_run_end\ndata: {}\n\n"
        )
        return httpx.Response(
            200, text=sse, headers={"Content-Type": "text/event-stream"}
        )

    mp.setattr(serve_module, "find_acli_binary", lambda path: "acli")
    mp.setattr(serve_module, "find_available_port", lambda: 1)
    mp.setattr(asyncio, "create_subprocess_exec", spawn)
    _mock_httpx(mp, handler)


def rovodev_serve(tmp, mp):
    install_fake_rovodev_serve(mp, [])
    return RovoDevServeInferencer(acli_path="acli", target_path=_dir(tmp, "work"))


LEAF_FIXTURES = {
    "claude_cli": claude_cli,
    "codex_cli": codex_cli,
    "devmate_cli": devmate_cli,
    "rovodev_cli": rovodev_cli,
    "kiro_cli": kiro_cli,
    "metamate_cli": metamate_cli,
    "tool_as": tool_as,
    "openclaw": openclaw,
    "claude_sdk": claude_sdk,
    "codex_sdk": codex_sdk,
    "devmate_sdk": devmate_sdk,
    "metamate_sdk": metamate_sdk,
    "rovochat": rovochat,
    "rovodev_serve": rovodev_serve,
}
