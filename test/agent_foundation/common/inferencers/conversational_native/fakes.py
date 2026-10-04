"""Test doubles for the native conversational orchestrator.

``FakeBackend`` replays scripted vendor turns. Tool steps call the REAL
AgentFoundation bridge handlers (from a foreign task, like a vendor SDK) and
the real ``SessionHooks``, so tests exercise production code paths.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
import uuid
from typing import Any, Optional

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.events import (
    MessageEnd,
    TextDelta,
    TurnEnd,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.session.backend import (
    BackendCapabilities,
    CallerTools,
    L2Channel,
    SessionOpenRequest,
    TurnRequest,
)

FAKE_CAPABILITIES = BackendCapabilities(
    kind="claude_sdk",
    caller_tools=CallerTools.INPROCESS,
    l2_channels=(L2Channel.HOOK, L2Channel.ENVELOPE),
    pinned_session_id=True,
    exact_fork=True,
    turn_stop_hook=True,
    subagent_attribution=True,
    compaction_signal=True,
    persistent_process=True,
    slash_passthrough=("compact",),
)


def text(value: str) -> tuple:
    return ("text", value)


def tools(*calls: tuple[str, dict]) -> tuple:
    """One assistant message that calls several tools."""
    return ("tools", list(calls))


def subagent_tool(name: str, args: dict) -> tuple:
    return ("subagent_tool", name, args)


def turn_end(backend: FakeBackend) -> TurnEnd:
    return TurnEnd(session_id=backend.session_id, stop_reason="end_turn", num_turns=1)


class FakeBackend:
    capabilities = FAKE_CAPABILITIES

    def __init__(
        self,
        spec: Any,
        scripts: list,
        *,
        capabilities: Optional[BackendCapabilities] = None,
        **_runtime: Any,
    ) -> None:
        if capabilities is not None:
            self.capabilities = capabilities
        self.spec = spec
        self._scripts = scripts
        self.open_request: Optional[SessionOpenRequest] = None
        self.turn_requests: list[TurnRequest] = []
        self.tool_results: list[tuple[str, str, bool]] = []
        self.l2_seen: list[str] = []
        self.closed = False
        self.interrupts = 0
        self.interrupted = asyncio.Event()
        self.models_set: list[str] = []
        self._session_id = ""

    @property
    def session_id(self) -> str:
        return self._session_id

    async def open(self, request: SessionOpenRequest) -> None:
        self.open_request = request
        if request.fork_from is not None:
            self._session_id = (
                f"fork-of-{request.fork_from[0]}-at-{request.fork_from[1]}"
            )
        else:
            self._session_id = request.session_id or uuid.uuid4().hex

    async def call_tool(
        self,
        name: str,
        args: dict,
        *,
        tool_use_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> tuple[Any, bool]:
        """Run one AF tool exactly as a vendor would (hooks around the real
        handler, from a foreign task). Returns ``(result, stop_requested)``."""
        hooks = self.open_request.hooks
        tool_use_id = tool_use_id or f"tu_{uuid.uuid4().hex[:6]}"
        deny = await hooks.before_af_tool(name, tool_use_id, agent_id)
        if deny:
            self.tool_results.append((name, deny, True))
            return None, False
        handlers = {t.name: t.handler for t in self.open_request.tools}
        result = await asyncio.get_running_loop().create_task(
            handlers[name](dict(args))
        )
        self.tool_results.append((name, result.text, result.is_error))
        return result, await hooks.after_af_tool(name, tool_use_id)

    async def run_turn(self, request: TurnRequest):
        self.turn_requests.append(request)
        hooks = self.open_request.hooks
        self.l2_seen.append(hooks.l2_for_turn())
        steps = self._scripts.pop(0) if self._scripts else [text("ok")]
        if callable(steps):
            # A programmable turn: ``steps(backend, request)`` is an async
            # generator yielding vendor events.
            async for event in steps(self, request):
                yield event
            return
        handlers = {t.name: t.handler for t in self.open_request.tools}
        pending_text = ""
        message_no = 0
        stopped = False
        for step in steps:
            if step[0] == "text":
                message_id = f"m{message_no}"
                pending_text += step[1]
                yield TextDelta(message_id=message_id, text=step[1])
            elif step[0] == "tools":
                message_id = f"m{message_no}"
                ids = [f"tu_{uuid.uuid4().hex[:6]}" for _ in step[1]]
                yield MessageEnd(
                    message_id=message_id,
                    text=pending_text,
                    tool_use_ids=tuple(ids),
                    af_tool_use_ids=tuple(ids),
                    message_uuid=f"uuid-{message_id}",
                )
                pending_text = ""
                message_no += 1
                for tool_use_id, (name, args) in zip(ids, step[1]):
                    deny = await hooks.before_af_tool(name, tool_use_id, None)
                    if deny:
                        self.tool_results.append((name, deny, True))
                        continue
                    result = await asyncio.get_running_loop().create_task(
                        handlers[name](dict(args))
                    )
                    self.tool_results.append((name, result.text, result.is_error))
                    if await hooks.after_af_tool(name, tool_use_id):
                        stopped = True
                if stopped:
                    break
            elif step[0] == "subagent_tool":
                deny = await hooks.before_af_tool(step[1], "tu_sub", "agent-1")
                self.tool_results.append((step[1], deny or "", bool(deny)))
        if pending_text and not stopped:
            yield MessageEnd(
                message_id=f"m{message_no}",
                text=pending_text,
                message_uuid=f"uuid-m{message_no}",
            )
        yield TurnEnd(session_id=self._session_id, stop_reason="end_turn", num_turns=1)

    def fork_message_map(self, source: str, forked: str) -> "_RenamingMap":
        """Like a real fork: every copied message gets a new id in ``forked``."""
        return _RenamingMap(forked)

    async def interrupt(self) -> None:
        self.interrupts += 1
        self.interrupted.set()

    async def set_model(self, model: str) -> None:
        self.models_set.append(model)
        self.spec.model = model

    async def close(self) -> None:
        self.closed = True


class FakeBackendFactory:
    """Callable ``backend_factory`` exposing ``capabilities`` like a class."""

    capabilities = FAKE_CAPABILITIES

    def __init__(
        self,
        scripts: Optional[list] = None,
        *,
        capabilities: Optional[BackendCapabilities] = None,
    ) -> None:
        self.scripts = scripts if scripts is not None else []
        self.instances: list[FakeBackend] = []
        if capabilities is not None:
            self.capabilities = capabilities

    def __call__(self, spec: Any, **runtime: Any) -> FakeBackend:
        backend = FakeBackend(
            spec, self.scripts, capabilities=self.capabilities, **runtime
        )
        self.instances.append(backend)
        return backend

    @property
    def last(self) -> FakeBackend:
        return self.instances[-1]


class ModelRefusingFactory(FakeBackendFactory):
    """Backends whose live model switch fails (e.g. the control channel), so
    the session is reopened on the new model."""

    def __call__(self, spec: Any, **runtime: Any) -> FakeBackend:
        backend = super().__call__(spec, **runtime)

        async def refuse(model: str) -> None:
            raise RuntimeError("set_model refused")

        backend.set_model = refuse
        return backend


class RecordingInteractive:
    """Minimal host transport: records streamed rounds and widgets, answers
    widgets from a scripted queue."""

    def __init__(self, answers: Optional[list] = None) -> None:
        self.answers = list(answers or [])
        self.streamed: list[str] = []
        self.widgets: list[Any] = []
        self.round_contexts: list[Any] = []
        self.turn_boundaries: list[int] = []
        self.boundary_cache_folders: list[str] = []
        self.persisted: list[Any] = []

    async def stream_token_batches(
        self, tokens, session_id, send_stream_end=False, turn_number=0
    ):
        chunks = []
        async for chunk, _meta in tokens:
            chunks.append(chunk)
        text = "".join(chunks)
        self.streamed.append(text)
        return text

    async def asend_response(
        self, text, flag=None, input_mode=None, prompt_data=None, **_kw
    ):
        self.widgets.append({"text": text, "input_mode": input_mode})

    async def aget_input(self):
        return self.answers.pop(0) if self.answers else None

    def set_round_context(self, ctx):
        self.round_contexts.append(ctx)

    async def send_turn_boundary(self, session_id, turn_number=0, cache_folder=""):
        self.turn_boundaries.append(turn_number)
        self.boundary_cache_folders.append(cache_folder)

    def persist_pending_widget(self, tools, action_tools, blob):
        self.persisted.append(
            {"tools": tools, "action_tools": action_tools, "blob": blob}
        )


class _RenamingMap:
    """Maps any old message id to its id in the forked session."""

    def __init__(self, forked: str) -> None:
        self._forked = forked

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str)

    def __getitem__(self, key: str) -> str:
        return f"{self._forked}:{key}"


def scripted_cli(stdout: str, stderr: str, rc: int) -> str:
    """A stand-in vendor CLI that prints ``stdout``, writes ``stderr`` and
    exits with ``rc`` (a POSIX shell script, so it runs under any test
    launcher's interpreter)."""
    directory = tempfile.mkdtemp(prefix="fake_cli_")
    out, err = os.path.join(directory, "stdout"), os.path.join(directory, "stderr")
    with open(out, "w") as f:
        f.write(stdout)
    with open(err, "w") as f:
        f.write(stderr)
    path = os.path.join(directory, "cli")
    with open(path, "w") as f:
        f.write(
            f"#!/bin/sh\ncat {shlex.quote(out)}\ncat {shlex.quote(err)} >&2\nexit {int(rc)}\n"
        )
    os.chmod(path, 0o700)
    return path


_GATED_CLI = """#!/bin/sh
d={directory}
n=$(( $(cat "$d/count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$d/count"
for arg; do last=$arg; done
printf '%s' "$last" > "$d/prompt$n"
k=0
while [ -f "$d/out${{n}}_$k" ]; do
  if [ "$k" -gt 0 ]; then
    while [ ! -f "$d/gate${{n}}_$k" ]; do sleep 0.02; done
  fi
  cat "$d/out${{n}}_$k"
  : > "$d/printed${{n}}_$k"
  k=$((k + 1))
done
exit 0
"""


class GatedCli:
    """A stand-in vendor CLI run once per turn: invocation ``n`` prints chunk
    0 of ``turns[n - 1]`` and each later chunk ``k`` once ``release(n, k)``
    was called (``printed(n, k)`` after it did), then exits 0; its last
    argument (the prompt) is kept."""

    def __init__(self, turns: list[list[str]]) -> None:
        self.directory = tempfile.mkdtemp(prefix="gated_cli_")
        for n, chunks in enumerate(turns, start=1):
            for k, chunk in enumerate(chunks):
                with open(os.path.join(self.directory, f"out{n}_{k}"), "w") as f:
                    f.write(chunk)
        self.path = os.path.join(self.directory, "cli")
        with open(self.path, "w") as f:
            f.write(_GATED_CLI.format(directory=shlex.quote(self.directory)))
        os.chmod(self.path, 0o700)

    def release(self, turn: int, chunk: int) -> None:
        open(os.path.join(self.directory, f"gate{turn}_{chunk}"), "w").close()

    def printed(self, turn: int, chunk: int) -> bool:
        return os.path.exists(os.path.join(self.directory, f"printed{turn}_{chunk}"))

    def prompt(self, turn: int) -> str:
        with open(os.path.join(self.directory, f"prompt{turn}")) as f:
            return f.read()
