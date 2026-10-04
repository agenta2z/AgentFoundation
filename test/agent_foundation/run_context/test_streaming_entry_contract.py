"""Public-entry contract (plan §4.4, I9).

(a) Every public entry runs its transport under the caller's ``run_context`` and
inside the instance's own invocation frame, opened by that entry: a stub leaf
covers the base templates, and claude_code, rovodev and devmate are probed at
``construct_command`` and OpenClaw at ``_ws_connect`` (codex and kiro share the
CLI entry shape). All four entries agree on the stub's output, and a configured
fan-out runs under the host ctx instead of the transport.

(a') Between yields, every streaming entry leaves the consumer's context exactly as
it was (``copy_context()`` unchanged): no run context, frame or leaf ContextVar is
held across a yield (B29).

(b) Every public-entry override under ``common/inferencers`` is listed in
``_ALLOWED`` with the delegation calls it must make; unlisted and stale entries
fail. Public streaming overrides are plain methods (no generator body), except
the documented Conversational bypass.
"""

import ast
import asyncio
import contextvars
import os
from typing import Any

import pytest
from agent_foundation.common.inferencers import inferencer_base
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw import (
    openclaw_inferencer as oc_module,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.openclaw.openclaw_inferencer import (
    OpenClawInferencer,
)
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    frame_for,
    RunContext,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from attr import attrib, attrs

from . import (
    test_golden_openclaw as oc_rig,
    test_golden_streaming_claude_code as cc_rig,
    test_golden_streaming_devmate as dm_rig,
    test_golden_streaming_rovodev as rovo_rig,
)

ENTRIES = ("ainfer", "infer", "ainfer_streaming", "infer_streaming")
STREAMING = ("ainfer_streaming", "infer_streaming")
CONVERSATIONAL = "agentic_inferencers/conversational/conversational_inferencer.py"


async def _drain(agen):
    return [chunk async for chunk in agen]


def _invoke(inst, entry, prompt, **kwargs):
    if entry == "ainfer":
        return asyncio.run(inst.ainfer(prompt, **kwargs))
    if entry == "infer":
        return inst.infer(prompt, **kwargs)
    if entry == "ainfer_streaming":
        return "".join(asyncio.run(_drain(inst.ainfer_streaming(prompt, **kwargs))))
    return "".join(inst.infer_streaming(prompt, **kwargs))


@attrs
class _Stub(StreamingInferencerBase):
    seen: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.seen.append(active_run_context())
        return "abc"

    async def _ainfer_streaming(self, prompt, **kwargs):
        self.seen.append(active_run_context())
        for chunk in ("a", "b", "c"):
            yield chunk
            await asyncio.sleep(0)


@attrs
class _FanoutStub(_Stub):
    fanout_ctx: list = attrib(factory=list, kw_only=True)

    @property
    def _delegates_execution(self) -> bool:
        return True

    def _should_fan_out(self) -> bool:
        return True

    def _run_fanout(self, inference_input, inference_config, kwargs):
        self.fanout_ctx.append(active_run_context())
        return "FANOUT", {}

    async def _arun_fanout(self, inference_input, inference_config, kwargs):
        self.fanout_ctx.append(active_run_context())
        return "FANOUT", {}


@pytest.mark.parametrize("entry", ENTRIES)
def test_stub_transport_runs_under_explicit_ctx(entry):
    stub = _Stub()
    ctx = RunContext.root().child("leaf")
    assert _invoke(stub, entry, "q", run_context=ctx) == "abc"
    assert stub.seen and all(seen is ctx for seen in stub.seen)


# The frame entry that each public entry opens (``_(a)infer_single`` for the
# non-streaming ones, the streaming templates otherwise).
FRAME_ENTRY = {
    "ainfer": "ainfer",
    "infer": "infer",
    "ainfer_streaming": "ainfer_streaming",
    "infer_streaming": "infer_streaming",
}


def _frame_view(inst, ctx):
    frame = frame_for(inst)
    if frame is None:
        return None
    return (frame.entry, frame.ctx is ctx, frame.mode)


@attrs
class _FrameStub(_Stub):
    frames: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.frames.append(_frame_view(self, active_run_context()))
        return super()._infer(inference_input, inference_config, **kwargs)

    async def _ainfer_streaming(self, prompt, **kwargs):
        self.frames.append(_frame_view(self, active_run_context()))
        async for chunk in super()._ainfer_streaming(prompt, **kwargs):
            yield chunk


@pytest.mark.parametrize("entry", ENTRIES)
def test_stub_transport_runs_inside_its_own_frame(entry):
    stub = _FrameStub()
    ctx = RunContext.root().child("leaf")
    assert _invoke(stub, entry, "q", run_context=ctx) == "abc"
    assert stub.frames == [(FRAME_ENTRY[entry], True, "host")]
    assert frame_for(stub) is None


@pytest.mark.parametrize("entry", ENTRIES)
def test_fanout_runs_under_explicit_ctx_instead_of_transport(entry):
    stub = _FanoutStub()
    ctx = RunContext.root().child("leaf")
    assert _invoke(stub, entry, "q", run_context=ctx) == "FANOUT"
    assert stub.fanout_ctx == [ctx] and stub.fanout_ctx[0] is ctx
    assert stub.seen == []


class _CtxProbe:
    def construct_command(self, *args, **kwargs):
        ctx = active_run_context()
        self.__dict__.setdefault("probe_ctx", []).append(ctx)
        self.__dict__.setdefault("probe_frame", []).append(_frame_view(self, ctx))
        return super().construct_command(*args, **kwargs)


class _ClaudeProbe(_CtxProbe, cc_rig._RecordingClaude):
    pass


class _RovoProbe(_CtxProbe, rovo_rig._RecordingRovoDev):
    pass


class _DevmateProbe(_CtxProbe, dm_rig._RecordingDevmate):
    pass


def _cli_leaf(leaf, tmp_path, monkeypatch):
    if leaf == "claude_code":
        return cc_rig._make(cc_rig._rig(tmp_path, monkeypatch), cls=_ClaudeProbe), None
    if leaf == "rovodev":
        paths = rovo_rig._rig(tmp_path, monkeypatch)
        return rovo_rig._make(paths, cls=_RovoProbe), None
    rig = dm_rig._rig(tmp_path)
    return dm_rig._make(rig, cls=_DevmateProbe), dm_rig._host_ctx(rig)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("leaf", ("claude_code", "rovodev", "devmate"))
def test_cli_transport_runs_under_explicit_ctx(leaf, entry, tmp_path, monkeypatch):
    inst, host = _cli_leaf(leaf, tmp_path, monkeypatch)
    host = host or RunContext.root().child("leaf")
    _invoke(inst, entry, "q", run_context=host)
    probed = inst.__dict__.get("probe_ctx", [])
    assert probed and all(seen is host for seen in probed)
    frames = inst.__dict__["probe_frame"]
    assert frames == [(FRAME_ENTRY[entry], True, "host")] * len(frames)


def _context_view():
    return dict(contextvars.copy_context())


def _consumer_views(inst, entry, **kwargs):
    """The consumer's context before the stream starts and after each yield."""
    views = [_context_view()]
    if entry == "infer_streaming":
        for _ in inst.infer_streaming("q", **kwargs):
            views.append(_context_view())
        return views

    async def main():
        views[0] = _context_view()
        async for _ in inst.ainfer_streaming("q", **kwargs):
            views.append(_context_view())
        return views

    return asyncio.run(main())


def _streaming_leaf(leaf, tmp_path, monkeypatch):
    if leaf == "stub":
        return _Stub(), RunContext.root().child("leaf")
    if leaf == "openclaw":
        gateway = oc_rig.FakeGateway()
        monkeypatch.setattr(oc_module, "run_subprocess", gateway.run_subprocess)
        inst = _OpenClawProbe(auth_token="tok", gateway=gateway)
        return inst, RunContext.root().child("leaf")
    inst, host = _cli_leaf(leaf, tmp_path, monkeypatch)
    return inst, host or RunContext.root().child("leaf")


@pytest.mark.parametrize("entry", STREAMING)
@pytest.mark.parametrize(
    "leaf", ("stub", "claude_code", "rovodev", "devmate", "openclaw")
)
@pytest.mark.parametrize("with_ctx", (False, True))
def test_consumer_context_is_unchanged_between_yields(
    leaf, entry, with_ctx, tmp_path, monkeypatch
):
    inst, host = _streaming_leaf(leaf, tmp_path, monkeypatch)
    kwargs = {"run_context": host} if with_ctx else {}
    views = _consumer_views(inst, entry, **kwargs)
    assert len(views) > 1
    assert all(view == views[0] for view in views[1:])


@attrs
class _OpenClawProbe(OpenClawInferencer):
    gateway: Any = attrib(default=None, kw_only=True)
    probe_ctx: list = attrib(factory=list, init=False)
    probe_frame: list = attrib(factory=list, init=False)

    async def _ws_connect(self):
        ctx = active_run_context()
        self.probe_ctx.append(ctx)
        self.probe_frame.append(_frame_view(self, ctx))
        return self.gateway.connect()


@pytest.mark.parametrize("entry", ENTRIES)
def test_openclaw_transport_runs_under_explicit_ctx(entry, monkeypatch):
    gateway = oc_rig.FakeGateway()
    monkeypatch.setattr(oc_module, "run_subprocess", gateway.run_subprocess)
    inst = _OpenClawProbe(auth_token="tok", gateway=gateway)
    host = RunContext.root().child("leaf")
    oc_rig._invoke(inst, entry, "q", {"run_context": host})
    assert inst.probe_ctx and all(seen is host for seen in inst.probe_ctx)
    # OpenClaw's sync ``infer`` is a thin ``_run_async(self.ainfer(...))`` adapter.
    frame_entry = "ainfer" if entry == "infer" else FRAME_ENTRY[entry]
    assert inst.probe_frame == [(frame_entry, True, "host")] * len(inst.probe_frame)


_EXT = "agentic_inferencers/external/"
_ALLOWED = {
    ("inferencer_base.py", "_PrototypeCloneFactory", "__call__"): ((), "clone factory"),
    ("inferencer_base.py", "_FreshCloneFactory", "__call__"): ((), "clone factory"),
    ("agentic_inferencers/conversational/inbox_driver.py", "RunTurn", "__call__"): (
        (),
        "callable Protocol type, not an entry",
    ),
    (
        "agentic_inferencers/conversational_native/bridge/mcp_http.py",
        "_ASGIEndpoint",
        "__call__",
    ): ((), "ASGI app of the local MCP server, not an entry"),
    ("inferencer_base.py", "InferencerBase", "infer"): (
        ("self._infer_dispatch",),
        "base",
    ),
    ("inferencer_base.py", "InferencerBase", "iter_infer"): (("self.infer",), "base"),
    ("inferencer_base.py", "InferencerBase", "parallel_infer"): (
        ("self._infer_single",),
        "base",
    ),
    ("inferencer_base.py", "InferencerBase", "__call__"): (("self.infer",), "base"),
    ("inferencer_base.py", "InferencerBase", "ainfer"): (
        ("self._ainfer_dispatch",),
        "base",
    ),
    ("inferencer_base.py", "InferencerBase", "aiter_infer"): (("self.ainfer",), "base"),
    ("inferencer_base.py", "InferencerBase", "aparallel_infer"): (
        ("self._aparallel_one",),
        "base",
    ),
    ("streaming_inferencer_base.py", "StreamingInferencerBase", "ainfer_streaming"): (
        ("self._ainfer_streaming_entry",),
        "template",
    ),
    ("streaming_inferencer_base.py", "StreamingInferencerBase", "infer_streaming"): (
        ("self._infer_streaming_entry",),
        "template",
    ),
    ("templated_inferencer.py", "TemplatedInferencer", "__call__"): (
        ("self.base_inferencer",),
        "renders, then calls its base",
    ),
    ("templated_inferencer.py", "TemplatedInferencer", "infer"): (
        ("self.__call__",),
        "adapter",
    ),
    ("agentic_functions/decorator.py", "_SyncAgenticFunction", "__call__"): (
        ("self.call",),
        "not an inferencer",
    ),
    ("agentic_functions/decorator.py", "_AsyncAgenticFunction", "__call__"): (
        ("self.acall",),
        "not an inferencer",
    ),
    (
        "agentic_inferencers/conversational/context_compressor.py",
        "InferencerContextCompressor",
        "__call__",
    ): ((), "not an inferencer"),
    (CONVERSATIONAL, "ConversationalInferencer", "ainfer_streaming"): (
        (),
        "documented bypass (plan §5.2)",
    ),
    (
        "agentic_inferencers/conversational/protocols.py",
        "ToolExecutorCallable",
        "__call__",
    ): ((), "protocol"),
    (
        "agentic_inferencers/conversational/protocols.py",
        "ContextCompressorCallable",
        "__call__",
    ): ((), "protocol"),
    ("mock_inferencers/mock_bta_components.py", "MockBreakdownInferencer", "ainfer"): (
        (),
        "duck-typed mock stage",
    ),
    ("mock_inferencers/mock_bta_components.py", "MockWorker", "ainfer"): (
        (),
        "duck-typed mock stage",
    ),
    ("mock_inferencers/mock_bta_components.py", "MockAggregator", "ainfer"): (
        (),
        "duck-typed mock stage",
    ),
    (
        "mock_inferencers/mock_clarification_inferencer.py",
        "MockClarificationInferencer",
        "__call__",
    ): ((), "mock"),
}
_CLI_SINGLE = {
    "claude_code/claude_code_cli_inferencer.py": "ClaudeCodeCliInferencer",
    "codex/codex_cli_inferencer.py": "CodexCliInferencer",
    "kiro/kiro_cli_inferencer.py": "KiroCliInferencer",
    "rovodev/rovodev_cli_inferencer.py": "RovoDevCliInferencer",
}
for _rel, _cls in _CLI_SINGLE.items():
    _ALLOWED[(_EXT + _rel, _cls, "ainfer")] = (("self._ainfer_single",), "CLI adapter")
    _ALLOWED[(_EXT + _rel, _cls, "infer")] = (("self._infer_single",), "CLI adapter")
_DEVMATE = (_EXT + "devmate/devmate_cli_inferencer.py", "DevmateCliInferencer")
_OPENCLAW = (_EXT + "openclaw/openclaw_inferencer.py", "OpenClawInferencer")
_ALLOWED.update(
    {
        (*_DEVMATE, "ainfer"): (("self._ainfer_single",), "CLI adapter"),
        (*_DEVMATE, "ainfer_streaming"): (
            ("super().ainfer_streaming",),
            "signature adapter",
        ),
        (*_DEVMATE, "infer_streaming"): (
            ("super().infer_streaming",),
            "signature adapter",
        ),
        (*_OPENCLAW, "ainfer"): (("self._ainfer_single",), "converged entry (B30)"),
        (*_OPENCLAW, "infer"): (("self.ainfer",), "sync adapter"),
        (*_OPENCLAW, "ainfer_streaming"): (
            ("super().ainfer_streaming",),
            "validates, then delegates",
        ),
        (*_OPENCLAW, "infer_streaming"): (
            ("super().infer_streaming",),
            "validates, then delegates",
        ),
    }
)
_ENTRY_NAMES = {name for _, _, name in _ALLOWED}


def _delegations(fn):
    calls = set()
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id == "self":
            calls.add("self." + node.func.attr)
        elif isinstance(owner, ast.Call) and getattr(owner.func, "id", "") == "super":
            calls.add("super()." + node.func.attr)
    return calls


def _overrides(root):
    for dirpath, _, files in os.walk(root, followlinks=True):
        for name in sorted(f for f in files if f.endswith(".py")):
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
            rel = os.path.relpath(path, root)
            for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
                for fn in cls.body:
                    if (
                        isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and fn.name in _ENTRY_NAMES
                    ):
                        yield (rel, cls.name, fn.name), fn


def test_public_entry_overrides_match_allow_list():
    root = os.path.dirname(inferencer_base.__file__)
    found = dict(_overrides(root))
    assert found, root
    assert sorted(set(found) - set(_ALLOWED)) == [], "unlisted public-entry overrides"
    assert sorted(set(_ALLOWED) - set(found)) == [], "stale _ALLOWED entries"
    for key, fn in found.items():
        missing = set(_ALLOWED[key][0]) - _delegations(fn)
        assert not missing, (key, missing)


def test_public_streaming_overrides_are_plain_methods():
    root = os.path.dirname(inferencer_base.__file__)
    for (rel, cls, meth), fn in _overrides(root):
        if meth not in STREAMING or rel == CONVERSATIONAL:
            continue
        assert isinstance(fn, ast.FunctionDef), (rel, cls, meth)
        yields = [n for n in ast.walk(fn) if isinstance(n, (ast.Yield, ast.YieldFrom))]
        assert not yields, (rel, cls, meth)
