"""Golden characterization of ``ClaudeCodeCliInferencer`` streaming entrypoints.

Pins the behaviour, per public entrypoint -- ``ainfer`` (the
``@bridge_entrypoint`` override), ``ainfer_streaming`` (inherited from the
streaming base, consumed fully) and ``infer_streaming`` (the base sync template
over the native ``_infer_streaming_pipeline``, consumed fully) -- and per
scenario:

* ``bare``: two sequential calls on one instance with no RunContext, so
  across-call session continuity (``--resume``) is visible;
* ``ctx_fresh``: one call with an explicit host ``run_context`` on a fresh
  instance;
* ``ctx_seeded``: the same call after seeding the instance backing session and
  the host ctx's live-handle slot with distinct ids, so the session each entry
  resolves, and where it writes the captured one, is visible (B26).

Each call records the output, the kwargs that reached ``construct_command`` and
the RunContext active inside it, the fake ``claude`` argv, cwd and stdin,
``active_session_id``, the instance backing and connection-scoped live-handle
store, ``get_streaming_result()`` and the captured stream ``result`` event's
session id. The fake ``claude`` in ``tmp_path`` honours ``--output-format``
(stream-json events with a ``session_id``, else plain text) and ``--resume``.

A second test pins B26b: with a ``bta_inferencer`` configured (fan-out stubbed
at ``_arun_fanout`` / ``_run_fanout``), every entrypoint fans out and none runs
the CLI locally.
"""

import asyncio
import json
import sys

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    enter_run,
    exit_run,
    RunContext,
)
from rich_python_utils.common_utils.function_helper import FallbackMode

from ._golden import check_golden, live_branch_label, Normalizer

ENTRIES = ("ainfer", "ainfer_streaming", "infer_streaming")
SCENARIOS = ("bare", "ctx_fresh", "ctx_seeded")
SEED_BACKING = "seed-backing-sid"
SEED_SLOT = "seed-slot-sid"
FANOUT_TEXT = "FANOUT_TEXT"

FAKE_CLAUDE = r"""
import json, os, sys, uuid

LOG = __LOG__
argv = sys.argv[1:]
fmt = argv[argv.index("--output-format") + 1] if "--output-format" in argv else None
stdin = sys.stdin.read() if fmt else None
sid = argv[argv.index("--resume") + 1] if "--resume" in argv else str(uuid.uuid4())
prompt = stdin if stdin is not None else argv[-1]
with open(LOG, "a") as f:
    f.write(json.dumps({"argv": argv, "cwd": os.getcwd(), "stdin": stdin}) + "\n")
sys.stderr.write("claude stderr noise\n")


def delta(text):
    inner = {"type": "text_delta", "text": text}
    return {"type": "stream_event",
            "event": {"type": "content_block_delta", "delta": inner}}


if fmt == "stream-json":
    for event in (
        {"type": "system", "subtype": "init", "session_id": sid},
        {"type": "stream_event", "event": {"type": "content_block_start"}},
        delta("reply "),
        delta("to %r" % prompt),
        {"type": "result", "subtype": "success", "is_error": False,
         "result": "reply to %r" % prompt, "session_id": sid},
    ):
        print(json.dumps(event))
else:
    print("plain reply to %r" % prompt)
    print("second line")
"""


class _RecordingClaude(ClaudeCodeCliInferencer):
    def construct_command(self, inference_input, **kwargs):
        ctx = active_run_context()
        self.__dict__.setdefault("golden_cc", []).append(
            {"kwargs": _plain(kwargs), "active_ctx": _ctx_view(ctx)}
        )
        return super().construct_command(inference_input, **kwargs)


class _FanoutClaude(_RecordingClaude):
    def _fanout_record(self, path, contract, inference_args):
        self.__dict__.setdefault("golden_fanout", []).append(
            {"path": path, "contract": _plain(contract), "args": _plain(inference_args)}
        )
        return FANOUT_TEXT, {}

    def _run_fanout(self, contract, inference_config, inference_args):
        return self._fanout_record("_run_fanout", contract, inference_args)

    async def _arun_fanout(self, contract, inference_config, inference_args):
        return self._fanout_record("_arun_fanout", contract, inference_args)


def _plain(value):
    if isinstance(value, RunContext):
        return f"<RunContext path={value.path} legacy_mint={value.legacy_mint}>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return f"<{type(value).__name__}>"


def _ctx_view(ctx):
    if ctx is None:
        return None
    return {"path": ctx.path, "legacy_mint": ctx.legacy_mint}


def _store_view(inst):
    store = inst.__dict__.get("_live_handle_store")
    if store is None:
        return None
    return {
        live_branch_label(key): _plain(dict(h._data))
        for key, h in sorted(store._by_path.items())
    }


def _result_view(result):
    if isinstance(result, str):
        return result
    fields = ("output", "raw_output", "stderr", "return_code", "success", "error")
    view = {f: getattr(result, f, None) for f in fields}
    view.update(type=type(result).__name__, session_id=result.session_id)
    return view


async def _drain(agen):
    return [chunk async for chunk in agen]


def _invoke(inst, entry, prompt, **kw):
    if entry == "ainfer":
        return _result_view(asyncio.run(inst.ainfer(prompt, **kw)))
    if entry == "ainfer_streaming":
        return "".join(asyncio.run(_drain(inst.ainfer_streaming(prompt, **kw))))
    return "".join(inst.infer_streaming(prompt, **kw))


def _rig(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_COMMAND", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_MAX_CONCURRENCY", raising=False)
    paths = {k: tmp_path / k for k in ("cli", "ws")}
    for p in paths.values():
        p.mkdir()
    paths["log"] = paths["cli"] / "calls.jsonl"
    script = FAKE_CLAUDE.replace("__LOG__", repr(str(paths["log"])))
    (paths["cli"] / "fake_claude.py").write_text(script, encoding="utf-8")
    return paths


def _make(paths, cls=_RecordingClaude, **kw):
    return cls(
        claude_command=f"{sys.executable} {paths['cli'] / 'fake_claude.py'}",
        target_path=str(paths["ws"]),
        **kw,
    )


def _cli_calls(paths):
    log = paths["log"]
    if not log.exists():
        return []
    lines = log.read_text(encoding="utf-8").splitlines()
    log.unlink()
    return [json.loads(line) for line in lines]


def _record_call(inst, paths, entry, prompt, **kw):
    output = _invoke(inst, entry, prompt, **kw)
    return {
        "output": output,
        "construct_command": inst.__dict__.pop("golden_cc", []),
        "cli": _cli_calls(paths),
        "active_session_id": inst.active_session_id,
        "session_backing": inst._session_id,
        "live_store": _store_view(inst),
        "get_streaming_result": _result_view(inst.get_streaming_result()),
    }


def _read_under(inst, ctx):
    token = enter_run(ctx)
    try:
        return inst.active_session_id
    finally:
        exit_run(token)


def _seed(inst, ctx):
    inst.active_session_id = SEED_BACKING
    token = enter_run(ctx)
    try:
        inst.active_session_id = SEED_SLOT
    finally:
        exit_run(token)


def _run_scenario(inst, paths, entry, scenario):
    if scenario == "bare":
        turns = [_record_call(inst, paths, entry, f"turn {i}") for i in (1, 2)]
        return {"calls": turns}
    ctx = RunContext.root().child("leaf")
    if scenario == "ctx_seeded":
        _seed(inst, ctx)
    call = _record_call(inst, paths, entry, "turn 1", run_context=ctx)
    call["active_session_id_under_ctx"] = _read_under(inst, ctx)
    return {"host_ctx": _ctx_view(ctx), "calls": [call]}


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_claude_code_streaming_golden(entry, scenario, tmp_path, monkeypatch):
    paths = _rig(tmp_path, monkeypatch)
    inst = _make(paths)
    norm = Normalizer({"<TMP>": tmp_path})
    data = norm.value(_run_scenario(inst, paths, entry, scenario))
    check_golden(f"streaming/claude_code_{entry}_{scenario}", data)


@pytest.mark.parametrize("entry", ENTRIES)
def test_claude_code_bta_fanout_golden(entry, tmp_path, monkeypatch):
    """B26b: with ``bta_inferencer`` set, every entrypoint, the sync
    ``infer_streaming`` template included, delegates to the fan-out and never
    runs the CLI locally."""
    paths = _rig(tmp_path, monkeypatch)
    bta = BreakdownThenAggregateInferencer(
        fallback_mode=FallbackMode.NEVER, max_retry=0
    )
    inst = _make(paths, cls=_FanoutClaude, bta_inferencer=bta)
    call = _record_call(inst, paths, entry, "turn 1")
    call["fanout"] = inst.__dict__.pop("golden_fanout", [])
    norm = Normalizer({"<TMP>": tmp_path})
    check_golden(f"streaming/claude_code_bta_{entry}", norm.value(call))
