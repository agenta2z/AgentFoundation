"""Golden characterization of ``DevmateCliInferencer`` streaming entrypoints.

Pins the behaviour of the three public entrypoints -- ``ainfer`` (the
``@bridge_entrypoint`` override that accumulates devmate's private streaming
pipeline), ``ainfer_streaming`` and ``infer_streaming`` (the inherited base
templates behind devmate's signature adapters, over devmate's async and sync
pipelines), each consumed fully -- for:

* ``two_bare``: two sequential bare calls on one instance (session continuity);
* ``two_bare_getter``: the same with ``get_streaming_result()`` read after
  each call, which is what advances the streaming entries' session;
* ``host_ctx``: one call with an explicit host ``run_context`` followed by a
  bare call on the same instance (B26);
* ``failure``: a call whose CLI exits non-zero, then a successful one;
* ``ainfer`` with and without an ``inference_config`` (B27);
* a configured ``bta_inferencer`` with the fan-out stubbed (B26b).

Each call records the output, the chunks seen by ``stream_observer``, the
kwargs, active RunContext and ``dump_output`` inside ``construct_command``,
the fake ``dm`` argv / cwd / stdin / ``TMPDIR``, ``dump_output``,
``active_session_id`` and ``_consecutive_error_count`` after the call, and the
files left under ``tmp_path`` (stream cache, session logs). The fake ``dm``
prints the dm banner, prompt echo, bullet answer and Logs/Resume footer,
honours ``--resume`` and exits 3 when the prompt contains ``FAIL``.
"""

import asyncio
import json
import shlex
import sys

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
    DevmateCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    enter_run,
    exit_run,
    RunContext,
)
from rich_python_utils.common_utils.function_helper import FallbackMode

from ._golden import check_golden, Normalizer, session_log_types, workspace_tree
from ._leaf_fixtures import FAKE_DM

ENTRIES = ("ainfer", "ainfer_streaming", "infer_streaming")
BARE_SCENARIOS = ("two_bare", "two_bare_getter")
INST_ID = "RecordingDevmateCli"
EXTRA_RULES = (
    (r"stream_[0-9a-f]{8}_", "stream_<RAND>_"),
    (r"/tmp/devmate-(?:\d+|<ID\d+>)", "/tmp/devmate-<UID>"),
)
RESULT_FIELDS = (
    "output",
    "raw_output",
    "stderr",
    "return_code",
    "success",
    "session_id",
    "trajectory_url",
    "error",
)


class _RecordingDevmate(DevmateCliInferencer):
    def construct_command(self, inference_input, **kwargs):
        self.__dict__.setdefault("golden_cc", []).append(
            {
                "input": _plain(inference_input),
                "kwargs": _plain(kwargs),
                "active_ctx": _ctx_view(active_run_context()),
                "dump_output": self.dump_output,
            }
        )
        return super().construct_command(inference_input, **kwargs)


class _FanoutDevmate(_RecordingDevmate):
    def _fanout_record(self, path, contract, inference_config, inference_args):
        self.__dict__.setdefault("golden_fanout", []).append(
            {
                "path": path,
                "contract": _plain(contract),
                "inference_config": _plain(inference_config),
                "args": _plain(inference_args),
                "active_ctx": _ctx_view(active_run_context()),
            }
        )
        return f"FANOUT({contract})", {}

    def _run_fanout(self, contract, inference_config, inference_args):
        return self._fanout_record(
            "_run_fanout", contract, inference_config, inference_args
        )

    async def _arun_fanout(self, contract, inference_config, inference_args):
        return self._fanout_record(
            "_arun_fanout", contract, inference_config, inference_args
        )


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


def _result_view(result):
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        view = {str(k): _plain(v) for k, v in result.items()}
    else:
        view = {f: _plain(getattr(result, f, None)) for f in RESULT_FIELDS}
    view["type"] = type(result).__name__
    return view


async def _drain(agen):
    return [chunk async for chunk in agen]


def _invoke(inst, entry, prompt, **kw):
    if entry == "ainfer":
        return _result_view(asyncio.run(inst.ainfer(prompt, **kw)))
    if entry == "ainfer_streaming":
        return asyncio.run(_drain(inst.ainfer_streaming(prompt, **kw)))
    return list(inst.infer_streaming(prompt, **kw))


def _attempt(inst, entry, prompt, **kw):
    try:
        return {"output": _invoke(inst, entry, prompt, **kw), "raised": None}
    except Exception as exc:
        raised = {"type": type(exc).__name__, "message": str(exc)}
        return {"output": None, "raised": raised}


def _take(items):
    taken = list(items)
    items.clear()
    return taken


def _cli_calls(log):
    if not log.exists():
        return []
    lines = log.read_text(encoding="utf-8").splitlines()
    log.unlink()
    return [json.loads(line) for line in lines]


def _record_call(inst, rig, entry, prompt, getter=False, **kw):
    record = _attempt(inst, entry, prompt, **kw)
    record.update(
        observed=_take(rig["observed"]),
        construct_command=inst.__dict__.pop("golden_cc", []),
        cli=_cli_calls(rig["log"]),
        dump_output_after=inst.dump_output,
        active_session_id=inst.active_session_id,
        consecutive_error_count=inst._consecutive_error_count,
    )
    if getter:
        record["get_streaming_result"] = _result_view(inst.get_streaming_result())
        record["active_session_id_after_getter"] = inst.active_session_id
    return record


def _read_under(inst, ctx):
    token = enter_run(ctx)
    try:
        return inst.active_session_id
    finally:
        exit_run(token)


def _rig(tmp_path):
    cli = tmp_path / "cli"
    cli.mkdir()
    (tmp_path / "repo" / ".sl").mkdir(parents=True)
    log = cli / "calls.jsonl"
    script = FAKE_DM.replace("__LOG__", repr(str(log)))
    (cli / "fake_dm.py").write_text(script, encoding="utf-8")
    return {"tmp": tmp_path, "cli": cli, "log": log, "observed": []}


def _make(rig, cls=_RecordingDevmate, **kw):
    fake = shlex.quote(str(rig["cli"] / "fake_dm.py"))
    return cls(
        target_path=str(rig["tmp"] / "repo"),
        cli_binary=f"{shlex.quote(sys.executable)} -I {fake}",
        cache_folder=str(rig["tmp"] / "cache"),
        dump_output=True,
        stream_observer=rig["observed"].append,
        id=INST_ID,
        **kw,
    )


def _golden_data(rig, body):
    norm = Normalizer({"<TMP>": rig["tmp"]}, extra=EXTRA_RULES)
    data = {"body": norm.value(body)}
    data["tree"] = workspace_tree(rig["tmp"], norm, content_for=_wants_content)
    data["session_logs"] = session_log_types(rig["tmp"], norm)
    return data


def _wants_content(rel):
    if rel.startswith("cli/"):
        return False
    return not (rel.startswith("logs/") or "/logs/" in rel)


def _host_ctx(rig):
    workspace = InferencerWorkspace(root=str(rig["tmp"] / "host"))
    return RunContext.root(workspace=workspace).child("devmate")


@pytest.mark.parametrize("scenario", BARE_SCENARIOS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_devmate_bare_calls_golden(entry, scenario, tmp_path):
    rig = _rig(tmp_path)
    inst = _make(rig)
    getter = scenario == "two_bare_getter"
    calls = [
        _record_call(inst, rig, entry, f"question {i}", getter=getter) for i in (1, 2)
    ]
    check_golden(f"streaming/devmate_{entry}_{scenario}", _golden_data(rig, calls))


@pytest.mark.parametrize("entry", ENTRIES)
def test_devmate_host_ctx_golden(entry, tmp_path):
    """B26: every entrypoint runs devmate's pipeline under the host
    ``run_context``, so it is active inside ``construct_command`` and never
    reaches its kwargs."""
    rig = _rig(tmp_path)
    inst = _make(rig)
    ctx = _host_ctx(rig)
    first = _record_call(inst, rig, entry, "host question", run_context=ctx)
    first["active_session_id_under_ctx"] = _read_under(inst, ctx)
    second = _record_call(inst, rig, entry, "bare follow-up")
    second["active_session_id_under_ctx"] = _read_under(inst, ctx)
    body = {"host_ctx": _ctx_view(ctx), "calls": [first, second]}
    check_golden(f"streaming/devmate_{entry}_host_ctx", _golden_data(rig, body))


@pytest.mark.parametrize("entry", ENTRIES)
def test_devmate_failure_golden(entry, tmp_path):
    rig = _rig(tmp_path)
    inst = _make(rig)
    calls = [
        _record_call(inst, rig, entry, prompt, getter=True)
        for prompt in ("FAIL please", "after failure")
    ]
    check_golden(f"streaming/devmate_{entry}_failure", _golden_data(rig, calls))


@pytest.mark.parametrize("config_id, config", [("none", None), ("truthy", {"k": 1})])
def test_devmate_ainfer_inference_config_golden(config_id, config, tmp_path):
    """B27: ``ainfer`` accumulates the private pipeline, whose keyword-only
    ``filter_session_info`` defaults to ``False``, so the accumulated stream
    (seen by ``stream_observer``) is unfiltered whatever ``inference_config``
    is."""
    rig = _rig(tmp_path)
    inst = _make(rig)
    kw = {} if config is None else {"inference_config": config}
    prompt = "inference config question"
    call = _record_call(inst, rig, "ainfer", prompt, getter=True, **kw)
    check_golden(
        f"streaming/devmate_ainfer_config_{config_id}", _golden_data(rig, call)
    )


@pytest.mark.parametrize("entry", ENTRIES)
def test_devmate_bta_fanout_golden(entry, tmp_path):
    """B26b: with ``bta_inferencer`` set, every entrypoint, the sync
    ``infer_streaming`` template included, delegates to the fan-out and never
    runs the CLI locally."""
    rig = _rig(tmp_path)
    bta = BreakdownThenAggregateInferencer(
        fallback_mode=FallbackMode.NEVER, max_retry=0
    )
    inst = _make(rig, cls=_FanoutDevmate, bta_inferencer=bta)
    call = _record_call(inst, rig, entry, "fanout question")
    call["fanout"] = inst.__dict__.pop("golden_fanout", [])
    check_golden(f"streaming/devmate_bta_{entry}", _golden_data(rig, call))
