"""Golden characterization of ``RovoDevCliInferencer`` streaming entrypoints.

Pins today's behaviour, per public entrypoint -- ``ainfer`` (the
``@bridge_entrypoint`` override), ``ainfer_streaming`` (the inherited base
template over rovodev's session-resolving pipeline, consumed fully) and
``infer_streaming`` (the inherited base thread bridge, consumed fully) -- and
per scenario:

* ``bare``: two sequential calls on one instance with no RunContext, so
  across-call session continuity (``--restore``) is visible;
* ``ctx_fresh``: one call with an explicit host ``run_context`` on a fresh
  instance;
* ``ctx_seeded``: the same call after seeding the instance backing session and
  the host ctx's live-handle slot with distinct ids, so the session each entry
  resolves, and where it writes the captured one, is visible (B26).

Each call records the output, the kwargs that reached ``construct_command`` and
the RunContext active inside it, the fake ``acli`` argv and cwd,
``active_session_id``, the instance backing and connection-scoped live-handle
store, ``get_final_output()`` and ``_last_clean_output`` (bare getters, written
by non-host calls only) and, under a host ctx, the ``final_output`` the call
published in its outcome (B23, P10). The fake ``acli`` in
``tmp_path`` writes the legacy ``--output-file`` and a ``session_context.json``
under a temp sessions dir that ``find_latest_session_id`` /
``ensure_session_metadata`` are pointed at; that dir's final tree is pinned too.
"""

import asyncio
import functools
import json
import sys
import tempfile

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev import (
    common as rovo_common,
    rovodev_cli_inferencer as rovo_mod,
)
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    enter_run,
    exit_run,
    read_outcome,
    RunContext,
)

from ._golden import check_golden, live_branch_label, Normalizer, workspace_tree
from ._leaf_fixtures import FAKE_ACLI

ENTRIES = ("ainfer", "ainfer_streaming", "infer_streaming")
SCENARIOS = ("bare", "ctx_fresh", "ctx_seeded")
SEED_BACKING = "seed-backing-sid"
SEED_SLOT = "seed-slot-sid"


class _RecordingRovoDev(rovo_mod.RovoDevCliInferencer):
    def construct_command(self, inference_input, **kwargs):
        ctx = active_run_context()
        self.__dict__.setdefault("golden_cc", []).append(
            {"kwargs": _plain(kwargs), "active_ctx": _ctx_view(ctx)}
        )
        return super().construct_command(inference_input, **kwargs)


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
    fields = ("output", "raw_output", "stderr", "return_code", "success")
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
    paths = {k: tmp_path / k for k in ("cli", "sessions", "tmp", "ws")}
    for p in paths.values():
        p.mkdir()
    paths["log"] = paths["cli"] / "calls.jsonl"
    script = FAKE_ACLI.replace("__LOG__", repr(str(paths["log"])))
    script = script.replace("__SESSIONS__", repr(str(paths["sessions"])))
    (paths["cli"] / "fake_acli.py").write_text(script, encoding="utf-8")
    sessions = str(paths["sessions"])
    monkeypatch.setattr(tempfile, "tempdir", str(paths["tmp"]))
    monkeypatch.setattr(
        rovo_mod,
        "find_latest_session_id",
        functools.partial(rovo_common.find_latest_session_id, sessions_dir=sessions),
    )
    monkeypatch.setattr(
        rovo_mod,
        "ensure_session_metadata",
        functools.partial(rovo_common.ensure_session_metadata, sessions_dir=sessions),
    )
    return paths


def _make(paths, cls=_RecordingRovoDev):
    return cls(
        acli_path=f"{sys.executable} {paths['cli'] / 'fake_acli.py'}",
        target_path=str(paths["ws"]),
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
        "get_final_output": inst.get_final_output(),
        "last_clean_output": getattr(inst, "_last_clean_output", None),
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
    call["final_output_under_ctx"] = read_outcome(ctx).final_output
    return {"host_ctx": _ctx_view(ctx), "calls": [call]}


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_rovodev_streaming_golden(entry, scenario, tmp_path, monkeypatch):
    paths = _rig(tmp_path, monkeypatch)
    inst = _make(paths)
    norm = Normalizer(
        {"<TMP>": tmp_path},
        extra=[(r"rovodev_output_\w+\.md", "rovodev_output_<RAND>.md")],
    )
    data = norm.value(_run_scenario(inst, paths, entry, scenario))
    data["sessions_dir"] = workspace_tree(paths["sessions"], norm)
    check_golden(f"streaming/rovodev_{entry}_{scenario}", data)
