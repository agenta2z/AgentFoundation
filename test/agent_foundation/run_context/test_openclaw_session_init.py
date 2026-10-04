"""OpenClaw initializes each session once per connection (plan v8 §13, P10; B19).

With ``always_initialize_new_session`` a call probes the pod for its session's
transcript and runs a warm-up turn when the session is new. The initialized
session ids are a Tier-3 set: a second session id gets its own probe and warm-up
instead of inheriting the first session's flag, a repeated id is probed once,
and a host branch records its sessions in its own live handles, never in the
instance backing other branches fall back to.
"""

from __future__ import annotations

import asyncio

from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import RunContext

from .test_golden_openclaw import _make, FakeGateway

WARM_UP = "Session initialization. Respond with exactly one word: Ready."


def _probes(gateway):
    return [c["session_id"] for c in gateway.log if c["method"] == "transcript_probe"]


def _warm_ups(gateway):
    return [c["session_id"] for c in gateway.log if c.get("prompt") == WARM_UP]


def _host(root):
    return RunContext.root(workspace=InferencerWorkspace(root=str(root)))


def test_each_session_id_gets_its_own_warm_up(monkeypatch):
    gateway = FakeGateway()
    inst = _make(monkeypatch, gateway)
    for sid in ("sess-a", "sess-b", "sess-a"):
        asyncio.run(inst.ainfer(f"{sid} question", session_id=sid))
    assert _probes(gateway) == ["sess-a", "sess-b"]
    assert _warm_ups(gateway) == ["sess-a", "sess-b"]


def test_a_session_with_a_transcript_is_probed_once_and_never_warmed_up(monkeypatch):
    gateway = FakeGateway(transcripts=("sess-old",))
    inst = _make(monkeypatch, gateway)
    for _ in range(2):
        asyncio.run(inst.ainfer("question", session_id="sess-old"))
    assert _probes(gateway) == ["sess-old"]
    assert _warm_ups(gateway) == []


def test_an_explicit_initialization_is_honoured_by_later_calls(monkeypatch):
    gateway = FakeGateway()
    inst = _make(monkeypatch, gateway)
    asyncio.run(inst.initialize_session("sess-x"))
    asyncio.run(inst.ainfer("question", session_id="sess-x"))
    assert _probes(gateway) == []
    assert _warm_ups(gateway) == ["sess-x"]


def test_host_branches_keep_their_initialized_sessions_off_the_instance(
    tmp_path, monkeypatch
):
    gateway = FakeGateway()
    inst = _make(monkeypatch, gateway)
    for root in ("a", "b"):
        asyncio.run(
            inst.ainfer(
                "question", session_id="sess-h", run_context=_host(tmp_path / root)
            )
        )
    assert _probes(gateway) == ["sess-h", "sess-h"]
    assert _warm_ups(gateway) == ["sess-h"]
    assert "_initialized_sessions_backing" not in vars(inst)
