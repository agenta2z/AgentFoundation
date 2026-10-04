"""SDK leaves read the session through the session policy (plan v8 §13, P10; B6 leaves).

codex SDK, devmate SDK, metamate SDK and rovochat read the instance backing
``_session_id`` directly: for auto-resume (codex SDK when it connects, devmate SDK on
every call) and for ``SDKInferencerResponse.session_id``. Under a host ctx the backing
is never this branch's session, so a branch's next call started over and its response
reported no session. They now read ``active_session_id``: this branch's slot under a
host ctx, the instance otherwise. RovoChat also closes its event stream inside the
call; it used to be finalized at event-loop shutdown, outside the call.
"""

from __future__ import annotations

import asyncio

from agent_foundation.common.inferencers.agentic_inferencers.external.codex.codex_sdk_inferencer import (
    CodexSdkInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_sdk_inferencer import (
    DevmateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.metamate_sdk_inferencer import (
    MetamateSDKInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.rovochat.rovochat_inferencer import (
    RovoChatInferencer,
)
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    RunContext,
)

from ._leaf_fixtures import (
    fixed_code_scope,
    install_fake_codex_sdk,
    install_fake_devmate_sdk,
    install_fake_metamate_sdk,
    install_fake_rovochat,
)


def _host(root):
    return RunContext.root(workspace=InferencerWorkspace(root=str(root)))


def _session_under(inf, ctx):
    token = enter_run(ctx)
    try:
        return inf.active_session_id
    finally:
        exit_run(token)


def _sdk_call(inf, prompt, **kwargs):
    return asyncio.run(inf.ainfer(prompt, return_sdk_response=True, **kwargs))


def test_devmate_sdk_resumes_the_branch_session_under_a_host_ctx(tmp_path, monkeypatch):
    log = []
    install_fake_devmate_sdk(monkeypatch, log)
    devmate = DevmateSDKInferencer(target_path=str(tmp_path))
    ctx = _host(tmp_path / "a")
    first = _sdk_call(devmate, "one", run_context=ctx)
    second = _sdk_call(devmate, "two", run_context=ctx)
    assert log == [("start_session", None), ("start_session", "dm-1")]
    assert (first.session_id, second.session_id) == ("dm-1", "dm-1")
    assert devmate.__dict__.get("_session_id") is None


def test_bare_devmate_sdk_calls_still_resume_on_the_instance(tmp_path, monkeypatch):
    log = []
    install_fake_devmate_sdk(monkeypatch, log)
    devmate = DevmateSDKInferencer(target_path=str(tmp_path))
    _sdk_call(devmate, "one")
    second = _sdk_call(devmate, "two")
    assert log == [("start_session", None), ("start_session", "dm-1")]
    assert second.session_id == devmate.__dict__.get("_session_id") == "dm-1"


def test_codex_sdk_sync_calls_resume_the_branch_thread(tmp_path, monkeypatch):
    log = []
    install_fake_codex_sdk(monkeypatch, log)
    codex = CodexSdkInferencer(target_path=str(tmp_path))
    ctx = _host(tmp_path / "a")
    codex.infer("one", run_context=ctx)
    codex.infer("two", run_context=ctx)
    opened = [entry for entry in log if entry[0] != "close"]
    assert opened == [("thread_start", "thread-1"), ("thread_resume", "thread-1")]


def test_metamate_sdk_reports_the_branch_session(tmp_path, monkeypatch):
    install_fake_metamate_sdk(monkeypatch, [])
    metamate = MetamateSDKInferencer(
        poll_interval_seconds=0.01, code_scope_judge=fixed_code_scope
    )
    ctx = _host(tmp_path / "a")
    response = _sdk_call(metamate, "q", run_context=ctx)
    assert response.session_id is not None
    assert response.session_id == _session_under(metamate, ctx)


def _rovochat(monkeypatch, log):
    install_fake_rovochat(monkeypatch, log)
    return RovoChatInferencer(
        base_url="https://rovo.example.test", cloud_id="cloud-1", uct_token="uct"
    )


def test_rovochat_reports_the_branch_session(tmp_path, monkeypatch):
    rovochat = _rovochat(monkeypatch, [])
    ctx = _host(tmp_path / "a")
    response = _sdk_call(rovochat, "q", run_context=ctx)
    assert response.session_id == _session_under(rovochat, ctx) == "conv-1"


def test_rovochat_closes_its_event_stream_inside_the_call(tmp_path, monkeypatch):
    log = []
    rovochat = _rovochat(monkeypatch, log)
    asyncio.run(rovochat.ainfer("q", run_context=_host(tmp_path / "a")))
    assert [entry for entry in log if entry[0] == "stream_closed"] == [
        ("stream_closed", "/")
    ]
