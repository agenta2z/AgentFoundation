"""The BTA resume identity manifest (plan v8 §5.12, P8 commit 4).

Under its lease a BTA call writes ``bta_manifest.json`` at its checkpoint root: the
identity header (input, definition, invocation arguments) before anything is read,
and the committed effective plan once ``_build_subgraph_spec`` returns. A resume
verifies the header and rebuilds the workers from the plan; it fails closed on a
mismatch, a corrupt tree, checkpoints without a manifest (unless
``trust_legacy``) and an identity it can't compute. Crashes are simulated with a
``BaseException`` raised from a stage, which no retry catches and which leaves the
tree exactly as process death would.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
    BtaResumeCorruptionError,
    BtaResumeIdentityMismatch,
    UnverifiedLegacyBtaResumeError,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.bta_checkpoints import (
    MANIFEST_FILE,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context.resume_identity import (
    ResumeIdentityUnavailableError,
)
from attr import attrib, attrs

KINDS = ("sync", "async")
CALLS = []
CRASH_AT = set()


class _Crash(BaseException):
    """Stands in for process death: no retry catches it."""


@pytest.fixture(autouse=True)
def _reset():
    CALLS.clear()
    CRASH_AT.clear()
    yield
    CALLS.clear()
    CRASH_AT.clear()


@attrs(slots=False)
class _Stage(InferencerBase):
    label: str = attrib(default="stage", kw_only=True)
    response: str = attrib(default="", kw_only=True)

    def _answer(self, inference_input):
        CALLS.append(self.label)
        if self.label in CRASH_AT:
            raise _Crash(self.label)
        return self.response or f"{self.label}:{inference_input}"

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._answer(inference_input)

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._answer(inference_input)


def _worker(sub_query, index):
    return _Stage(label=f"w{index}")


def _bta(root, **kwargs):
    defaults = {
        "breakdown_inferencer": _Stage(label="bd", response="1. q0\n2. q1\n3. q2"),
        "worker_inferencers": _worker,
        "aggregator_inferencer": _Stage(label="agg"),
        "breakdown_format": "numbered_list",
        "enable_result_save": True,
        "resume_with_saved_results": True,
        "max_retry": 1,
        "workspace": InferencerWorkspace(root=str(root)),
    }
    defaults.update(kwargs)
    bta = BreakdownThenAggregateInferencer(**defaults)
    bta.name = "rm"
    return bta


def _call(bta, kind, text="task"):
    if kind == "sync":
        return bta.infer(text)
    return asyncio.run(bta.ainfer(text))


def _manifest(root):
    with open(root / "checkpoints" / MANIFEST_FILE, encoding="utf-8") as f:
        return json.load(f)


def _crash(root, kind, at, **kwargs):
    CRASH_AT.add(at)
    with pytest.raises(_Crash):
        _call(_bta(root, **kwargs), kind)
    CRASH_AT.clear()
    CALLS.clear()


def test_a_fresh_run_writes_the_header_and_commits_the_plan(tmp_path):
    assert _call(_bta(tmp_path), "sync").startswith("agg:")
    manifest = _manifest(tmp_path)
    header = manifest["header"]
    assert header["unverifiable"] is None
    assert header["input"]["type"] == "builtins:str" and header["input"]["length"] == 4
    assert manifest["plan"] == {
        "sub_queries": ["q0", "q1", "q2"],
        "workers": ["rm.worker_00", "rm.worker_01", "rm.worker_02"],
    }


@pytest.mark.parametrize("kind", KINDS)
def test_a_fresh_instance_resumes_a_matching_manifest(kind, tmp_path):
    first = _call(_bta(tmp_path), kind)
    CALLS.clear()
    assert _call(_bta(tmp_path), kind) == first
    assert CALLS == []


@pytest.mark.parametrize("kind", KINDS)
def test_crash_with_only_the_header_reruns_everything(kind, tmp_path):
    _crash(tmp_path, kind, "bd")
    assert _manifest(tmp_path)["plan"] is None
    assert _call(_bta(tmp_path), kind).startswith("agg:")
    assert CALLS[0] == "bd" and sorted(CALLS[1:4]) == ["w0", "w1", "w2"]


def _crashing_in_build(sub_query, index):
    if index == 1:
        raise _Crash("while building the workers")
    return _Stage(label=f"w{index}")


_crashing_in_build.resume_identity = "w<index>"
_worker.resume_identity = "w<index>"


def _promote_breakdown(root, descriptions):
    """What a breakdown stage with a ``checkpoint_scope="parent"`` extraction
    leaves in the BTA's checkpoints."""
    path = root / "checkpoints" / "breakdown" / "decomposed_subtasks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"subtasks": [{"description": d} for d in descriptions]}),
        encoding="utf-8",
    )


def test_crash_after_the_breakdown_replans_from_it(tmp_path):
    with pytest.raises(_Crash):
        _call(
            _bta(tmp_path, worker_inferencers=_crashing_in_build, max_breakdown=2),
            "sync",
        )
    assert _manifest(tmp_path)["plan"] is None
    _promote_breakdown(tmp_path, ["p0", "p1", "p2"])
    CALLS.clear()
    assert _call(_bta(tmp_path, max_breakdown=2), "sync").startswith("agg:")
    assert "bd" not in CALLS
    plan = _manifest(tmp_path)["plan"]["sub_queries"]
    assert [q["args"]["description"] for q in plan] == ["p0", "p1"]


@pytest.mark.parametrize("kind", KINDS)
def test_crash_after_the_commit_rebuilds_the_workers_from_the_plan(kind, tmp_path):
    _crash(tmp_path, kind, "w0", max_breakdown=2)
    assert _manifest(tmp_path)["plan"]["sub_queries"] == ["q0", "q1"]
    assert _call(_bta(tmp_path, max_breakdown=2), kind).startswith("agg:")
    # Async workers run concurrently, so w1 may have persisted before w0 died.
    assert "bd" not in CALLS and "w0" in CALLS and "w2" not in CALLS


def test_crash_after_some_workers_reruns_only_the_rest(tmp_path):
    _crash(tmp_path, "sync", "w2")
    assert _call(_bta(tmp_path), "sync").startswith("agg:")
    assert CALLS == ["w2", "agg"]


def test_crash_after_the_aggregator_finalizes_from_the_saved_result(tmp_path):
    bta = _bta(tmp_path)

    def crash_in_finalize(response):
        raise _Crash("finalize")

    bta._finalize_output = crash_in_finalize
    with pytest.raises(_Crash):
        bta.infer("task")
    CALLS.clear()
    assert _bta(tmp_path).infer("task").startswith("agg:")
    assert CALLS == []


@pytest.mark.parametrize(
    ("change", "part"),
    (
        ({"text": "another task"}, "input"),
        ({"max_breakdown": 1}, "definition"),
    ),
)
def test_a_mismatch_fails_before_any_stage_runs(change, part, tmp_path):
    _call(_bta(tmp_path), "sync")
    CALLS.clear()
    text = change.pop("text", "task")
    with pytest.raises(BtaResumeIdentityMismatch, match=part):
        _call(_bta(tmp_path, **change), "sync", text=text)
    assert CALLS == []


def test_a_manifest_write_cut_short_leaves_the_root_fresh(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / f".{MANIFEST_FILE}.k2j4x9").write_text('{"schema', encoding="utf-8")
    assert _call(_bta(tmp_path), "sync").startswith("agg:")
    assert _manifest(tmp_path)["plan"]["sub_queries"] == ["q0", "q1", "q2"]


def test_a_corrupt_manifest_fails_closed(tmp_path):
    _call(_bta(tmp_path), "sync")
    (tmp_path / "checkpoints" / MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    CALLS.clear()
    with pytest.raises(BtaResumeCorruptionError):
        _call(_bta(tmp_path), "sync")
    assert CALLS == []


def test_worker_results_without_a_committed_plan_are_corruption(tmp_path):
    _crash(tmp_path, "sync", "w2")
    path = tmp_path / "checkpoints" / MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["plan"] = None
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BtaResumeCorruptionError, match="without a committed plan"):
        _call(_bta(tmp_path), "sync")
    assert CALLS == []


def test_a_call_that_does_not_resume_never_vouches_for_an_earlier_calls_tree(
    tmp_path,
):
    """It overwrites only the checkpoints it reaches, so a resume of it would
    return the earlier call's aggregation."""
    _call(_bta(tmp_path), "sync", text="old task")
    _crash(tmp_path, "sync", "w1", resume_with_saved_results=False)
    assert not (tmp_path / "checkpoints" / MANIFEST_FILE).exists()
    with pytest.raises(UnverifiedLegacyBtaResumeError, match="did not resume"):
        _call(_bta(tmp_path), "sync")
    assert CALLS == []


def test_a_call_that_does_not_resume_vouches_for_a_clean_root(tmp_path):
    _crash(tmp_path, "sync", "w1", resume_with_saved_results=False)
    assert _manifest(tmp_path)["plan"]["sub_queries"] == ["q0", "q1", "q2"]
    assert _call(_bta(tmp_path), "sync").startswith("agg:")
    assert CALLS == ["w1", "w2", "agg"]


def _legacy_tree(root):
    _call(_bta(root), "sync")
    os.remove(root / "checkpoints" / MANIFEST_FILE)
    CALLS.clear()


def test_checkpoints_without_a_manifest_fail_closed(tmp_path):
    _legacy_tree(tmp_path)
    with pytest.raises(UnverifiedLegacyBtaResumeError, match="trust_legacy"):
        _call(_bta(tmp_path), "sync")
    assert CALLS == []
    assert not (tmp_path / ".attempts").exists()


def test_trust_legacy_resumes_with_a_warning_and_writes_no_manifest(tmp_path, caplog):
    _legacy_tree(tmp_path)
    with caplog.at_level(logging.WARNING):
        result = _call(_bta(tmp_path, resume_identity_policy="trust_legacy"), "sync")
    assert result.startswith("agg:")
    assert "trust_legacy" in caplog.text
    assert not (tmp_path / "checkpoints" / MANIFEST_FILE).exists()


def test_a_legacy_resume_rebuilds_the_list_the_breakdown_saved(tmp_path):
    """The breakdown node's saved result is the list after truncation; the
    promoted breakdown predates it (B35)."""
    _crash(tmp_path, "sync", "w1", max_breakdown=2)
    os.remove(tmp_path / "checkpoints" / MANIFEST_FILE)
    _promote_breakdown(tmp_path, ["p0", "p1", "p2"])
    trusted = _bta(tmp_path, max_breakdown=2, resume_identity_policy="trust_legacy")
    assert _call(trusted, "sync").startswith("agg:")
    assert CALLS == ["w1", "agg"]


def test_the_certify_helper_makes_a_legacy_tree_resumable(tmp_path):
    _legacy_tree(tmp_path)
    path = _bta(tmp_path).certify_resume_manifest("task")
    assert os.path.basename(path) == MANIFEST_FILE
    assert _manifest(tmp_path)["plan"]["sub_queries"] == ["q0", "q1", "q2"]
    assert _call(_bta(tmp_path), "sync").startswith("agg:")
    assert CALLS == []


def test_certify_rebinds_a_mismatched_manifest_and_keeps_its_plan(tmp_path, caplog):
    """E.g. after an upgrade changed the definition's identity: the committed
    plan still describes the workers on disk."""
    _crash(tmp_path, "sync", "w1", max_breakdown=2)
    path = tmp_path / "checkpoints" / MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["header"]["definition"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        _bta(tmp_path, max_breakdown=2).certify_resume_manifest("task")
    assert "written for another definition" in caplog.text
    assert _manifest(tmp_path)["plan"] == manifest["plan"]
    assert _call(_bta(tmp_path, max_breakdown=2), "sync").startswith("agg:")
    assert CALLS == ["w1", "agg"]


def test_certify_refuses_an_unprovable_identity(tmp_path):
    _legacy_tree(tmp_path)
    opaque = _bta(tmp_path, worker_inferencers=lambda sub_query, index: _Stage())
    with pytest.raises(ResumeIdentityUnavailableError):
        opaque.certify_resume_manifest("task")
    assert not (tmp_path / "checkpoints" / MANIFEST_FILE).exists()


def test_an_unverifiable_identity_runs_fresh_but_never_resumes(tmp_path, caplog):
    def opaque():
        return _bta(tmp_path, worker_inferencers=lambda sub_query, index: _Stage())

    with caplog.at_level(logging.WARNING):
        assert _call(opaque(), "sync").startswith("agg:")
    assert "can't be resumed" in caplog.text
    assert "lambda" in _manifest(tmp_path)["header"]["unverifiable"]
    CALLS.clear()
    with pytest.raises(ResumeIdentityUnavailableError, match="cannot verify"):
        _call(opaque(), "sync")
    assert CALLS == []


class _Query:
    """A sub-query that is not a JSON value."""

    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


def _object_parser(raw_output):
    return [{"query": f"q{i}", "args": {"source": _Query("s")}} for i in range(2)]


def test_a_plan_that_cannot_be_replayed_resumes_only_when_trusted(tmp_path, caplog):
    _crash(tmp_path, "sync", "w1", breakdown_parser=_object_parser)
    plan = _manifest(tmp_path)["plan"]
    assert "unavailable" in plan and plan["workers"] == ["rm.worker_00", "rm.worker_01"]
    with pytest.raises(ResumeIdentityUnavailableError, match="can't be replayed"):
        _call(_bta(tmp_path, breakdown_parser=_object_parser), "sync")
    assert CALLS == []
    trusted = _bta(
        tmp_path, breakdown_parser=_object_parser, resume_identity_policy="trust_legacy"
    )
    with caplog.at_level(logging.WARNING):
        assert _call(trusted, "sync").startswith("agg:")
    assert "trust_legacy" in caplog.text
    assert CALLS == ["w1", "agg"]
