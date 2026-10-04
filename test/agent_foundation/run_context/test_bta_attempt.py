"""A BTA run is a loop of attempts, each with its own state (plan v8 §5.7, P6 c2).

``_open_attempt`` puts a fresh ``_BtaAttempt`` in the frame and runs the
``_begin_attempt`` hook; everything the run used to keep on the instance (the
original query, the promoted-breakdown memo, the aggregation guidance, the topology
guard and pending topology, ``use_async``) lives on the attempt. An interactive
"rerun" review starts the next attempt instead of re-entering ``_ainfer``, and the
tail (reconcile, ``_finalize_response``, ``_conclude_attempt``) runs once, for the
attempt whose result is returned. MFI's pre- and post-processing are those two
hooks.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Any

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import frame_for
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrib, attrs

KINDS = ("sync", "async")


@attrs
class _Leaf(InferencerBase):
    response: str = attrib(default="ok", kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return f"{self.response}:{inference_input}"


class _Interactive:
    """Answers results reviews from a script."""

    def __init__(self, answers):
        self.answers = list(answers)

    async def asend_response(self, prompt, flag=None, input_mode=None, **kwargs):
        pass

    async def aget_input(self):
        return self.answers.pop(0)


def _render_guidance(key, active_template_root_space=None, master_version=None, **feed):
    return feed.get("aggregation_guidance")


@attrs(slots=False)
class _GuidanceAggregator(TemplatedInferencerBase):
    """A templated aggregator whose rendered prompt is the guidance in its feed."""

    # A recorder, not configuration: it stays out of the resume identity.
    rendered: list = attrib(factory=list, init=False)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        self.rendered.append(inference_input)
        return "aggregated"


@attrs(slots=False)
class _RecordingBTA(BreakdownThenAggregateInferencer):
    attempts: list = attrib(factory=list, kw_only=True)
    concluded: list = attrib(factory=list, kw_only=True)

    def _begin_attempt(self, attempt, inference_input):
        self.attempts.append(attempt)
        super()._begin_attempt(attempt, inference_input)

    def _conclude_attempt(self, result):
        self.concluded.append(result)
        return super()._conclude_attempt(result)


def _workers(sub_query, index):
    return _Leaf(response=f"w{index}")


def _call(inf, kind, text="go", **kwargs):
    if kind == "sync":
        return inf.infer(text, **kwargs)
    return asyncio.run(inf.ainfer(text, **kwargs))


# -- the attempt loop ---------------------------------------------------------------


def test_a_rerun_review_runs_a_second_attempt_and_the_tail_once(tmp_path):
    bta = _RecordingBTA(
        breakdown_inferencer=_Leaf(),
        worker_inferencers=_workers,
        aggregator_inferencer=_Leaf(response="agg"),
        predefined_sub_queries=["q0"],
        interactive=_Interactive(["rerun", "approve"]),
        enable_checkpoint_results_review=True,
        workspace=InferencerWorkspace(root=str(tmp_path)),
    )
    assert _call(bta, "async") == "agg:go"
    first, second = bta.attempts
    assert first is not second
    assert first.original_query == second.original_query == "go"
    assert bta.concluded == ["agg:go"]


@pytest.mark.parametrize("kind", KINDS)
def test_a_run_without_review_is_one_attempt(kind, tmp_path):
    bta = _RecordingBTA(
        breakdown_inferencer=_Leaf(),
        worker_inferencers=_workers,
        aggregator_inferencer=_Leaf(response="agg"),
        predefined_sub_queries=["q0"],
        workspace=InferencerWorkspace(root=str(tmp_path)),
    )
    _call(bta, kind)
    (attempt,) = bta.attempts
    assert attempt.use_async is (kind == "async")
    assert len(bta.concluded) == 1


@pytest.mark.parametrize("kind", KINDS)
def test_the_stages_run_on_the_attempts_path(kind, tmp_path):
    """An async attempt builds async node functions (stages entered through
    ``ainfer``), a sync one sync functions (``infer``); the BTA keeps no
    ``use_async`` of its own."""
    seen = []

    @attrs
    class _Probe(InferencerBase):
        response: str = attrib(default="probe", kw_only=True)

        def _infer(self, inference_input, inference_config=None, **kwargs):
            seen.append((self.response, frame_for(self).entry))
            return self.response

    bta = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Probe(response="1. q0"),
        worker_inferencers=lambda sub_query, index: _Probe(response="worker"),
        disable_aggregator=True,
        workspace=InferencerWorkspace(root=str(tmp_path)),
    )
    _call(bta, kind)
    entry = "ainfer" if kind == "async" else "infer"
    assert seen == [("1. q0", entry), ("worker", entry)]
    assert "use_async" not in vars(bta)


# -- MFI's hooks ------------------------------------------------------------------


@attrs(slots=False)
class _CountingMFI(MultiFlowInferencer):
    counts: dict = attrib(factory=dict, kw_only=True)

    def _count(self, name):
        self.counts[name] = self.counts.get(name, 0) + 1

    def _apply_runtime_input_propagation(self, attempt, inference_input):
        self._count("propagate")
        super()._apply_runtime_input_propagation(attempt, inference_input)

    def _reset_cross_flow_state(self):
        self._count("reset")
        super()._reset_cross_flow_state()


def _mfi(tmp_path, parsed, **extra):
    def parser(raw):
        parsed.append(raw)
        return f"parsed<{raw}>"

    flows = [
        {
            "input": f"task{i}",
            "initial_inferencer": _Leaf(response=f"f{i}"),
            "followup_inferencer": _Leaf(response=f"f{i}f"),
            "end_condition": lambda state, out: True,
            "max_dynamic_steps": 1,
        }
        for i in range(2)
    ]
    return _CountingMFI(
        flow_configs=flows,
        aggregator_inferencer=_Leaf(response="agg"),
        response_parser=parser,
        workspace=InferencerWorkspace(root=str(tmp_path)),
        **extra,
    )


def test_mfi_prepares_every_attempt_and_post_processes_once_per_call(tmp_path):
    parsed = []
    mfi = _mfi(
        tmp_path,
        parsed,
        interactive=_Interactive(["rerun", "approve"]),
        enable_checkpoint_results_review=True,
    )
    assert _call(mfi, "async", "hello") == "parsed<agg:hello>"
    assert mfi.counts == {"propagate": 2, "reset": 2}
    assert parsed == ["agg:hello"]


@pytest.mark.parametrize("kind", KINDS)
def test_mfi_sync_and_async_run_the_same_hooks(kind, tmp_path):
    parsed = []
    mfi = _mfi(tmp_path, parsed)
    assert _call(mfi, kind, "hello") == "parsed<agg:hello>"
    assert mfi.counts == {"propagate": 1, "reset": 1}
    assert parsed == ["agg:hello"]


# -- reuse --------------------------------------------------------------------------


def _breakdown_json(guidance):
    return (
        "```json\n"
        + json.dumps(
            {"subtasks": [{"description": "D0"}], "aggregation_guidance": guidance}
        )
        + "\n```"
    )


def _drop_aggregation_results(ws_root):
    checkpoints = os.path.join(ws_root, "checkpoints")
    for name in os.listdir(checkpoints):
        if name.startswith(("aggregator_result", "reuse_result")):
            path = os.path.join(checkpoints, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)


def _promote(ws_root, guidance):
    path = os.path.join(ws_root, "checkpoints", "breakdown", "decomposed_subtasks.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"subtasks": [{"description": "D0"}], "aggregation_guidance": guidance}, f
        )


@pytest.mark.parametrize("kind", KINDS)
def test_a_resumed_second_call_gets_its_own_guidance_not_the_first_calls(
    kind, tmp_path
):
    """B3: a reused BTA's second call resumes; its aggregator gets the promoted
    breakdown's guidance, not the guidance the first call parsed in-process."""
    aggregator = _GuidanceAggregator(
        template_manager=_render_guidance, template_root_space="aggregation"
    )
    bta = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Leaf(response=_breakdown_json("FIRST")),
        worker_inferencers=_workers,
        aggregator_inferencer=aggregator,
        breakdown_format="json_subtasks",
        enable_result_save=True,
        resume_with_saved_results=True,
        workspace=InferencerWorkspace(root=str(tmp_path)),
    )
    bta.name = "reuse"
    _call(bta, kind)
    _drop_aggregation_results(str(tmp_path))
    _promote(str(tmp_path), "SECOND")
    _call(bta, kind)
    assert aggregator.rendered == ["FIRST", "SECOND"]


def test_no_run_state_is_left_on_the_instance(tmp_path):
    bta = BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Leaf(response=_breakdown_json("G")),
        worker_inferencers=_workers,
        aggregator_inferencer=_Leaf(response="agg"),
        breakdown_format="json_subtasks",
        workspace=InferencerWorkspace(root=str(tmp_path)),
    )
    _call(bta, "async")
    leftovers: Any = {
        "_cached_original_query",
        "_promoted_breakdown_cache",
        "_pending_topology",
        "_graph_topology_emitted",
        "_last_aggregation_guidance",
    } & set(vars(bta))
    assert leftovers == set()
