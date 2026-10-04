"""Orchestrators publish the task contract of their input-side author (plan v8
§5.3, P6 c4; B1, B13).

A BTA records the contract each successful worker published (read at the worker's
own ctx, ``_task_contract_state_at``) and relays one: the lowest successful worker
index, whatever order the workers finished in (B13), on the sync path too (B1). MFI
relays its winner's flow first. A Dual relays the contract it captured from its
proposer at propose-completion. Each publishes through ``_outcome_for`` at its own
node; outside a host ctx the documented getter fields are their compat projections,
and a host call writes none of them.
"""

from __future__ import annotations

import ast
import asyncio
import json
import pathlib
from typing import Any

import agent_foundation
import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    ConsensusConfig,
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
    MultiFlowDualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import read_outcome, RunContext
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrib, attrs
from rich_python_utils.common_utils.function_helper import FallbackMode

KINDS = ("sync", "async")


class _Templates:
    """Renders ``[root|key] input``; the task contract is ``Contract for <input>``."""

    def __call__(
        self, key, *, active_template_root_space=None, master_version=None, **feed
    ):
        return f"[{active_template_root_space}|{key}] {feed.get('input')}"

    def get_raw_template(self, *a, **k):
        return "x"

    def load_variables(self, variable_specs=None, root_space="", **k):
        if "task_instructions" not in (variable_specs or {}):
            return {}
        return {"task_instructions": "Contract for {{ input }}"}

    def _resolve_templated_feed(self, feed, root_space=""):
        text = feed["task_instructions"].replace("{{ input }}", str(feed.get("input")))
        return {**feed, "task_instructions": text}

    def add_template_root(self, *a, **k):
        pass

    def __deepcopy__(self, memo):
        return self


@attrs(slots=False)
class _Author(TemplatedInferencerBase):
    """A templated leaf; ``delay`` slows its async call, ``fail`` makes it raise."""

    delay: float = attrib(default=0.0, kw_only=True)
    fail: bool = attrib(default=False, kw_only=True)
    finished: Any = attrib(default=None, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        if self.fail:
            raise RuntimeError("author failed")
        if self.finished is not None:
            self.finished.append(inference_input)
        return f"done({inference_input})"

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        await asyncio.sleep(self.delay)
        return self._infer(inference_input)


def _author(**kwargs):
    return _Author(
        template_manager=_Templates(),
        template_root_space="plan",
        template_key="initial",
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
        **kwargs,
    )


@attrs
class _Pure(InferencerBase):
    response: str = attrib(default="ok", kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self.response


def _call(inf, kind, text="go", **kwargs):
    if kind == "sync":
        return inf.infer(text, **kwargs)
    return asyncio.run(inf.ainfer(text, **kwargs))


def _host(tmp_path):
    return RunContext.root(workspace=InferencerWorkspace(root=str(tmp_path)))


# -- BTA ----------------------------------------------------------------------------


def _bta(workers, **kwargs):
    return BreakdownThenAggregateInferencer(
        breakdown_inferencer=_Pure(),
        worker_inferencers=workers,
        predefined_sub_queries=["q0", "q1"],
        disable_aggregator=True,
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
        **kwargs,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_bta_relays_the_lowest_successful_worker_index(kind, tmp_path):
    """B13: worker 0 finishes last on the async path; its contract is still the
    one relayed. B1: the sync path relays one too."""
    finished = []

    def workers(sub_query, index):
        return _author(delay=0.05 if index == 0 else 0.0, finished=finished)

    ctx = _host(tmp_path)
    _call(_bta(workers), kind, run_context=ctx)
    if kind == "async":
        assert [prompt[-2:] for prompt in finished] == ["q1", "q0"]
    outcome = read_outcome(ctx)
    assert outcome.task_contract.text == "Contract for q0"
    assert outcome.summary.selected_contract_index == 0
    assert outcome.task_contract.source_path.endswith("worker_00")


def test_a_failed_worker_is_never_relayed(tmp_path):
    def workers(sub_query, index):
        return _author(fail=index == 0)

    ctx = _host(tmp_path)
    _call(_bta(workers, min_successful_workers=1), "async", run_context=ctx)
    outcome = read_outcome(ctx)
    assert outcome.task_contract.text == "Contract for q1"
    assert outcome.summary.selected_contract_index == 1


@pytest.mark.parametrize("kind", KINDS)
def test_two_host_calls_each_publish_their_own_contract(kind, tmp_path):
    bta = _bta(lambda sub_query, index: _author())
    first, second = _host(tmp_path / "one"), _host(tmp_path / "two")
    bta.predefined_sub_queries = ["first"]
    _call(bta, kind, run_context=first)
    bta.predefined_sub_queries = ["second"]
    _call(bta, kind, run_context=second)
    assert read_outcome(first).task_contract.text == "Contract for first"
    assert read_outcome(second).task_contract.text == "Contract for second"
    assert bta._worker_task_instructions == ""


@pytest.mark.parametrize("kind", KINDS)
def test_a_bare_call_projects_the_relayed_contract_onto_the_getter(kind, tmp_path):
    bta = _bta(
        lambda sub_query, index: _author(),
        workspace=InferencerWorkspace(root=str(tmp_path)),
    )
    _call(bta, kind)
    assert bta._proposer_task_instructions() == "Contract for q0"
    assert bta.last_call_summary.selected_task_contract == "Contract for q0"


def test_a_host_leaf_call_writes_no_getter_field(tmp_path):
    leaf = _author()
    ctx = _host(tmp_path)
    leaf.infer("task", run_context=ctx)
    assert read_outcome(ctx).task_contract.text == "Contract for task"
    assert leaf._last_rendered_task_instructions == ""
    leaf.infer("bare")
    assert leaf._last_rendered_task_instructions == "Contract for bare"


# -- MFI ----------------------------------------------------------------------------


def _mfi(winner=None):
    flows = [
        {
            "input": name,
            "initial_inferencer": _author(),
            "followup_inferencer": _author(),
            "end_condition": lambda state, out: True,
            "max_dynamic_steps": 1,
        }
        for name in ("task-x", "task-y")
    ]
    return MultiFlowInferencer(
        flow_configs=flows,
        aggregator_inferencer=_Pure(response="AGG"),
        winner_parser=(lambda raw: winner),
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
    )


@pytest.mark.parametrize("winner", (None, 1))
@pytest.mark.parametrize("kind", KINDS)
def test_mfi_relays_the_winners_flow_first(kind, winner, tmp_path):
    ctx = _host(tmp_path)
    _call(_mfi(winner), kind, run_context=ctx)
    expected = "Contract for task-y" if winner == 1 else "Contract for task-x"
    outcome = read_outcome(ctx)
    assert outcome.task_contract.text == expected
    assert outcome.summary.selected_contract_index == (winner or 0)


# -- Dual and the chain ----------------------------------------------------------------


def _dual(base, **kwargs):
    return DualInferencer(
        base_inferencer=base,
        review_inferencer=_Pure(
            response="## Review\nSeverity: COSMETIC\nApproved: true"
        ),
        fixer_inferencer=None,
        consensus_config=ConsensusConfig(),
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
        **kwargs,
    )


def test_dual_publishes_its_proposers_contract(tmp_path):
    ctx = _host(tmp_path)
    _call(_dual(_author()), "async", "task", run_context=ctx)
    outcome = read_outcome(ctx)
    assert outcome.task_contract.text == "Contract for task"
    assert outcome.task_contract.source_path == "/propose"


@pytest.mark.parametrize("winner", (None, 1))
def test_the_chain_leaf_lwi_bta_mfi_dual_relays_the_winners_author(winner, tmp_path):
    """leaf → LWI (each flow) → MFI (a BTA) → Dual, under a host ctx."""
    ctx = _host(tmp_path)
    _call(_dual(_mfi(winner)), "async", "task", run_context=ctx)
    expected = "Contract for task-y" if winner == 1 else "Contract for task-x"
    assert read_outcome(ctx).task_contract.text == expected
    assert read_outcome(ctx.child("propose")).task_contract.text == expected


@pytest.mark.parametrize("winner", (None, 1))
def test_the_chain_outside_a_host_ctx_relays_through_the_getters(winner, tmp_path):
    dual = _dual(_mfi(winner), workspace=InferencerWorkspace(root=str(tmp_path)))
    _call(dual, "async", "task")
    expected = "Contract for task-y" if winner == 1 else "Contract for task-x"
    assert dual._proposer_task_instructions() == expected
    assert dual.base_inferencer._worker_task_instructions == expected


def _verdict(approved):
    issues = [
        {
            "severity": "MAJOR",
            "category": "logic",
            "description": "gap",
            "location": "n/a",
            "suggestion": "fix",
        }
    ]
    body = {
        "approved": approved,
        "severity": "MINOR" if approved else "MAJOR",
        "issues": [] if approved else issues,
        "reasoning": "r",
    }
    return f"```json\n{json.dumps(body)}\n```"


@attrs(slots=False)
class _RoleAuthor(_Author):
    """Answers reviews from a shared script of verdicts and records its prompts."""

    verdicts: list = attrib(factory=list, kw_only=True)
    prompts: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        prompt = str(inference_input)
        self.prompts.append(prompt)
        if prompt.startswith("[review"):
            return self.verdicts.pop(0) if self.verdicts else _verdict(True)
        return f"done({prompt})"


_DUAL_AGGREGATION = (
    "<FinalPlan>\nintegrated\n</FinalPlan>\n"
    '```json winner_pick\n{"winner_index": 0, "reason": "t"}\n```\n'
    '```json ranking\n{"ranking": [0, 1], "reason": "t"}\n```'
)


def test_the_mfdual_fixer_rerender_leaves_the_propose_contract(tmp_path):
    """MultiFlowDual reuses the winning flow's leaf as its fixer; the fixer's
    re-render publishes at the fix node, so the Dual still relays the contract its
    proposer rendered."""
    verdicts = [_verdict(False), _verdict(True)]
    leaves = [
        _RoleAuthor(
            template_manager=_Templates(),
            template_root_space="plan",
            template_key="initial",
            fallback_mode=FallbackMode.NEVER,
            max_retry=0,
            verdicts=verdicts,
        )
        for _ in range(2)
    ]
    flows = [
        {
            "input": name,
            "initial_inferencer": leaf,
            "followup_inferencer": _Pure(),
            "end_condition": lambda state, out: True,
            "max_dynamic_steps": 1,
        }
        for name, leaf in zip(("task-x", "task-y"), leaves)
    ]
    mfdual = MultiFlowDualInferencer(
        flow_configs=flows,
        multi_flow_aggregator_inferencer=_Pure(response=_DUAL_AGGREGATION),
        reviewer_strategy="runner_up",
        fixer_strategy="winner",
        review_template_root_space="review",
        followup_template_root_space="followup",
        consensus_config=ConsensusConfig(max_iterations=2),
        fallback_mode=FallbackMode.NEVER,
        max_retry=0,
    )
    ctx = _host(tmp_path)
    asyncio.run(mfdual.ainfer("go", run_context=ctx))
    assert any(prompt.startswith("[followup") for prompt in leaves[0].prompts)
    assert read_outcome(ctx).task_contract.text == "Contract for task-x"


# -- stop gate --------------------------------------------------------------------------


def _getter_call_sites():
    """``(module, enclosing function)`` of every call of a child's
    ``_proposer_task_instructions`` getter in the package source."""
    root = pathlib.Path(agent_foundation.__file__).parent
    sites = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                named = (
                    isinstance(node, ast.Attribute)
                    and node.attr == "_proposer_task_instructions"
                ) or (
                    isinstance(node, ast.Constant)
                    and node.value == "_proposer_task_instructions"
                )
                if named:
                    sites.add((path.relative_to(root).as_posix(), func.name))
    return sites


def test_no_parent_reads_a_childs_last_call_getter():
    """P6 stop gate: the only reader of a child's getter is the true no-ctx branch
    of ``_task_contract_state_at``."""
    assert _getter_call_sites() == {
        ("common/inferencers/inferencer_base.py", "_task_contract_state_at")
    }
