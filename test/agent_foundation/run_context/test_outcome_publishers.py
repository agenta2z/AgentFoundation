"""Typed outcome channel, generic half (plan v8 §5.3, P4).

A templated leaf publishes the task contract it rendered in its call
(``NodeOutcomeState.task_contract``), and a ``LinearWorkflowInferencer``
publishes its first step child's. Parents read a child's contract at the exact
child ctx through ``_task_contract_at``: the typed channel under any ctx, the
documented getter only in true no-ctx.
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)
from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.run_context import (
    enter_run,
    exit_run,
    read_outcome,
    RunContext,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from attr import attrib, attrs

KINDS = ("sync", "async")
CONTRACT_MARK = "Conclude your deliverable"


def _template_manager():
    from agent_foundation.resources import PROMPT_TEMPLATES_ROOT
    from rich_python_utils.string_utils.formatting.jinja2_format import format_template
    from rich_python_utils.string_utils.formatting.template_manager import (
        TemplateManager,
    )

    return TemplateManager(
        templates=str(PROMPT_TEMPLATES_ROOT),
        template_formatter=format_template,
        active_template_root_space="plan",
        active_template_type="main",
        predefined_variables=True,
        default_template_key="initial",
        enable_templated_feed=True,
    )


@attrs(slots=False)
class _Author(TemplatedInferencerBase):
    """Renders the research_propose proposer contract, which embeds its own
    output path."""

    fail: bool = attrib(default=False, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        if self.fail:
            raise RuntimeError("model failed")
        return f"done:{inference_input}"

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._infer(inference_input, inference_config, **kwargs)


def _author(output_path, **kwargs):
    leaf = _Author(**kwargs)
    leaf.template_manager = _template_manager()
    leaf.template_root_space = "plan"
    leaf.template_key = "initial"
    leaf.template_master_version = "research_propose"
    leaf.template_variables = {"task_instructions": "research_propose"}
    leaf.output_path = output_path
    leaf.has_local_access = True
    return leaf


def _call(inf, kind, text="q", **kwargs):
    if kind == "sync":
        return inf.infer(text, **kwargs)
    return asyncio.run(inf.ainfer(text, **kwargs))


def _host(slot="leaf"):
    return RunContext.root().child(slot)


# -- the templated leaf ---------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_a_templated_leaf_publishes_the_contract_it_rendered(kind, tmp_path):
    out = str(tmp_path / "author" / "outputs" / "output.md")
    leaf, ctx = _author(out), _host()
    _call(leaf, kind, run_context=ctx)
    contract = read_outcome(ctx).task_contract
    assert CONTRACT_MARK in contract.text
    assert out in contract.text
    assert contract.sha256 == hashlib.sha256(contract.text.encode()).hexdigest()
    assert (contract.role, contract.source_path) == (None, "/leaf")


@pytest.mark.parametrize("kind", KINDS)
def test_a_failed_leaf_call_publishes_nothing(kind, tmp_path):
    leaf, ctx = _author(str(tmp_path / "o.md"), fail=True), _host()
    with pytest.raises(RuntimeError, match="model failed"):
        _call(leaf, kind, run_context=ctx)
    assert read_outcome(ctx) is None


def test_one_leaf_in_two_branches_publishes_each_branchs_own_render(tmp_path):
    out_a, out_b = str(tmp_path / "a" / "o.md"), str(tmp_path / "b" / "o.md")
    leaf = _author(out_a)
    a, b = _host("a"), _host("b")
    leaf.infer("q", run_context=a)
    leaf.output_path = out_b
    leaf.infer("q", run_context=b)
    assert out_a in read_outcome(a).task_contract.text
    assert out_b in read_outcome(b).task_contract.text


# -- parents read at the exact child ctx -------------------------------------------


@attrs
class _Parent(InferencerBase):
    """Calls its author at a child slot, then reads the author's contract there."""

    author: _Author = attrib(kw_only=True)
    read: list = attrib(factory=list, kw_only=True)

    def _infer(self, inference_input, inference_config=None, **kwargs):
        child_ctx = self._rc_child("author")
        self.author.infer(inference_input, run_context=child_ctx)
        self.read.append(self._task_contract_at(self.author, child_ctx))
        return "ok"

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        child_ctx = self._rc_child("author")
        await self.author.ainfer(inference_input, run_context=child_ctx)
        self.read.append(self._task_contract_at(self.author, child_ctx))
        return "ok"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("host", (True, False), ids=("host", "legacy"))
def test_a_parent_reads_its_childs_contract_at_the_child_ctx(kind, host, tmp_path):
    parent = _Parent(author=_author(str(tmp_path / "o.md")))
    _call(parent, kind, run_context=RunContext.root() if host else None)
    (read,) = parent.read
    assert CONTRACT_MARK in read


def test_with_a_ctx_an_absent_outcome_reads_empty_never_the_getter(tmp_path):
    leaf = _author(str(tmp_path / "o.md"))
    leaf._last_rendered_task_instructions = "a stale getter value"
    assert InferencerBase._task_contract_at(leaf, _host()) == ""


def test_true_no_ctx_falls_back_to_the_documented_getter(tmp_path):
    leaf = _author(str(tmp_path / "o.md"))
    leaf._last_rendered_task_instructions = "the getter value"
    assert InferencerBase._task_contract_at(leaf, None) == "the getter value"
    assert InferencerBase._task_contract_at(object(), None) == ""


# -- the workflow ---------------------------------------------------------------


def _static_lwi(author, tmp_path):
    return LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(name="propose", inferencer=author),
            WorkflowStepConfig(
                name="polish", inferencer=_author(str(tmp_path / "polish.md"))
            ),
        ],
    )


def _dynamic_lwi(author, tmp_path):
    return LinearWorkflowInferencer(
        dynamic_mode=True,
        default_initial_inferencer=author,
        default_followup_inferencer=_author(str(tmp_path / "followup.md")),
        end_condition=lambda state, result: True,
        max_dynamic_steps=2,
    )


@pytest.mark.parametrize(
    "build", (_static_lwi, _dynamic_lwi), ids=("static", "dynamic")
)
def test_a_workflow_publishes_its_first_step_childs_contract(build, tmp_path):
    first = str(tmp_path / "first" / "outputs" / "output.md")
    lwi, ctx = build(_author(first), tmp_path), _host("lwi")
    asyncio.run(lwi.ainfer("q", run_context=ctx))
    contract = read_outcome(ctx).task_contract
    assert first in contract.text
    assert contract.source_path.startswith("/lwi/")


def test_a_workflow_whose_first_step_fails_publishes_nothing(tmp_path):
    lwi = _static_lwi(_author(str(tmp_path / "o.md"), fail=True), tmp_path)
    ctx = _host("lwi")
    with pytest.raises(RuntimeError, match="model failed"):
        asyncio.run(lwi.ainfer("q", run_context=ctx))
    assert read_outcome(ctx) is None


def test_a_preview_render_never_publishes_or_changes_the_published_contract(tmp_path):
    """Dual previews a leaf's prompt with ``_render_prompt`` under the leaf's ctx but
    outside the leaf's invocation; only the leaf's own call publishes."""
    out = str(tmp_path / "author" / "o.md")
    leaf, ctx = _author(out), _host()
    leaf.infer("q", run_context=ctx)
    published = read_outcome(ctx)
    leaf.output_path = str(tmp_path / "preview" / "o.md")
    token = enter_run(ctx)
    try:
        leaf._render_prompt("q")
    finally:
        exit_run(token)
    assert read_outcome(ctx) is published
    assert out in published.task_contract.text
