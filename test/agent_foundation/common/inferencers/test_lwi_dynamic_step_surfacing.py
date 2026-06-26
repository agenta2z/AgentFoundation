"""M7 regression guard: a dynamic-mode LWI must surface its LAST step's output to
the flow's outputs/ — under a live run-context.

Root cause this guards against: M7 dispatches each dynamic step under ctx node
`step_{N}` (path-mirrored on-disk workspace children/step_N), but _finalize_output
(+ propagation) address children/initial|round{NN}. The naming diverged, so the
flow surfaced a non-existent path and its outputs/final_deliverables stayed empty,
which in turn made the parent aggregator embed raw <Response> text instead of a
clean (See file: <path>) reference.

This test runs the REAL dynamic dispatch under the ctx bridge (unlike the
construction-only test_lwi_symlinks_to_last_child) so the step writes to its
ctx-resolved workspace and the divergence is exercised end-to-end.
"""

import asyncio
import os

from attr import attrs, attrib

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.run_context import RunContext, enter_run, exit_run
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
)


@attrs
class _StepMock(InferencerBase):
    """A leaf step that writes its <Response> to outputs/output.md (via the base
    _finalize_output) at whatever workspace it resolves under the active ctx."""

    scripted_response: str = attrib(default="<Response>step</Response>")

    def _write_output(self):
        ws = self._workspace
        if ws is not None:
            out = ws.output_path("output.md")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                f.write(self.scripted_response)

    def _infer(self, inference_input, inference_config=None, **kw):
        self._write_output()
        return self.scripted_response

    async def _ainfer(self, inference_input, inference_config=None, **kw):
        self._write_output()
        return self.scripted_response


def _run_two_step_dynamic_flow(flow_root):
    flow_ws = InferencerWorkspace(root=flow_root)
    # Construct WITHOUT a workspace: the flow's workspace arrives via the active
    # run-context (as it does deep in the real BTA/MFI dispatch), so propagation
    # runs under a ctx and the step instances are not pre-pinned — reproducing the
    # step_N path-mirror that diverges from _finalize_output's round{NN}.
    lwi = LinearWorkflowInferencer(
        dynamic_mode=True,
        default_initial_inferencer=_StepMock(scripted_response="<Response>INITIAL artifact</Response>"),
        default_followup_inferencer=_StepMock(scripted_response="<Response>ROUND01 artifact</Response>"),
        end_condition=lambda s, r: len(s.get("dynamic_step_results", [])) >= 2,
        max_dynamic_steps=2,
        output_path="output.md",
    )
    ctx = RunContext.root(workspace=flow_ws)
    tok = enter_run(ctx)
    try:
        asyncio.run(lwi.ainfer("design the --dry-run CLI flag"))
    finally:
        exit_run(tok)
    return flow_ws


def test_dynamic_flow_surfaces_last_step_output(tmp_path):
    """The flow's outputs/output.md must be populated with the LAST step's artifact."""
    flow_root = str(tmp_path / "flow")
    _run_two_step_dynamic_flow(flow_root)

    own = os.path.join(flow_root, "outputs", "output.md")
    assert os.path.exists(own), (
        "flow did NOT surface its last step's output to outputs/output.md — the "
        "dynamic-step workspace naming (step_N) diverged from _finalize_output (round{NN})"
    )
    assert "ROUND01" in open(own).read(), "must surface the LAST step (round01), not an earlier one"


def test_dynamic_step_writes_under_canonical_round_dir(tmp_path):
    """The last step's on-disk output must live under children/round01/ (what
    propagation, _finalize_output, and the MFI flow-output resolver address) — not
    under children/step_1/."""
    flow_root = str(tmp_path / "flow")
    _run_two_step_dynamic_flow(flow_root)

    canonical = os.path.join(flow_root, "children", "round01", "outputs", "output.md")
    assert os.path.exists(canonical), (
        "last step output must land under children/round01/ (canonical naming), "
        f"found instead: {sorted(os.listdir(os.path.join(flow_root, 'children'))) if os.path.isdir(os.path.join(flow_root, 'children')) else 'no children dir'}"
    )
