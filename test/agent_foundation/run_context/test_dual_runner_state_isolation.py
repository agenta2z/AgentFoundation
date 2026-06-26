"""Codex Commit 3 audit: Dual/MFDual per-run runtime fields (`_current_attempt`,
`_current_inference_config`, `_current_extra_inference_args`, `_current_round_ws`,
`_last_output_child_ws`, `step_configs`) are virtualized into the active ctx
(`ctx.node().scratch`) via `_run_get`/`_run_set` (and a compat-property for `step_configs`),
so ONE shared Dual/MFDual instance is concurrency-isolated across distinct RunContexts.
Legacy / no-ctx falls back to the instance `__dict__` backing (byte-identical).

(The dead-agent that virtualized the fields landed the code but dropped on an API error
before writing this test; the resolver was verified working by hand and is captured here.)
"""

from attr import attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.run_context import (
    RunContext,
    enter_run,
    exit_run,
)


@attrs(slots=False)
class _Leaf(InferencerBase):
    def _infer(self, x, inference_config=None, **kw):
        return x


def _dual():
    return DualInferencer(base_inferencer=_Leaf(), review_inferencer=_Leaf())


def test_run_fields_isolated_across_two_ctxs_on_one_shared_instance():
    d = _dual()
    rA, rB = RunContext.root(workspace=None), RunContext.root(workspace=None)

    tA = enter_run(rA)
    try:
        d._run_set("_current_inference_config", {"a": 1})
        d._run_set("_current_attempt", 1)
        d.step_configs = ["A"]
    finally:
        exit_run(tA)

    tB = enter_run(rB)
    try:
        d._run_set("_current_inference_config", {"b": 2})
        d._run_set("_current_attempt", 2)
        d.step_configs = ["B"]
    finally:
        exit_run(tB)

    # Re-enter each: its own per-node scratch survived, the other's run never clobbered it.
    tA = enter_run(rA)
    try:
        assert d._run_get("_current_inference_config") == {"a": 1}
        assert d._run_get("_current_attempt") == 1
        assert d.step_configs == ["A"]
    finally:
        exit_run(tA)

    tB = enter_run(rB)
    try:
        assert d._run_get("_current_inference_config") == {"b": 2}
        assert d._run_get("_current_attempt") == 2
        assert d.step_configs == ["B"]
    finally:
        exit_run(tB)

    # The shared instance itself was never mutated by either run.
    assert "_current_inference_config" not in d.__dict__
    assert d._run_get("_current_inference_config", "DEFAULT") == "DEFAULT"


def test_step_configs_in_place_patch_isolated_per_ctx():
    """The propose-override patches ``self.step_configs[0] = ...`` in place; the getter
    must resolve the per-run list so the patch stays isolated under a real ctx."""
    d = _dual()
    rA, rB = RunContext.root(workspace=None), RunContext.root(workspace=None)

    tA = enter_run(rA)
    try:
        d.step_configs = ["a0", "a1"]
        d.step_configs[0] = "a0_patched"  # in-place patch (the :852 pattern)
    finally:
        exit_run(tA)

    tB = enter_run(rB)
    try:
        d.step_configs = ["b0", "b1"]
    finally:
        exit_run(tB)

    tA = enter_run(rA)
    try:
        assert d.step_configs == ["a0_patched", "a1"]
    finally:
        exit_run(tA)


def test_legacy_no_ctx_uses_instance_backing_byte_identical():
    d = _dual()
    # No active context: writes/reads go to the instance backing.
    d._run_set("_current_attempt", 7)
    d.step_configs = ["legacy"]
    assert d._run_get("_current_attempt") == 7
    assert d.step_configs == ["legacy"]
    assert d.__dict__.get("_current_attempt_backing") == 7


def test_legacy_mint_uses_instance_backing():
    """A legacy-minted root (bare ``infer()``) is treated like no-ctx for run fields, so
    post-call instance reads still work."""
    from agent_foundation.common.inferencers.run_context import bridge

    d = _dual()
    root = bridge.mint_root(workspace=None)
    tok = enter_run(root)
    try:
        d._run_set("_current_attempt", 9)
        assert d._run_get("_current_attempt") == 9
    finally:
        exit_run(tok)
    # legacy-mint mirrored to the backing -> readable after the ctx is gone
    assert d._run_get("_current_attempt") == 9
