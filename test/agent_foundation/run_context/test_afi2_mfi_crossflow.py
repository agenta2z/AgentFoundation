"""AF1 / C3 (Part B) — MFI cross-flow buffers are per-run (ctx.node().attempt), not on ``self``.

Verifies the compat-property + walk-up mechanism directly (no LLM-mocked full ``ainfer``):

  * ONE shared MFI run across TWO distinct RunContexts keeps each run's
    ``latest_per_flow`` / ``judgments`` isolated (write under ctx A, assert ctx B unaffected);
  * the flow worker closures resolve the MFI's attempt node by walking UP their ancestor
    paths — so a write made while a flow CHILD ctx is active (a descendant of the MFI node,
    which is what BTA re-enters via ``flow.ainfer(run_context=flow_child)`` even in a worker
    thread) lands on the MFI's own attempt node, visible cross-flow within that run only;
  * a real (non-legacy) ctx NEVER writes the instance backing → a shared MFI is
    concurrency-isolated across N RunContexts;
  * the legacy / no-ctx path still uses the instance backing (byte-identical).
"""

from attr import attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (  # noqa: E501
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.run_context import (
    MultiFlowAttemptState,
    RunContext,
    active_run_context,
    enter_run,
    exit_run,
    mint_root,
)


@attrs
class _Leaf(InferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        return ""


def _make_mfi():
    return MultiFlowInferencer(
        flow_configs=[
            {
                "input": "t0",
                "initial_inferencer": _Leaf(),
                "followup_inferencer": _Leaf(),
                "end_condition": lambda s, r: True,
                "max_dynamic_steps": 1,
            },
            {
                "input": "t1",
                "initial_inferencer": _Leaf(),
                "followup_inferencer": _Leaf(),
                "end_condition": lambda s, r: True,
                "max_dynamic_steps": 1,
            },
        ],
        disable_aggregator=True,
    )


def test_crossflow_isolated_across_distinct_run_contexts():
    """ONE shared MFI; two distinct ctxs each keep their own per-run cross-flow buffers.

    Write through the SAME instance under ctx A, then assert ctx B is completely
    unaffected — the buffers live in each ctx's ``MultiFlowAttemptState``, not on ``self``.
    """
    mfi = _make_mfi()
    # Two distinct, independent roots (model two concurrent BTA subtasks sharing one MFI).
    ctx_a = RunContext.root(workspace=None)  # non-legacy
    ctx_b = RunContext.root(workspace=None)  # non-legacy

    # Seed a fresh attempt state on each MFI node (what _reset_cross_flow_state does
    # at the top of each _ainfer/_infer attempt).
    tok = enter_run(ctx_a)
    mfi._reset_cross_flow_state()
    exit_run(tok)
    tok = enter_run(ctx_b)
    mfi._reset_cross_flow_state()
    exit_run(tok)

    # Write through the shared instance under ctx A (in-place mutation, exactly as the
    # flow closures do: ``outer._latest_per_flow[idx] = ...`` / ``_all_judgments.append``).
    tok = enter_run(ctx_a)
    mfi._latest_per_flow[0] = "A-flow0"
    mfi._latest_per_flow[1] = "A-flow1"
    mfi._all_judgments.append((0, 0, "judge-A"))
    exit_run(tok)

    # Under ctx A: the writes are visible.
    tok = enter_run(ctx_a)
    assert mfi._latest_per_flow == {0: "A-flow0", 1: "A-flow1"}
    assert mfi._all_judgments == [(0, 0, "judge-A")]
    exit_run(tok)

    # Under ctx B: completely unaffected (fresh attempt state — no cross-talk).
    tok = enter_run(ctx_b)
    assert mfi._latest_per_flow == {0: None, 1: None}
    assert mfi._all_judgments == []
    exit_run(tok)

    # The two ctxs hold DISTINCT MultiFlowAttemptState objects on distinct nodes.
    attempt_a = ctx_a.node().attempt
    attempt_b = ctx_b.node().attempt
    assert isinstance(attempt_a, MultiFlowAttemptState)
    assert isinstance(attempt_b, MultiFlowAttemptState)
    assert attempt_a is not attempt_b
    assert attempt_a.latest_per_flow == {0: "A-flow0", 1: "A-flow1"}
    assert attempt_a.judgments == [(0, 0, "judge-A")]
    assert attempt_b.latest_per_flow == {0: None, 1: None}
    assert attempt_b.judgments == []

    # A real (non-legacy) ctx must NEVER write the instance backing → no shared-instance
    # bleed across concurrent runs.
    assert mfi._latest_per_flow_backing == {0: None, 1: None}
    assert mfi._all_judgments_backing == []


def test_crossflow_walkup_resolves_mfi_node_from_flow_child_ctx():
    """A write made while a FLOW CHILD ctx is active resolves the MFI's attempt node by
    walking UP the ancestor paths — the exact mechanism the flow worker closures rely on.

    BTA runs the per-flow workers in worker threads that start with the DEFAULT context,
    but re-enters the bridge via ``flow.ainfer(run_context=flow_child)``; so inside a
    closure ``active_run_context()`` is the flow's child ctx (a DESCENDANT of the MFI node).
    The buffers must still reach the MFI's own ``MultiFlowAttemptState`` for cross-flow
    visibility, and stay isolated per run.
    """
    mfi = _make_mfi()
    ctx_a = RunContext.root(workspace=None)
    ctx_b = RunContext.root(workspace=None)

    # MFI runs at the root of each ctx; the workers run under flow child ctxs.
    tok = enter_run(ctx_a)
    mfi._reset_cross_flow_state()
    exit_run(tok)
    tok = enter_run(ctx_b)
    mfi._reset_cross_flow_state()
    exit_run(tok)

    # Run "inside flow 0 of ctx A": the active ctx is the flow's child (deep descendant),
    # not the MFI node. The closure-style in-place mutations must walk UP to the MFI node.
    flow0_a = ctx_a.child("flow_0").child("children").child("initial")
    tok = enter_run(flow0_a)
    assert active_run_context().path == "/flow_0/children/initial"
    mfi._latest_per_flow[0] = "A0"
    mfi._all_judgments.append((0, 0, "jA0"))
    exit_run(tok)

    # Run "inside flow 1 of ctx A": a DIFFERENT flow child ctx; it must see flow 0's write
    # (cross-flow peer visibility) AND add its own — all on the SAME MFI attempt node.
    flow1_a = ctx_a.child("flow_1")
    tok = enter_run(flow1_a)
    assert mfi._latest_per_flow.get(0) == "A0"  # peer visibility within run A
    mfi._latest_per_flow[1] = "A1"
    mfi._all_judgments.append((1, 0, "jA1"))
    exit_run(tok)

    # The MFI node of ctx A now holds both flows' data; ctx B is untouched.
    attempt_a = ctx_a.node().attempt
    assert attempt_a.latest_per_flow == {0: "A0", 1: "A1"}
    assert attempt_a.judgments == [(0, 0, "jA0"), (1, 0, "jA1")]

    # Even read from a deep descendant of ctx B, the walk-up finds B's (empty) MFI node,
    # never A's — full isolation across the two runs.
    flow0_b = ctx_b.child("flow_0").child("children").child("initial")
    tok = enter_run(flow0_b)
    assert mfi._latest_per_flow == {0: None, 1: None}
    assert mfi._all_judgments == []
    exit_run(tok)

    # No real-ctx write ever touched the instance backing.
    assert mfi._latest_per_flow_backing == {0: None, 1: None}
    assert mfi._all_judgments_backing == []


def test_crossflow_legacy_no_ctx_uses_instance_backing():
    """With no active ctx, the cross-flow buffers fall back to the instance backing
    (byte-identical to the pre-virtualization behavior)."""
    mfi = _make_mfi()
    assert active_run_context() is None

    # Reset with no ctx -> resets the instance backings in place.
    mfi._reset_cross_flow_state()
    assert mfi._latest_per_flow == {0: None, 1: None}
    assert mfi._all_judgments == []
    # The property resolves to the backing object itself (no ctx).
    assert mfi._latest_per_flow is mfi._latest_per_flow_backing
    assert mfi._all_judgments is mfi._all_judgments_backing

    # In-place mutation lands on the backing and is readable post-write (still no ctx).
    mfi._latest_per_flow[0] = "legacy0"
    mfi._all_judgments.append((0, 1, "legacy-judge"))
    assert mfi._latest_per_flow_backing == {0: "legacy0", 1: None}
    assert mfi._all_judgments_backing == [(0, 1, "legacy-judge")]


def test_crossflow_legacy_mint_mirrors_backing_for_post_call_read():
    """A bare ``infer()`` legacy-mints a root whose store is discarded on exit; the reset
    makes the attempt state share the backing objects so the closures' in-call writes are
    still readable POST-CALL (ctx is None -> backing). Mirrors the dispatch-state rule."""
    mfi = _make_mfi()
    legacy = mint_root()  # legacy_mint=True
    assert legacy.legacy_mint is True

    tok = enter_run(legacy)
    mfi._reset_cross_flow_state()
    # Under the legacy ctx, writes go through the attempt state, which IS the backing.
    mfi._latest_per_flow[0] = "lm0"
    mfi._all_judgments.append((0, 0, "lm-judge"))
    # The attempt state and the instance backing are the SAME objects under legacy_mint.
    assert legacy.node().attempt.latest_per_flow is mfi._latest_per_flow_backing
    assert legacy.node().attempt.judgments is mfi._all_judgments_backing
    exit_run(tok)

    # Post-call: ctx is None, so reads hit the mirrored backing — data survives.
    assert active_run_context() is None
    assert mfi._latest_per_flow == {0: "lm0", 1: None}
    assert mfi._all_judgments == [(0, 0, "lm-judge")]
