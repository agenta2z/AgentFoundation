"""C10 / Part G — runner-state virtualization concurrency acceptance.

Proves that a **single shared** ``LinearWorkflowInferencer`` instance, run under
two **distinct, concurrent** ``RunContext`` branches, keeps its
``LinearWorkflow``/``Workflow`` runner state machine isolated per branch: the
working ``state`` dict, the loop counters (``loop_counts``), and the execution
sequence (``exec_seq``) of one branch never bleed into the other.

This is the concurrency proof the dispatch/role purity gate (Part D) defers to
the runner-state track (R6): before Part G these fields lived on ``self`` (a
plain attrib), so two concurrent runs on one instance clobbered each other.
After virtualization each per-run runner attribute is a compat-property routed
to the per-run context node (the typed ``LinearWorkflowState`` carrier for the
serialized fields, ``ctx.node().scratch`` for the transient ones), so concurrent
runs stay isolated.

Covers both carrier shapes:

* **plain LWI** — ``ctx.node().call`` is the working dict (``_state``/
  ``_pending_state``) and the serialized runner fields ride a per-path
  ``LinearWorkflowState`` lazily held in ``ctx.node().scratch``;
* **typed ``.call``** (the MFDual shape) — ``ctx.node().call`` is an
  ``InferencerStateBase`` whose ``runner`` carrier (a ``LinearWorkflowState``)
  holds the working state, mirroring ``_pending_state`` exactly.
"""

import asyncio

import attrs

from agent_foundation.common.inferencers.run_context import (
    InferencerStateBase,
    RunContext,
    active_run_context,
    enter_run,
    exit_run,
)
from agent_foundation.common.inferencers.run_context.state import (
    LinearWorkflowState,
    register_state,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (  # noqa: E501
    LinearWorkflowInferencer,
    WorkflowStepConfig,
)


# A minimal typed ``.call`` state with a ``runner`` carrier — the MFDual shape,
# so we exercise the ``call.runner.state`` resolution path of the compat-props.
@register_state
@attrs.define
class _TypedCallWithRunner(InferencerStateBase):
    tag: str | None = None
    runner: LinearWorkflowState = attrs.field(factory=LinearWorkflowState)


def _make_shared_lwi(barrier: asyncio.Barrier):
    """Build ONE shared LWI whose single looping step.

    * seeds per-run state from the (per-branch) ``original_input``,
    * appends a per-iteration marker into ``state["trace"]`` (so a bleed across
      branches would show up as a polluted trace / loop count),
    * rendezvous on ``barrier`` on the FIRST iteration so both concurrent runs
      are provably mid-flight (interleaved) at the same time,
    * loops ``loop_back_to`` itself until it has run ``state["target_iters"]``
      times — a per-branch count, so ``loop_counts``/``exec_seq`` differ per run.
    """

    async def _looping_step(step_input, state):
        # First entry: rendezvous so both branches are simultaneously in-flight.
        if not state.get("_barrier_done"):
            state["_barrier_done"] = True
            await barrier.wait()
        state.setdefault("trace", []).append(state["original_input"])
        state["iters_done"] = state.get("iters_done", 0) + 1
        return state["iters_done"]

    def _loop_cond(state, result):
        # Keep looping until we've done the per-branch target number of iters.
        return state["iters_done"] < state["target_iters"]

    return LinearWorkflowInferencer(
        step_configs=[
            WorkflowStepConfig(
                name="loop_step",
                step_fn=_looping_step,
                loop_back_to="loop_step",
                loop_condition=_loop_cond,
                max_loop_iterations=20,
                enable_result_save=False,
            ),
        ],
        initial_state_factory=lambda inp: {
            "original_input": inp,
            "target_iters": inp,  # input == how many iterations this branch runs
        },
        response_builder=lambda s: s,
    )


def _runner_carrier_for(ctx: RunContext) -> LinearWorkflowState | None:
    """The per-path serialized runner carrier the compat-properties resolved to.

    Mirrors ``LinearWorkflowInferencer._runner_carrier`` resolution: a typed
    ``.call`` exposes its ``runner``; a plain dict/None ``.call`` uses the
    lazily-created carrier in ``ctx.node().scratch``.
    """
    call = ctx.node().call
    runner = getattr(call, "runner", None)
    if isinstance(call, InferencerStateBase) and runner is not None:
        return runner
    return ctx.node().scratch.get(
        LinearWorkflowInferencer._RUNNER_SCRATCH_KEY
    )


def test_shared_lwi_two_concurrent_ctxs_isolate_runner_state():
    """ONE shared LWI under two concurrent ctxs -> isolated state/loop_counts/exec_seq."""

    async def _run():
        barrier = asyncio.Barrier(2)
        lwi = _make_shared_lwi(barrier)  # the ONE shared instance

        root = RunContext.root(workspace=None)
        ctx_a = root.child("branch_a")
        ctx_b = root.child("branch_b")

        # branch_a loops 3 times; branch_b loops 6 times -> distinct counters.
        res_a, res_b = await asyncio.gather(
            lwi.ainfer(3, run_context=ctx_a),
            lwi.ainfer(6, run_context=ctx_b),
        )

        # --- Results: each branch saw ONLY its own input, the right # of times. ---
        assert res_a["iters_done"] == 3
        assert res_b["iters_done"] == 6
        assert res_a["trace"] == [3, 3, 3]
        assert res_b["trace"] == [6, 6, 6, 6, 6, 6]
        assert res_a["original_input"] == 3
        assert res_b["original_input"] == 6

        # --- Working state dict: distinct objects, no cross-branch keys. ---
        call_a = ctx_a.node().call
        call_b = ctx_b.node().call
        assert call_a is not call_b
        assert call_a["original_input"] == 3
        assert call_b["original_input"] == 6

        # --- Serialized runner carrier: independent loop_counts + exec_seq. ---
        carrier_a = _runner_carrier_for(ctx_a)
        carrier_b = _runner_carrier_for(ctx_b)
        assert carrier_a is not None and carrier_b is not None
        assert carrier_a is not carrier_b
        # branch_a looped back 2x (3 entries), branch_b 5x (6 entries).
        assert sum(carrier_a.loop_counts.values()) == 2
        assert sum(carrier_b.loop_counts.values()) == 5
        # exec_seq advances per saved result; with saves off both stay 0 — the
        # point is they are SEPARATE objects, proven by the loop_counts split.
        assert carrier_a.exec_seq == carrier_b.exec_seq == 0

    asyncio.run(_run())


def test_shared_lwi_three_concurrent_ctxs_no_bleed():
    """Scale to N=3 concurrent branches on one instance — each fully isolated."""

    async def _run():
        barrier = asyncio.Barrier(3)
        lwi = _make_shared_lwi(barrier)

        root = RunContext.root(workspace=None)
        ctxs = [root.child(f"parallel_{i}") for i in range(3)]
        inputs = [2, 4, 7]

        results = await asyncio.gather(
            *(lwi.ainfer(n, run_context=c) for n, c in zip(inputs, ctxs))
        )

        for n, c, res in zip(inputs, ctxs, results):
            assert res["iters_done"] == n, (n, res["iters_done"])
            assert res["trace"] == [n] * n
            assert c.node().call["original_input"] == n
            carrier = _runner_carrier_for(c)
            assert carrier is not None
            assert sum(carrier.loop_counts.values()) == n - 1

        # All carriers are distinct objects (the crux: no shared instance attr).
        carriers = [_runner_carrier_for(c) for c in ctxs]
        assert len({id(x) for x in carriers}) == 3

    asyncio.run(_run())


def test_shared_lwi_typed_call_runner_carrier_path():
    """Typed ``.call`` (MFDual shape): runner state rides ``call.runner`` per branch."""

    async def _run():
        barrier = asyncio.Barrier(2)
        lwi = _make_shared_lwi(barrier)
        # state_factory makes ``.call`` a TYPED state with a ``runner`` carrier,
        # exercising the ``call.runner.state`` resolution (not the dict path).
        lwi.state_factory = lambda inp: _TypedCallWithRunner(tag=f"t{inp}")

        root = RunContext.root(workspace=None)
        ctx_a = root.child("typed_a")
        ctx_b = root.child("typed_b")

        res_a, res_b = await asyncio.gather(
            lwi.ainfer(2, run_context=ctx_a),
            lwi.ainfer(5, run_context=ctx_b),
        )

        assert res_a["iters_done"] == 2
        assert res_b["iters_done"] == 5

        call_a = ctx_a.node().call
        call_b = ctx_b.node().call
        # Typed states preserved per branch; the runner carrier holds the state.
        assert isinstance(call_a, _TypedCallWithRunner) and call_a.tag == "t2"
        assert isinstance(call_b, _TypedCallWithRunner) and call_b.tag == "t5"
        assert call_a.runner is not call_b.runner
        assert call_a.runner.state["original_input"] == 2
        assert call_b.runner.state["original_input"] == 5
        assert sum(call_a.runner.loop_counts.values()) == 1  # 2 iters -> 1 loop-back
        assert sum(call_b.runner.loop_counts.values()) == 4  # 5 iters -> 4 loop-backs

    asyncio.run(_run())


def test_bare_call_runner_state_byte_identical_to_pending_state():
    """A bare ``ainfer`` (no caller ctx) legacy-mints a root whose Tier-1 store is
    discarded on exit — so post-call ``_state`` is ``None``, exactly like the
    pre-existing ``_pending_state`` (byte-identical legacy behaviour).  During the
    run (legacy ctx active) the runner fields are observable and correct.
    """
    captured = {}

    async def _run():
        barrier = asyncio.Barrier(1)  # solo party -> no block for one run
        lwi = _make_shared_lwi(barrier)

        # Wrap the step so it snapshots the runner fields mid-run (ctx active).
        orig = lwi.step_configs[0].step_fn

        async def _spy(step_input, state):
            out = await orig(step_input, state)
            captured["state_is_pending"] = lwi._state is lwi._pending_state
            captured["loop_total"] = sum(lwi._loop_counts.values())
            captured["original_input"] = lwi._state["original_input"]
            return out

        lwi.step_configs[0].step_fn = _spy

        res = await lwi.ainfer(4)
        assert res["iters_done"] == 4

        # During the run the runner state was live and coherent.
        assert captured["state_is_pending"] is True
        assert captured["original_input"] == 4
        assert captured["loop_total"] == 3  # 4 iters -> 3 loop-backs

        # Post-call: no active ctx; legacy root discarded -> None, == _pending_state.
        assert active_run_context() is None
        assert lwi._state is None
        assert lwi._pending_state is None
        assert lwi._state == lwi._pending_state

    asyncio.run(_run())


def test_explicit_root_runner_state_survives_post_call():
    """An explicit (non-legacy) app-host root keeps its Tier-1 store, so the
    runner state is readable from the node after a bare-instance call."""

    async def _run():
        barrier = asyncio.Barrier(1)
        lwi = _make_shared_lwi(barrier)

        root = RunContext.root(workspace=None)  # legacy_mint=False -> store kept
        res = await lwi.ainfer(4, run_context=root)
        assert res["iters_done"] == 4

        # The working dict survives on the (non-discarded) node.
        node_call = root.node().call
        assert node_call["original_input"] == 4
        assert node_call["iters_done"] == 4
        carrier = _runner_carrier_for(root)
        assert carrier is not None
        assert sum(carrier.loop_counts.values()) == 3

    asyncio.run(_run())


def test_state_and_pending_state_stay_coherent_under_ctx():
    """``_state`` and ``_pending_state`` resolve to the SAME object (M1 coherence).

    The Dual review step reads ``self._state`` then falls back to
    ``self._pending_state``; they must stay consistent under every home.
    """
    lwi = LinearWorkflowInferencer(
        step_configs=[WorkflowStepConfig(name="s", step_fn=lambda x, s: x)],
    )

    # (a) No ctx -> both use the instance backing, same object.
    lwi._pending_state = {"k": 1}
    assert lwi._state is lwi._pending_state
    assert lwi._state == {"k": 1}

    # (b) Plain dict .call under a ctx -> both are ctx.node().call.
    root = RunContext.root(workspace=None)
    tok = enter_run(root)
    try:
        lwi._pending_state = {"k": 2}
        assert lwi._state is lwi._pending_state
        assert root.node().call == {"k": 2}
        # Writing via _state is visible via _pending_state and the node.
        lwi._state = {"k": 3}
        assert lwi._pending_state == {"k": 3}
        assert root.node().call == {"k": 3}
    finally:
        exit_run(tok)

    # (c) Typed .call with a runner carrier -> both are call.runner.state.
    root2 = RunContext.root(workspace=None)
    root2.node().call = _TypedCallWithRunner(tag="x")
    tok2 = enter_run(root2)
    try:
        lwi._pending_state = {"k": 4}
        assert lwi._state is lwi._pending_state
        assert root2.node().call.runner.state == {"k": 4}
    finally:
        exit_run(tok2)
