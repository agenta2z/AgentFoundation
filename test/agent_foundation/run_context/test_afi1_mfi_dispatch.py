"""AF1 / C2 — MFI dispatch state is per-run (ctx.node().call), not on ``self``.

Verifies the compat-property mechanism directly (no LLM-mocked full ``ainfer`` needed):
  * concurrent sibling ctxs each keep their own winner/alias (no cross-talk);
  * a real (non-legacy) ctx never pollutes the instance backing (concurrency isolation);
  * a legacy-minted bare call mirrors to the backing so the post-call getter still works;
  * a *reused* node is reset per call (state_factory is populate-once, so the explicit
    ``_reset_dispatch_state_for_call`` in ``_init_call_state`` is what clears a stale winner).
"""

from attr import attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (  # noqa: E501
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.run_context import (
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


def test_dispatch_isolated_across_sibling_ctxs():
    """ONE shared MFI; two sibling ctxs each keep their own dispatch — no cross-talk."""
    mfi = _make_mfi()
    root = RunContext.root()  # non-legacy
    c0, c1 = root.child("parallel_0"), root.child("parallel_1")

    tok = enter_run(c0)
    mfi._last_winner_idx = 0
    mfi._last_reviewer_alias = "a0"
    exit_run(tok)

    tok = enter_run(c1)
    mfi._last_winner_idx = 1
    mfi._last_reviewer_alias = "a1"
    exit_run(tok)

    tok = enter_run(c0)
    assert mfi.get_winner_flow_idx() == 0
    assert mfi.get_chosen_reviewer_alias() == "a0"
    exit_run(tok)

    tok = enter_run(c1)
    assert mfi.get_winner_flow_idx() == 1
    assert mfi.get_chosen_reviewer_alias() == "a1"
    exit_run(tok)

    # A real (non-legacy) ctx must NEVER write the instance backing → no shared-instance bleed.
    assert mfi._last_winner_idx_backing is None
    assert mfi._last_reviewer_alias_backing is None


def test_legacy_mint_mirrors_backing_for_post_call_getter():
    """A bare ``infer()`` legacy-mints a root whose store is discarded; the writer mirrors
    to the backing so the post-call getter (ctx is None) still resolves the winner."""
    mfi = _make_mfi()
    legacy = mint_root()  # legacy_mint=True
    assert legacy.legacy_mint is True

    tok = enter_run(legacy)
    mfi._last_winner_idx = 7
    mfi._last_ranking = [7, 2]
    exit_run(tok)

    assert active_run_context() is None  # post-call
    assert mfi.get_winner_flow_idx() == 7  # read from the mirrored backing
    assert mfi.get_runner_up_flow_idx() == 2
    assert mfi._last_winner_idx_backing == 7


def test_reused_node_is_reset_per_call():
    """state_factory is populate-once, so a reused node would keep a stale winner without
    the explicit per-call reset (now in ``_init_call_state``, after ctx activation)."""
    mfi = _make_mfi()
    root = RunContext.root()  # same ctx/node reused across two "calls"

    tok = enter_run(root)
    mfi._init_call_state("q1")
    mfi._last_winner_idx = 3
    assert mfi.get_winner_flow_idx() == 3
    exit_run(tok)

    tok = enter_run(root)
    mfi._init_call_state("q2")  # populate-once won't re-seed; the reset must clear winner
    assert mfi.get_winner_flow_idx() is None
    exit_run(tok)


def test_ranking_unset_is_none_byte_identical():
    """Under a ctx with nothing dispatched, ``get_ranking()`` is None (not [])."""
    mfi = _make_mfi()
    root = RunContext.root()
    tok = enter_run(root)
    mfi._init_call_state("q")
    assert mfi.get_ranking() is None
    assert mfi.get_winner_flow_idx() is None
    exit_run(tok)
