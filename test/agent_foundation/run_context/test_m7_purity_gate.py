"""§2.6 — the M7 per-class purity GATE, applied to real converted classes.

The plan idealized an *empty* allow-list ("no per-run self-mutation at all"), but two
deliberate, documented carve-outs keep a small, fixed set of fields on ``self``:

  * **D6 — dispatch-state stays on the instance** (``_last_winner_idx`` /
    ``_last_reviewer_alias`` / ``_last_fixer_alias`` / ``_last_ranking`` on MultiFlow;
    the reviewer/fixer *reference* picks on MFDual). They deliberately OUTLIVE the
    call for POST-call readers (``get_winner_flow_idx()`` + ~10 tests). Routing them
    through a finally-cleared context would return ``None`` and break dispatch.
  * **§2.12 option-(a) — workspace + deliverable flags stay instance-backed** under a
    context (``switch_role``). They are read POST-dispatch on the instance (e.g.
    MFDual's alias-dispatched fixer asserted via ``fixer.output_is_deliverable``);
    the ``_workspace`` getter is intentionally instance-pure. Proven by
    ``test_alias_dispatched_fixer_inherits_output_is_deliverable``.

So the *correct* gate is **purity modulo the documented carve-outs**: exercise each
per-run mutation site under an active context and assert the instance-``__dict__``
delta is a SUBSET of that site's documented allow-list. This catches any NEW orphan
per-run field (the failure mode the plan cares about) while encoding the deliberate
carve-outs explicitly — converting the recurring "is M7 complete?" audit into an
objective, self-documenting test.
"""

from attr import attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.run_context import (
    RunContext,
    enter_run,
    exit_run,
)
from agent_foundation.common.inferencers.run_context.purity import purity_snapshot


@attrs(slots=False)
class _Leaf(InferencerBase):
    def _infer(self, x, inference_config=None, **kw):
        return x


# --- §2.12 option-(a): switch_role's deliberate instance-backed carve-outs --------
# Each entry is read POST-dispatch on the instance (option-(a)) or is workspace-derived
# state reconfigured by the _workspace setter cascade (§2.12). NOTHING ELSE may mutate.
_ROLE_CARVEOUTS = frozenset({
    "output_is_deliverable",          # §2.12 option-(a): post-call reader
    "is_deliverable_boundary",        # §2.12 option-(a): post-call reader
    "_role_history",                  # audit trail (deliberate, never read for dispatch)
    "_InferencerBase__workspace",     # §2.12 option-(a): workspace backing (getter is ctx-aware)
    "logger",                         # workspace-DERIVED (reconfigured by the setter cascade)
    "_logger_awaiting_workspace",     # workspace-DERIVED
    "_resolved_logger_configs",       # workspace-DERIVED
    "_ws_log_relpaths",               # workspace-DERIVED: static {logger: relpath} tag (always "logs/session.jsonl")
    "_pending_role_changes",          # transient template-layer stash, cleared in-method
    "_session_id",                    # reset_session (Tier-3-aware; legacy backing)
})

# --- D6: MultiFlow dispatch-state deliberately stays on the instance --------------
_DISPATCH_CARVEOUTS = frozenset({
    "_last_winner_idx",
    "_last_reviewer_alias",
    "_last_fixer_alias",
    "_last_ranking",
})

# --- §2.5: MFDual reviewer/fixer *reference* picks are deliberate dispatch-state ---
# ("reviewer/fixer reference picks already safe" — read post-call; the full
# shared-instance model was dropped as over-engineering, D4). The reviewers panel is
# populated the same way (reviewer_match_all_non_winners).
_MFDUAL_DISPATCH_CARVEOUTS = frozenset({
    "review_inferencer",
    "fixer_inferencer",
    "reviewers",
    "_current_round_ws",
})


def _delta_keys(holder):
    d = holder[0]
    return set(d.added) | set(d.changed)


def test_switch_role_self_mutation_is_confined_to_documented_carveouts():
    """Under a context, switch_role mutates ONLY the §2.12 option-(a) carve-outs —
    no orphan per-run field leaks onto self."""
    leaf = _Leaf()
    root = RunContext.root(workspace=InferencerWorkspace(root="/tmp/pg"))
    tok = enter_run(root)
    try:
        with purity_snapshot(leaf) as h:
            leaf.switch_role(
                "reviewer",
                workspace=InferencerWorkspace(root="/tmp/pg/review"),
                output_is_deliverable=True,
                is_deliverable_boundary=True,
            )
    finally:
        exit_run(tok)
    leaked = _delta_keys(h) - _ROLE_CARVEOUTS
    assert not leaked, f"switch_role leaked non-carveout per-run fields: {sorted(leaked)}"


def test_switch_role_without_workspace_touches_only_flags_and_audit():
    """The minimal carve-out: with no workspace, only the flags + audit trail change."""
    leaf = _Leaf()
    root = RunContext.root(workspace=None)
    tok = enter_run(root)
    try:
        with purity_snapshot(leaf) as h:
            leaf.switch_role("reviewer", output_is_deliverable=True)
    finally:
        exit_run(tok)
    assert _delta_keys(h) == {"output_is_deliverable", "_role_history"}


def test_multiflow_dispatch_state_is_confined_to_d6_carveouts():
    """MultiFlow's per-call dispatch state stays on self (D6) and nothing else leaks."""
    mfi = MultiFlowInferencer(
        flow_configs=[
            {"initial_inferencer": _Leaf(), "input": "a"},
            {"initial_inferencer": _Leaf(), "input": "b"},
        ],
        disable_aggregator=True,
    )
    root = RunContext.root(workspace=None)
    tok = enter_run(root)
    try:
        with purity_snapshot(mfi) as h:
            mfi._last_winner_idx = 1
            mfi._last_reviewer_alias = "claude"
            mfi._last_ranking = [1, 0]
    finally:
        exit_run(tok)
    leaked = _delta_keys(h) - _DISPATCH_CARVEOUTS
    assert not leaked, f"MultiFlow leaked non-dispatch per-run fields: {sorted(leaked)}"


def test_mfdual_reviewer_match_all_non_winners_dispatch_is_confined():
    """MFDual's reviewer-panel dispatch (reviewer_match_all_non_winners) mutates only
    the §2.5 reference-pick carve-outs — review_inferencer/fixer_inferencer/reviewers —
    and nothing else leaks onto the shared instance."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
        MultiFlowDualInferencer,
    )

    w, n1, n2 = _Leaf(), _Leaf(), _Leaf()
    d = MultiFlowDualInferencer(
        flow_configs=[
            {"initial_inferencer": w, "input": "a"},
            {"initial_inferencer": n1, "input": "b"},
            {"initial_inferencer": n2, "input": "c"},
        ],
        multi_flow_disable_aggregator=True,
        reviewer_strategy="all_non_winners",
    )
    d.base_inferencer._last_winner_idx = 0  # w wins -> n1, n2 are the panel
    root = RunContext.root(workspace=None)
    tok = enter_run(root)
    # C5: roles are now resolved PER-RUN into ctx scratch (not the shared instance), so the
    # §2.5 carve-out is gone — the gate now asserts FULL purity: NOTHING leaks onto self.
    before = {k: id(v) for k, v in vars(d).items()}
    try:
        d._select_reviewer_and_fixer()
        # the panel resolved per-run, readable via _role_get while the ctx is active
        assert d._role_get("review_inferencer") is n1
        assert d._role_get("reviewers") == [n2]
        after = {k: id(v) for k, v in vars(d).items()}
        mutated = {k for k in after if before.get(k) != after[k]}
    finally:
        exit_run(tok)
    # FULL purity: the reference picks did NOT mutate the shared instance (they live in
    # ctx scratch) — so a concurrent run on the same instance cannot clobber them.
    assert not mutated, f"MFDual dispatch leaked onto the shared instance: {sorted(mutated)}"
    # and the instance attributes are untouched (still the construction definitions)
    assert d.review_inferencer is None       # was never configured; resolution didn't touch self
    assert d.reviewers is None               # ditto


def test_mfdual_concurrent_dispatch_role_isolation():
    """C5 shared-instance guarantee: ONE MFDual instance resolves DIFFERENT reviewer panels
    under two distinct RunContexts with no cross-contamination — the roles live on each
    ctx's own node (scratch), so a second run cannot clobber the first."""
    from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
        MultiFlowDualInferencer,
    )

    w, n1, n2 = _Leaf(), _Leaf(), _Leaf()
    d = MultiFlowDualInferencer(
        flow_configs=[
            {"initial_inferencer": w, "input": "a"},
            {"initial_inferencer": n1, "input": "b"},
            {"initial_inferencer": n2, "input": "c"},
        ],
        multi_flow_disable_aggregator=True,
        reviewer_strategy="all_non_winners",
    )
    rootA = RunContext.root(workspace=None)
    rootB = RunContext.root(workspace=None)

    # Run under ctx A with winner=0 -> panel head n1, rest [n2]
    d.base_inferencer._last_winner_idx = 0
    tokA = enter_run(rootA)
    try:
        d._select_reviewer_and_fixer()
        a_review, a_panel = d._role_get("review_inferencer"), d._role_get("reviewers")
    finally:
        exit_run(tokA)

    # Run under ctx B with winner=1 -> panel head w, rest [n2]
    d.base_inferencer._last_winner_idx = 1
    tokB = enter_run(rootB)
    try:
        d._select_reviewer_and_fixer()
        b_review, b_panel = d._role_get("review_inferencer"), d._role_get("reviewers")
    finally:
        exit_run(tokB)

    assert a_review is n1 and a_panel == [n2]
    assert b_review is w and b_panel == [n2]
    # Re-entering ctx A still sees A's resolution — B's run on rootB never touched it.
    tokA2 = enter_run(rootA)
    try:
        assert d._role_get("review_inferencer") is n1
        assert d._role_get("reviewers") == [n2]
    finally:
        exit_run(tokA2)
    # The shared instance itself was never mutated by either run.
    assert d.review_inferencer is None and d.reviewers is None


def test_gate_actually_catches_an_unexpected_orphan_field():
    """Negative control: the gate FAILS if a non-carve-out per-run field mutates —
    proving it would catch a real M7 regression (a new orphan field)."""
    leaf = _Leaf()
    root = RunContext.root(workspace=None)
    tok = enter_run(root)
    try:
        with purity_snapshot(leaf) as h:
            leaf.__dict__["_some_new_orphan_counter"] = 7  # simulate a missed field
    finally:
        exit_run(tok)
    leaked = _delta_keys(h) - _ROLE_CARVEOUTS
    assert leaked == {"_some_new_orphan_counter"}  # gate sees it -> would fail a real run
