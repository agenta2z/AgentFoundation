"""AF1 / Decision 5 (Commit 1) — MFI prompt-feed writes are per-run, not instance state.

MultiFlowInferencer passes per-flow / per-call data (``upstream_artifacts``) into a
SHARED child leaf's wrapper template. Pre-fix it did this by MUTATING the child
instance's ``template_extra_feed`` at runtime, so under shared-instance reuse (one MFI
across N concurrent ctxs, or one followup/aggregator child reused across N flows) the
writes clobbered each other — the exact state-separation hazard.

The fix routes those per-call feed values through a CALL-SCOPED ctx channel (published
into a RunContext handle, merged by the leaf at render time via a walk-up of the active
ctx tree) instead of the shared child instance. These tests prove:

  * under a real ctx, invoking MFI's aggregator/followup builders does NOT leave
    ``upstream_artifacts`` on the shared child instance's ``template_extra_feed``;
  * two concurrent ctxs each see only their OWN published feed (no cross-bleed);
  * the rendered prompt STILL contains the upstream content (behavior preserved) —
    a real ``TemplatedInferencerBase`` leaf merges the ctx-published feed;
  * the legacy / no-ctx path still writes the instance dict (byte-identical).
"""

from attr import attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE,
    TemplatedInferencerBase,
    _resolve_ctx_feed_override,
)
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


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@attrs(slots=False)
class _FeedLeaf(TemplatedInferencerBase):
    """A templated leaf whose ``template_manager`` renders ``{{ upstream_artifacts }}``
    and ``{{ input }}`` from the feed — so we can assert the rendered prompt reflects
    the ctx-published per-call feed (the leaf merges it in ``_build_template_feed``)."""

    def __attrs_post_init__(self):
        # A stub template_manager: called as ``tm(key, active_template_root_space=...,
        # master_version=..., **feed)`` by ``_render_prompt``; we render a tiny string
        # so the test can assert the upstream content + the separate input slot.
        def _tm(key, active_template_root_space=None, master_version=None, **feed):
            return (
                f"INPUT={feed.get('input', '')}|"
                f"UPSTREAM={feed.get('upstream_artifacts', '<none>')}"
            )

        self.template_manager = _tm
        # Satisfy ``_render_prompt``'s "must name a template" guard.
        self.template_root_space = "plan"

    def _infer(self, inference_input, inference_config=None, **kwargs):  # pragma: no cover
        return inference_input


@attrs
class _Leaf(InferencerBase):
    def _infer(self, inference_input, inference_config=None, **kwargs):
        return ""


def _make_mfi(*, aggregator):
    """A 2-flow MFI with upstream injection enabled and an aggregator that has its own
    ``aggregator_prompt`` (so MFI installs its default aggregator prompt builder)."""
    return MultiFlowInferencer(
        flow_configs=[
            {
                "input": "task_0",
                "initial_inferencer": _Leaf(),
                "followup_inferencer": _Leaf(),
                "end_condition": lambda s, r: True,
                "max_dynamic_steps": 2,
            },
            {
                "input": "task_1",
                "initial_inferencer": _Leaf(),
                "followup_inferencer": _Leaf(),
                "end_condition": lambda s, r: True,
                "max_dynamic_steps": 2,
            },
        ],
        visible_flows="all",
        inject_upstream_artifacts=True,
        aggregator_inferencer=aggregator,
        aggregator_prompt="AGG over {{ worker_plans }}",
    )


# ---------------------------------------------------------------------------
# Aggregator path — feed goes to the ctx, not the shared aggregator instance
# ---------------------------------------------------------------------------


def test_aggregator_feed_published_to_ctx_not_instance():
    agg = _FeedLeaf()
    mfi = _make_mfi(aggregator=agg)
    builder = mfi._make_default_aggregator_prompt_builder()

    ctx = RunContext.root(workspace=None)  # non-legacy
    tok = enter_run(ctx)
    try:
        # Builder runs under the MFI/BTA ctx; BTA later invokes the aggregator with
        # run_context=self._rc_child("aggregator"), so the feed must land there.
        agg_input = builder(["plan_0"], original_query="master")
    finally:
        exit_run(tok)

    # inject mode: the builder returns JUST the original query as the agg input
    # (the upstream goes into the separate {{ upstream_artifacts }} slot).
    assert agg_input == "master"

    # The shared aggregator instance was NOT mutated.
    assert "upstream_artifacts" not in agg.template_extra_feed

    # The feed WAS published into the aggregator's child ctx handle.
    agg_ctx = ctx.child("aggregator")
    override = agg_ctx.handles.get(TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE)
    assert isinstance(override, dict)
    assert "upstream_artifacts" in override
    assert "plan_0" in override["upstream_artifacts"]


def test_aggregator_rendered_prompt_contains_upstream_content():
    """Behavior preserved: with the feed published to the aggregator's ctx, the leaf's
    own render (under THAT ctx) merges it → the rendered prompt carries the upstream."""
    agg = _FeedLeaf()
    mfi = _make_mfi(aggregator=agg)
    builder = mfi._make_default_aggregator_prompt_builder()

    ctx = RunContext.root(workspace=None)
    tok = enter_run(ctx)
    try:
        agg_input = builder(["plan_0", "plan_1"], original_query="master")
        # Simulate BTA invoking the aggregator under run_context=_rc_child("aggregator"):
        # the leaf renders under its own ctx and must see the published upstream feed.
        agg_ctx = ctx.child("aggregator")
        inner_tok = enter_run(agg_ctx)
        try:
            rendered = agg._render_prompt(agg_input)
        finally:
            exit_run(inner_tok)
    finally:
        exit_run(tok)

    # {{ input }} carries the original task; {{ upstream_artifacts }} carries the plans.
    assert "INPUT=master" in rendered
    assert "plan_0" in rendered and "plan_1" in rendered
    # And the instance dict stayed clean.
    assert "upstream_artifacts" not in agg.template_extra_feed


# ---------------------------------------------------------------------------
# Followup path — feed published at the flow node, found by the step child (walk-up)
# ---------------------------------------------------------------------------


def test_followup_feed_published_to_ctx_not_instance_and_walks_up():
    followup = _FeedLeaf()
    mfi = _make_mfi(aggregator=_FeedLeaf())

    # The flow's LWI ctx (BTA enters it via flow.ainfer(run_context=flow_child)).
    flow_ctx = RunContext.root(workspace=None).child("flow_0")
    tok = enter_run(flow_ctx)
    try:
        # MFI's followup builder publishes at the active (flow LWI) node.
        published = mfi._publish_child_template_feed(
            followup, None, {"upstream_artifacts": "PEERS_AND_PREV"}
        )
        assert published is True  # ctx channel, not instance write
    finally:
        exit_run(tok)

    # The shared followup instance was NOT mutated.
    assert "upstream_artifacts" not in followup.template_extra_feed

    # The followup leaf runs under LWI's step_{i} child (a DESCENDANT of the flow node);
    # it resolves the override by walking UP, and renders it.
    step_ctx = flow_ctx.child("step_1")
    inner_tok = enter_run(step_ctx)
    try:
        assert active_run_context().path == "/flow_0/step_1"
        # Walk-up resolution finds the override published at /flow_0.
        resolved = _resolve_ctx_feed_override()
        assert resolved == {"upstream_artifacts": "PEERS_AND_PREV"}
        rendered = followup._render_prompt("task_0")
    finally:
        exit_run(inner_tok)

    assert "INPUT=task_0" in rendered
    assert "PEERS_AND_PREV" in rendered
    assert "upstream_artifacts" not in followup.template_extra_feed


def test_followup_latest_step_value_wins():
    """Each step overwrites ``upstream_artifacts`` at the flow node — latest wins,
    matching the old instance-overwrite semantics (no stale bleed across steps)."""
    followup = _FeedLeaf()
    mfi = _make_mfi(aggregator=_FeedLeaf())
    flow_ctx = RunContext.root(workspace=None).child("flow_0")

    tok = enter_run(flow_ctx)
    try:
        mfi._publish_child_template_feed(followup, None, {"upstream_artifacts": "STEP1"})
        mfi._publish_child_template_feed(followup, None, {"upstream_artifacts": "STEP2"})
    finally:
        exit_run(tok)

    step_ctx = flow_ctx.child("step_2")
    inner_tok = enter_run(step_ctx)
    try:
        assert _resolve_ctx_feed_override() == {"upstream_artifacts": "STEP2"}
    finally:
        exit_run(inner_tok)


# ---------------------------------------------------------------------------
# Concurrency — two ctxs sharing ONE MFI + ONE child do not bleed feed values
# ---------------------------------------------------------------------------


def test_two_concurrent_ctxs_do_not_bleed_feed():
    """ONE shared MFI and ONE shared aggregator instance, two distinct RunContexts:
    each ctx's published upstream feed is isolated, and the shared instance is clean."""
    agg = _FeedLeaf()  # the SAME shared aggregator instance for both runs
    mfi = _make_mfi(aggregator=agg)
    builder = mfi._make_default_aggregator_prompt_builder()

    ctx_a = RunContext.root(workspace=None)
    ctx_b = RunContext.root(workspace=None)

    tok = enter_run(ctx_a)
    try:
        builder(["A_plan_0", "A_plan_1"], original_query="master_A")
    finally:
        exit_run(tok)

    tok = enter_run(ctx_b)
    try:
        builder(["B_plan_0", "B_plan_1"], original_query="master_B")
    finally:
        exit_run(tok)

    # Each ctx holds its OWN aggregator-feed override; no cross-talk.
    override_a = ctx_a.child("aggregator").handles.get(TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE)
    override_b = ctx_b.child("aggregator").handles.get(TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE)
    assert "A_plan_0" in override_a["upstream_artifacts"]
    assert "B_plan_0" in override_b["upstream_artifacts"]
    assert "B_plan_0" not in override_a["upstream_artifacts"]
    assert "A_plan_0" not in override_b["upstream_artifacts"]

    # The shared aggregator instance was never mutated by EITHER run.
    assert "upstream_artifacts" not in agg.template_extra_feed

    # Each ctx renders its own upstream content under its own aggregator child ctx.
    for ctx, expect in ((ctx_a, "A_plan_0"), (ctx_b, "B_plan_0")):
        agg_ctx = ctx.child("aggregator")
        inner_tok = enter_run(agg_ctx)
        try:
            rendered = agg._render_prompt("master")
        finally:
            exit_run(inner_tok)
        assert expect in rendered


# ---------------------------------------------------------------------------
# Legacy / no-ctx — byte-identical instance write
# ---------------------------------------------------------------------------


def test_legacy_no_ctx_uses_instance_feed():
    """With no active ctx, the per-call feed falls back to the child instance's
    ``template_extra_feed`` (byte-identical to the pre-virtualization behavior)."""
    agg = _FeedLeaf()
    mfi = _make_mfi(aggregator=agg)
    assert active_run_context() is None

    published = mfi._publish_child_template_feed(
        agg, "aggregator", {"upstream_artifacts": "LEGACY_UPSTREAM"}
    )
    assert published is False  # legacy: instance write, not ctx channel
    assert agg.template_extra_feed.get("upstream_artifacts") == "LEGACY_UPSTREAM"

    # And the leaf renders the instance feed (no ctx → no override; reads instance).
    rendered = agg._render_prompt("master")
    assert "INPUT=master" in rendered
    assert "LEGACY_UPSTREAM" in rendered


def test_legacy_mint_bare_call_uses_instance_feed():
    """A bare ``infer()`` legacy-mints a root. Under it, the publish still routes to
    the ctx channel (the call's store node), but a legacy-minted run's child handles are
    discarded on exit — so the instance backing remains the durable post-call surface for
    legacy callers. Here we assert the ctx-channel publish happens (active ctx present)
    and that, importantly, a real (non-legacy) shared run never pollutes the instance."""
    agg = _FeedLeaf()
    mfi = _make_mfi(aggregator=agg)
    legacy = mint_root()  # legacy_mint=True
    assert legacy.legacy_mint is True

    tok = enter_run(legacy)
    try:
        published = mfi._publish_child_template_feed(
            agg, "aggregator", {"upstream_artifacts": "LM_UPSTREAM"}
        )
        # A ctx IS active (legacy-minted), so we still use the ctx channel — and the
        # leaf renders it within the call.
        assert published is True
        agg_ctx = legacy.child("aggregator")
        inner_tok = enter_run(agg_ctx)
        try:
            rendered = agg._render_prompt("master")
        finally:
            exit_run(inner_tok)
        assert "LM_UPSTREAM" in rendered
    finally:
        exit_run(tok)

    # Even under legacy-mint, the ctx-channel publish did not mutate the instance dict.
    assert "upstream_artifacts" not in agg.template_extra_feed
