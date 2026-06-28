"""MultiFlowDualInferencer — convenience class for the common composition pattern.

A :class:`DualInferencer` whose ``base_inferencer`` is auto-constructed as a
:class:`MultiFlowInferencer` running N parallel dynamic-mode LWI flows that
cross-pollinate via cross-flow visibility, with an aggregator integrating the
final plan. Review and fix are performed by independent reviewer/fixer
inferencers (set as ordinary :class:`DualInferencer` attributes).

This class collapses the hand-wired composition::

    multi_flow = MultiFlowInferencer(flow_configs=..., visible_flows="all", ...)
    dual = DualInferencer(
        base_inferencer = multi_flow,
        review_inferencer = reviewer_cli,
        fixer_inferencer = fixer_cli,
        ...
    )

into a single construction step::

    mfdi = MultiFlowDualInferencer(
        flow_configs = ...,
        visible_flows = "all",
        review_inferencer = reviewer_cli,
        fixer_inferencer = fixer_cli,
        ...
    )

Round 7 — dynamic dispatch (rule-based and/or alias-based)::

    mfdi = MultiFlowDualInferencer(
        flow_configs=[...],
        multi_flow_winner_parser=parse_winner_tag,
        multi_flow_aggregator_prompt=AGGREGATOR_W_WINNER_TEMPLATE,
        # Rule-based: avoid self-review + winner-as-fixer
        review_default=kiro_cli,
        review_priority_pool=[claude_cli],
        fixer_strategy=FixerStrategy.WINNER,
    )

After each propose step (which runs MultiFlow), :meth:`_select_reviewer_and_fixer`
mutates ``self.review_inferencer`` and ``self.fixer_inferencer`` based on
the dispatch state MultiFlow exposed (``_last_winner_idx``,
``_last_reviewer_alias``, ``_last_fixer_alias``).

⚠️ **Production usability note for YAML configs**: When ``_target_: MultiFlowDual``
is constructed from YAML, distinct YAML nodes typically produce distinct
Python instances — and the ``is``-based self-review check would fail silently.
**Always use the alias-based path with YAML**: declare your inferencers once,
expose them via shared interpolation refs, populate ``inferencer_pool`` with
``{alias: ${ref}}`` entries, and reference inferencers by alias string in
``flow_configs``, ``review_default``, and ``review_priority_pool``. Aliases
resolve to the SAME pool instance everywhere → ``is`` comparison works.

Naming note: the class is *not* called ``MultiFlowPlanThenImplementInferencer``
because :class:`PlanThenImplementInferencer` denotes a four-step
plan/approval/implement/analysis workflow that is structurally different
from this propose/review/fix loop.
"""

import enum
import logging
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
    _VisibilitySpec,
)
from agent_foundation.common.inferencers.flow_parsers import (
    parse_finalplan_tag,
    parse_ranking_tag,
    parse_winner_tag,
)
from agent_foundation.common.inferencers.template_defaults import (
    FOLLOWUP_AGGREGATION_DEFAULTS,
    AGGREGATION_DEFAULTS,
)
from rich_python_utils.common_objects.workflow.workflow import Workflow
from rich_python_utils.io_utils.artifact import artifact_type

_logger = logging.getLogger(__name__)


@artifact_type(Workflow, type="json", group="workflows")
class ReviewerStrategy(str, enum.Enum):
    """Rule-based reviewer-selection strategy (an LLM ``reviewer_alias`` overrides it)."""

    CONFIGURED = "configured"            # review_default + review_priority_pool (avoid self-review)
    RUNNER_UP = "runner_up"              # runner-up flow's inferencer (ranking pos 1 / first non-winner)
    ALL_NON_WINNERS = "all_non_winners"  # §3 panel: every non-winner flow becomes a reviewer


class FixerStrategy(str, enum.Enum):
    """Rule-based fixer-selection strategy (an LLM ``fixer_alias`` overrides it)."""

    BASE = "base"                        # keep the configured fixer_inferencer
    WINNER = "winner"                    # winning flow's inferencer


@attrs
class MultiFlowDualInferencer(DualInferencer):
    """DualInferencer whose ``base_inferencer`` is an auto-constructed MultiFlow.

    Set ``flow_configs`` and the ``multi_flow_*`` attributes; the propose-phase
    MultiFlow is built in :meth:`__attrs_post_init__`. Reviewer and fixer are
    ordinary :class:`DualInferencer` attributes — either concrete
    :class:`InferencerBase` instances or, for Round 7 dynamic dispatch, alias
    strings resolved via ``inferencer_pool``.

    DO NOT pass ``base_inferencer`` directly — construction will raise
    :class:`ValueError` to surface the mistake immediately.

    ⚠️ **YAML usage warning**: distinct YAML nodes produce distinct Python
    instances; the rule-based self-review check uses ``is`` identity. **Use
    the alias-based path with YAML**: populate ``inferencer_pool`` with shared
    refs and reference inferencers by alias string. See module docstring.

    Example (rule-based dispatch — main use case)::

        mfdi = MultiFlowDualInferencer(
            flow_configs=[
                {"input": q, "initial_inferencer": claude,
                 "followup_inferencer": claude,
                 "end_condition": parse_stop, "max_dynamic_steps": 4},
                {"input": q, "initial_inferencer": kiro,
                 "followup_inferencer": kiro,
                 "end_condition": parse_stop, "max_dynamic_steps": 4},
            ],
            visible_flows="all",
            multi_flow_aggregator_inferencer=synthesizer,
            multi_flow_aggregator_prompt=AGGREGATOR_W_WINNER_TEMPLATE,
            multi_flow_winner_parser=parse_winner_tag,
            multi_flow_response_parser=parse_finalplan_tag,
            review_default=kiro_cli,                 # default reviewer
            review_priority_pool=[claude_cli],       # if Kiro wins → swap to Claude
            fixer_strategy=FixerStrategy.WINNER,                 # fixer = winning flow's CLI
            consensus_config=ConsensusConfig(max_iterations=2),
        )
        result = await mfdi.ainfer("Build a secure REST API")
    """

    # === Slot-based template role defaults (consumed by config_utils._walk) ===
    # Inherits Dual's review_inferencer.template_key="review" via MRO.
    # Adds defaults for the forwarded MultiFlow slots:
    #   - multi_flow_aggregator_inferencer → aggregation triplet
    #   - flow_configs.*.followup_inferencer → conditional aggregation triplet
    #     (gated on visible_flows!="self" AND inject_upstream_artifacts=True;
    #     both are class-default-aware — MFDual defaults visible_flows="all"
    #     AND inject_upstream_artifacts=True, so bare MFDual YAMLs get the
    #     aggregation framing without any flag wiring).
    SLOT_DEFAULTS: ClassVar[Dict[str, Any]] = {
        # multi_flow_aggregator + per-flow followup both depend on the full
        # structured-aggregation triplet — task_response_format renders the
        # winner_pick / iteration_judgment schemas when their flags fire.
        "multi_flow_aggregator_inferencer": AGGREGATION_DEFAULTS,
        "flow_configs.*.followup_inferencer": FOLLOWUP_AGGREGATION_DEFAULTS,
    }

    _workspace_propagation_skip: frozenset = frozenset({"multi_flow_aggregator_inferencer"})

    # ─── MultiFlow-specific config (forwarded into the auto-constructed MultiFlow) ───
    flow_configs: List[dict] = attrib(factory=list)
    visible_flows: _VisibilitySpec = attrib(default="all")

    multi_flow_aggregator_inferencer: Optional[InferencerBase] = attrib(default=None)
    multi_flow_aggregator_prompt: Optional[str] = attrib(default=None)
    multi_flow_aggregator_prompt_builder: Optional[Callable] = attrib(default=None)
    multi_flow_followup_prompt: Optional[str] = attrib(default=None)
    multi_flow_initial_prompt: Optional[str] = attrib(default=None)
    multi_flow_judgment_parser: Optional[Callable] = attrib(default=None)
    multi_flow_response_parser: Optional[Callable] = attrib(default=parse_finalplan_tag)
    """Default ``parse_finalplan_tag``: extracts ``<FinalPlan>...</FinalPlan>``
    content from the aggregator output, falls back to the raw string when no
    tag is present (graceful — passthrough for use cases that don't wrap)."""
    multi_flow_max_concurrency: Optional[int] = attrib(default=None)
    multi_flow_disable_aggregator: bool = attrib(default=False)
    multi_flow_prompt_formatter: Optional[Callable] = attrib(default=None)

    # ─── Round 7: dispatch parsers forwarded into MultiFlow ───
    multi_flow_winner_parser: Optional[Callable] = attrib(default=parse_winner_tag)
    """Default ``parse_winner_tag``: extracts the winning flow index from the
    aggregator's ``winner_pick`` JSON block (or legacy ``<Winner>flow_K</Winner>``
    tag), returns ``None`` if neither is present (graceful — dispatch falls back
    to ``review_default``). Forwarded to MultiFlow."""

    multi_flow_reviewer_alias_parser: Optional[Callable] = attrib(default=None)
    """Parser for ``<Reviewer>alias</Reviewer>``. Forwarded to MultiFlow."""

    multi_flow_fixer_alias_parser: Optional[Callable] = attrib(default=None)
    """Parser for ``<Fixer>alias</Fixer>``. Forwarded to MultiFlow."""

    # ─── Runtime input propagation (opt-in, forwarded to MultiFlow) ───
    propagate_runtime_input: bool = attrib(default=False)
    """Forward to MultiFlow's ``propagate_runtime_input``. When True, MultiFlow
    rewrites each flow's input from the runtime ``inference_input`` per call.
    Required when MFDual is used as a PTI planner so PTI's runtime subtask
    description actually reaches the flows. Resume not supported when True."""

    cross_flow_sync: bool = attrib(default=False)
    """Forward to MultiFlow's ``cross_flow_sync`` (the cross-flow step barrier).
    When True, each flow's round x+1 waits for ALL still-active flows' round x
    before reading peer artifacts — closing the "(no output yet)" race. Opt-in;
    async-only. See INTEGRATED_cross_flow_coordination_plan.md."""

    coordinated_stop: bool = attrib(default=False)
    """Forward to MultiFlow's ``coordinated_stop`` (alias for ``cross_flow_sync``;
    the unanimous stop vote is deferred). Kept for config-surface symmetry."""

    runtime_input_template: Optional[str] = attrib(default=None)
    """Forward to MultiFlow's ``runtime_input_template``. Optional Jinja
    template wrapping each flow's input with feed
    ``{input, flow_idx, n_flows}`` for per-flow perspective seeding."""

    inject_upstream_artifacts: bool = attrib(default=True)
    """Forward to MultiFlow's ``inject_upstream_artifacts``. When True (the
    MFDual default), the auto-constructed MultiFlow pushes formatted peer
    outputs into target inferencers' ``template_extra_feed["upstream_artifacts"]``
    and returns the per-flow input as ``inference_input`` so wrapper templates
    can use ``{{ upstream_artifacts }}`` and ``{{ input }}`` as separate slots.

    Default differs from MultiFlow's (False): MFDual is purpose-built for
    cross-flow aggregation with peer visibility (``visible_flows="all"`` is
    also MFDual's default), so injection is the natural mode. Set to ``False``
    only when the followup is a bare leaf without a wrapper template that
    consumes the slot — e.g., the ``multi-flow-dual.yaml`` topology preset
    that opts out explicitly to preserve simple prev_result passthrough."""

    # ─── Round 7: dispatch configuration on this class ───
    inferencer_pool: Dict[str, InferencerBase] = attrib(factory=dict)
    """Optional alias → instance mapping. Used both for resolving alias strings
    in ``review_default`` / ``review_priority_pool`` and for resolving aliases
    returned by the MultiFlow alias parsers (LLM-driven dispatch)."""

    review_default: Union[InferencerBase, str, None] = attrib(default=None)
    """Default reviewer for rule-based dispatch. May be an :class:`InferencerBase`
    instance OR a string alias resolvable via ``inferencer_pool``. When set,
    used as the reviewer unless the resolved default is the same instance as
    the MultiFlow winner — then the priority pool is consulted."""

    review_priority_pool: List[Union[InferencerBase, str]] = attrib(factory=list)
    """Fallback reviewers (instances or aliases) tried in order when the
    resolved ``review_default`` is the same instance as the MultiFlow winner.
    Self-review avoidance."""

    reviewer_strategy: ReviewerStrategy = attrib(
        default=ReviewerStrategy.CONFIGURED, converter=ReviewerStrategy
    )
    """How the per-attempt reviewer is chosen after each propose step (rule-based; an
    LLM ``reviewer_alias`` always overrides this). Accepts the enum or its string value
    (e.g. ``runner_up``) from YAML:
      - ``CONFIGURED`` (default): use ``review_default`` + ``review_priority_pool`` with
        self-review avoidance against the winner.
      - ``RUNNER_UP``: the runner-up flow's inferencer — ranking position 1, falling back
        to the first non-winner in ``flow_configs`` order. Requires ≥2 flows; auto-enables
        ``winner_pick`` and the aggregator ``include_ranking`` flag. Symmetric with
        ``FixerStrategy.WINNER`` (winner->fixer, runner-up->reviewer).
      - ``ALL_NON_WINNERS``: §3 panel — ALL non-winner flows become reviewers
        (``review_inferencer`` = panelist 0, the rest independent panelists run under
        ``review/panelist_i`` child ctxs, merged via ``merge_reviews``). Requires ≥2
        flows; auto-enables ``winner_pick``."""

    fixer_strategy: FixerStrategy = attrib(
        default=FixerStrategy.BASE, converter=FixerStrategy
    )
    """How the per-attempt fixer is chosen (an LLM ``fixer_alias`` always overrides).
    Accepts the enum or its string value from YAML:
      - ``BASE`` (default): keep the configured ``fixer_inferencer`` as-is.
      - ``WINNER``: set ``fixer_inferencer`` to the MultiFlow winner's inferencer after
        each propose step."""

    multi_flow_ranking_parser: Optional[Callable] = attrib(default=None)
    """Parser for ranking block in aggregator output. Auto-set to
    ``parse_ranking_tag`` when ``reviewer_strategy`` is ``RUNNER_UP`` and no
    explicit parser is provided."""

    # ─── Optional template overrides for role reassignment via switch_role ───
    review_template_key: Optional[str] = attrib(default=None)
    review_template_root_space: Optional[str] = attrib(default=None)
    followup_template_key: Optional[str] = attrib(default=None)
    followup_template_root_space: Optional[str] = attrib(default=None)

    winner_pick: bool = attrib(default=False)
    """Named feature toggle for the winner-pick pattern. When True,
    auto-injects ``include_winner_pick: true`` into
    ``multi_flow_aggregator_inferencer.template_extra_feed`` so the
    aggregator's wrapper template renders the JSON schema instructing
    the LLM to emit a ``winner_pick`` block. The default
    :func:`parse_winner_tag` (set on ``multi_flow_winner_parser``)
    extracts the index. ``setdefault`` semantics — explicit user values
    in ``template_extra_feed`` win.

    Pair with ``fixer_strategy=WINNER`` for full Round 7 dispatch
    (winner identified by aggregator → fixer set to that flow's CLI).
    Use without ``fixer_strategy=WINNER`` for observability — the JSON
    block is emitted and parsed but the parsed index isn't auto-routed."""

    # ─── Legacy dispatch predicates derived from the strategy enums (the canonical
    # config). Read-only — set ``reviewer_strategy`` / ``fixer_strategy`` to configure;
    # these keep the validation + ``_select_reviewer_and_fixer`` ladder readable. ───
    @property
    def reviewer_match_second(self) -> bool:
        return self.reviewer_strategy is ReviewerStrategy.RUNNER_UP

    @property
    def reviewer_match_all_non_winners(self) -> bool:
        return self.reviewer_strategy is ReviewerStrategy.ALL_NON_WINNERS

    @property
    def fixer_match_winner(self) -> bool:
        return self.fixer_strategy is FixerStrategy.WINNER

    def __attrs_post_init__(self):
        # Snapshot YAML-configured originals BEFORE any override
        # (reviewer_match_second, fixer_match_winner, or DualInferencer's
        # super().__attrs_post_init__() defaults). The identity guard in
        # _reassign_role_workspace compares the current value against these
        # snapshots to detect runtime role swaps that need workspace isolation.
        self._fixer_inferencer_original = self.fixer_inferencer
        self._review_inferencer_original = self.review_inferencer

        # Catch the most common misuse early: caller setting base_inferencer.
        if self.base_inferencer is not None:
            raise ValueError(
                "MultiFlowDualInferencer auto-constructs base_inferencer from "
                "flow_configs; do not pass base_inferencer directly. Use "
                "DualInferencer if you want manual control over base_inferencer."
            )
        if not self.flow_configs:
            raise ValueError(
                "MultiFlowDualInferencer requires at least one entry in flow_configs"
            )

        # If the caller specified a class-level initial_prompt template for the
        # MultiFlow, propagate it into each flow_config that doesn't already
        # supply its own. Per-flow override wins.
        if self.multi_flow_initial_prompt is not None:
            for cfg in self.flow_configs:
                cfg.setdefault("initial_prompt", self.multi_flow_initial_prompt)

        # reviewer_match_second validation + auto-enable dependencies.
        if self.reviewer_match_second:
            if len(self.flow_configs) < 2:
                raise ValueError(
                    "reviewer_match_second requires at least 2 entries in "
                    f"flow_configs (got {len(self.flow_configs)})"
                )
            self.winner_pick = True

        # §3 reviewer_match_all_non_winners (panel) — same winner-relative dependency.
        if self.reviewer_match_all_non_winners:
            if len(self.flow_configs) < 2:
                raise ValueError(
                    "reviewer_match_all_non_winners requires at least 2 entries in "
                    f"flow_configs (got {len(self.flow_configs)})"
                )
            self.winner_pick = True

        # `winner_pick: true` is a named feature toggle that auto-sets the
        # ``include_winner_pick`` template flag on the multi_flow_aggregator's
        # template_extra_feed (its wrapper template renders the JSON schema
        # asking the LLM to emit a `winner_pick` block). Done BEFORE the
        # MultiFlow auto-construction below so the mutated aggregator is
        # forwarded into the inner MultiFlow with the flag set. ``setdefault``
        # preserves any explicit user value.
        if self.winner_pick and self.multi_flow_aggregator_inferencer is not None:
            agg = self.multi_flow_aggregator_inferencer
            if hasattr(agg, "template_extra_feed"):
                if agg.template_extra_feed is None:
                    agg.template_extra_feed = {}
                agg.template_extra_feed.setdefault("include_winner_pick", True)

        # reviewer_match_second: inject ranking template flag + default parser.
        if self.reviewer_match_second and self.multi_flow_aggregator_inferencer is not None:
            agg = self.multi_flow_aggregator_inferencer
            if hasattr(agg, "template_extra_feed"):
                if agg.template_extra_feed is None:
                    agg.template_extra_feed = {}
                agg.template_extra_feed.setdefault("include_ranking", True)
            if self.multi_flow_ranking_parser is None:
                self.multi_flow_ranking_parser = parse_ranking_tag

        # Auto-construct the MultiFlow propose engine. Forward
        # ``workspace`` and ``checkpoint_dir`` so MultiFlow's own
        # checkpoint storage piggy-backs on whichever the parent uses; if
        # the parent later assigns ``_result_root_override`` to children,
        # BTA's existing machinery will redirect checkpoint paths accordingly.
        # (The legacy ``workspace_root`` shorthand was removed 2026-05-05.)
        self.base_inferencer = MultiFlowInferencer(
            flow_configs=self.flow_configs,
            visible_flows=self.visible_flows,
            aggregator_inferencer=self.multi_flow_aggregator_inferencer,
            aggregator_prompt=self.multi_flow_aggregator_prompt,
            aggregator_prompt_builder=self.multi_flow_aggregator_prompt_builder,
            multiflow_followup_prompt=self.multi_flow_followup_prompt,
            judgment_parser=self.multi_flow_judgment_parser,
            response_parser=self.multi_flow_response_parser,
            max_concurrency=self.multi_flow_max_concurrency,
            disable_aggregator=self.multi_flow_disable_aggregator,
            prompt_formatter=self.multi_flow_prompt_formatter,
            workspace=self.workspace,
            checkpoint_dir=self.checkpoint_dir,
            # Round 7: forward dispatch parsers
            winner_parser=self.multi_flow_winner_parser,
            reviewer_alias_parser=self.multi_flow_reviewer_alias_parser,
            fixer_alias_parser=self.multi_flow_fixer_alias_parser,
            ranking_parser=self.multi_flow_ranking_parser,
            # Runtime input propagation (opt-in, lets MFDual be used as PTI planner)
            propagate_runtime_input=self.propagate_runtime_input,
            runtime_input_template=self.runtime_input_template,
            # Cross-flow step coordination (opt-in lock-step visibility barrier)
            cross_flow_sync=self.cross_flow_sync,
            coordinated_stop=self.coordinated_stop,
            # Upstream-artifact injection (opt-in, wires {{ upstream_artifacts }}
            # slot in target inferencers' wrapper templates separate from {{ input }})
            inject_upstream_artifacts=self.inject_upstream_artifacts,
        )

        # Round 7: seed initial review_inferencer / fixer_inferencer from
        # rule-based config if not already set. The dispatch in propose's tail
        # may overwrite these per-attempt based on the winner.
        if self.review_inferencer is None and self.review_default is not None:
            self.review_inferencer = self._resolve_id(self.review_default)
        if (
            self.review_inferencer is None
            and self.reviewer_match_second
            and len(self.flow_configs) > 1
        ):
            self.review_inferencer = self.flow_configs[1].get("initial_inferencer")
        if (
            self.fixer_inferencer is None
            and self.fixer_match_winner
            and self.review_inferencer is not None
        ):
            # Placeholder until propose runs and we know the winner.
            # Use the reviewer as a benign default so DualInferencer's
            # 2-agent fallback doesn't kick in unexpectedly.
            self.fixer_inferencer = self.review_inferencer

        # When dynamic dispatch is configured (reviewer_match_second or
        # fixer_match_winner), review/fixer inferencers may be aliased to
        # flow_configs entries. Skip construction-time workspace propagation
        # for them — _reassign_role_workspace assigns the correct workspace
        # at runtime after each propose step.
        if self.reviewer_match_second or self.fixer_match_winner:
            self._workspace_propagation_skip = frozenset(
                self._workspace_propagation_skip | {"review_inferencer", "fixer_inferencer"}
            )

        # Defer to DualInferencer for prompt-template setup, parser defaults,
        # workspace, sub-inferencer wiring, etc.
        super().__attrs_post_init__()

        if self.review_inferencer is None:
            _logger.warning(
                "MultiFlowDualInferencer: no reviewer mechanism configured "
                "(review_inferencer, review_default, or reviewer_strategy=runner_up/all_non_winners "
                "are all unset). The review step will fail at runtime."
            )


    # ------------------------------------------------------------------
    # Part B — Workspace isolation for runtime role swaps (added 2026-05-09)
    # ------------------------------------------------------------------

    def _resolve_role_template(self, role_name):
        """Resolve template_key and template_root_space for a role name.

        Returns ``(template_key, template_root_space)`` — either from the
        explicit per-role attribs (``review_template_key``, etc.) or from the
        canonical :mod:`template_defaults` bundles. Returns ``(None, None)``
        for unrecognised roles (no template override).
        """
        from agent_foundation.common.inferencers.template_defaults import (
            REVIEW_TEMPLATE_DEFAULTS, FOLLOWUP_TEMPLATE_DEFAULTS)
        if role_name in ("reviewer", "review_inferencer"):
            return (self.review_template_key or REVIEW_TEMPLATE_DEFAULTS.template_key,
                    self.review_template_root_space or getattr(REVIEW_TEMPLATE_DEFAULTS, 'template_root_space', None))
        elif role_name in ("fixer", "fixer_inferencer"):
            return (self.followup_template_key or FOLLOWUP_TEMPLATE_DEFAULTS.template_key,
                    self.followup_template_root_space or getattr(FOLLOWUP_TEMPLATE_DEFAULTS, 'template_root_space', None))
        return (None, None)

    def _reassign_role_workspace(
        self, inferencer, role_name: str, *, panelist_slot: Optional[str] = None,
    ) -> None:
        """Force a fresh workspace + session because role has changed.

        When MFDual reuses an instance across roles (e.g., winning flow's
        initial_inferencer becomes fixer_inferencer), its original workspace
        (``flow_N_initial/``) belongs to the propose-phase. Continuing to write
        to it would mix fix-phase artifacts with propose-phase artifacts.

        Delegates to ``inferencer.switch_role()`` which handles workspace
        assignment, template overrides, session reset, and audit trail in a
        single call.

        Identity guard: skip reassignment when the inferencer is the role's
        ORIGINALLY-CONFIGURED instance (snapshot in __attrs_post_init__).
        Prevents clobbering a deliberately-configured fixer's workspace.

        Args:
            inferencer: the candidate inferencer to (possibly) reassign.
            role_name: 'fixer_inferencer' or 'review_inferencer' — used as the
                child workspace directory name.
            panelist_slot: when set (e.g. ``"review/panelist_1"``), use this as the
                context-path slot instead of the default ``"review"`` or ``"fix"``.
                Required for ``all_non_winners`` panel reviewers — each panelist
                needs its own context path so their ``switch_role`` creator tags
                don't collide (heterogeneous CLI types have different creator tuples).
                Matches the path the Dual review step will actually run them under.
        """
        if inferencer is None or self._workspace is None:
            return
        # Identity guard
        original = getattr(self, f"_{role_name}_original", None)
        if inferencer is original:
            return
        # Do NOT create a workspace dir here — the Dual review/fix step creates the
        # round-scoped workspace (round_NN/children/review/children/panelist_NN/ or
        # round_NN/children/fix/) and assigns it before ainfer(). Creating a pre-round
        # workspace here produced empty scaffold dirs (children/review/, children/fix/)
        # that were never populated and only caused confusion.
        kwargs = dict(
            output_is_deliverable=(True if role_name == "fixer_inferencer" else None),
        )
        # Pass template kwargs only when the inferencer supports them
        # (TemplatedInferencerBase subclasses accept template_key etc.)
        from agent_foundation.common.inferencers.templated_inferencer_base import TemplatedInferencerBase
        if isinstance(inferencer, TemplatedInferencerBase):
            new_key, new_root = self._resolve_role_template(role_name)
            kwargs["template_key"] = new_key
            kwargs["template_root_space"] = new_root
        # §2.7 isolation: record the role-switch state (RoleState/session) on the
        # role's OWN child ctx (``./review`` or ``./fix``) — the SAME node the
        # review/fix dispatch targets via ``run_context=self._rc_child(slot)``.
        # This runs inside ``_step_propose_impl`` (a step_fn) where the active ctx
        # is the worker node, so without this scoping the reviewer (runner-up) and
        # fixer (winner) leaves — distinct creators — both tag the worker path and
        # trip the creator-collision guard on the consensus re-run. It also fixes a
        # latent template-locality bug: the dispatch renders at ``./review``/``./fix``
        # and (no ancestor walk-up) would otherwise miss the worker-node RoleState.
        from agent_foundation.common.inferencers.run_context import (
            enter_run,
            exit_run,
        )

        _role_slot = panelist_slot or {"review_inferencer": "review", "fixer_inferencer": "fix"}.get(
            role_name, role_name
        )
        # For compound slots (e.g. "review/panelist_01"), chain .child() calls
        # to build a nested context path. _rc_child() sanitizes "/" to "_" which
        # would create a flat path that doesn't match the nested execution path.
        if "/" in _role_slot:
            from agent_foundation.common.inferencers.run_context import active_run_context
            _ctx = active_run_context()
            _rc = _ctx
            if _ctx is not None:
                for _seg in _role_slot.split("/"):
                    _rc = _rc.child(_seg)
        else:
            _rc = self._rc_child(_role_slot)
        _tok = enter_run(_rc) if _rc is not None else None
        try:
            # Publish the runtime-resolved role into the role's OWN child node
            # scratch (``./review`` | ``./fix``). ``_select_reviewer_and_fixer``
            # wrote it to the WORKER node (via ``_role_set`` under
            # ``_step_propose_impl``'s worker ctx), but the review/fix pre-render
            # AND dispatch call ``_role_get`` while the active ctx is THIS child
            # node — and ``_role_get`` is node-scoped (reads the active node's
            # scratch with NO ancestor walk-up). Without this publish the child
            # read misses the worker-node value and falls back to the STALE
            # construction-time ``self.<role>`` placeholder (``flow_configs[1]``,
            # the runner-up seed). When that placeholder's class differs from the
            # runtime leaf, the pre-render's ``_effective_role`` claims this node
            # with a DIFFERENT creator than ``switch_role`` (below) just did →
            # ``CollisionError``. Publishing here keeps the child-scoped read
            # coherent with the worker-scoped selection. (No-ctx / legacy-mint:
            # ``_rc`` is None → skipped; the role lives on the instance attribute,
            # which the legacy ``_role_get`` reads — byte-identical.)
            if _rc is not None:
                self._role_set(role_name, inferencer, _rc)
            inferencer.switch_role(new_role=role_name, **kwargs)
        finally:
            if _tok is not None:
                exit_run(_tok)

    # ------------------------------------------------------------------
    # Round 7 — dispatch helpers
    # ------------------------------------------------------------------

    def _resolve_id(
        self, x: Union[InferencerBase, str, None]
    ) -> Optional[InferencerBase]:
        """Resolve an instance-or-alias to a concrete :class:`InferencerBase`.

        Resolution order:

        1. ``None``                                → ``None``
        2. Already an :class:`InferencerBase`      → returned as-is
        3. String present in ``inferencer_pool``   → ``inferencer_pool[x]``
        4. otherwise                               → :class:`KeyError`
           with a message listing the known pool keys

        We deliberately do NOT support dynamic class import from a dotted
        path: configured instances (with ``target_path``, ``cache_folder``,
        etc.) belong in the pool, and "instantiate from a class path" would
        produce an un-configured instance.
        """
        if x is None:
            return None
        if isinstance(x, InferencerBase):
            return x
        if isinstance(x, str):
            if x in self.inferencer_pool:
                return self.inferencer_pool[x]
            raise KeyError(
                f"MultiFlowDualInferencer: identifier {x!r} not found in "
                f"inferencer_pool (known keys: {sorted(self.inferencer_pool.keys())!r}). "
                f"Either add it to inferencer_pool or pass an InferencerBase instance."
            )
        raise TypeError(
            f"MultiFlowDualInferencer: expected InferencerBase | str | None, "
            f"got {type(x).__name__}"
        )

    def _select_reviewer_and_fixer(self) -> None:
        """Mutate ``self.review_inferencer`` and ``self.fixer_inferencer`` based
        on dispatch state exposed by the inner MultiFlow.

        Resolution precedence (reviewer):
          1. LLM-driven alias from MultiFlow's alias parser, if present.
          2. ``reviewer_match_second``: runner-up from ranking or first non-winner.
          3. Rule-based: ``review_default`` with avoid-self-review via priority pool.

        Resolution precedence (fixer):
          1. LLM-driven alias from MultiFlow's fixer alias parser.
          2. ``fixer_match_winner``: winner's flow inferencer.

        When neither path applies, leaves the inferencer at its current value
        (typically the construction-time default).
        """
        from agent_foundation.common.inferencers.run_context import (
            active_run_context,
            enter_run,
            exit_run,
        )

        mfi = self.base_inferencer
        if not isinstance(mfi, MultiFlowInferencer):
            return  # nothing to dispatch on

        # C5: the ctx whose node owns the per-run resolved roles (MFDual's own node).
        # Roles are written via _role_set: under a real ctx → ctx.node().scratch (no
        # self-mutation, so a shared instance is safe across concurrent runs); under
        # legacy / legacy-mint → the instance attribute (byte-identical).
        mfdual_ctx = active_run_context()

        # Phase 1: read the inner MultiFlow dispatch under the PROPOSE child node — the
        # surviving store node MFI actually wrote. Under a real shared-instance ctx, mfi's
        # own getters (which read the *active* node) resolve MFDual's node or MFI's instance
        # backing (only populated under legacy); entering the propose child makes them
        # resolve the MultiFlowState MFI published there.
        _pc = self._rc_child("propose")
        _tok = enter_run(_pc) if _pc is not None else None
        try:
            winner = mfi.get_winner_inferencer()
            review_alias = mfi.get_chosen_reviewer_alias()
            fixer_alias = mfi.get_chosen_fixer_alias()
            non_winners = mfi.get_non_winner_inferencers()
            runner_up = mfi.get_runner_up_inferencer()
            first_non_winner = mfi.get_first_non_winner_inferencer()
        finally:
            if _tok is not None:
                exit_run(_tok)

        # Sanity warnings: dispatch was configured but winner was not detected.
        # Caused by missing winner_parser, malformed aggregator output, or the
        # aggregator's prompt not instructing the LLM to emit <Winner>flow_X</Winner>.
        # Suppressed when the LLM-driven alias path is in use (those don't need
        # a winner). Edge case: if <Reviewer> is present but <Winner> is missing,
        # the reviewer warning is correctly suppressed (alias dispatch covers
        # reviewer); the fixer warning still fires because fixer dispatch DOES
        # need a winner.
        if winner is None:
            if self.review_default is not None and not review_alias:
                _logger.warning(
                    "MultiFlowDualInferencer: review_default is configured but "
                    "the winner could not be identified from MultiFlow output. "
                    "Self-review-avoidance cannot fire — reviewer will be set "
                    "to review_default as-is. Verify multi_flow_winner_parser "
                    "is configured AND the aggregator's prompt instructs the "
                    "LLM to emit <Winner>flow_X</Winner>."
                )
            if self.fixer_match_winner and not fixer_alias:
                _logger.warning(
                    "MultiFlowDualInferencer: fixer_strategy=winner but the "
                    "winner could not be identified. Fixer will keep its "
                    "construction-time value (typically falling through to "
                    "base_inferencer)."
                )

        # ----- Reviewer ----- (compute chosen reviewer + optional panel, then store once)
        chosen: Optional[InferencerBase] = None
        panel_set = False
        panel_value = None
        # 1) LLM-driven alias takes precedence
        if review_alias:
            try:
                chosen = self._resolve_id(review_alias)
            except KeyError as exc:
                _logger.warning(
                    "MultiFlowDual: reviewer_alias %r could not be resolved "
                    "(not in inferencer_pool): %s — falling back to rule-based dispatch",
                    review_alias,
                    exc,
                )
        if chosen is not None:
            pass
        elif self.reviewer_match_all_non_winners:
            # 2) §3 PANEL: every non-winner flow becomes a reviewer. Populate the
            # inherited Dual ``reviewers`` panel (review_inferencer = panelist 0, the
            # rest independent panelists); the Dual review step runs them under
            # ``review/panelist_i`` child ctxs and merges via merge_reviews.
            #
            # When winner is None (aggregator output broken → winner_parser failed),
            # fall back to ALL flows as reviewers — every flow is a "non-winner" when
            # no winner was detected. This ensures the review ALWAYS runs with
            # all_non_winners, acting as a recovery gate for broken propose output.
            _candidates = non_winners if (winner is not None and non_winners) else None
            if _candidates is None:
                _all_flows = [
                    cfg.get("initial_inferencer") for cfg in mfi.flow_configs
                    if cfg.get("initial_inferencer") is not None
                ]
                if _all_flows:
                    _candidates = _all_flows
                    if winner is None:
                        _logger.warning(
                            "MultiFlowDual: reviewer_strategy=all_non_winners but no "
                            "winner detected (broken aggregator output?). Falling back "
                            "to ALL flows as reviewers — review will still run."
                        )
            if _candidates:
                chosen = _candidates[0]
                panel_set = True
                panel_value = _candidates[1:] if len(_candidates) > 1 else None
            else:
                _logger.warning(
                    "MultiFlowDual: reviewer_strategy=all_non_winners but no "
                    "candidate flows found. Reviewer unchanged."
                )
        elif self.reviewer_match_second and winner is not None:
            # 2) Runner-up as reviewer (ranking-based or declaration-order fallback)
            ru = runner_up if runner_up is not None else first_non_winner
            if ru is not None and ru is not winner:
                chosen = ru
            else:
                _logger.warning(
                    "MultiFlowDual: reviewer_strategy=runner_up but no valid "
                    "runner-up found. Reviewer unchanged."
                )
        elif self.review_default is not None:
            # 3) Rule-based: avoid self-review
            default = self._resolve_id(self.review_default)
            if winner is not None and winner is default:
                fallback: Optional[InferencerBase] = None
                for cand in self.review_priority_pool:
                    try:
                        cand_inf = self._resolve_id(cand)
                    except KeyError as exc:
                        _logger.warning(
                            "MultiFlowDual: priority_pool entry %r could not be "
                            "resolved: %s — skipping", cand, exc,
                        )
                        continue
                    if cand_inf is not None and cand_inf is not winner:
                        fallback = cand_inf
                        break
                if fallback is not None:
                    chosen = fallback
                else:
                    _logger.warning(
                        "MultiFlowDual: review_default is the winner and "
                        "priority_pool exhausted; using default (self-review)."
                    )
                    chosen = default
            else:
                chosen = default

        if chosen is not None:
            self._role_set("review_inferencer", chosen, mfdual_ctx)
        if panel_set:
            self._role_set("reviewers", panel_value, mfdual_ctx)

        # ----- Fixer -----
        chosen_fix: Optional[InferencerBase] = None
        if fixer_alias:
            try:
                chosen_fix = self._resolve_id(fixer_alias)
            except KeyError as exc:
                _logger.warning(
                    "MultiFlowDual: fixer_alias %r could not be resolved "
                    "(not in inferencer_pool): %s — falling back to rule-based dispatch",
                    fixer_alias,
                    exc,
                )
        if chosen_fix is None and self.fixer_match_winner and winner is not None:
            chosen_fix = winner
        if chosen_fix is not None:
            self._role_set("fixer_inferencer", chosen_fix, mfdual_ctx)

    def _collect_candidate_inferencers(self) -> List[InferencerBase]:
        """Raw collection of every inferencer that could plausibly be
        selected as reviewer or fixer at runtime, plus the base inferencer
        itself. May contain duplicates (the same instance can appear in
        multiple slots/pools) — callers go through
        :meth:`_iter_child_inferencers` which dedups on identity.

        Note: ``MultiFlowDualInferencer`` mutates ``review_inferencer`` /
        ``fixer_inferencer`` mid-run via :meth:`_select_reviewer_and_fixer`,
        so any parent→child operation must walk every candidate (not just
        the current ``(base, review, fixer)`` triple).
        """
        cands: List[InferencerBase] = []
        if self.base_inferencer is not None:
            cands.append(self.base_inferencer)
        for slot in (self.review_inferencer, self.fixer_inferencer):
            if slot is not None:
                cands.append(slot)
        # review_default and priority pool (resolved if alias-typed)
        if self.review_default is not None:
            try:
                resolved = self._resolve_id(self.review_default)
                if resolved is not None:
                    cands.append(resolved)
            except KeyError:
                pass
        for c in self.review_priority_pool:
            try:
                resolved = self._resolve_id(c)
                if resolved is not None:
                    cands.append(resolved)
            except KeyError:
                continue
        # Flow inferencers (initial + followup). We rely on these being
        # idempotent on aconnect — ClaudeCodeCli / KiroCli are; document
        # the contract for any custom flow inferencer.
        if isinstance(self.base_inferencer, MultiFlowInferencer):
            for cfg in self.base_inferencer.flow_configs:
                init_inf = cfg.get("initial_inferencer")
                fup_inf = cfg.get("followup_inferencer")
                if init_inf is not None:
                    cands.append(init_inf)
                if fup_inf is not None and fup_inf is not init_inf:
                    cands.append(fup_inf)
        # Pool entries
        for v in self.inferencer_pool.values():
            if v is not None:
                cands.append(v)
        return cands

    def _iter_child_inferencers(self):
        """Broad candidate set including pool / default / priority_pool entries.

        Yields each candidate at most once (id-based dedup) per the
        ``InferencerBase._iter_child_inferencers`` contract. Used uniformly
        by lifecycle methods (``aconnect`` / ``adisconnect`` —
        inherited from ``DualInferencer``) and recursive ``pre_retry``.
        """
        seen: set = set()
        for inf in self._collect_candidate_inferencers():
            if inf is None or id(inf) in seen:
                continue
            seen.add(id(inf))
            yield inf

    # ------------------------------------------------------------------
    # Round 7 — overrides
    # ------------------------------------------------------------------

    async def _step_propose_impl(self, step_input, state):
        """Run the parent propose step (which calls MultiFlow), then dispatch
        reviewer / fixer based on the dispatch state MultiFlow exposed."""
        prev_reviewer = self.review_inferencer
        prev_fixer = self.fixer_inferencer
        result = await DualInferencer._step_propose_impl(self, step_input, state)
        self._select_reviewer_and_fixer()
        # Part B: when winner/loser are runtime-assigned to fixer/reviewer roles,
        # give them a fresh workspace + session so role artifacts don't mix
        # with propose-phase artifacts in flow_N_initial/. Identity guard inside
        # the helper protects originally-configured instances.
        # C5: resolve the per-run roles via _role_get (ctx scratch under a real ctx,
        # else the instance attribute) — never read self.review_inferencer directly here,
        # which under a shared-instance ctx is the static definition, not the selection.
        from agent_foundation.common.inferencers.inferencer_workspace import indexed_child_name
        _panel_list = self._role_get("reviewers") or []
        _has_panel = bool(_panel_list)
        # Primary reviewer: in panel mode, the execution path is
        # review/panelist_00 (matching panelists 1+ at review/panelist_01, etc.).
        # Single reviewer: the execution path is just review/.
        if _has_panel:
            self._reassign_role_workspace(
                self._role_get("review_inferencer"), "review_inferencer",
                panelist_slot=f"review/{indexed_child_name('panelist', 0)}",
            )
        else:
            self._reassign_role_workspace(self._role_get("review_inferencer"), "review_inferencer")
        self._reassign_role_workspace(self._role_get("fixer_inferencer"), "fixer_inferencer")
        # §3 panel: the extra non-winner panelists (reviewers list) must ALSO be
        # switched into the reviewer role/template — else they would review with
        # their original flow role.
        for _i, _panelist in enumerate(_panel_list, start=1):
            self._reassign_role_workspace(
                _panelist, "review_inferencer",
                panelist_slot=f"review/{indexed_child_name('panelist', _i)}",
            )
        # NOTE: reset_session is handled by switch_role() inside
        # _reassign_role_workspace — no separate call needed here.

        mfi = self.base_inferencer
        if isinstance(mfi, MultiFlowInferencer):
            from agent_foundation.common.inferencers.run_context import (
                enter_run,
                exit_run,
            )

            # Read dispatch under the propose child node (see _select_reviewer_and_fixer).
            _pc = self._rc_child("propose")
            _tok = enter_run(_pc) if _pc is not None else None
            try:
                _w_idx = mfi._last_winner_idx
                _rank = mfi._last_ranking
            finally:
                if _tok is not None:
                    exit_run(_tok)
            # Winner dispatch info is recorded in round_log.jsonl (by the Dual
            # review step); no separate filesystem nav entry needed.

        return result

    # NOTE: aconnect / adisconnect are inherited from DualInferencer.
    # DualInferencer.aconnect/adisconnect iterate `_iter_child_inferencers`
    # (with `hasattr` guards), and our override above yields the broad
    # candidate set. So the inherited methods Just Work — no need for
    # bespoke overrides here.
