"""MultiFlowInferencer — N parallel dynamic LWI flows with optional cross-flow visibility.

A specialization of :class:`BreakdownThenAggregateInferencer` where:

* The breakdown phase is disabled — sub-queries come from ``flow_configs[i]["input"]``.
* Each worker is a :class:`LinearWorkflowInferencer` running in dynamic mode.
* Flows can optionally see each other's latest outputs via ``visible_flows``.
* The aggregator (when configured) integrates the final result from all flows.

Inheriting from BTA gives MultiFlow first-class access to BTA's graph-level
features (graph_reporter, interactive checkpoints, expansion infrastructure,
checkpoint/resume, max_concurrency, workspace) without re-declaration.

Usage::

    mfi = MultiFlowInferencer(
        flow_configs=[
            {
                "input": "Research authentication best practices",
                "initial_inferencer": researcher,
                "followup_inferencer": researcher,
                "end_condition": lambda s, r: "DONE" in str(r),
                "max_dynamic_steps": 5,
            },
            {
                "input": "Design the API schema",
                "initial_inferencer": designer,
                "followup_inferencer": designer,
                "max_dynamic_steps": 3,
            },
        ],
        visible_flows="all",
        aggregator_inferencer=synthesizer,
        max_concurrency=2,
    )
    result = await mfi.ainfer("Build a secure REST API")
"""

import logging
import os
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

from attr import attrib, attrs

from rich_python_utils.common_utils.async_utils import call_maybe_async

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
)
from agent_foundation.common.inferencers.template_defaults import (
    FOLLOWUP_AGGREGATION_DEFAULTS,
)

_logger = logging.getLogger(__name__)


# Keys characteristic of an LWI dynamic-mode workflow state dict. If a tuple
# element has these (and no `output` field), it's a worker's state, not an
# aggregator's response — filter it out of aggregator-output selection.
_LWI_STATE_DICT_MARKERS = frozenset((
    "dynamic_step_results",
    "iteration_records",
    "__expansion_count",
    "_prev_iteration",
))


def _looks_like_lwi_state(d: Dict[str, Any]) -> bool:
    """True if `d` looks like an LWI workflow state dict (worker output),
    not an inferencer response dict (aggregator output)."""
    if not isinstance(d, dict):
        return False
    return any(k in d for k in _LWI_STATE_DICT_MARKERS)


# ---------------------------------------------------------------------------
# Followup-input formatting (per-step iteration)
# ---------------------------------------------------------------------------
#
# Default formatter for the per-step "upstream artifacts" text used at
# step ≥ 1 of each flow's LWI. Output is either passed as the LWI's
# ``inference_input`` directly (legacy passthrough mode) or pushed into the
# followup_inferencer's ``template_extra_feed["upstream_artifacts"]`` slot
# (when ``inject_upstream_artifacts=True``); in the latter mode the wrapper
# template's ``{{ input }}`` slot carries the original task separately, so
# the task does NOT appear in the formatted output below (avoids duplication
# with the wrapper).
#
# Exposed as module-level constants so prompt-engineering tweaks live in
# one place. Override by subclassing MultiFlowInferencer and replacing
# ``_format_followup_input``, or bypass entirely via ``cfg["followup_prompt"]``
# (legacy template path) or ``cfg["dynamic_input_builder"]`` (full callback).

_FOLLOWUP_OWN_PREVIOUS_HEADER = (
    "You previously produced this artifact (flow {flow_idx}, step {prev_step_idx}):\n"
    "{your_prev}"
)
_FOLLOWUP_PEER_INTRO = "Here are artifacts produced by other parallel flows:"
_FOLLOWUP_PEER_BLOCK_HEADER = "=== Flow {idx} ==="
_FOLLOWUP_PEER_EMPTY_PLACEHOLDER = "(no output yet)"
_FOLLOWUP_NO_PEERS = (
    "(No peer outputs visible yet — this iteration sees only your own previous output.)"
)


# ---------------------------------------------------------------------------
# Visibility helpers
# ---------------------------------------------------------------------------

_VisibilitySpec = Union[str, List[int], Tuple[int, ...]]


def _resolve_visible_indices(
    flow_idx: int,
    n_flows: int,
    spec: _VisibilitySpec,
) -> List[int]:
    """Translate a ``visible_flows`` spec into concrete flow indices.

    * ``"self"`` -> ``[flow_idx]``
    * ``"all"`` -> ``[0, 1, ..., n_flows - 1]`` (includes self)
    * list/tuple of ints -> filtered to in-range indices

    Raises :class:`ValueError` for any other shape.
    """
    if spec == "self":
        return [flow_idx]
    if spec == "all":
        return list(range(n_flows))
    if isinstance(spec, (list, tuple)):
        return [int(i) for i in spec if 0 <= int(i) < n_flows]
    raise ValueError(
        f"visible_flows must be 'self', 'all', or a list of int — got {spec!r}"
    )


# ---------------------------------------------------------------------------
# MultiFlowInferencer
# ---------------------------------------------------------------------------


@attrs(slots=False)
class MultiFlowInferencer(BreakdownThenAggregateInferencer):
    """Run N parallel dynamic LWI flows and aggregate their outputs.

    A subclass of :class:`BreakdownThenAggregateInferencer` where:

    * ``breakdown_inferencer`` is forced to ``None`` (no LLM breakdown).
    * ``predefined_sub_queries`` is auto-derived from ``[c["input"] for c in flow_configs]``.
    * ``worker_inferencers`` is auto-built to produce a fresh dynamic-mode LWI per flow.
    * ``aggregator_inferencer`` (inherited) integrates the final result.

    Each ``flow_configs`` entry is a dict accepting:

    ====================  =================================================  ==========
    Key                   Meaning                                            Default
    ====================  =================================================  ==========
    ``input``             Per-flow query (becomes step 0's input)            *required*
    ``initial_inferencer``First step inferencer                              *required*
    ``followup_inferencer``Subsequent step inferencer                        ``None``
    ``end_condition``     ``(state, result) -> bool``                        ``None``
    ``max_dynamic_steps`` Cap per flow                                       ``10``
    ``inferencer_factory``Factory for class-typed inferencers                ``None``
    ``dynamic_input_builder`` Custom builder; bypasses template machinery    ``None``
    ``response_builder``  Final result builder for the LWI                   ``None``
    ``visible_flows``     Per-flow visibility override                       *class default*
    ``initial_prompt``    Template (key or raw Jinja2) for step 0            ``None``
    ``followup_prompt``   Template for step ≥ 1                              ``None``
    ``iteration_judgment`` Named feature toggle (see below)                  ``False``
    ====================  =================================================  ==========

    When no ``initial_prompt`` is set, step 0 receives the raw ``input`` field.
    When no ``followup_prompt`` and class-level
    ``multiflow_followup_prompt`` are set, step ≥ 1 receives the previous
    step's result directly (legacy passthrough).

    ``iteration_judgment: true`` is a coordinated feature toggle that bundles
    two coupled setups (without it, each half is inert):

    1. Sets ``end_condition`` to ``parse_decision_stop`` — extracts the
       ``iteration_judgment`` JSON block from each step's output and
       terminates the flow early when the LLM emits ``"decision": "stop"``.
    2. Sets ``followup_inferencer.template_extra_feed["include_iteration_judgment"]``
       to ``True`` — the followup's wrapper template (e.g.
       ``plan/main/_variables/task_response_format/aggregation.jinja2``)
       renders the JSON schema instructing the LLM to emit the judgment.

    Both halves use ``setdefault`` semantics — explicit user overrides win.
    """

    # === Slot-based template role defaults (consumed by config_utils._walk) ===
    # Inherits BTA's breakdown/aggregator defaults via MRO. The
    # ``flow_configs.*.followup_inferencer`` wildcard path applies aggregation
    # framing to each per-flow followup IFF (a) peers are visible
    # (visible_flows != "self") AND (b) injection wiring is engaged
    # (inject_upstream_artifacts=True). The condition is class-default-aware:
    # MultiFlow defaults visible_flows="self" + inject=False (simple parallel
    # sampling, no peer aggregation), so bare MultiFlow YAMLs skip the
    # default; MFDual subclass defaults both to enable aggregation, so bare
    # MFDual YAMLs apply the default automatically.
    SLOT_DEFAULTS: ClassVar[Dict[str, Any]] = {
        "flow_configs.*.followup_inferencer": FOLLOWUP_AGGREGATION_DEFAULTS,
    }

    # MultiFlow-specific config (NEW vs. previous composition wrapper)
    flow_configs: List[dict] = attrib(factory=list)
    visible_flows: _VisibilitySpec = attrib(default="self")

    # Prompt-template machinery (mirrors DualInferencer's pattern)
    prompt_formatter: Optional[Callable] = attrib(default=None)
    multiflow_followup_prompt: Optional[str] = attrib(default=None)
    aggregator_prompt: Optional[str] = attrib(default=None)

    # Optional parsers (informational + post-processing)
    judgment_parser: Optional[Callable[[str], Optional[str]]] = attrib(default=None)
    response_parser: Optional[Callable[[str], str]] = attrib(default=None)

    # ----- Round 7: opt-in dispatch state for downstream MultiFlowDualInferencer ----
    # All three default to None — when None, MultiFlow behaves exactly as in
    # Round 5/6 (no winner / alias identification). Round 7 backward-compat.

    winner_parser: Optional[Callable[[str], Optional[int]]] = attrib(default=None)
    """Parser for ``<Winner>flow_X</Winner>`` style tags in the aggregator's
    output. Returns the winning flow's index (0-based), or None if not found."""

    reviewer_alias_parser: Optional[Callable[[str], Optional[str]]] = attrib(default=None)
    """Optional parser for ``<Reviewer>alias</Reviewer>`` style tags. When set,
    the LLM aggregator can choose a reviewer alias from a downstream pool.
    Returns the chosen alias name, or None when not present."""

    fixer_alias_parser: Optional[Callable[[str], Optional[str]]] = attrib(default=None)
    """Same as ``reviewer_alias_parser``, for ``<Fixer>alias</Fixer>``."""

    ranking_parser: Optional[Callable[[str], Optional[list]]] = attrib(default=None)
    """Parser for flow ranking in aggregator output. Returns flow indices
    ordered best-to-worst, or None if not found."""

    # ----- Runtime input propagation (opt-in) -----
    # When True, ``_ainfer`` / ``_infer`` mutate ``flow_configs[i]["input"]``
    # and ``predefined_sub_queries`` from the runtime ``inference_input`` before
    # delegating to BTA's worker spawning. Fixes the "predefined_sub_queries
    # snapshotted at construction" problem that prevents MFDual from being used
    # as a PTI planner (PTI calls ``planner.ainfer(state["current_input"])``;
    # without this flag, the runtime input is dropped and flows run against the
    # static placeholder strings declared in YAML).
    #
    # Default False preserves existing behavior for callers (tests, configs)
    # that rely on static ``flow_configs[i]["input"]`` values.
    #
    # Caveat: NOT compatible with checkpoint/resume — the mutation is per-call
    # and resume reconstruction of ``_cached_sub_queries`` may use stale values.
    # Acceptable for non-resume use cases (e.g. shallow real-CLI tests).
    propagate_runtime_input: bool = attrib(default=False)
    """Opt-in: rewrite each flow's input from runtime ``inference_input`` per call."""

    runtime_input_template: Optional[str] = attrib(default=None)
    """Optional Jinja template to wrap runtime input per-flow. Receives feed
    ``{"input": runtime_input, "flow_idx": i, "n_flows": N}``. Lets callers add
    flow-specific perspective seeds (e.g. flow 0 = "foundational angle",
    flow 1 = "incremental angle"). ``None`` = use runtime input verbatim for
    all flows. Only consulted when ``propagate_runtime_input=True``."""

    # ----- Upstream artifact injection (opt-in) -----
    # When True, MultiFlow injects formatted upstream artifacts into the
    # target inferencer's ``template_extra_feed["upstream_artifacts"]`` BEFORE
    # invoking it, AND returns just the per-flow ``input`` (or original_query
    # for the final aggregator) as ``inference_input``. This lets the target
    # inferencer's wrapper template reference ``{{ upstream_artifacts }}`` and
    # ``{{ input }}`` as separate slots — the upstream content lives in its
    # own slot, the original task lives in ``{{ input }}``.
    #
    # Affects both:
    #   - per-flow followup_inferencer (steps ≥ 1): upstream_artifacts =
    #     rendered followup_prompt (formatted your_prev + visible_plans)
    #   - final aggregator_inferencer: upstream_artifacts = rendered
    #     aggregator_prompt (formatted worker_plans)
    #
    # Default False preserves existing behavior (upstream content goes into
    # ``inference_input`` directly, ``{{ upstream_artifacts }}`` slot is
    # undefined in the wrapper template).
    inject_upstream_artifacts: bool = attrib(default=False)
    """Opt-in: push formatted upstream artifacts into target inferencer's
    ``template_extra_feed["upstream_artifacts"]`` per call, while letting
    ``inference_input`` carry the original task. Lets wrapper templates with a
    separate ``{{ upstream_artifacts }}`` slot work cleanly."""

    # Part C — Coordinated stop mode (added 2026-05-09; OPT-IN, default False).
    # When True, MFInferencer runs flows in lock-step: gather all flows' results
    # per step, then check for unanimous stop before proceeding. Default False
    # preserves today's per-flow independent execution via BTA's WorkGraph.
    # Implementation deferred to PR #2 (see plan §C); this attribute is the
    # Phase C1 scaffold so that downstream tooling (YAML configs, type hints)
    # can already reference the public surface area. When set to True with the
    # current scaffold, a NotImplementedError is raised to prevent silent
    # fallback to independent mode (no silent failure).
    coordinated_stop: bool = attrib(default=False)
    """Opt-in coordinated lock-step execution across flows. See plan §C.

    Backward-compatible ALIAS for ``cross_flow_sync`` (the visibility barrier). The
    unanimous *stop vote* — the second half of the original §C2 — is deferred to Part 2;
    today ``coordinated_stop=True`` installs the step barrier only (early finishers may
    still depart). See
    ``_docs/_plan/inferencer_architecture/INTEGRATED_cross_flow_coordination_plan.md``.
    """

    # Cross-flow step synchronization (the visibility barrier). When enabled, each flow's
    # round x+1 waits for ALL still-active flows' round x before reading peer artifacts —
    # closing the "(no output yet)" race (a fast flow building round01 before a slow peer
    # finished initial). Pure-asyncio rendezvous installed at the LWI round boundary; no
    # WorkGraph node promotion, all per-step checkpoint/resume preserved.
    cross_flow_sync: bool = attrib(default=False)
    """Opt-in cross-flow step barrier: round x+1 waits for all flows' round x."""

    # Class-level key under which the transient ``CrossFlowRendezvous`` is stashed on the
    # MFI node's ``scratch`` (ClassVar → ignored by attrs; never serialized).
    _RENDEZVOUS_SCRATCH_KEY: ClassVar[str] = "cross_flow_rendezvous"

    # Internal state — reset at the top of each ainfer/infer call.
    # Declared as init=False so attrs doesn't include them in __init__.
    # M-AF1/Part B: the two per-ATTEMPT cross-flow buffers are now compat-PROPERTIES
    # (``_latest_per_flow`` / ``_all_judgments``, defined below) over these name-backings.
    # Under a RunContext they read/write a per-run ``MultiFlowAttemptState`` at the MFI's
    # OWN ``ctx.node().attempt`` (resolved from a flow closure by walking UP the ancestor
    # paths in the shared store — the flow workers run in worker threads / fresh contexts
    # that do NOT inherit a parent ContextVar, but BTA RE-ENTERS the bridge via
    # ``flow.ainfer(run_context=flow_child)``, so ``active_run_context()`` inside a closure
    # is the flow's child ctx, a DESCENDANT of the MFI node). The backing is the legacy /
    # no-ctx store (never touched under a real ctx → a single shared MFI is concurrency-safe
    # across N RunContexts).
    _latest_per_flow_backing: Dict[int, Any] = attrib(factory=dict, init=False)
    # Per-flow output PATH backing — mirrors ``_latest_per_flow_backing`` (legacy/no-ctx
    # store); under a RunContext the live map is ``MultiFlowAttemptState.latest_per_flow_path``.
    _latest_per_flow_path_backing: Dict[int, Any] = attrib(factory=dict, init=False)
    _all_judgments_backing: List[Tuple[int, int, str]] = attrib(factory=list, init=False)
    # Transient cross-flow rendezvous (lock-step barrier) for the legacy/no-ctx path; under
    # a RunContext the live object lives on the MFI node's ``scratch`` (never serialized).
    _cross_flow_rendezvous_backing: Any = attrib(default=None, init=False)
    # M-AF1: the 4 per-CALL dispatch fields are now compat-PROPERTIES (defined below) over
    # these name-backings. Under a RunContext they read/write the per-run ``MultiFlowState``
    # at ``ctx.node().call``; the backing is the legacy/no-ctx store AND the legacy-mint
    # post-call-getter mirror (never written under a real caller ctx → concurrency-isolated).
    _last_winner_idx_backing: Optional[int] = attrib(default=None, init=False)
    _last_reviewer_alias_backing: Optional[str] = attrib(default=None, init=False)
    _last_fixer_alias_backing: Optional[str] = attrib(default=None, init=False)
    _last_ranking_backing: Optional[list] = attrib(default=None, init=False)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __attrs_post_init__(self):
        if not self.flow_configs:
            raise ValueError(
                "MultiFlowInferencer requires at least one entry in flow_configs"
            )
        for i, cfg in enumerate(self.flow_configs):
            if not isinstance(cfg, dict):
                raise TypeError(
                    f"flow_configs[{i}] must be a dict (got {type(cfg).__name__})"
                )
            if "input" not in cfg:
                # When ``propagate_runtime_input=True``, ``cfg["input"]`` is
                # overwritten per call by ``_apply_runtime_input_propagation``,
                # so the YAML value is dead weight — author can omit it and
                # we fill in an empty placeholder. When the flag is False,
                # the input IS the per-flow query and must be supplied.
                if self.propagate_runtime_input:
                    cfg["input"] = ""
                else:
                    raise ValueError(
                        f"flow_configs[{i}] is missing required key 'input' "
                        f"(set 'propagate_runtime_input: true' if you intend "
                        f"to fill it from the runtime inference_input per call)"
                    )
            # `iteration_judgment: true` is a named feature toggle that bundles
            # two coupled setups: (a) `end_condition` parses the LLM's
            # `iteration_judgment` JSON block to decide stop/continue, and
            # (b) the followup_inferencer's wrapper template renders that
            # JSON schema (so the LLM knows to emit it). Both halves are
            # inert without the other, so we set them as a coordinated
            # bundle. ``setdefault`` preserves any explicit user override.
            if cfg.get("iteration_judgment", False):
                # Lazy import to avoid module-load cycles.
                from agent_foundation.common.inferencers.flow_parsers import (
                    parse_decision_stop,
                )
                cfg.setdefault("end_condition", parse_decision_stop)
                followup = cfg.get("followup_inferencer")
                if followup is not None and hasattr(followup, "template_extra_feed"):
                    if followup.template_extra_feed is None:
                        followup.template_extra_feed = {}
                    followup.template_extra_feed.setdefault(
                        "include_iteration_judgment", True
                    )

        # Forbid conflicting BTA-level config — MultiFlow owns these fields.
        if self.breakdown_inferencer is not None:
            raise ValueError(
                "MultiFlowInferencer disables breakdown; do not pass breakdown_inferencer"
            )
        if self.predefined_sub_queries is not None:
            raise ValueError(
                "MultiFlowInferencer derives predefined_sub_queries from flow_configs; "
                "do not pass predefined_sub_queries directly"
            )

        # Initialize the legacy/no-ctx cross-flow backings (one None slot per flow).
        # Under a RunContext the live buffers are the per-run ``MultiFlowAttemptState``
        # at the MFI's node; these backings are the legacy fallback. Construction runs
        # with no active ctx, so seed the backings directly (and the compat-property
        # getter resolves them fresh each read — no by-reference capture needed).
        self._latest_per_flow_backing = {i: None for i in range(len(self.flow_configs))}
        self._latest_per_flow_path_backing = {i: None for i in range(len(self.flow_configs))}
        self._all_judgments_backing = []

        # Wire BTA fields from flow_configs. The flow-builder closure is an
        # arity-2 ``(sub_query, index) -> LWI`` callable; BTA's resolver dispatches
        # it via the else-arm (a plain callable, not a LazyConfigFactory).
        self.predefined_sub_queries = [c["input"] for c in self.flow_configs]
        # breakdown_inferencer stays None (default).
        if self.worker_inferencers is None:
            self.worker_inferencers = self._build_worker_factory()

        # Surgical 3-way aggregator-source guard: when the aggregator is
        # enabled, at least ONE of these must be configured, otherwise the
        # aggregation step has no instructions to give the model:
        #   (A) ``aggregator_prompt`` (str)         — legacy Jinja string.
        #   (B) ``aggregator_prompt_builder`` (Callable) — user-supplied
        #                                              prompt-feed builder.
        #   (C) ``aggregator_inferencer`` (Inferencer) — the leaf renders
        #         its own template via SLOT_DEFAULTS (modern Path B; this
        #         is what ``breakdown-multiflow-plan.yaml`` and friends use).
        # Raising here converts a previously-silent misconfiguration (the
        # aggregator step would run with an empty / unset prompt and emit
        # garbage) into an actionable error at construction time.
        if not self.disable_aggregator:
            has_prompt_str     = self.aggregator_prompt is not None
            has_prompt_builder = self.aggregator_prompt_builder is not None
            has_inferencer     = self.aggregator_inferencer is not None
            if not (has_prompt_str or has_prompt_builder or has_inferencer):
                raise ValueError(
                    "MultiFlowInferencer aggregator is enabled but no prompt "
                    "source is configured. Provide ONE of: "
                    "(1) aggregator_prompt (str) — legacy Jinja template; "
                    "(2) aggregator_prompt_builder (Callable) — user-supplied "
                    "builder; "
                    "(3) aggregator_inferencer (InferencerBase) — leaf with "
                    "its own SLOT_DEFAULTS-resolved template (recommended). "
                    "Otherwise set disable_aggregator=True to skip aggregation."
                )

        # Install default aggregator prompt builder when the user supplied a
        # template but no custom builder. BTA's own builder (if any) wins.
        if (
            self.aggregator_prompt is not None
            and self.aggregator_prompt_builder is None
            and not self.disable_aggregator
        ):
            self.aggregator_prompt_builder = self._make_default_aggregator_prompt_builder()

        # Defer to BTA for workspace, subgraph_registry, and the rest.
        super().__attrs_post_init__()

        # M-AF1: per-run dispatch state lives in a typed ``MultiFlowState`` at
        # ``ctx.node().call`` (seeded once per call by ``_init_call_state``); default the
        # factory so the dispatch compat-properties resolve against a typed node. An
        # explicit caller-supplied ``state_factory`` wins.
        if self.state_factory is None:
            from agent_foundation.common.inferencers.run_context import MultiFlowState

            self.state_factory = lambda _inp: MultiFlowState()

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_to_text(value: Any) -> str:
        """Best-effort extract a textual representation from a step result.

        CLI inferencers (e.g., ``ClaudeCodeCliInferencer``) return structured
        objects (``TerminalInferencerResponse`` or similar) or dicts with an
        ``"output"`` key. We unwrap to the textual content so prompt templates
        render the actual response rather than a Python repr of the wrapper.

        Field preference (first non-empty string wins):
          1. ``output`` — the cleaned response text
          2. ``raw_output`` — fallback when ``output`` got over-filtered to
             empty (we've seen Claude CLI occasionally do this)
          3. ``str(value)`` as a last resort

        Plain strings are returned as-is; ``None`` becomes ``""``.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace")
            except Exception:
                return repr(value)

        def _extract(name):
            if isinstance(value, dict):
                v = value.get(name)
            else:
                v = getattr(value, name, None)
            return v if isinstance(v, str) else None

        for field in ("output", "raw_output"):
            text = _extract(field)
            if text:  # non-empty string wins
                return text

        # Empty-string output fields → return empty rather than repr().
        # An InferencerResponse with output="" really means "no text".
        if (
            isinstance(value, dict) and "output" in value
        ) or hasattr(value, "output"):
            return ""
        return str(value)

    def _render_template(self, template: str, feed: Dict[str, Any]) -> str:
        """Render a template (key or raw Jinja2) with the given feed.

        When ``prompt_formatter`` is set we treat ``template`` as a key and
        delegate; otherwise we render ``template`` as a raw Jinja2 string.
        """
        if self.prompt_formatter is not None:
            try:
                return self.prompt_formatter(template_key=template, feed=feed)
            except TypeError:
                # Permit a generic callable that accepts (template, feed).
                return self.prompt_formatter(template, feed)

        from jinja2 import Template  # imported lazily; jinja2 is already a transitive dep

        return Template(template).render(**feed)

    # ------------------------------------------------------------------
    # Workspace propagation override
    # ------------------------------------------------------------------

    def _propagate_workspace_to_children(self, parent_workspace):
        """MultiFlow override: delegates flow_configs propagation to LWI.

        Flow-internal inferencers (initial_inferencer, followup_inferencer)
        receive workspaces from LWI's ``_propagate_workspace_to_children``
        when BTA assigns each LWI worker its workspace. MultiFlow only
        propagates direct attrs (aggregator_inferencer, etc.) via base.
        """
        super()._propagate_workspace_to_children(parent_workspace)

    # ------------------------------------------------------------------
    # Worker naming override (Fix #6: flow_N_workflow instead of worker_N)
    # ------------------------------------------------------------------

    def _worker_child_name(self, index: int) -> str:
        """MultiFlow override: name workers ``flow_N`` instead of ``worker_N``.

        Each flow's LWI worker owns its entire sub-tree::

            base_inferencer/
            ├── flow_0/               # LWI root (owns children/initial/, round01/, ...)
            ├── flow_1/
            └── aggregator/
        """
        from agent_foundation.common.inferencers.inferencer_workspace import indexed_child_name
        return indexed_child_name("flow", index)

    def _is_worker_child_name(self, name: str) -> bool:
        """Match ``flow_NN`` names produced by the override above."""
        if not name.startswith("flow_"):
            return False
        return name[len("flow_"):].isdigit()

    # ------------------------------------------------------------------
    # Per-flow visibility resolution
    # ------------------------------------------------------------------

    def _resolve_flow_visibility(self, flow_idx: int) -> List[int]:
        cfg = self.flow_configs[flow_idx]
        spec = cfg.get("visible_flows", self.visible_flows)
        return _resolve_visible_indices(flow_idx, len(self.flow_configs), spec)

    # ------------------------------------------------------------------
    # Followup-input formatting (per-step iteration default)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Part A — Path-aware peer/own-flow visibility (added 2026-05-09)
    # ------------------------------------------------------------------

    def _resolve_flow_output_path(
        self, flow_idx: int, path_snapshot: Optional[Dict[int, Any]] = None
    ) -> Optional[str]:
        """Resolve the on-disk path of flow ``flow_idx``'s most-recent output.

        Returns the deliverable path if the flow has a finalized deliverable,
        otherwise the working output path; ``None`` if neither is available
        (e.g., the flow hasn't run yet or its workspace isn't on disk).

        PRIMARY (M7): read the per-run ``_latest_per_flow_path`` map captured from the LIVE
        run-context as each flow produced output (see ``_wrapped_dynamic_input_builder`` /
        ``_last_dynamic_step_text``). This is the only correct source under ctx/workspace
        decoupling — the flow-config leaf instances' ``_workspace`` is NOT the live per-run
        workspace (workspaces are published into the run-context, not onto the shared leaf),
        so the legacy resolution below silently returned ``None`` for every followup step.

        LEGACY FALLBACK (no ctx / never captured): the original resolution off the
        flow-config inferencer instances' ``_workspace``, kept byte-identical so existing
        no-ctx unit tests (which set ``inferencer._workspace`` directly) still pass.

        Delegates to the shared ``resolve_canonical_output_path`` helper for canonical
        3-tier resolution (Tier 1: deliverables / Tier 2: outputs/output.md / Tier 3: None).
        """
        # PRIMARY: ctx-captured per-run path (own = prior step; peer = last step).
        # Under cross-flow coordination a FROZEN round N-1 snapshot is passed in so peer
        # paths are lock-step consistent (not a fast peer's already-advanced round N path);
        # otherwise read the live per-run map.
        source = path_snapshot if path_snapshot is not None else (self._latest_per_flow_path or {})
        tracked = source.get(flow_idx)
        if tracked and os.path.exists(tracked):
            return tracked

        # LEGACY FALLBACK: resolve off the flow-config inferencer instances' _workspace.
        from agent_foundation.common.inferencers.inferencer_workspace import (  # noqa: E501
            resolve_canonical_output_path,
        )

        try:
            cfg = self.flow_configs[flow_idx]
        except (IndexError, KeyError, TypeError):
            return None
        # Try followup_inferencer first (most recent state); fall back to initial
        for key in ("followup_inferencer", "initial_inferencer"):
            inferencer = cfg.get(key) if isinstance(cfg, dict) else None
            if inferencer is None:
                continue
            path = resolve_canonical_output_path(
                getattr(inferencer, "_workspace", None),
                deliverables_fallback="first_match",
            )
            if path is not None:
                return path
        return None

    # ------------------------------------------------------------------
    # Followup-input formatting (per-step iteration default)
    # ------------------------------------------------------------------

    def _format_followup_input(
        self,
        *,
        your_prev: str,
        flow_idx: int,
        step_idx: int,
        visible_plans: Dict[int, Optional[str]],
        peer_path_snapshot: Optional[Dict[int, Any]] = None,
    ) -> str:
        """Default formatter for per-step iteration input (used at step >= 1).

        Composes the flow's previous output and visible peers' latest outputs
        into a single text blob. Used as either the LWI's ``inference_input``
        directly (legacy passthrough) or pushed into the followup_inferencer's
        ``template_extra_feed["upstream_artifacts"]`` slot
        (``inject_upstream_artifacts=True``); the original task is supplied by
        the wrapper's ``{{ input }}`` slot in the latter mode, so it is NOT
        included here.

        Subclasses can override; callers can bypass via
        ``cfg["followup_prompt"]`` (legacy template path) or
        ``cfg["dynamic_input_builder"]`` (full callback).
        """
        # Part A: include own-flow's prior output so the LLM can refine from
        # the full artifact, not just the (possibly summarized) text excerpt.
        # For flows WITH local access: pass a file path (the CLI reads it).
        # For flows WITHOUT local access: inline the full file content.
        own_path = self._resolve_flow_output_path(flow_idx, path_snapshot=peer_path_snapshot)
        flow_inf = self.flow_configs[flow_idx].get("followup_inferencer")
        flow_has_local = getattr(flow_inf, "has_local_access", False) if flow_inf else False

        own_prev_text = your_prev
        if own_path and not flow_has_local and os.path.isfile(own_path):
            try:
                own_prev_text = open(own_path, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                pass

        own_block = _FOLLOWUP_OWN_PREVIOUS_HEADER.format(
            flow_idx=flow_idx,
            prev_step_idx=step_idx - 1,
            your_prev=own_prev_text,
        )
        if own_path and flow_has_local:
            own_block += (
                f"\n\nYour previous full artifact is on disk at:\n"
                f"  `{own_path}`\n"
                f"You may re-read or copy this file as a starting point for incremental edits."
            )
        parts = [own_block]
        if visible_plans:
            peer_segments = []
            for idx, plan in visible_plans.items():
                peer_path = self._resolve_flow_output_path(idx, path_snapshot=peer_path_snapshot)
                # For non-local flows: inline the full peer artifact content
                # instead of a useless path reference.
                peer_text = plan or _FOLLOWUP_PEER_EMPTY_PLACEHOLDER
                if peer_path and not flow_has_local and os.path.isfile(peer_path):
                    try:
                        peer_text = open(peer_path, encoding="utf-8").read()
                    except (OSError, UnicodeDecodeError):
                        pass
                segment = (
                    f"{_FOLLOWUP_PEER_BLOCK_HEADER.format(idx=idx)}\n"
                    f"{peer_text}"
                )
                if peer_path and flow_has_local:
                    segment += (
                        f"\n\nThe full peer artifact is available at:\n"
                        f"  `{peer_path}`"
                    )
                peer_segments.append(segment)
            peer_blocks = "\n\n".join(peer_segments)
            parts.append(f"{_FOLLOWUP_PEER_INTRO}\n\n{peer_blocks}")
        else:
            parts.append(_FOLLOWUP_NO_PEERS)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Per-call child template-feed publishing (Decision 5 — prompt-feed
    # writes are runtime state, not instance state)
    # ------------------------------------------------------------------

    def _publish_child_template_feed(
        self, child_inf: Any, child_slot: str, feed: Dict[str, Any]
    ) -> bool:
        """Pass per-call/per-flow data (e.g. ``upstream_artifacts``) into a SHARED
        child leaf's wrapper template WITHOUT mutating the child's instance
        ``template_extra_feed`` — the Decision-5 / state-separation requirement.

        Under an active RunContext: publish ``feed`` into the child's run-context
        handle (the leaf merges it at render time via the ctx walk-up in
        :func:`TemplatedInferencerBase._resolve_ctx_feed_override`). The instance
        ``template_extra_feed`` is left untouched, so a single shared MFI across N
        concurrent ctxs — or a single followup/aggregator child reused across N
        flows — never clobbers a concurrent call's feed.

        Legacy / no active ctx: fall back to writing the child instance's
        ``template_extra_feed`` exactly as before — **byte-identical** with the
        pre-virtualization behavior.

        ``child_slot`` is the deterministic ctx slot the orchestrator threads as
        ``run_context=`` into the child's own ``ainfer`` (e.g. ``"aggregator"``).
        For children whose exact child-ctx slot the publisher cannot know (the
        per-flow followup runs under LWI's ``step_{i}`` node), pass the orchestrator's
        OWN active path by using ``child_slot=None`` — the value is published at the
        active node and the descendant leaf finds it by walking up.

        Returns ``True`` when published via the ctx channel (no instance write),
        ``False`` when it fell back to the legacy instance write.
        """
        from agent_foundation.common.inferencers.run_context import (
            active_run_context,
        )
        from agent_foundation.common.inferencers.templated_inferencer_base import (
            TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE,
        )

        ctx = active_run_context()
        if ctx is not None:
            target_ctx = ctx.child(child_slot) if child_slot else ctx
            existing = target_ctx.handles.get(
                TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE, None
            )
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged.update(feed)
            target_ctx.handles.set(TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE, merged)
            return True

        # Legacy / no-ctx: byte-identical instance write.
        if child_inf is not None and hasattr(child_inf, "template_extra_feed"):
            if child_inf.template_extra_feed is None:
                child_inf.template_extra_feed = {}
            child_inf.template_extra_feed.update(feed)
        return False

    # ------------------------------------------------------------------
    # Aggregator prompt builder (default, installed when aggregator_prompt is set)
    # ------------------------------------------------------------------

    def _make_default_aggregator_prompt_builder(self) -> Callable:
        """Return a BTA-compatible ``aggregator_prompt_builder`` that renders
        ``self.aggregator_prompt`` against a structured feed.
        """
        outer = self

        def _builder(
            worker_results,
            original_query: str = "",
            worker_output_paths=None,
            bta=None,  # noqa: ARG001 — accepted for BTA's call signature
        ) -> str:
            n = len(outer.flow_configs)
            # Prefer _latest_per_flow (keyed by declaration index, populated
            # by each worker's final response_builder) over BTA's positional
            # worker_results — BTA orders results by completion, which scrambles
            # per-flow labels when one flow fails fast.
            worker_plans = {}
            results_list = list(worker_results) if worker_results is not None else []
            for i in range(n):
                text = outer._latest_per_flow.get(i)
                if text is None:
                    # Fallback: positional worker_result (legacy / when a
                    # flow's response_builder didn't publish to _latest_per_flow)
                    fallback = results_list[i] if i < len(results_list) else None
                    text = outer._coerce_to_text(fallback)
                worker_plans[i] = text

            all_judgments_summary = ""
            if outer._all_judgments:
                all_judgments_summary = "\n".join(
                    f"Flow {fi} step {si}: {judg}"
                    for (fi, si, judg) in outer._all_judgments
                )

            feed = {
                "input": original_query or "",
                "worker_plans": worker_plans,
                "all_judgments_summary": all_judgments_summary,
                # NOTE: worker_output_paths kept in builder signature for
                # BTA call-compatibility but not fed into the template
                # because the legacy default template didn't consume it.
                # User-supplied ``aggregator_prompt`` strings that DO want
                # output paths can use ``aggregator_prompt_builder`` instead.
            }

            # ``_make_default_aggregator_prompt_builder`` is gated on
            # ``aggregator_prompt is not None`` (see ``__attrs_post_init__``),
            # so this attribute is always a non-None string here.
            template = outer.aggregator_prompt
            rendered = outer._render_template(template, feed)

            # Upstream artifact injection: when enabled, push the rendered
            # peer-output formatting into the aggregator inferencer's extra
            # feed under ``upstream_artifacts`` and return JUST the original
            # query as ``inference_input``. The aggregator's wrapper template
            # then has ``{{ input }} = original_query`` and
            # ``{{ upstream_artifacts }} = formatted peers`` as separate slots.
            #
            # Without this flag, the entire rendered string is the
            # ``inference_input`` and ``{{ upstream_artifacts }}`` is undefined.
            if outer.inject_upstream_artifacts and outer.aggregator_inferencer is not None:
                # Decision 5: pass the per-call upstream artifacts to the
                # aggregator's wrapper template via the CALL-SCOPED ctx channel
                # (published into the aggregator's child run-context handle), NOT
                # by mutating the shared aggregator instance's
                # ``template_extra_feed`` — which would clobber concurrent calls
                # under shared-instance reuse. BTA invokes the aggregator with
                # ``run_context=self._rc_child("aggregator")`` (this builder runs
                # under BTA's active ctx), so the ``"aggregator"`` slot is the
                # exact child node the leaf renders under. Legacy / no-ctx falls
                # back to the instance write (byte-identical).
                #
                # NOTE: Per-flow output paths are embedded inline within
                # ``rendered`` upstream_artifacts text. No structured
                # ``worker_output_paths`` variable is injected because no
                # MFI aggregator template currently consumes it.
                outer._publish_child_template_feed(
                    outer.aggregator_inferencer,
                    "aggregator",
                    {"upstream_artifacts": rendered},
                )
                return original_query or ""

            return rendered

        return _builder

    # ------------------------------------------------------------------
    # Worker factory (auto-built from flow_configs)
    # ------------------------------------------------------------------

    def _build_worker_factory(self) -> Callable:
        """Return ``(sub_query, index) -> LinearWorkflowInferencer`` for BTA."""
        outer = self
        configs = self.flow_configs

        def _factory(sub_query, index):
            cfg = configs[index]
            visible = outer._resolve_flow_visibility(index)
            user_dynamic_builder = cfg.get("dynamic_input_builder")
            initial_template = cfg.get("initial_prompt")
            followup_template = cfg.get("followup_prompt") or outer.multiflow_followup_prompt

            # Wrapped dynamic_input_builder for steps ≥ 1.
            #
            # Always:
            #   1. publish this flow's latest output into outer._latest_per_flow
            #   2. expose visible peers via state["visible_flow_outputs"]
            #   3. (optional) collect <Judgment>-style metadata via judgment_parser
            #
            # Then either delegate to the user's builder, render a template,
            # or pass prev_result through unchanged (legacy behaviour).
            async def _wrapped_dynamic_input_builder(state, prev_result):
                # Store the textual form of prev_result so cross-flow visibility
                # surfaces real text, not Python wrapper reprs.
                prev_text = outer._coerce_to_text(prev_result)
                outer._latest_per_flow[index] = prev_text
                # Capture this flow's PRIOR-STEP output path from the LIVE run-context (the
                # active ctx here IS the flow's LWI node, whose ``.workspace`` is the flow
                # workspace) so the followup own_path/peer_path resolves from a real per-run
                # path — NOT the stale leaf-instance ``_workspace`` (M7). Target the prior
                # step's ``children/<name>/outputs/output.md`` (what ``prev_text`` summarizes),
                # NOT the surfaced flow deliverable (written only at flow completion).
                _step_idx_now = state.get("dynamic_step_count", 0)
                if _step_idx_now >= 1:
                    from agent_foundation.common.inferencers.run_context import (
                        active_run_context,
                    )
                    from agent_foundation.common.inferencers.inferencer_workspace import (
                        resolve_canonical_output_path,
                    )
                    _ctx_now = active_run_context()
                    if _ctx_now is not None and _ctx_now.workspace is not None:
                        _prev_name = LinearWorkflowInferencer._dynamic_child_name(
                            _step_idx_now - 1,
                            (state or {}).get("consensus_iteration_id", 0),
                        )
                        outer._latest_per_flow_path[index] = resolve_canonical_output_path(
                            _ctx_now.workspace.child(_prev_name),
                            deliverables_fallback="none",
                        )
                # --- Cross-flow step barrier (lock-step coordination) ---------------
                # This flow has now PUBLISHED its prior-round output (text @
                # ``_latest_per_flow[index]`` and path @ ``_latest_per_flow_path[index]``
                # above). Before READING peers below, block until every still-active peer
                # has likewise published its prior round. Ordering invariant — and the
                # whole fix for the "(no output yet)" race: publish -> barrier -> read.
                # A flow that stopped/crashed has ``leave()``'d the rendezvous (see the
                # worker wrapper), so it never hangs survivors here.
                # Peer-view source: the live buffer by default. Under coordination the
                # barrier returns a FROZEN round N-1 snapshot (text + path) captured the
                # instant all active flows had published — read THAT instead of the live
                # buffer, so a fast peer racing ahead into round N can't contaminate this
                # flow's read (lock-step consistency; every flow sees the same round N-1).
                _peer_text = outer._latest_per_flow
                _peer_path_snapshot = None
                if outer._coordination_enabled:
                    _rdv = outer._resolve_rendezvous()
                    if _rdv is not None:
                        _snap = await _rdv.arrive_and_wait(
                            index,
                            snapshot_fn=lambda: (
                                dict(outer._latest_per_flow),
                                dict(outer._latest_per_flow_path),
                            ),
                        )
                        if _snap is not None:
                            _peer_text, _peer_path_snapshot = _snap
                visible_plans = {
                    i: _peer_text.get(i) for i in visible if i != index
                }
                state["visible_flow_outputs"] = visible_plans

                if outer.judgment_parser is not None:
                    try:
                        judg = outer.judgment_parser(prev_text)
                    except Exception as exc:  # noqa: BLE001 — best-effort; warn
                        _logger.warning(
                            "MultiFlow judgment_parser raised on flow %d: %s",
                            index,
                            exc,
                        )
                    else:
                        if judg:
                            outer._all_judgments.append(
                                (index, state.get("dynamic_step_count", 0), str(judg))
                            )

                if user_dynamic_builder is not None:
                    # call_maybe_async: a sync user builder runs inline; a future async one
                    # is awaited (the wrapper is async now, so a bare call would leak an
                    # un-awaited coroutine).
                    return await call_maybe_async(user_dynamic_builder, state, prev_result)

                step_idx = state.get("dynamic_step_count", 0)

                # Compute the upstream-artifacts text. Three modes, in priority:
                #   1. ``followup_template`` is set → render template (legacy
                #      escape hatch; lets callers customize per-step formatting
                #      via Jinja).
                #   2. ``inject_upstream_artifacts`` is True → built-in Python
                #      formatter ``_format_followup_input`` (default for the
                #      injection-into-extra-feed flow).
                #   3. neither → legacy LWI passthrough (pass prev_result
                #      through unchanged); preserves behavior for tests that
                #      configure neither.
                if followup_template is not None:
                    feed = {
                        "input": cfg.get("input", ""),
                        "your_prev": prev_text,
                        "visible_plans": visible_plans,
                        "all_plans": {
                            i: _peer_text.get(i) for i in range(len(configs))
                        },
                        "flow_idx": index,
                        "step_idx": step_idx,
                    }
                    upstream_text = outer._render_template(followup_template, feed)
                elif outer.inject_upstream_artifacts:
                    upstream_text = outer._format_followup_input(
                        your_prev=prev_text,
                        flow_idx=index,
                        step_idx=step_idx,
                        visible_plans=visible_plans,
                        peer_path_snapshot=_peer_path_snapshot,
                    )
                else:
                    return prev_result

                # Upstream artifact injection: when enabled, push the formatted
                # text (your_prev + visible_plans) into the followup_inferencer's
                # extra feed under ``upstream_artifacts`` and return the per-flow
                # runtime input as ``inference_input``. The followup_inferencer's
                # wrapper template then has ``{{ input }} = cfg["input"]`` (the
                # runtime task) and ``{{ upstream_artifacts }} = formatted peers``
                # as separate slots. Without this flag, the full formatted output
                # is the inference_input and ``{{ upstream_artifacts }}`` is
                # undefined.
                #
                # Decision 5: route the per-FLOW upstream artifacts through the
                # CALL-SCOPED ctx channel instead of mutating the shared followup
                # instance's ``template_extra_feed`` (which would clobber other
                # flows reusing the same followup leaf, or concurrent ctxs sharing
                # one MFI). This builder runs under the flow's LWI ctx; LWI invokes
                # the followup with ``run_context=self._rc_child("step_{i}")`` (a
                # DESCENDANT of that node), so we publish at the LWI's active node
                # (``child_slot=None``) and the followup leaf finds it by walking up
                # at render time. Legacy / no-ctx falls back to the instance write
                # (byte-identical).
                if outer.inject_upstream_artifacts:
                    followup_inf = cfg.get("followup_inferencer")
                    outer._publish_child_template_feed(
                        followup_inf,
                        None,
                        {"upstream_artifacts": upstream_text},
                    )
                    return cfg.get("input", "")

                return upstream_text

            # When an initial_prompt template is set, render it lazily at the
            # start of the LWI run via ``initial_state_factory``. The factory
            # receives the LWI's actual ``inference_input`` (= ``sub_query``
            # passed in by BTA), so the template can reference the per-flow
            # input via ``{{ input }}``.
            initial_state_factory: Optional[Callable] = None
            if initial_template is not None:
                def _initial_state_factory(
                    inference_input,
                    _cfg=cfg,
                    _idx=index,
                    _tmpl=initial_template,
                ):
                    feed = {
                        "input": inference_input,
                        "flow_idx": _idx,
                        "step_idx": 0,
                    }
                    return {"original_input": outer._render_template(_tmpl, feed)}

                initial_state_factory = _initial_state_factory

            # Default response_builder for the LWI worker: return the LAST
            # dynamic step's result. Without this, LWI returns the full state
            # dict, which BTA's aggregator then receives as a worker plan —
            # nonsensical for downstream consumers expecting a single output
            # per flow. Per-flow override via cfg["response_builder"] wins.
            cfg_response_builder = cfg.get("response_builder")
            if cfg_response_builder is None:
                def _last_dynamic_step_text(
                    state,
                    _idx=index,
                    _coerce=outer._coerce_to_text,
                    _outer=outer,
                ):
                    results = (state or {}).get("dynamic_step_results") or []
                    text = "" if not results else _coerce(results[-1])
                    # Publish the FINAL output to _latest_per_flow keyed by
                    # flow_idx so the aggregator builder can read in the
                    # declaration order — bypassing BTA's worker_results
                    # ordering, which reflects WORKER COMPLETION ORDER (a
                    # fast-failing flow ends up at index 0, scrambling the
                    # aggregator's per-flow labels).
                    _outer._latest_per_flow[_idx] = text
                    # Mirror the LAST-step output PATH so PEER flows (sibling nodes
                    # unreachable from another flow's ctx ancestor walk) expose their on-disk
                    # path via this shared per-attempt map, exactly as their TEXT does above.
                    # Resolve the last step's children/<name>/outputs/output.md (reliably on
                    # disk), mirroring the own-flow prior-step capture.
                    _n_steps = len((state or {}).get("dynamic_step_results") or [])
                    if _n_steps >= 1:
                        from agent_foundation.common.inferencers.run_context import (
                            active_run_context,
                        )
                        from agent_foundation.common.inferencers.inferencer_workspace import (
                            resolve_canonical_output_path,
                        )
                        _ctx_done = active_run_context()
                        if _ctx_done is not None and _ctx_done.workspace is not None:
                            _last_name = LinearWorkflowInferencer._dynamic_child_name(
                                _n_steps - 1,
                                (state or {}).get("consensus_iteration_id", 0),
                            )
                            _outer._latest_per_flow_path[_idx] = (
                                resolve_canonical_output_path(
                                    _ctx_done.workspace.child(_last_name),
                                    deliverables_fallback="none",
                                )
                            )
                    return text
                cfg_response_builder = _last_dynamic_step_text

            _initial = cfg.get("initial_inferencer")
            _worker = LinearWorkflowInferencer(
                dynamic_mode=True,
                default_initial_inferencer=_initial,
                default_followup_inferencer=cfg.get("followup_inferencer"),
                output_path=getattr(_initial, "output_path", None),
                end_condition=cfg.get("end_condition"),
                max_dynamic_steps=cfg.get("max_dynamic_steps", 10),
                inferencer_factory=cfg.get("inferencer_factory"),
                dynamic_input_builder=_wrapped_dynamic_input_builder,
                response_builder=cfg_response_builder,
                initial_state_factory=initial_state_factory,
            )
            if outer._coordination_enabled:
                # Deadlock-safe deregister (the barrier's load-bearing invariant: every
                # seeded participant eventually leaves). When THIS flow's loop ends — normal
                # stop, max-steps, OR an exception — depart the rendezvous so any peer
                # blocked at a later round barrier is released instead of hanging. Wrapping
                # ``_ainfer`` guarantees once-per-flow on ANY exit (the ``finally`` runs on
                # success and on raise). ``leave`` is idempotent, so a WorkGraph node retry
                # that re-runs ``_ainfer`` (and departs again) is harmless. The rdv ref is
                # captured at ENTRY (ctx active, rdv already seeded by
                # ``_reset_cross_flow_state`` before workers launched) and used in the
                # finally, where the flow's ctx is still active.
                #
                # Also tag the worker with its flow index so BTA's worker_fn can run a
                # belt-and-suspenders depart in its OWN finally — covering the paths that
                # satisfy a worker WITHOUT ever entering ``_ainfer`` (backup-resume cache
                # hit / cancel at the worker boundary). leave() is idempotent, so the two
                # departs never conflict.
                _worker._cross_flow_index = index
                _orig_ainfer = _worker._ainfer

                async def _ainfer_with_leave(*a, _idx=index, _orig=_orig_ainfer, **kw):
                    _rdv = outer._resolve_rendezvous()
                    try:
                        return await _orig(*a, **kw)
                    finally:
                        if _rdv is not None:
                            await _rdv.leave(_idx)

                _worker._ainfer = _ainfer_with_leave
            return _worker

        return _factory

    # ------------------------------------------------------------------
    # Inference entry points (override only to manage cross-flow state)
    # ------------------------------------------------------------------

    def _reset_cross_flow_state(self) -> None:
        """Reset PER-ATTEMPT cross-flow worker state. Called at the top of
        ``_ainfer`` / ``_infer``, so it runs on every retry attempt.

        Note: dispatch state (``_last_winner_idx``, ``_last_reviewer_alias``,
        ``_last_fixer_alias``) is NOT reset here — see
        :meth:`_reset_dispatch_state_for_call`. Splitting the reset by lifetime
        prevents a successful early-attempt's parsed winner from being
        clobbered by a malformed retry attempt.

        M-AF1/Part B: under a RunContext, install a FRESH ``MultiFlowAttemptState`` on the
        MFI's OWN node (``ctx.node().attempt``) — the cross-flow buffers are now per-run
        attempt state, so a shared MFI across N RunContexts keeps each run's buffers
        isolated. The flow closures (running under flow child ctxs) resolve this same node
        by walking UP their ancestor paths. With no ctx (legacy), reset the instance
        backings as before (byte-identical).
        """
        from agent_foundation.common.inferencers.run_context import (
            MultiFlowAttemptState,
            active_run_context,
        )

        fresh_latest = {i: None for i in range(len(self.flow_configs))}
        fresh_paths = {i: None for i in range(len(self.flow_configs))}
        fresh_judgments: List[Tuple[int, int, str]] = []

        ctx = active_run_context()
        if ctx is not None:
            # Fresh attempt state on the MFI's own node (claims it as creator so the
            # walk-up from flow closures finds THIS node's ``MultiFlowAttemptState``).
            node = ctx.node(creator=(type(self).__qualname__, ctx.path))
            node.attempt = MultiFlowAttemptState(
                latest_per_flow=fresh_latest,
                latest_per_flow_path=fresh_paths,
                judgments=fresh_judgments,
            )
            if self._coordination_enabled:
                # Transient lock-step barrier on the MFI's OWN node. ``scratch`` is excluded
                # from ``to_json`` (never serialized); the rendezvous is re-created fresh on
                # every attempt/resume — correct by construction (it is per-process-run).
                node.scratch[self._RENDEZVOUS_SCRATCH_KEY] = self._build_rendezvous()
            if ctx.legacy_mint:
                # Legacy-minted bare call: the store is discarded on exit, but post-call
                # getters (ctx is None) read the instance backing. Make the backing the
                # SAME objects the attempt state holds, so the closures' in-place mutations
                # (during the call, via the walk-up) ALSO land on the backing -> the
                # post-call ``mfi._all_judgments`` / ``_latest_per_flow`` reads still work.
                # Never executed under a real (non-legacy) ctx -> the shared backing stays
                # pristine -> a shared MFI across N concurrent RunContexts is isolated.
                self._latest_per_flow_backing = fresh_latest
                self._latest_per_flow_path_backing = fresh_paths
                self._all_judgments_backing = fresh_judgments
        else:
            # Legacy / no-ctx: reset the instance backings in place.
            self._latest_per_flow_backing.clear()
            self._latest_per_flow_backing.update(fresh_latest)
            self._latest_per_flow_path_backing.clear()
            self._latest_per_flow_path_backing.update(fresh_paths)
            self._all_judgments_backing.clear()
            if self._coordination_enabled:
                self._cross_flow_rendezvous_backing = self._build_rendezvous()

    def _reset_dispatch_state_for_call(self) -> None:
        """Reset PER-CALL dispatch state. Called once per top-level
        ``ainfer`` / ``infer`` invocation (not per retry attempt) so that
        a winner parsed by an early successful retry attempt survives a
        later malformed retry of the same call.
        """
        self._last_winner_idx = None
        self._last_reviewer_alias = None
        self._last_fixer_alias = None
        self._last_ranking = None

    # ------------------------------------------------------------------
    # M-AF1: dispatch-state compat-properties (per-run ctx.node().call vs backing)
    # ------------------------------------------------------------------
    def _dispatch_get(self, field: str, backing: Any) -> Any:
        """Read a dispatch field from the per-run ``MultiFlowState`` at ``ctx.node().call``
        when a context is active, else the instance backing (legacy / post-call getter).

        Uses ``creator=None`` (read-only) so reading another inferencer's node — e.g. an
        outer MFDual reading ``mfi._last_winner_idx`` while MFDual's own node is active —
        never raises a creator collision; a non-``MultiFlowState`` ``call`` falls through
        to the backing.
        """
        from agent_foundation.common.inferencers.run_context import (
            MultiFlowState,
            active_run_context,
        )

        ctx = active_run_context()
        if ctx is not None:
            call = ctx.node().call
            if isinstance(call, MultiFlowState):
                return getattr(call, field)
        return backing

    def _dispatch_set(self, field: str, value: Any, backing_attr: str) -> None:
        """Write a dispatch field. ALWAYS writes the per-run ``MultiFlowState`` at
        ``ctx.node().call`` when a context is active (the in-loop MFDual->MFI handoff reads
        that node, even under a legacy-minted bare call); ADDITIONALLY mirrors to the
        instance backing under ``ctx.legacy_mint`` (or no ctx) for the post-call getter.
        Never writes the backing under a real (non-legacy) ctx -> concurrency-isolated.
        """
        from agent_foundation.common.inferencers.run_context import (
            MultiFlowState,
            active_run_context,
        )

        ctx = active_run_context()
        if ctx is not None:
            node = ctx.node(creator=(type(self).__qualname__, ctx.path))
            if not isinstance(node.call, MultiFlowState):
                node.call = MultiFlowState()
            setattr(node.call, field, value)
            if ctx.legacy_mint:
                setattr(self, backing_attr, value)
        else:
            setattr(self, backing_attr, value)

    @property
    def _last_winner_idx(self) -> Optional[int]:
        return self._dispatch_get("winner_idx", self._last_winner_idx_backing)

    @_last_winner_idx.setter
    def _last_winner_idx(self, value: Optional[int]) -> None:
        self._dispatch_set("winner_idx", value, "_last_winner_idx_backing")

    @property
    def _last_reviewer_alias(self) -> Optional[str]:
        return self._dispatch_get("reviewer_alias", self._last_reviewer_alias_backing)

    @_last_reviewer_alias.setter
    def _last_reviewer_alias(self, value: Optional[str]) -> None:
        self._dispatch_set("reviewer_alias", value, "_last_reviewer_alias_backing")

    @property
    def _last_fixer_alias(self) -> Optional[str]:
        return self._dispatch_get("fixer_alias", self._last_fixer_alias_backing)

    @_last_fixer_alias.setter
    def _last_fixer_alias(self, value: Optional[str]) -> None:
        self._dispatch_set("fixer_alias", value, "_last_fixer_alias_backing")

    @property
    def _last_ranking(self) -> Optional[list]:
        return self._dispatch_get("ranking", self._last_ranking_backing)

    @_last_ranking.setter
    def _last_ranking(self, value: Optional[list]) -> None:
        self._dispatch_set("ranking", value, "_last_ranking_backing")

    # ------------------------------------------------------------------
    # M-AF1/Part B: cross-flow attempt-state compat-properties
    # (per-run ``MultiFlowAttemptState`` at the MFI's node vs the instance backing)
    # ------------------------------------------------------------------
    def _resolve_attempt_node(self) -> Any:
        """Resolve the MFI's OWN run-context node for the active branch.

        The cross-flow buffers (``latest_per_flow`` / ``judgments``) AND the transient
        lock-step rendezvous both live on this single node. They are accessed from the flow
        worker closures, which run INSIDE ``flow.ainfer(run_context=flow_child)`` — i.e.
        under the flow's child ctx (a DESCENDANT of the MFI node), even when BTA runs the
        flow in a worker thread (the thread starts with the default context, but BTA
        re-enters the bridge with the explicit ``flow_child``). So ``active_run_context()``
        here is the flow node; we walk UP its ancestor paths in the SHARED store and return
        the first node whose ``.attempt`` is a ``MultiFlowAttemptState`` (the MFI's node,
        seeded by ``_reset_cross_flow_state``). Returns ``None`` when no ctx is active or no
        ancestor carries one (legacy / pre-reset) — callers then use the instance backing.
        """
        from agent_foundation.common.inferencers.run_context import (
            MultiFlowAttemptState,
            active_run_context,
        )

        ctx = active_run_context()
        if ctx is None:
            return None
        store = ctx._store
        # Enumerate ancestor paths from the active (flow) path up to the root, e.g.
        # "/bta/flow_0" -> ["/bta/flow_0", "/bta", "/"]. ``ctx.path`` is "/"-segmented.
        segments = [s for s in ctx.path.split("/") if s]
        candidate_paths = []
        for i in range(len(segments), -1, -1):
            candidate_paths.append("/" + "/".join(segments[:i]))
        # De-dup while preserving order ("/" can repeat when path == "/").
        seen: set = set()
        for path in candidate_paths:
            if path in seen:
                continue
            seen.add(path)
            node = store.peek(path)
            if node is not None and isinstance(node.attempt, MultiFlowAttemptState):
                return node
        return None

    def _resolve_attempt_state(self) -> Any:
        """The MFI's per-run ``MultiFlowAttemptState`` (see :meth:`_resolve_attempt_node`)."""
        node = self._resolve_attempt_node()
        return node.attempt if node is not None else None

    # ------------------------------------------------------------------
    # Cross-flow step barrier (lock-step coordination)
    # ------------------------------------------------------------------

    @property
    def _coordination_enabled(self) -> bool:
        """True when the cross-flow step barrier should be installed — the explicit
        ``cross_flow_sync`` knob or the backward-compatible ``coordinated_stop`` alias."""
        return bool(self.cross_flow_sync or self.coordinated_stop)

    def _resolve_active_flow_indices(self) -> set:
        """Flow indices that will actually run their ``_ainfer`` this attempt and thus
        participate in the barrier.

        Fresh run → all flows (MFI disables breakdown, so workers are 1:1 with
        ``flow_configs``). The resume path narrows this to exclude flows whose worker result
        already exists on disk (they LOAD instead of running, so they must not be barrier
        participants — otherwise they are ghosts that never arrive/leave → deadlock). See
        the INTEGRATED plan §2.3.
        """
        return set(range(len(self.flow_configs)))

    def _build_rendezvous(self) -> Any:
        """Create a fresh ``CrossFlowRendezvous`` seeded with the participating flows."""
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.cross_flow_rendezvous import (
            CrossFlowRendezvous,
        )

        return CrossFlowRendezvous(self._resolve_active_flow_indices())

    def _resolve_rendezvous(self) -> Any:
        """The active per-run ``CrossFlowRendezvous`` (lock-step barrier), or ``None``.

        Under a ctx it lives on the MFI node's transient ``scratch`` (resolved by the same
        ancestor walk-up as the attempt state); otherwise the instance backing.
        """
        node = self._resolve_attempt_node()
        if node is not None:
            return node.scratch.get(self._RENDEZVOUS_SCRATCH_KEY)
        return self._cross_flow_rendezvous_backing

    @property
    def _latest_per_flow(self) -> Dict[int, Any]:
        """Cross-flow latest-output buffer: the per-run attempt state under a ctx
        (resolved by walking up to the MFI node), else the instance backing."""
        attempt = self._resolve_attempt_state()
        if attempt is not None:
            return attempt.latest_per_flow
        return self._latest_per_flow_backing

    @_latest_per_flow.setter
    def _latest_per_flow(self, value: Dict[int, Any]) -> None:
        attempt = self._resolve_attempt_state()
        if attempt is not None:
            attempt.latest_per_flow = value
        else:
            self._latest_per_flow_backing = value

    @property
    def _latest_per_flow_path(self) -> Dict[int, Any]:
        """Cross-flow latest-output PATH buffer (mirrors ``_latest_per_flow``): the per-run
        attempt state under a ctx, else the instance backing. Defaults to an empty dict when
        the backing was never initialized (e.g. a unit-test stub that bypasses attrs init)."""
        attempt = self._resolve_attempt_state()
        if attempt is not None:
            return attempt.latest_per_flow_path
        return getattr(self, "_latest_per_flow_path_backing", None) or {}

    @_latest_per_flow_path.setter
    def _latest_per_flow_path(self, value: Dict[int, Any]) -> None:
        attempt = self._resolve_attempt_state()
        if attempt is not None:
            attempt.latest_per_flow_path = value
        else:
            self._latest_per_flow_path_backing = value

    @property
    def _all_judgments(self) -> List[Tuple[int, int, str]]:
        """Cross-flow judgment accumulator: the per-run attempt state under a ctx
        (resolved by walking up to the MFI node), else the instance backing."""
        attempt = self._resolve_attempt_state()
        if attempt is not None:
            return attempt.judgments
        return self._all_judgments_backing

    @_all_judgments.setter
    def _all_judgments(self, value: List[Tuple[int, int, str]]) -> None:
        attempt = self._resolve_attempt_state()
        if attempt is not None:
            attempt.judgments = value
        else:
            self._all_judgments_backing = value

    def _init_call_state(self, inference_input):
        """M-AF1: seed the per-run ``MultiFlowState`` (via ``state_factory``) THEN reset the
        per-call dispatch fields. Runs AFTER ``enter_run`` (the base calls it post-bridge),
        once per call — so the reset clears THIS call's node (``state_factory`` is
        populate-once, so a *reused* node is not otherwise reset). Replaces the old
        ``ainfer``/``infer`` overrides that reset *before* the ctx was active.
        """
        super()._init_call_state(inference_input)
        self._reset_dispatch_state_for_call()

    # ------------------------------------------------------------------
    # Recursive pre_retry — declare children for the retry-time hook.
    # MultiFlow's children are: aggregator (inherited from BTA) plus
    # every flow's initial/followup inferencer. Worker session resets on
    # MultiFlow retry are benign:
    #   - Cache-hit case (typical on retry): workers load
    #     `final_result.json`, no new ainfer calls fire. Reset is
    #     effectively a no-op.
    #   - Cache-miss case: workers re-execute. Each retry attempt of
    #     MultiFlow gets a fresh worker session; within ONE attempt,
    #     step-0 → step-N still chain via active_session_id (set after
    #     each step's response). Across attempts, sessions reset, which
    #     is the desired retry semantic.
    # ------------------------------------------------------------------

    def _iter_child_inferencers(self):
        # Aggregator from parent BTA
        yield from super()._iter_child_inferencers()
        # Each flow's initial/followup inferencer, dedup'd
        seen = set()
        for cfg in self.flow_configs:
            for key in ("initial_inferencer", "followup_inferencer"):
                inf = cfg.get(key)
                if inf is not None and id(inf) not in seen:
                    seen.add(id(inf))
                    yield inf

    def _iter_child_slots(self):
        """§9.3/N-Major1: BTA breakdown/aggregator slots plus each flow's
        initial/followup inferencer keyed by ``flow_{i}_{role}``."""
        yield from super()._iter_child_slots()
        seen = set()
        for i, cfg in enumerate(self.flow_configs):
            for key in ("initial_inferencer", "followup_inferencer"):
                inf = cfg.get(key)
                if inf is not None and id(inf) not in seen:
                    seen.add(id(inf))
                    yield (f"flow_{i}_{key.split('_')[0]}", inf)

    # ------------------------------------------------------------------
    # Round 7 — dispatch-state extraction and public accessors
    # ------------------------------------------------------------------

    def _extract_dispatch_state(self, raw: Any) -> None:
        """Best-effort extraction of `<Winner>` / `<Reviewer>` / `<Fixer>` tags
        from the aggregator's output.

        Each parser is independent — a failure in one is logged at WARNING
        level and leaves the corresponding ``_last_*`` field at its default.
        Skipped entirely when ``disable_aggregator`` is True or when ``raw``
        is not a string (e.g., the worker tuple returned in disabled-aggregator
        mode).
        """
        if self.disable_aggregator or not isinstance(raw, str):
            return
        if self.winner_parser is not None:
            try:
                idx = self.winner_parser(raw)
                if idx is not None and 0 <= int(idx) < len(self.flow_configs):
                    self._last_winner_idx = int(idx)
            except Exception as exc:  # noqa: BLE001 — best effort
                _logger.warning("MultiFlow winner_parser raised: %s", exc)
        if self.reviewer_alias_parser is not None:
            try:
                alias = self.reviewer_alias_parser(raw)
                if alias is not None:
                    self._last_reviewer_alias = str(alias)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("MultiFlow reviewer_alias_parser raised: %s", exc)
        if self.fixer_alias_parser is not None:
            try:
                alias = self.fixer_alias_parser(raw)
                if alias is not None:
                    self._last_fixer_alias = str(alias)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("MultiFlow fixer_alias_parser raised: %s", exc)
        if self.ranking_parser is not None:
            try:
                ranking = self.ranking_parser(raw)
                if ranking is not None and isinstance(ranking, list):
                    valid = [int(i) for i in ranking
                             if 0 <= int(i) < len(self.flow_configs)]
                    valid = list(dict.fromkeys(valid))
                    if valid:
                        self._last_ranking = valid
                        if self._last_winner_idx is None:
                            self._last_winner_idx = valid[0]
            except Exception as exc:  # noqa: BLE001
                _logger.warning("MultiFlow ranking_parser raised: %s", exc)

    def get_winner_flow_idx(self) -> Optional[int]:
        """Index of the winning flow (per the most recent ainfer/infer call),
        or None when no winner was identified."""
        return self._last_winner_idx

    def get_winner_inferencer(self) -> Optional[Any]:
        """The ``initial_inferencer`` of the winning flow, or None when no
        winner is known.

        Returns the *Python instance* of the winning flow's first-step
        inferencer; downstream consumers (e.g., :class:`MultiFlowDualInferencer`)
        use this for self-review-avoidance comparisons via ``is``.
        """
        if self._last_winner_idx is None:
            return None
        if 0 <= self._last_winner_idx < len(self.flow_configs):
            return self.flow_configs[self._last_winner_idx].get("initial_inferencer")
        return None

    def get_chosen_reviewer_alias(self) -> Optional[str]:
        """The alias the aggregator chose for the reviewer (LLM-driven dispatch),
        or None when ``reviewer_alias_parser`` is unset / didn't match."""
        return self._last_reviewer_alias

    def get_chosen_fixer_alias(self) -> Optional[str]:
        """The alias the aggregator chose for the fixer, or None."""
        return self._last_fixer_alias

    def get_ranking(self) -> Optional[list]:
        """Flow indices ordered best-to-worst, or None."""
        return self._last_ranking

    def get_runner_up_flow_idx(self) -> Optional[int]:
        """Index of the second-best flow, or None."""
        if self._last_ranking is not None and len(self._last_ranking) > 1:
            return self._last_ranking[1]
        return None

    def get_runner_up_inferencer(self) -> Optional[Any]:
        """The ``initial_inferencer`` of the runner-up flow, or None."""
        idx = self.get_runner_up_flow_idx()
        if idx is not None and 0 <= idx < len(self.flow_configs):
            return self.flow_configs[idx].get("initial_inferencer")
        return None

    def get_first_non_winner_inferencer(self) -> Optional[Any]:
        """First flow inferencer that is not the winner (declaration order).
        Fallback when ranking is unavailable."""
        winner = self.get_winner_inferencer()
        for cfg in self.flow_configs:
            inf = cfg.get("initial_inferencer")
            if inf is not None and inf is not winner:
                return inf
        return None

    def get_non_winner_inferencers(self) -> List[Any]:
        """ALL flow inferencers that are not the winner (declaration order, deduped
        by identity). Backs MFDual's ``reviewer_match_all_non_winners`` panel (§3)."""
        winner = self.get_winner_inferencer()
        out: List[Any] = []
        seen: set = set()
        for cfg in self.flow_configs:
            inf = cfg.get("initial_inferencer")
            if inf is not None and inf is not winner and id(inf) not in seen:
                seen.add(id(inf))
                out.append(inf)
        return out

    def _normalize_aggregator_output(self, raw: Any) -> Any:
        """Pick the aggregator's textual output from BTA's result.

        BTA's async ``_ainfer`` can return:
          - ``(worker_lwi_state, aggregator_output)`` — heterogeneous tuple
          - ``aggregator_output`` directly (sync path)

        Aggregator output itself can be:
          - A plain string (simple inferencers)
          - An :class:`InferencerResponse`-like object with ``.output`` attr
            (CLI inferencers, e.g., ``TerminalInferencerResponse``)
          - A dict with ``"output"`` key

        We normalize so downstream callers (DualInferencer's propose step,
        ``response_parser``) always see a string when an aggregator is active.
        Falls back to ``raw`` unchanged when no string can be extracted.

        When ``disable_aggregator=True`` (no aggregator), we return ``raw``
        unchanged — callers expect the per-worker tuple in that mode.
        """
        if self.disable_aggregator or self.aggregator_inferencer is None:
            return raw

        # Diagnostic logging for the BTA→MultiFlow handoff. Real-CLI runs
        # surfaced multiple subtle shape mismatches here (workers' LWI state
        # dicts vs. aggregator's TerminalInferencerResponse, ordering inside
        # the WorkGraph result tuple, etc.). Kept at DEBUG level so it's
        # available for future investigation without polluting normal logs.
        if _logger.isEnabledFor(logging.DEBUG):
            try:
                if isinstance(raw, tuple):
                    summary = []
                    for i, x in enumerate(raw):
                        if x is None:
                            summary.append(f"[{i}]=None")
                        elif isinstance(x, str):
                            summary.append(f"[{i}]=str({len(x)}ch:{x[:40]!r})")
                        elif isinstance(x, dict):
                            summary.append(f"[{i}]=dict(keys={list(x.keys())[:6]})")
                        elif hasattr(x, "output"):
                            out = getattr(x, "output", "")
                            summary.append(f"[{i}]={type(x).__name__}(output={out[:40]!r})")
                        else:
                            summary.append(f"[{i}]={type(x).__name__}")
                    _logger.debug(
                        "_normalize_aggregator_output: raw shape=tuple[%d] elements=%s",
                        len(raw), " | ".join(summary),
                    )
                else:
                    _logger.debug(
                        "_normalize_aggregator_output: raw type=%s repr=%r",
                        type(raw).__name__, str(raw)[:200],
                    )
            except Exception as exc:  # noqa: BLE001
                _logger.debug(
                    "_normalize_aggregator_output diagnostic logging failed: %s", exc
                )

        # 1. Unwrap tuple wrapping. The aggregator's response is the one we
        #    want; workers may also appear in the tuple. Tuple ORDERING is
        #    NOT guaranteed (depends on WorkGraph traversal — empirically
        #    we've seen both `(worker_str, agg_response)` and
        #    `(agg_response, worker_str)`).
        #
        #    Selection priority (most-aggregator-like first):
        #      Tier 1: Element with `.output` attr but NOT a string (i.e.,
        #              `TerminalInferencerResponse`-like). Workers'
        #              dynamic-mode return shape is a plain str (LWI's
        #              last step text), so a wrapped response object is
        #              a strong signal of "aggregator's CLI response."
        #      Tier 2: Element that's a dict with `"output"` key but
        #              NOT an LWI state dict (filtered by marker keys).
        #      Tier 3: A plain string (only valid when aggregator is a
        #              simple in-process inferencer that returns text
        #              directly — typical for unit tests).
        #      Tier 4: Last non-None element (defensive fallback).
        #
        #    We prefer the LAST element within each tier (later in the
        #    tuple = closer to the aggregator's position in BTA's
        #    topological run order, when the order does cooperate).
        if isinstance(raw, tuple):
            non_none = [x for x in raw if x is not None]
            if not non_none:
                pass  # All-None — fall through unchanged
            elif len(non_none) == 1:
                raw = non_none[0]
            else:
                tier1 = [x for x in non_none
                         if hasattr(x, "output") and not isinstance(x, str)]
                tier2 = [x for x in non_none
                         if isinstance(x, dict) and "output" in x
                         and not _looks_like_lwi_state(x)]
                tier3 = [x for x in non_none if isinstance(x, str)]
                if tier1:
                    raw = tier1[-1]
                elif tier2:
                    raw = tier2[-1]
                elif tier3:
                    raw = tier3[-1]
                else:
                    raw = non_none[-1]

        # 2/3. Delegate to _coerce_to_text for InferencerResponse-like objects
        # and dicts, with the same field preference (output → raw_output).
        # Plain strings short-circuit at the top of _coerce_to_text.
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict) or hasattr(raw, "output"):
            return self._coerce_to_text(raw)
        return raw

    def _maybe_strip_response(self, raw: Any) -> Any:
        """Apply ``response_parser`` to a string aggregator output.

        Skips parsing when ``response_parser`` is unset or when ``raw`` is
        not a string (parsers are designed for textual aggregator output).
        """
        if self.response_parser is None or not isinstance(raw, str):
            return raw
        try:
            return self.response_parser(raw)
        except Exception as exc:  # noqa: BLE001 — best-effort; warn
            _logger.warning("MultiFlow response_parser failed: %s", exc)
            return raw

    # NOTE (M-AF1): the former ``ainfer``/``infer`` overrides only existed to call
    # ``_reset_dispatch_state_for_call()`` *before* ``super()`` — i.e. before the bridge
    # activated this call's RunContext, which would clear the wrong place. The per-call
    # reset now runs in ``_init_call_state`` (invoked by the base AFTER ``enter_run``),
    # so these pass-through overrides are removed.

    def _apply_runtime_input_propagation(self, inference_input: Any) -> None:
        """Rewrite each flow's input from the runtime ``inference_input``.

        No-op unless ``propagate_runtime_input=True``. When enabled, mutates
        ``flow_configs[i]["input"]`` and ``predefined_sub_queries`` so the
        downstream BTA worker spawning (which reads ``predefined_sub_queries``)
        sees the runtime input rather than the static YAML placeholder.

        If ``runtime_input_template`` is set, each flow's input is rendered
        through it with feed ``{input, flow_idx, n_flows}`` — useful for
        per-flow perspective seeding (different angles for different flows).

        See class docstring for the resume-incompatibility caveat.
        """
        if not self.propagate_runtime_input:
            return
        n_flows = len(self.flow_configs)
        # Compute the effective per-flow inputs WITHOUT mutating the definition.
        effective = []
        for i, cfg in enumerate(self.flow_configs):
            if self.runtime_input_template is not None:
                effective.append(
                    self._render_template(
                        self.runtime_input_template,
                        {"input": inference_input, "flow_idx": i, "n_flows": n_flows},
                    )
                )
            else:
                effective.append(inference_input)

        from agent_foundation.common.inferencers.run_context import (
            MultiFlowState,
            active_run_context,
        )

        _ctx = active_run_context()
        if _ctx is not None:
            # M5 read-flip: under a RunContext, publish the runtime sub-queries to
            # the context node and DO NOT mutate ``flow_configs[i]["input"]`` /
            # ``self.predefined_sub_queries`` — the definition stays immutable
            # (the §2.4 invariant). BTA reads them via
            # ``_get_effective_predefined_sub_queries`` (ctx-preferring).
            _node = _ctx.node(creator=(type(self).__qualname__, _ctx.path))
            if isinstance(_node.call, MultiFlowState):
                # M-AF1/G9: ``state_factory`` made ``call`` a typed ``MultiFlowState`` —
                # set the typed fields (the inherited BTA reader reads
                # ``effective_sub_queries`` on its typed branch); the old dict-guard
                # would have silently skipped this.
                _node.call.effective_sub_queries = effective
                _node.call.flow_inputs = effective
            elif _node.call is None or isinstance(_node.call, dict):
                _node.call = dict(_node.call or {})
                _node.call["predefined_sub_queries"] = effective
                _node.call["flow_inputs"] = effective
        else:
            # Legacy path (no context): mutate as before — byte-identical.
            for i, cfg in enumerate(self.flow_configs):
                cfg["input"] = effective[i]
            self.predefined_sub_queries = effective

    async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
        # Part C — cross-flow step coordination. When enabled (``cross_flow_sync`` or its
        # ``coordinated_stop`` alias), the lock-step barrier is installed transparently:
        # ``_reset_cross_flow_state`` seeds the rendezvous and the per-flow input builders
        # arrive at it each round (publish -> barrier -> read). No special-casing here —
        # the async fan-out (BTA WorkGraph ``asyncio.gather``) is exactly the single-loop
        # context the rendezvous needs. See the INTEGRATED plan.
        self._apply_runtime_input_propagation(inference_input)
        self._reset_cross_flow_state()
        raw = await BreakdownThenAggregateInferencer._ainfer(
            self, inference_input, inference_config=inference_config, **_inference_args
        )
        raw = self._normalize_aggregator_output(raw)
        # Round 7: best-effort extract dispatch state BEFORE response_parser
        # strips structured tags — parsers operate on the raw aggregator output.
        self._extract_dispatch_state(raw)
        return self._maybe_strip_response(raw)

    def _infer(self, inference_input, inference_config=None, **_inference_args):
        # Part C — cross-flow coordination requires the ASYNC path: the step barrier is an
        # asyncio rendezvous that must be awaited, and the flows must share one event loop
        # (BTA's async ``gather``). The sync path runs flows without that loop, so loud-fail
        # rather than silently dropping coordination. Use ``ainfer`` instead.
        if self._coordination_enabled:
            raise NotImplementedError(
                "MultiFlowInferencer cross-flow coordination (cross_flow_sync / "
                "coordinated_stop) requires the async path — call ainfer(), not infer(). "
                "The step barrier is an awaitable rendezvous and needs the single-event-loop "
                "context of BTA's async fan-out."
            )
        self._apply_runtime_input_propagation(inference_input)
        self._reset_cross_flow_state()
        raw = BreakdownThenAggregateInferencer._infer(
            self, inference_input, inference_config=inference_config, **_inference_args
        )
        raw = self._normalize_aggregator_output(raw)
        self._extract_dispatch_state(raw)
        return self._maybe_strip_response(raw)
