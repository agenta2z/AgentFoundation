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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from attr import attrib, attrs

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.linear_workflow_inferencer import (
    LinearWorkflowInferencer,
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
# Default templates (raw Jinja2 — used when no shared TemplateManager is set)
# ---------------------------------------------------------------------------

DEFAULT_MULTIFLOW_FOLLOWUP_TEMPLATE = """\
Original task:
{{ input }}

Your previous output:
{{ your_prev }}

{% if visible_plans %}
Other teams' latest outputs:
{% for idx, plan in visible_plans.items() %}
--- Flow {{ idx }} ---
{{ plan if plan else '(no output yet)' }}

{% endfor %}
{% endif %}\
Continue iterating; integrate the best ideas from any other teams' outputs.
"""

DEFAULT_AGGREGATOR_PROMPT_TEMPLATE = """\
Original task:
{{ input }}

Each team's final output:
{% for idx, plan in worker_plans.items() %}
=== Flow {{ idx }} ===
{{ plan if plan else '(no output)' }}

{% endfor %}\
{% if all_judgments_summary %}
Per-iteration judgments collected during the multi-flow phase:
{{ all_judgments_summary }}

{% endif %}\
Produce a final integrated synthesis drawing on the best of every team's work.
"""


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
    * ``worker_factory`` is auto-built to produce a fresh dynamic-mode LWI per flow.
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
    ====================  =================================================  ==========

    When no ``initial_prompt`` is set, step 0 receives the raw ``input`` field.
    When no ``followup_prompt`` and class-level
    ``multiflow_followup_prompt`` are set, step ≥ 1 receives the previous
    step's result directly (legacy passthrough).
    """

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

    # Internal state — reset at the top of each ainfer/infer call.
    # Declared as init=False so attrs doesn't include them in __init__.
    _latest_per_flow: Dict[int, Any] = attrib(factory=dict, init=False)
    _all_judgments: List[Tuple[int, int, str]] = attrib(factory=list, init=False)
    _last_winner_idx: Optional[int] = attrib(default=None, init=False)
    _last_reviewer_alias: Optional[str] = attrib(default=None, init=False)
    _last_fixer_alias: Optional[str] = attrib(default=None, init=False)

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
                raise ValueError(f"flow_configs[{i}] is missing required key 'input'")

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

        # Initialize cross-flow visibility tracking dict (one None slot per flow).
        # Worker factory closures read this dict by reference, so we mutate it
        # in place (do NOT reassign) at the top of each ainfer/infer call.
        self._latest_per_flow = {i: None for i in range(len(self.flow_configs))}
        self._all_judgments = []

        # Wire BTA fields from flow_configs.
        self.predefined_sub_queries = [c["input"] for c in self.flow_configs]
        # breakdown_inferencer stays None (default).
        if self.worker_factory is None:
            self.worker_factory = self._build_worker_factory()

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
    # Per-flow visibility resolution
    # ------------------------------------------------------------------

    def _resolve_flow_visibility(self, flow_idx: int) -> List[int]:
        cfg = self.flow_configs[flow_idx]
        spec = cfg.get("visible_flows", self.visible_flows)
        return _resolve_visible_indices(flow_idx, len(self.flow_configs), spec)

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
                "worker_output_paths": list(worker_output_paths or []),
            }

            template = outer.aggregator_prompt or DEFAULT_AGGREGATOR_PROMPT_TEMPLATE
            return outer._render_template(template, feed)

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
            def _wrapped_dynamic_input_builder(state, prev_result):
                # Store the textual form of prev_result so cross-flow visibility
                # surfaces real text, not Python wrapper reprs.
                prev_text = outer._coerce_to_text(prev_result)
                outer._latest_per_flow[index] = prev_text
                visible_plans = {
                    i: outer._latest_per_flow.get(i) for i in visible if i != index
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
                    return user_dynamic_builder(state, prev_result)

                if followup_template is None:
                    # No template configured — preserve legacy LWI behaviour
                    # (pass the previous result through unchanged).
                    return prev_result

                feed = {
                    "input": cfg.get("input", ""),
                    "your_prev": prev_text,
                    "visible_plans": visible_plans,
                    "all_plans": {
                        i: outer._latest_per_flow.get(i) for i in range(len(configs))
                    },
                    "flow_idx": index,
                    "step_idx": state.get("dynamic_step_count", 0),
                }
                return outer._render_template(followup_template, feed)

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
                    return text
                cfg_response_builder = _last_dynamic_step_text

            return LinearWorkflowInferencer(
                dynamic_mode=True,
                default_initial_inferencer=cfg.get("initial_inferencer"),
                default_followup_inferencer=cfg.get("followup_inferencer"),
                end_condition=cfg.get("end_condition"),
                max_dynamic_steps=cfg.get("max_dynamic_steps", 10),
                inferencer_factory=cfg.get("inferencer_factory"),
                dynamic_input_builder=_wrapped_dynamic_input_builder,
                response_builder=cfg_response_builder,
                initial_state_factory=initial_state_factory,
            )

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
        """
        # In-place mutation — closures hold a reference to this dict.
        self._latest_per_flow.clear()
        self._latest_per_flow.update({i: None for i in range(len(self.flow_configs))})
        self._all_judgments.clear()

    def _reset_dispatch_state_for_call(self) -> None:
        """Reset PER-CALL dispatch state. Called once per top-level
        ``ainfer`` / ``infer`` invocation (not per retry attempt) so that
        a winner parsed by an early successful retry attempt survives a
        later malformed retry of the same call.
        """
        self._last_winner_idx = None
        self._last_reviewer_alias = None
        self._last_fixer_alias = None

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

    async def ainfer(self, inference_input, inference_config=None, **_inference_args):
        """Override: reset dispatch state ONCE per top-level call (not per
        retry attempt). Cross-flow worker state is still reset per-attempt
        in :meth:`_ainfer`. See plan Fix 2 for lifetime split rationale.
        """
        self._reset_dispatch_state_for_call()
        return await super().ainfer(inference_input, inference_config, **_inference_args)

    def infer(self, inference_input, inference_config=None, **_inference_args):
        """Sync mirror of :meth:`ainfer` override."""
        self._reset_dispatch_state_for_call()
        return super().infer(inference_input, inference_config, **_inference_args)

    async def _ainfer(self, inference_input, inference_config=None, **_inference_args):
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
        self._reset_cross_flow_state()
        raw = BreakdownThenAggregateInferencer._infer(
            self, inference_input, inference_config=inference_config, **_inference_args
        )
        raw = self._normalize_aggregator_output(raw)
        self._extract_dispatch_state(raw)
        return self._maybe_strip_response(raw)
