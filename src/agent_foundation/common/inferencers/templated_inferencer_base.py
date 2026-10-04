"""
TemplatedInferencerBase - InferencerBase variant for inferencers that render
their own ``inference_input`` through a Jinja ``TemplateManager`` before LLM call.

Architectural role (post axes-refactor 2026-05-16)::

    InferencerBase (template-free)
    ├── TemplatedInferencerBase  ← THIS FILE — one of three orthogonal axes
    ├── StreamingInferencerBase  — streaming axis (decoupled)
    ├── TerminalInferencerBase   — terminal-exec axis (decoupled)
    │
    ├── TerminalTemplatedInferencerBase(TIB, TemplatedIB) — MI convenience
    ├── TerminalSessionInferencerBase(TIB, SIB)           — MI convenience
    ├── TerminalSessionTemplatedInferencerBase(TSIB, TemplatedIB) — MI convenience
    │
    ├── ApiInferencerBase(TemplatedIB)       — API leaves inherit directly
    ├── RemoteInferencerBase(TemplatedIB)    — remote leaves inherit directly
    │
    └── (orchestrators) — Dual, BTA, LWI, PTI, MultiFlow, MultiFlowDual,
        Conversational* — inherit InferencerBase directly, not this class.

    Leaves opt into templating via direct inheritance from this class OR
    via the convenience MI classes. StreamingInferencerBase and
    TerminalInferencerBase no longer inherit from this class; streaming-only
    and terminal-only leaves are no longer accidental recipients of
    cascaded template state.

Why a separate base?
    Cascade injection of ``_template_manager`` via ``_-prefix`` (in
    ``rich_python_utils.config_utils._instantiate``) walks every descendant
    that has a ``template_manager`` constructor param. By keeping the
    template fields off ``InferencerBase``, orchestrators never receive
    cascade-injected template state — so they cannot accidentally try to
    render their own input through an unconfigured template.

    Leaves opt into rendering by inheriting from this class AND setting
    ``template_root_space`` (or ``template_key``). Forgetting the namespace
    raises a loud ``ValueError`` from ``_render_prompt`` rather than silently
    no-rendering.

Slot-based role defaults
------------------------
Some template fields are role-derived: BTA's breakdown slot always wants
``template_root_space="task_breakdown"``; any aggregator slot consuming
upstream artifacts wants the ``aggregation`` triplet; any review slot
wants ``template_key="review"``. Repeating these in every YAML is brittle
(partial-triplet drift) and noisy.

Each orchestrator class declares a ``SLOT_DEFAULTS`` ClassVar mapping
slot names (or dotted paths with ``*`` wildcards) to a reusable
:class:`agent_foundation.common.inferencers.template_defaults.InferencerTemplateDefaults`
bundle. The Hydra walker (``rich_python_utils.config_utils._instantiate``,
step 1d) fills missing template fields on the slot child before
construction: scalar fill for ``template_root_space``/``template_key``;
per-key dict merge for ``template_variables``/``template_extra_feed``
(user-supplied keys always win). The named bundles
(``BREAKDOWN_TEMPLATE_DEFAULTS``, ``AGGREGATION_TEMPLATE_DEFAULTS``,
``REVIEW_TEMPLATE_DEFAULTS``, ``FOLLOWUP_AGGREGATION_DEFAULTS``) are the
single source of truth for "what the role wants"; YAMLs only spell out
the use-case-specific choices (``template_root_space: implementation``
vs ``plan``).

Opt-out: set ``_disable_slot_defaults_: true`` on any orchestrator node
to skip the entire injection at that node. Per-key opt-out: set the key
to an empty string (``task_instructions: ""``) — the empty value
survives the merge and the template renders that variable empty.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Dict, Optional

import attrs as attrs_mod
from agent_foundation.common.inferencers.inferencer_base import (
    _field_defaults,
    _init_param_names,
    ACTOR_SCOPED_FEED_KEYS,
    InferencerBase,
    RoleTransition,
    TEMPLATE_EXTRA_FEED_ATTR,
)
from agent_foundation.common.inferencers.run_context import (
    active_run_context,
    frame_for,
    InvocationFrame,
    NodeOutcomeState,
    publish_result,
    RenderedTaskContractState,
    ROLE_STATE_ATTRS,
    RoleState,
    RuntimeKey,
)
from agent_foundation.common.inferencers.template_feed_scope import (  # noqa: F401
    publish_propagated,
    resolve_ctx_feed_override,
    resolve_propagated,
    TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE,
    TEMPLATE_PROPAGATED_FEED_HANDLE,
    TEMPLATE_PROPAGATED_MODES_HANDLE,
)
from attr import attrib, attrs
from rich_python_utils.config_utils import collect_slot_defaults


@attrs_mod.frozen
class _EffectiveRole:
    """The template attributes one render uses: the active ``RoleState``'s value
    where it sets one, else the definition's."""

    template_key: Any
    template_root_space: Any
    template_extra_feed: Any
    template_variables: Any
    template_version: Any
    template_master_version: Any
    modes: Any


def _deep_merge_into(target: dict, source: dict) -> None:
    """Recursively merge ``source`` into ``target``. Dicts at matching keys
    merge; non-dict leaves overwrite. Used to fold ``load_variables`` output
    into the feed without clobbering sibling sub-namespaces (e.g., mode
    injection must not wipe out ``feed["instructions"]["behavior"]``).
    """
    for k, v in source.items():
        existing = target.get(k)
        if isinstance(existing, dict) and isinstance(v, dict):
            _deep_merge_into(existing, v)
        else:
            target[k] = v


def _resolve_ctx_feed_override() -> Optional[dict]:
    """The active RunContext's per-call feed override (see ``template_feed_scope``)."""
    return resolve_ctx_feed_override()


@attrs(slots=False)
class TemplatedInferencerBase(InferencerBase):
    """Base class for inferencers that render their own ``inference_input``
    through a Jinja ``TemplateManager``.

    Adds opt-in template fields and overrides the no-op stubs on
    ``InferencerBase`` (``_render_prompt``, ``_propagate_to_children``,
    ``supports_prompt_rendering``) with real implementations.

    Note: ``_finalize_output`` lives on ``InferencerBase`` (gated on
    ``output_path`` + ``has_local_access``) — file-writing is workspace
    functionality, not template-specific. Both leaves AND orchestrators
    benefit from inherited file-writing without having templates.
    """

    _HOST_PURE_CERTIFIED = True

    # === Template-based prompt rendering (opt-in) ===
    # When template_manager is set, inference_input is treated as the raw
    # user query.  Before reaching _infer(), the base class renders a Jinja2
    # template via template_manager, binding the raw input to {{ input }}.
    template_manager: Optional[Any] = attrib(default=None)
    template_key: str = attrib(default="")
    template_root_space: Optional[str] = attrib(default=None)
    template_extra_feed: dict = attrib(factory=dict)
    template_variables: dict = attrib(factory=dict)
    # Default version for variable lookups. Used when a key in
    # ``template_variables`` has a None/empty value -- per-key explicit
    # values still win. Distinct from ``TemplateManager.template_version``
    # (deployment-level default): this is the per-inferencer override that
    # flows into ``load_variable`` for variable lookups made by THIS
    # inferencer.
    template_version: Optional[str] = attrib(default=None)
    # Master version for variable lookups. When set, the variable cascade
    # searches inside a ``{name}/{master_version}/`` subdirectory instead of
    # flat ``{name}/{version}.ext``. Orthogonal to ``template_version``:
    # master selects the *family* (e.g., "aggregation"), version selects the
    # *variant* within that family (e.g., "create_role").
    template_master_version: Optional[str] = attrib(default=None)
    # Mode flags (e.g. "deep_mode", "elegant_mode") that toggle conditional
    # blocks in templates AND auto-load instruction text from
    # ``_variables/instructions/modes/<name>.jinja2``. For each entry:
    #   - ``feed["enable_<name>"] = bool(enabled)`` is exposed to Jinja2.
    #   - When enabled, the corresponding mode file (if present) is loaded
    #     and merged into ``feed["instructions"]["modes"][<name>]`` for
    #     ``{{ instructions.modes.<name> }}`` access.
    # Adding a new mode = drop a file in ``_variables/instructions/modes/``
    # and set ``modes: {<name>: true}`` in YAML. No code changes needed.
    # Defaults to deep_mode + elegant_mode ON — matches the user's standing
    # instructions ("ultrathink", "elegant proper solution"). Override per
    # YAML topology when specific runs need different behavior.
    modes: dict = attrib(factory=lambda: {"deep_mode": True, "elegant_mode": True})

    # Snapshot of the ``task_instructions`` THIS leaf actually rendered, recorded at
    # its own render by ``_capture_rendered_task_instructions``. Orchestrators relay
    # the *proposer's* snapshot into reviewer/fixer reference blocks verbatim, so
    # ``<OriginalTaskInstructions>`` shows what the author was really told instead of
    # a per-leaf re-render bound to the consumer's own context (see
    # ``DualInferencer._representative_proposer``). ``init=False``: runtime state,
    # never a constructor arg, never serialized. The bare compat projection of
    # ``_RENDERED_CONTRACT``: written outside host mode only.
    _last_rendered_task_instructions: str = attrib(default="", init=False)

    # The task contract this leaf rendered in its current invocation; published as
    # its node's ``NodeOutcomeState.task_contract`` when the invocation succeeds.
    _RENDERED_CONTRACT = RuntimeKey(
        "TemplatedInferencerBase.rendered_contract",
        compat={"_last_rendered_task_instructions": "text"},
    )

    # ------------------------------------------------------------------
    # Template feed construction
    # ------------------------------------------------------------------

    def _build_template_feed(
        self,
        inference_input: str,
        *,
        extra_feed: Optional[dict] = None,
    ) -> dict:
        """Build the template variable feed dict.

        Merges (in priority order, lowest first):

        1. ``template_variables`` — variant selectors resolved to file content
           via ``template_manager.load_variable()``.  E.g.,
           ``{"task_preamble": "skill_tool_creation"}`` loads
           ``_variables/task_preamble/skill_tool_creation.jinja2``.
        2. ``template_extra_feed`` — literal key-value overrides.
        3. ``extra_feed`` — per-call feed overrides (Phase 1, leaf-owned
           template rendering). Caller MUST NOT include reserved keys
           ({"input", "__template_space__"}) — ValueError raised at top.
        4. ``{{ input }}`` bound to ``inference_input`` (sacrosanct).
        5. ``output_path`` (if inferencer has local file access).

        Override this method to customize feed construction (e.g., add
        dynamic variables from external sources).
        """
        # ── Phase 1 (Q11): reserved-key guard. Per-call extra_feed cannot
        # clobber sacrosanct slots — silent override of {{ input }} would
        # be invisible until production. Raise loud at the boundary.
        if extra_feed:
            PROTECTED = {"input", "__template_space__"}
            collisions = PROTECTED & extra_feed.keys()
            if collisions:
                raise ValueError(
                    f"{type(self).__name__}._build_template_feed: extra_feed "
                    f"contains reserved key(s) {sorted(collisions)} which would "
                    f"clobber sacrosanct slots. Reserved: {sorted(PROTECTED)}. "
                    f"Caller must remove these keys before passing extra_feed."
                )
        feed: dict = {}
        role = self._effective_role_state()
        modes = self._effective_modes(role)

        # Build effective specs: user template_variables + per-enabled-mode
        # entries, unified into a single load_variables call. Multi-dot keys
        # (e.g., "instructions.modes.deep_mode") are handled natively by the
        # enhanced load_variables (which splits on ALL dots, not just the first).
        effective_specs: dict = dict(role.template_variables or {})

        # enable_<name> flags are set unconditionally so {%- if enable_X %}
        # can short-circuit even when False. Mode content is loaded only for
        # enabled modes via load_variables.
        for mode_name, enabled in modes.items():
            feed[f"enable_{mode_name}"] = bool(enabled)
            if enabled:
                effective_specs.setdefault(f"instructions.modes.{mode_name}", None)

        rendering_manager = self._rendering_manager()
        if (
            effective_specs
            and rendering_manager
            and hasattr(rendering_manager, "load_variables")
        ):
            try:
                resolved = rendering_manager.load_variables(
                    variable_specs=effective_specs,
                    root_space=role.template_root_space or "",
                    default_version=role.template_version or "",
                    master_version=role.template_master_version,
                )
            except FileNotFoundError as e:
                import logging

                logging.getLogger(__name__).debug(
                    "Variable not found, degrading gracefully: %s", e
                )
                resolved = {}
            _deep_merge_into(feed, resolved)
        elif role.template_variables:
            for var_name, value in (role.template_variables or {}).items():
                feed[var_name] = value if value else ""

        self._layer_extra_feed(feed, role, extra_feed)
        if role.template_root_space:
            feed["__template_space__"] = role.template_root_space

        if inference_input:
            feed["input"] = inference_input
        # Expose ``has_local_access`` so templates can gate shell/filesystem-only
        # advice (e.g., ``{% if has_local_access %}...{% endif %}``).  Critical
        # for prompts shared between CLI agents (RovoDev, ClaudeCodeCli) and
        # API-only inferencers (RovoChat, Claude API) — the latter cannot run
        # shell commands, so advice about ``grep``/``find``/``cat`` is noise.
        # Coerced to bool via ``getattr`` to handle property-style overrides
        # uniformly and default safely to False for any rare subclass that
        # doesn't define the attribute.
        feed["has_local_access"] = bool(getattr(self, "has_local_access", False))
        resolved = self.resolve_output_path()
        if resolved and os.path.isabs(resolved) and self.has_local_access:
            feed["output_path"] = resolved
        if self.target_path:
            feed.setdefault("target_path", self.target_path)
        ws = getattr(self, "_workspace", None)
        if ws is not None and hasattr(ws, "root") and feed["has_local_access"]:
            feed["workspace_root"] = str(ws.root)
            feed["workspace_outputs"] = os.path.join(str(ws.root), "outputs")
        if self._delegates_execution:
            for key in ACTOR_SCOPED_FEED_KEYS:
                feed.pop(key, None)
        return feed

    # ------------------------------------------------------------------
    # Stub overrides — provide real implementations for InferencerBase's
    # no-op stubs (_render_prompt, _propagate_to_children, supports_prompt_rendering).
    # ------------------------------------------------------------------

    @property
    def supports_prompt_rendering(self) -> bool:
        """True when configured with a template_manager — lets callers query
        whether this inferencer can render a template, without needing to
        actually trigger a render.
        """
        return self.template_manager is not None

    def _rendering_manager(self) -> Optional[Any]:
        """Render-time ``TemplateManager`` with this inferencer's variable
        extensions applied.

        Derived LAZILY on first render (never in ``__attrs_post_init__``) and
        memoized: only after every post-init has finished adding template roots
        (e.g. ``StreamingInferencerBase`` appends its recovery root there) does
        the fork snapshot the fully-built manager. Returns the shared
        ``template_manager`` unchanged -- byte-identical rendering -- when the
        master switch is off, when no ``prompt_templates/_variables`` root is
        discovered for this class, or when no manager is configured.
        """
        if self._extension_manager_cache is not None:
            return self._extension_manager_cache
        tm = self.template_manager
        roots = (
            type(self)._discover_inferencer_variable_roots()
            if tm is not None and self.enable_inferencer_variable_expansion
            else []
        )
        if roots:
            skip_keys = self._inferencer_variable_skip_keys()
            self._warn_unknown_override_keys(roots, skip_keys)
            tm = tm.with_variable_extensions(roots, disabled_keys=skip_keys)
        self._extension_manager_cache = tm
        return self._extension_manager_cache

    def _render_prompt(
        self,
        inference_input: Any,
        *,
        extra_feed: Optional[dict] = None,
    ) -> Any:
        """Render a template-based prompt if ``template_manager`` is configured.

        Called by ``_infer_single`` / ``_ainfer_single`` after
        ``input_preprocessor`` and before ``_infer``.

        Args:
            inference_input: The user/orchestrator input string.
            extra_feed: Optional per-call feed dict (Phase 1, leaf-owned
                template rendering). When provided, merged into the
                template feed via ``_build_template_feed(extra_feed=...)``.
                MUST NOT contain reserved keys ({"input",
                "__template_space__"}) — see ``_build_template_feed``.

        Behavior:

        - If ``template_manager`` is None → pass input through unchanged
          (this leaf was constructed without a template manager — fine).
        - If ``template_manager`` is set but neither ``template_root_space``
          nor ``template_key`` is configured → **raise ``ValueError``**.
          This is misconfiguration: a leaf that explicitly opted into
          templates (via ``template_manager``) but didn't specify which
          template to render. The previous silent pass-through hid bugs.
        - Otherwise → render the template via ``template_manager`` with
          this inferencer's ``template_key`` and ``active_template_root_space``,
          populated by ``_build_template_feed``.

        SUBCLASS OVERRIDE NOTE: Overrides MUST accept ``extra_feed`` (or
        ``**kwargs``) to avoid TypeError when callers pass it. The call
        site (``_*_single``) uses a conditional kwarg pass to support
        legacy overrides — see Round-7 audit in the leaf-rendering plan.
        """
        if self.template_manager is None:
            return inference_input
        # M7 read-flip: resolve the effective role from the active context's
        # RoleState (set by switch_role) when present, else the instance fields.
        eff_key, eff_root, eff_master = self._effective_role()
        if not eff_root and not eff_key:
            raise ValueError(
                f"{type(self).__name__}: template_manager is set but neither "
                f"template_root_space nor template_key is configured — cannot "
                f"resolve a specific template. Either set template_root_space "
                f"(e.g. 'plan' / 'task_breakdown' / 'implementation') or "
                f"template_key (e.g. 'review'). If this inferencer is an "
                f"orchestrator that shouldn't render its own input, it should "
                f"inherit from InferencerBase, not TemplatedInferencerBase."
            )
        feed = self._build_template_feed(inference_input, extra_feed=extra_feed)
        self._capture_rendered_task_instructions(feed, eff_root, eff_master)
        return self._rendering_manager()(
            eff_key,
            active_template_root_space=eff_root,
            master_version=eff_master,
            **feed,
        )

    def _capture_rendered_task_instructions(
        self, feed: dict, eff_root: Optional[str], eff_master: Optional[str]
    ) -> None:
        """Record the fully-resolved ``task_instructions`` THIS leaf just rendered.

        ``task_instructions`` is a predefined variable RE-RESOLVED per leaf, and some
        variants embed actor-scoped placeholders (e.g. ``{{ output_path }}``,
        ``{% if separate_proposal_files %}``). A downstream consumer that re-renders
        it binds those to ITSELF — which is how a reviewer's
        ``<OriginalTaskInstructions>`` came to quote the reviewer's own output path,
        an instruction no author ever received. Recording the value here lets the
        orchestrator relay the *author's* text verbatim instead (see
        ``DualInferencer._representative_proposer``).

        Must run at the leaf's OWN render: ``output_path``/``workspace_outputs``
        resolve through the ACTIVE run-context's workspace, and only during this
        render is that this leaf's workspace. Best-effort — on any failure the
        snapshot stays "" and the consumer simply omits the reference block.
        """
        from agent_foundation.common.inferencers.template_constants import (
            VAR_TASK_INSTRUCTIONS,
        )

        tm = self._rendering_manager()
        if tm is None:
            return
        try:
            raw = feed.get(VAR_TASK_INSTRUCTIONS)
            if not raw and hasattr(tm, "load_variables"):
                # Not pre-placed in the feed — this leaf lets ``__call__``
                # auto-discover the variable. Reproduce that selection with the
                # public loader, most-specific first: an explicitly declared variant,
                # then the master_version (which names the sub-directory holding the
                # variant, and doubles as the version selector), then the generic
                # default. Mirrors how the tool config itself pins a variant
                # (``template_variables.task_instructions: research_propose``).
                role = self._effective_role_state()
                _declared = (role.template_variables or {}).get(VAR_TASK_INSTRUCTIONS)
                for _selector in (_declared, eff_master, None):
                    _loaded = tm.load_variables(
                        variable_specs={VAR_TASK_INSTRUCTIONS: _selector},
                        root_space=eff_root or "",
                        default_version=role.template_version or "",
                        master_version=eff_master,
                    )
                    raw = (_loaded or {}).get(VAR_TASK_INSTRUCTIONS)
                    if raw:
                        break
            if not isinstance(raw, str) or not raw.strip():
                return
            rendered = raw
            if ("{{" in raw or "{%" in raw) and hasattr(tm, "_resolve_templated_feed"):
                # The same seam ``TemplateManager.__call__`` uses, so the snapshot
                # matches what this render actually emitted.
                rendered = tm._resolve_templated_feed(
                    {**feed, VAR_TASK_INSTRUCTIONS: raw},
                    root_space=eff_root or "",
                ).get(VAR_TASK_INSTRUCTIONS)
            # Brace-free invariant: an unresolved placeholder would be re-rendered
            # against the CONSUMER's feed downstream — exactly the leak this prevents.
            if not rendered or "{{" in rendered or "{%" in rendered:
                return
            self._record_rendered_contract(rendered)
        except Exception as exc:  # best-effort snapshot; never break the render
            self.log_debug(
                f"task_instructions snapshot skipped: {type(exc).__name__}: {exc}",
                "TaskInstructionsSnapshot",
            )

    def _record_rendered_contract(self, text: str) -> None:
        """Publish the contract this render produced (``publish_result``): into
        this leaf's invocation, its compat getter outside host mode. A render
        outside the leaf's invocation (a preview on another object, a direct hook
        call) records nothing under a host ctx and writes the getter otherwise."""
        frame = frame_for(self)
        ctx = frame.ctx if frame is not None else active_run_context()
        publish_result(
            self,
            self._RENDERED_CONTRACT,
            RenderedTaskContractState.of(
                text,
                role=self._active_role_name(),
                source_path=ctx.path if ctx is not None else "",
            ),
        )

    def _outcome_for(self, frame: InvocationFrame) -> Optional[NodeOutcomeState]:
        """A templated leaf publishes the task contract it rendered in this call."""
        contract = frame.get(self._RENDERED_CONTRACT)
        return None if contract is None else NodeOutcomeState(task_contract=contract)

    def _proposer_task_instructions(self) -> str:
        """A templated leaf IS an author: report the contract it rendered itself."""
        return self._last_rendered_task_instructions or ""

    def _active_role_state(self):
        """The ``RoleState`` a context-scoped ``switch_role`` recorded for this
        inferencer at the active context node, or ``None``."""
        ctx = active_run_context()
        if ctx is None:
            return None
        state = ctx.node(creator=(type(self).__qualname__, ctx.path)).role_state
        return state if isinstance(state, RoleState) else None

    def _effective_role_state(self) -> _EffectiveRole:
        """M7/B16: every template attribute a render uses — the active context's
        ``RoleState`` value where it sets one, else the instance field. This is the
        full overlay a no-ctx ``switch_role`` applies by writing the fields, so a
        role under a context renders exactly like the same role without one.
        Byte-identical without a context (the instance values)."""
        state = self._active_role_state()
        return _EffectiveRole(
            **{
                name: (
                    getattr(state, name)
                    if state is not None and getattr(state, name) is not None
                    else getattr(self, name)
                )
                for name in ROLE_STATE_ATTRS
            }
        )

    @staticmethod
    def _layer_extra_feed(
        feed: dict, role: _EffectiveRole, extra_feed: Optional[dict]
    ) -> None:
        """Layer the literal feed overrides onto ``feed``, lowest first."""
        feed.update(role.template_extra_feed or {})
        # A templated ancestor's own feed (B17), published for its descendants:
        # over this inferencer's own feed, as the legacy instance push merged it.
        propagated = resolve_propagated(TEMPLATE_PROPAGATED_FEED_HANDLE)
        if propagated:
            feed.update(propagated)
        # Per-call ctx-scoped override (published by an orchestrator into a
        # RunContext handle; resolved by walking up the active ctx tree). Sits
        # ABOVE the instance ``template_extra_feed`` (so an orchestrator can pass
        # per-flow/per-call data — e.g. ``upstream_artifacts`` — without mutating
        # the shared child instance) and BELOW the explicit per-call ``extra_feed``
        # kwarg (a direct caller still wins). Byte-identical when no override is
        # published (``_resolve_ctx_feed_override`` returns None).
        ctx_override = _resolve_ctx_feed_override()
        if ctx_override:
            feed.update(ctx_override)
        if extra_feed:
            feed.update(extra_feed)

    def _effective_modes(self, role: Optional[_EffectiveRole] = None) -> dict:
        """The modes one render uses (B17): the role's or the definition's, under
        the modes a templated ancestor published for its descendants."""
        if role is None:
            role = self._effective_role_state()
        propagated = resolve_propagated(TEMPLATE_PROPAGATED_MODES_HANDLE)
        return {**(role.modes or {}), **(propagated or {})}

    def _effective_role(self):
        """M7: (template_key, template_root_space, template_master_version) of the
        effective role (``_effective_role_state``)."""
        role = self._effective_role_state()
        return role.template_key, role.template_root_space, role.template_master_version

    def _active_role_name(self) -> Optional[str]:
        state = self._active_role_state()
        if state is not None:
            return state.new_role
        return getattr(self, "_applied_role", None)

    def _fanout_role_overrides(self, proto, slot: str) -> Dict[str, Any]:
        """``fresh_instance`` overrides re-roling this inferencer into ``proto``'s
        ``slot``: its own template selectors reset, its effective root space, an
        empty feed, then ``proto``'s slot bundle on top (``modes`` merged)."""
        cls = type(proto)
        node: Dict[str, Any] = {}
        bundle = collect_slot_defaults(cls).get(slot)
        if bundle is not None:
            parent_node = {
                "_target_": f"{cls.__module__}.{cls.__qualname__}",
                **(proto.__dict__.get("_init_recipe") or {}),
            }
            bundle.apply_to(node, parent_node=parent_node)
        overrides = _field_defaults(type(self), self._ROLE_SELECTOR_ATTRS)
        role = self._effective_role_state()
        overrides["template_root_space"] = role.template_root_space
        overrides["template_extra_feed"] = {}
        overrides.update(node)
        overrides["modes"] = {**self._effective_modes(role), **node.get("modes", {})}
        overrides.update(
            _field_defaults(type(self), proto.SELF_SLOT_DROPS.get(slot, ()))
        )
        if not overrides["template_root_space"]:
            raise ValueError(
                f"{type(self).__name__} has no template_root_space to render the "
                f"blank {slot} of its bta_inferencer in; configure the slot explicitly"
            )
        params = set(_init_param_names(type(self)))
        return {k: v for k, v in overrides.items() if k in params}

    def _propagate_to_children(self):
        """Hand ``template_extra_feed`` and ``modes`` down to child inferencers.

        Parent's keys take precedence (update semantics) — runtime context
        set by the orchestrator overrides yaml defaults on children.

        Under a ctx (B17), they are published at this inferencer's own node for
        its descendants, which read them at render time (``_build_template_feed``,
        ``_effective_modes``); no child instance or factory is rewritten, so a
        shared child never accumulates another parent's feed. With no ctx they are
        pushed into the child instances (and factory keywords), a setup-time API:
        each ``InferencerBase`` does this 1 layer, and recursive inference
        propagates through the full hierarchy.

        ``template_version`` and ``template_master_version`` are deliberately
        NOT propagated. They are slot-specific: a BTA's aggregator needs
        ``master_version="aggregation"`` but its breakdown and workers do not.
        Per-slot targeting is handled by ``SLOT_DEFAULTS`` at Hydra
        instantiation time, not by parent-to-child cascade.

        Uses ``_for_each_child_inferencer`` (defined on InferencerBase, the
        generic walker) to discover child instances, partials, and duck-typed
        callables across attrs/dict/list fields.
        """
        ctx = active_run_context()
        if ctx is not None:
            if self._has_child_inferencers():
                role = self._effective_role_state()
                publish_propagated(
                    ctx, TEMPLATE_PROPAGATED_FEED_HANDLE, role.template_extra_feed or {}
                )
                publish_propagated(
                    ctx, TEMPLATE_PROPAGATED_MODES_HANDLE, role.modes or {}
                )
            return
        # Propagate template_extra_feed (the original behavior).
        if self.template_extra_feed:
            self._propagate_dict_attr_to_children(
                self.template_extra_feed,
                TEMPLATE_EXTRA_FEED_ATTR,
            )
        # Propagate modes — same merge semantics so a parent topology can
        # set `modes: {deep_mode: true}` once and have it cascade to every
        # descendant inferencer (no per-child YAML edits required).
        if self.modes:
            self._propagate_dict_attr_to_children(self.modes, "modes")

    def _has_child_inferencers(self) -> bool:
        found = []
        self._for_each_child_inferencer(
            lambda child, field_name, key: found.append(child),
            lambda p, field_name, key: found.append(p),
        )
        return bool(found)

    def _propagate_dict_attr_to_children(self, source: dict, attr_name: str):
        """Helper: merge ``source`` into each child's ``attr_name`` dict.

        Children without this attribute are skipped (they can't receive it).
        Partials get merged kwargs.
        """

        def _on_instance(child, field_name, key):
            existing = getattr(child, attr_name, None)
            if existing is None:
                return
            existing.update(source)

        def _on_partial(p, field_name, key):
            existing = p.keywords.get(attr_name, {})
            merged = {**existing, **source}
            return functools.partial(p.func, **{**p.keywords, attr_name: merged})

        self._for_each_child_inferencer(_on_instance, _on_partial)

    # ------------------------------------------------------------------
    # Layered switch_role() — template-aware extension
    # ------------------------------------------------------------------

    _ROLE_RELEVANT_ATTRS = InferencerBase._ROLE_RELEVANT_ATTRS + (
        "template_key",
        "template_root_space",
        "template_extra_feed",
        "template_variables",
        "template_version",
        "template_master_version",
        "modes",
    )

    _SUPPORTS_BTA_ROLE_MAPPING = True

    # The attributes that select this inferencer's own task template.
    _ROLE_SELECTOR_ATTRS = (
        "template_key",
        "template_version",
        "template_master_version",
        "template_variables",
    )

    def switch_role(
        self,
        new_role,
        *,
        template_key=None,
        template_root_space=None,
        template_extra_feed=None,
        template_variables=None,
        template_version=None,
        template_master_version=None,
        modes=None,
        **base_kwargs,
    ):
        """Template-aware role switch: apply template attrs BEFORE the base
        layer's workspace + session reset, so the new template state is in
        place when the inferencer next renders.

        Template attrs that are not None are handed to the base layer's audit
        trail as a ``RoleTransition``. Under a context they are recorded into the
        node's typed ``RoleState`` (the render reads them through
        ``_effective_role_state``); without one they are set on self, and a switch
        that sets any of them also records ``new_role`` as ``_applied_role``: the
        role the instance fields carry.

        All remaining ``**base_kwargs`` are forwarded to
        ``InferencerBase.switch_role()`` (workspace, deliverable flags, etc.).
        """
        _ctx_active = active_run_context() is not None
        changes = {}
        for attr, val in {
            "template_key": template_key,
            "template_root_space": template_root_space,
            "template_extra_feed": template_extra_feed,
            "template_variables": template_variables,
            "template_version": template_version,
            "template_master_version": template_master_version,
            "modes": modes,
        }.items():
            if val is not None:
                # M7 read-flip: under a context, run-state (role) goes to the
                # context node (recorded below) and NOT onto ``self`` — the
                # definition stays pure (the render pipeline reads via
                # ``_effective_role``). Without a context, mutate ``self``
                # (legacy / byte-identical).
                if not _ctx_active:
                    setattr(self, attr, val)
                changes[attr] = val
        if changes:
            if not _ctx_active:
                object.__setattr__(self, "_applied_role", new_role)
            # Record the role change into the active context node (no-op without one).
            self._record_role_state(new_role, changes)
        super().switch_role(
            new_role, _role_changes=RoleTransition(changes), **base_kwargs
        )

    def _record_role_state(self, new_role, changes):
        """M7: mirror a role switch into ``ctx.node.role_state`` as a ``RoleState``,
        one typed field per attribute it sets."""
        ctx = active_run_context()
        if ctx is None:
            return
        node = ctx.node(creator=(type(self).__qualname__, ctx.path))
        node.role_state = RoleState(
            new_role=new_role,
            changes=dict(changes),
            **{name: changes.get(name) for name in ROLE_STATE_ATTRS},
        )
