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
from typing import Any, Optional

from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import (
    InferencerBase,
    TEMPLATE_EXTRA_FEED_ATTR,
)


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


# ---------------------------------------------------------------------------
# Per-call template-feed override (Tier-3 handle, ctx-scoped, walk-up read)
# ---------------------------------------------------------------------------
#
# An ORCHESTRATOR (e.g. MultiFlowInferencer) needs to pass per-flow / per-call
# data into a SHARED child leaf's wrapper template (``{{ upstream_artifacts }}``)
# WITHOUT mutating the child's instance ``template_extra_feed`` — because under
# shared-instance reuse (one child reused across N flows / one orchestrator
# across N concurrent RunContexts) an instance write clobbers concurrent calls
# (the Decision-5 / state-separation hazard). The seam mirrors the proven
# ``_publish_workspace_to_ctx`` → ctx-handle-read pattern: the orchestrator
# publishes a per-call feed dict into a RunContext handle (under a key the leaf
# never collides with), and the leaf merges it at render time.
#
# Resolution is a WALK-UP of the active ctx's ancestor paths (the same mechanism
# MFI's ``_resolve_attempt_state`` uses): the override is published at the
# orchestrator's own node or a child node, and the leaf — which renders under
# its OWN (descendant) ctx — finds it by walking up. This keeps the publish
# robust without the orchestrator reconstructing the leaf's exact child path
# (e.g. LWI's ``step_{i}`` slot, which the dynamic-input builder cannot know).
TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE: str = "__template_extra_feed_override__"


def _resolve_ctx_feed_override() -> Optional[dict]:
    """Return the per-call ``template_extra_feed`` override dict published into the
    active RunContext's handle tree (walking UP ancestor paths), else ``None``.

    Byte-identical with no active ctx / no published override (returns ``None`` →
    the feed is built exactly as before). Handles are path-keyed and NOT inherited
    by child contexts (``handles.py``), so a leaf rendering at ``/flow_0/step_1``
    must walk up to find an override published at ``/flow_0`` (or the root).
    """
    from agent_foundation.common.inferencers.run_context import active_run_context

    ctx = active_run_context()
    if ctx is None:
        return None
    store = ctx._handle_store
    # Enumerate ancestor paths from the active path up to the root, e.g.
    # "/flow_0/step_1" -> ["/flow_0/step_1", "/flow_0", "/"].
    segments = [s for s in ctx.path.split("/") if s]
    seen: set = set()
    for i in range(len(segments), -1, -1):
        path = "/" + "/".join(segments[:i])
        if path in seen:
            continue
        seen.add(path)
        handles = store.peek(path)
        if handles is None:
            continue
        override = handles.get(TEMPLATE_EXTRA_FEED_OVERRIDE_HANDLE, None)
        if isinstance(override, dict) and override:
            return override
    return None


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

        # Build effective specs: user template_variables + per-enabled-mode
        # entries, unified into a single load_variables call. Multi-dot keys
        # (e.g., "instructions.modes.deep_mode") are handled natively by the
        # enhanced load_variables (which splits on ALL dots, not just the first).
        effective_specs: dict = dict(self.template_variables or {})

        # enable_<name> flags are set unconditionally so {%- if enable_X %}
        # can short-circuit even when False. Mode content is loaded only for
        # enabled modes via load_variables.
        for mode_name, enabled in (self.modes or {}).items():
            feed[f"enable_{mode_name}"] = bool(enabled)
            if enabled:
                effective_specs.setdefault(
                    f"instructions.modes.{mode_name}", None
                )

        if effective_specs and self.template_manager and hasattr(self.template_manager, "load_variables"):
            try:
                resolved = self.template_manager.load_variables(
                    variable_specs=effective_specs,
                    root_space=self.template_root_space or "",
                    default_version=self.template_version or "",
                    master_version=self.template_master_version,
                )
            except FileNotFoundError as e:
                import logging
                logging.getLogger(__name__).debug(
                    "Variable not found, degrading gracefully: %s", e
                )
                resolved = {}
            _deep_merge_into(feed, resolved)
        elif self.template_variables:
            for var_name, value in (self.template_variables or {}).items():
                feed[var_name] = value if value else ""

        feed.update(self.template_extra_feed)
        # Per-call ctx-scoped override (published by an orchestrator into a
        # RunContext handle; resolved by walking up the active ctx tree). Sits
        # ABOVE the instance ``template_extra_feed`` (so an orchestrator can pass
        # per-flow/per-call data — e.g. ``upstream_artifacts`` — without mutating
        # the shared child instance) and BELOW the explicit per-call ``extra_feed``
        # kwarg (a direct caller still wins). Byte-identical when no override is
        # published (``_resolve_ctx_feed_override`` returns None).
        _ctx_override = _resolve_ctx_feed_override()
        if _ctx_override:
            feed.update(_ctx_override)
        if extra_feed:
            feed.update(extra_feed)
        if self.template_root_space:
            feed["__template_space__"] = self.template_root_space

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
        return self.template_manager(
            eff_key,
            active_template_root_space=eff_root,
            master_version=eff_master,
            **feed,
        )

    def _effective_role(self):
        """M7: (template_key, template_root_space, template_master_version) —
        from the active context's RoleState when set, else the instance fields.
        Byte-identical without a context (returns the instance values)."""
        from agent_foundation.common.inferencers.run_context import (
            RoleState,
            active_run_context,
        )

        key = self.template_key
        root = self.template_root_space
        master = self.template_master_version
        ctx = active_run_context()
        if ctx is not None:
            call = ctx.node(creator=(type(self).__qualname__, ctx.path)).call
            if isinstance(call, RoleState):
                if call.template_key is not None:
                    key = call.template_key
                if call.template_root_space is not None:
                    root = call.template_root_space
                if call.template_version is not None:
                    master = call.template_version
        return key, root, master

    def _propagate_to_children(self):
        """Push ``template_extra_feed`` and ``modes`` to child inferencers.

        Parent's keys take precedence (update semantics) — runtime context
        set by the orchestrator overrides yaml defaults on children. Each
        ``InferencerBase`` does this 1 layer; recursive inference naturally
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
        # Propagate template_extra_feed (the original behavior).
        if self.template_extra_feed:
            self._propagate_dict_attr_to_children(
                self.template_extra_feed, TEMPLATE_EXTRA_FEED_ATTR,
            )
        # Propagate modes — same merge semantics so a parent topology can
        # set `modes: {deep_mode: true}` once and have it cascade to every
        # descendant inferencer (no per-child YAML edits required).
        if self.modes:
            self._propagate_dict_attr_to_children(self.modes, "modes")

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
            return functools.partial(
                p.func, **{**p.keywords, attr_name: merged}
            )

        self._for_each_child_inferencer(_on_instance, _on_partial)

    # ------------------------------------------------------------------
    # Layered switch_role() — template-aware extension
    # ------------------------------------------------------------------

    _ROLE_RELEVANT_ATTRS = InferencerBase._ROLE_RELEVANT_ATTRS + (
        "template_key", "template_root_space", "template_extra_feed",
        "template_variables", "template_version", "template_master_version",
        "modes",
    )

    def switch_role(self, new_role, *, template_key=None, template_root_space=None,
                    template_extra_feed=None, template_variables=None,
                    template_version=None, template_master_version=None,
                    modes=None, **base_kwargs):
        """Template-aware role switch: apply template attrs BEFORE the base
        layer's workspace + session reset, so the new template state is in
        place when the inferencer next renders.

        Template attrs that are not None are set on self and stashed in
        ``_pending_role_changes`` so the base layer's audit trail merges them.

        All remaining ``**base_kwargs`` are forwarded to
        ``InferencerBase.switch_role()`` (workspace, deliverable flags, etc.).
        """
        from agent_foundation.common.inferencers.run_context import (
            active_run_context,
        )

        _ctx_active = active_run_context() is not None
        changes = {}
        for attr, val in {"template_key": template_key, "template_root_space": template_root_space,
                          "template_extra_feed": template_extra_feed, "template_variables": template_variables,
                          "template_version": template_version, "template_master_version": template_master_version,
                          "modes": modes}.items():
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
            object.__setattr__(self, "_pending_role_changes", changes)
            # Record the role change into the active context node (no-op without one).
            self._record_role_state(new_role, changes)
        super().switch_role(new_role, **base_kwargs)

    def _record_role_state(self, new_role, changes):
        """M7: mirror a role switch into ``ctx.node.call`` as a ``RoleState``."""
        from agent_foundation.common.inferencers.run_context import (
            RoleState,
            active_run_context,
        )

        ctx = active_run_context()
        if ctx is None:
            return
        node = ctx.node(creator=(type(self).__qualname__, ctx.path))
        node.call = RoleState(
            new_role=new_role,
            template_key=changes.get("template_key"),
            template_root_space=changes.get("template_root_space"),
            template_version=changes.get("template_version"),
            modes=changes.get("modes"),
            changes=dict(changes),
        )
