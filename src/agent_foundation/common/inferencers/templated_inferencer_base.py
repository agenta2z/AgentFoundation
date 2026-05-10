"""
TemplatedInferencerBase - InferencerBase variant for inferencers that render
their own ``inference_input`` through a Jinja ``TemplateManager`` before LLM call.

Architectural role::

    InferencerBase (template-free)
    ├── TemplatedInferencerBase  ← THIS FILE — leaves & their bases inherit
    │   ├── ApiInferencerBase    (and the leaves under it)
    │   ├── StreamingInferencerBase
    │   ├── RemoteInferencerBase
    │   └── TerminalInferencerBase
    └── (orchestrators) — Dual, BTA, LWI, PTI, MultiFlow, MultiFlowDual,
        Conversational* — DO NOT inherit from this; they delegate inference
        to children rather than rendering their own input.

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
        # Resolve template_variables via load_variables() (unified batch API
        # with cascade: space/type → space → global _variables/).
        # Prefix conventions: "@variant" = strict, "=literal" = force literal,
        # "variant" = try file then fall back to literal, None = use template_version.
        if self.template_variables and self.template_manager:
            if hasattr(self.template_manager, "load_variables"):
                resolved = self.template_manager.load_variables(
                    variable_specs=self.template_variables,
                    root_space=self.template_root_space,
                    default_version=self.template_version or "",
                )
                feed.update(resolved)
            else:
                for var_name, value in self.template_variables.items():
                    feed[var_name] = value if value else ""
        feed.update(self.template_extra_feed)
        # Phase 1 (leaf-owned rendering): per-call extra_feed merge — wins
        # over class-level template_extra_feed but loses to {{ input }} and
        # __template_space__ (sacrosanct, set after this).
        if extra_feed:
            feed.update(extra_feed)
        # Inject __template_space__ so .variables.yaml aliases like
        # `__action__: __template_space__` resolve to the active space
        # ("plan" / "implementation" / "task_breakdown"). Sourced from
        # template_root_space (semantically the same value).
        if self.template_root_space:
            feed["__template_space__"] = self.template_root_space

        # Mode handling: for each declared mode, set enable_<name> bool
        # and (if enabled) load instruction content into nested dict for
        # {{ instructions.modes.<name> }} Jinja2 access.
        # Note: load_variables() in TemplateManager only splits on the
        # FIRST dot (so "instructions.modes.deep_mode" resolves wrong).
        # We use _cascade_load_variable directly for proper nested access.
        if self.modes:
            self._inject_mode_flags_and_content(feed)

        feed["input"] = inference_input
        resolved = self.resolve_output_path()
        if resolved and os.path.isabs(resolved) and self.has_local_access:
            feed["output_path"] = resolved
        return feed

    def _inject_mode_flags_and_content(self, feed: dict) -> None:
        """Populate `enable_<name>` flags and `instructions.modes.<name>`
        content from `self.modes`.

        Flag derivation is unconditional (so `{%- if enable_X %}` can
        check the value even when False/missing). Content loading happens
        only for enabled modes — disabled modes contribute nothing.

        Errors are handled defensively but observably:
          - FileNotFoundError → debug log (mode declared but no instruction
            file; rendering proceeds, the conditional block emits nothing
            of value).
          - Any other exception → warning log (don't silently swallow real
            bugs; explicitly NOT `except Exception: pass`).
        """
        import logging
        logger = logging.getLogger(__name__)

        for mode_name, enabled in self.modes.items():
            feed[f"enable_{mode_name}"] = bool(enabled)
            if not enabled:
                continue
            if not self.template_manager:
                continue
            # Use _cascade_load_variable for proper nested-folder lookup.
            # Args: var_name (folder path with /), version (file stem),
            #       root_space, tmpl_type.
            try:
                content = self.template_manager._cascade_load_variable(
                    "instructions/modes",
                    mode_name,
                    self.template_root_space or "",
                    "main",
                )
            except FileNotFoundError:
                logger.debug(
                    "Mode '%s' enabled but no instruction file at "
                    "_variables/instructions/modes/%s.jinja2 — "
                    "{{ instructions.modes.%s }} will render empty.",
                    mode_name, mode_name, mode_name,
                )
                continue
            except Exception as e:
                logger.warning(
                    "Failed to load mode instructions for '%s': %s",
                    mode_name, e,
                )
                continue

            if content is None:
                # File doesn't exist (cascade returned None, didn't raise).
                continue

            # Merge into feed as {"instructions": {"modes": {<name>: <content>}}}
            instructions = feed.setdefault("instructions", {})
            if not isinstance(instructions, dict):
                # Don't clobber a non-dict 'instructions' set by extra_feed
                logger.warning(
                    "feed['instructions'] is %s, not dict — cannot inject "
                    "modes.%s. Skipping.",
                    type(instructions).__name__, mode_name,
                )
                continue
            modes_ns = instructions.setdefault("modes", {})
            if not isinstance(modes_ns, dict):
                logger.warning(
                    "feed['instructions']['modes'] is %s, not dict — "
                    "cannot inject modes.%s. Skipping.",
                    type(modes_ns).__name__, mode_name,
                )
                continue
            modes_ns[mode_name] = content

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
        if not self.template_root_space and not self.template_key:
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
            self.template_key,
            active_template_root_space=self.template_root_space,
            **feed,
        )

    def _propagate_to_children(self):
        """Push ``template_extra_feed`` and ``modes`` to child inferencers.

        Parent's keys take precedence (update semantics) — runtime context
        set by the orchestrator overrides yaml defaults on children. Each
        ``InferencerBase`` does this 1 layer; recursive inference naturally
        propagates through the full hierarchy.

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
