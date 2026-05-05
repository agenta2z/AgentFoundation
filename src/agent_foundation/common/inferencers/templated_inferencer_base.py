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

    # ------------------------------------------------------------------
    # Template feed construction
    # ------------------------------------------------------------------

    def _build_template_feed(self, inference_input: str) -> dict:
        """Build the template variable feed dict.

        Merges (in priority order, lowest first):

        1. ``template_variables`` — variant selectors resolved to file content
           via ``template_manager.load_variable()``.  E.g.,
           ``{"task_preamble": "skill_tool_creation"}`` loads
           ``_variables/task_preamble/skill_tool_creation.jinja2``.
        2. ``template_extra_feed`` — literal key-value overrides.
        3. ``{{ input }}`` bound to ``inference_input``.
        4. ``output_path`` (if inferencer has local file access).

        Override this method to customize feed construction (e.g., add
        dynamic variables from external sources).
        """
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
        feed["input"] = inference_input
        resolved = self.resolve_output_path()
        if resolved and os.path.isabs(resolved) and self.has_local_access:
            feed["output_path"] = resolved
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

    def _render_prompt(self, inference_input: Any) -> Any:
        """Render a template-based prompt if ``template_manager`` is configured.

        Called by ``_infer_single`` / ``_ainfer_single`` after
        ``input_preprocessor`` and before ``_infer``.

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
        feed = self._build_template_feed(inference_input)
        return self.template_manager(
            self.template_key,
            active_template_root_space=self.template_root_space,
            **feed,
        )

    def _propagate_to_children(self):
        """Push ``template_extra_feed`` to child inferencers found in attrs fields.

        Parent's keys take precedence (update semantics) — runtime context
        set by the orchestrator overrides yaml defaults on children. Each
        ``InferencerBase`` does this 1 layer; recursive inference naturally
        propagates through the full hierarchy.

        Uses ``_for_each_child_inferencer`` (defined on InferencerBase, the
        generic walker) to discover child instances, partials, and duck-typed
        callables across attrs/dict/list fields.
        """
        if not self.template_extra_feed:
            return

        feed = self.template_extra_feed
        attr_name = TEMPLATE_EXTRA_FEED_ATTR

        def _on_instance(child, field_name, key):
            # Children that don't have template_extra_feed (e.g., orchestrators
            # that inherit InferencerBase directly) are skipped — nowhere to
            # merge into. Children that DO have it (TemplatedInferencerBase
            # descendants + duck-typed callables) receive the merged dict.
            existing = getattr(child, attr_name, None)
            if existing is None:
                return
            existing.update(feed)

        def _on_partial(p, field_name, key):
            existing = p.keywords.get(attr_name, {})
            merged = {**existing, **feed}
            return functools.partial(
                p.func, **{**p.keywords, attr_name: merged}
            )

        self._for_each_child_inferencer(_on_instance, _on_partial)
