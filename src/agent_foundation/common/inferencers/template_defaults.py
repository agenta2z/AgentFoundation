"""Reusable bundles of template field defaults for inferencer slots.

Each :class:`InferencerTemplateDefaults` instance describes one semantic
role (breakdown, aggregation, review, etc.) and knows how to merge its
defaults into a YAML node before Hydra instantiation.

Application semantics:
- Per-key merge for dict fields (user-supplied keys win)
- Scalar fill for ``template_root_space`` / ``template_key`` (only fills
  if the field is absent from the node)
- Optional ``parent_node`` for conditional application (see
  :class:`ConditionalTemplateDefaults`)

Module-level constants (:data:`AGGREGATION_TEMPLATE_DEFAULTS`, etc.) are
the named, reusable bundles that orchestrator classes wire into their
``SLOT_DEFAULTS`` ClassVar. Importing the same constant from multiple
orchestrators (e.g. BTA's aggregator AND MFDual's
multi_flow_aggregator) is the point — "aggregation framing" lives in
exactly one place.

The Hydra walker in ``rich_python_utils.config_utils._instantiate``
duck-types these objects via ``hasattr(obj, "apply_to")`` and so does
NOT depend on this module — the framework primitive lives here, the
walker just consumes it generically.
"""

from __future__ import annotations

import copy
import importlib
import logging
from typing import Any, Callable, Optional

import attr

from .template_constants import (
    FIELD_TEMPLATE_EXTRA_FEED,
    FIELD_TEMPLATE_KEY,
    FIELD_TEMPLATE_MASTER_VERSION,
    FIELD_TEMPLATE_ROOT_SPACE,
    FIELD_TEMPLATE_VARIABLES,
    FIELD_TEMPLATE_VERSION,
    KEY_FOLLOWUP,
    KEY_REVIEW,
    SPACE_TASK_BREAKDOWN,
    VAR_TASK_INSTRUCTIONS,
    VAR_TASK_PREAMBLE,
    VAR_TASK_RESPONSE_FORMAT,
    VARIANT_AGGREGATION,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core primitive
# ---------------------------------------------------------------------------


class InferencerTemplateDefaults:
    """Reusable bundle of template field defaults for one slot role.

    Construct with any subset of the five template fields. ``apply_to``
    merges into a YAML node: scalar fields fill iff absent; dict fields
    merge per-key (user-supplied keys override defaults).
    """

    def __init__(
        self,
        *,
        template_root_space: Optional[str] = None,
        template_key: Optional[str] = None,
        template_variables: Optional[dict] = None,
        template_extra_feed: Optional[dict] = None,
        template_version: Optional[str] = None,
        template_master_version: Optional[str] = None,
        modes: Optional[dict] = None,
    ):
        self.template_root_space = template_root_space
        self.template_key = template_key
        self.template_variables = (
            dict(template_variables) if template_variables else {}
        )
        self.template_extra_feed = (
            dict(template_extra_feed) if template_extra_feed else {}
        )
        self.template_version = template_version
        self.template_master_version = template_master_version
        self.modes = dict(modes) if modes else {}

    def apply_to(
        self, node: dict, parent_node: Optional[dict] = None
    ) -> None:
        """Mutate ``node`` in place: fill missing fields, per-key merge dicts.

        ``parent_node`` is unused here but accepted for ABI compatibility
        with :class:`ConditionalTemplateDefaults`.
        """
        if not isinstance(node, dict):
            return
        if (
            self.template_root_space is not None
            and FIELD_TEMPLATE_ROOT_SPACE not in node
        ):
            node[FIELD_TEMPLATE_ROOT_SPACE] = self.template_root_space
        if (
            self.template_key is not None
            and FIELD_TEMPLATE_KEY not in node
        ):
            node[FIELD_TEMPLATE_KEY] = self.template_key
        if (
            self.template_version is not None
            and FIELD_TEMPLATE_VERSION not in node
        ):
            node[FIELD_TEMPLATE_VERSION] = self.template_version
        if (
            self.template_master_version is not None
            and FIELD_TEMPLATE_MASTER_VERSION not in node
        ):
            node[FIELD_TEMPLATE_MASTER_VERSION] = self.template_master_version
        if self.template_variables:
            self._merge_dict(node, FIELD_TEMPLATE_VARIABLES, self.template_variables)
        if self.template_extra_feed:
            self._merge_dict(node, FIELD_TEMPLATE_EXTRA_FEED, self.template_extra_feed)
        if self.modes:
            self._merge_dict(node, "modes", self.modes)

    @staticmethod
    def _merge_dict(node: dict, field: str, defaults: dict) -> None:
        """Per-key merge: defaults are the base, user-supplied keys win."""
        existing = node.get(field) or {}
        merged = copy.deepcopy(defaults)
        merged.update(existing)
        node[field] = merged


class InferencerTemplateVersionDefaults(InferencerTemplateDefaults):
    """Variant of :class:`InferencerTemplateDefaults` that declares a single
    ``template_version`` and a list of variable names, instead of a
    per-key dict.

    The constructor sugars ``variable_names=["a", "b"]`` into
    ``template_variables={"a": None, "b": None}`` so that ``_build_template_feed``
    sees empty values and falls back to the default version. This lets the
    constants stay terse::

        AGGREGATION_DEFAULTS = InferencerTemplateVersionDefaults(
            template_master_version="aggregation",
            variable_names=["task_preamble", "task_instructions", "task_response_format"],
        )

    is equivalent in effect to::

        InferencerTemplateDefaults(
            template_master_version="aggregation",
            template_variables={"task_preamble": None, "task_instructions": None, "task_response_format": None},
        )

    All apply_to merge semantics are inherited unchanged from the parent.
    """

    def __init__(
        self,
        *,
        template_version: Optional[str] = None,
        template_master_version: Optional[str] = None,
        variable_names: Optional[list] = None,
        template_root_space: Optional[str] = None,
        template_key: Optional[str] = None,
        template_variables: Optional[dict] = None,
        template_extra_feed: Optional[dict] = None,
        modes: Optional[dict] = None,
    ):
        # Sugar variable_names -> template_variables with None values.
        # Per-key explicit values in template_variables (if also supplied)
        # win over the variable_names-derived None entries.
        merged_variables: dict = {}
        if variable_names:
            merged_variables.update({name: None for name in variable_names})
        if template_variables:
            merged_variables.update(template_variables)
        super().__init__(
            template_root_space=template_root_space,
            template_key=template_key,
            template_variables=merged_variables or None,
            template_extra_feed=template_extra_feed,
            template_version=template_version,
            template_master_version=template_master_version,
            modes=modes,
        )


class ConditionalTemplateDefaults(InferencerTemplateDefaults):
    """Apply only when a predicate on the parent YAML node returns True.

    The predicate receives the parent orchestrator node (the YAML dict
    declaring the slot whose child is being defaulted), letting the
    condition gate on parent-level flags like ``inject_upstream_artifacts``.
    """

    def __init__(self, *, condition: Callable[[dict], bool], **kwargs):
        super().__init__(**kwargs)
        self.condition = condition

    def apply_to(
        self, node: dict, parent_node: Optional[dict] = None
    ) -> None:
        if parent_node is None or not self.condition(parent_node):
            return
        super().apply_to(node, parent_node=parent_node)


# ---------------------------------------------------------------------------
# Class-default lookup helper (for conditional defaults that need to know
# what an orchestrator's attribute defaults to when the YAML omits the key)
# ---------------------------------------------------------------------------


def _resolve_attrib_default(target: Any, field_name: str, fallback: Any) -> Any:
    """Look up an attrs field's default value on the class named by ``target``.

    ``target`` can be a fully-qualified import string (``"a.b.C"``) or
    ``None``. On any failure (unresolvable import, attrs.NOTHING default,
    factory failure, etc.) the supplied ``fallback`` is returned.

    Used by gating predicates so a YAML that omits a flag (e.g.
    ``visible_flows``) is still gated on the class default the user is
    implicitly opting into.
    """
    if not isinstance(target, str):
        return fallback
    try:
        module_path, _, attr_name = target.rpartition(".")
        if not module_path:
            return fallback
        module = importlib.import_module(module_path)
        cls = getattr(module, attr_name, None)
        if cls is None:
            return fallback
        for f in attr.fields(cls):
            if f.name != field_name:
                continue
            if f.default is attr.NOTHING:
                return fallback
            if isinstance(f.default, attr.Factory):
                try:
                    return f.default.factory()
                except Exception:  # noqa: BLE001 — best-effort
                    return fallback
            return f.default
        return fallback
    except Exception as exc:  # noqa: BLE001 — best-effort
        _logger.debug(
            "_resolve_attrib_default(%r, %r) failed: %s", target, field_name, exc
        )
        return fallback


# ---------------------------------------------------------------------------
# Module-level constants — named, reusable defaults
# ---------------------------------------------------------------------------


BREAKDOWN_TEMPLATE_DEFAULTS = InferencerTemplateDefaults(
    template_root_space=SPACE_TASK_BREAKDOWN,
)
"""For BTA's ``breakdown_inferencer``: render against the
``task_breakdown`` template space."""


AGGREGATION_DEFAULTS = InferencerTemplateVersionDefaults(
    template_version=VARIANT_AGGREGATION,
    template_master_version=VARIANT_AGGREGATION,
    modes={"deep_mode": False},
)
"""Aggregation framing — preamble + instructions + response_format.

``master_version="aggregation"`` routes variable lookups into the
``aggregation/`` subdirectory (e.g., ``task_preamble/aggregation/default.jinja2``).
Per-variable overrides in YAML (e.g., ``task_instructions: skill_tool_creation``)
select a specific variant within that subdirectory.

Used by BTA's ``aggregator_inferencer``, MFDual's
``multi_flow_aggregator_inferencer``, and :data:`FOLLOWUP_AGGREGATION_DEFAULTS`
(per-flow followup)."""


REVIEW_TEMPLATE_DEFAULTS = InferencerTemplateDefaults(
    template_key=KEY_REVIEW,
    modes={"elegant_mode": False},
)
"""For any review slot: render the canonical ``review`` template variant.

Template variables (``task_instructions``, ``task_preamble``, etc.) are
auto-discovered by the TemplateManager's ``predefined_variables: true``
infrastructure — no explicit ``variable_names`` needed.  The
VariableLoader's Pass-3 folder-default fallback finds
``_variables/<name>/default.jinja2`` even when ``version=""``."""


FOLLOWUP_TEMPLATE_DEFAULTS = InferencerTemplateDefaults(
    template_key=KEY_FOLLOWUP,
)
"""For any followup/fixer slot: render the canonical ``followup`` template variant.

Wired into ``Dual.SLOT_DEFAULTS["fixer_inferencer"]`` so the fixer leaf
renders ``plan/main/followup.jinja2`` (or the active space equivalent)
with auto-discovered template variables."""


def _aggregation_applicable(parent: dict) -> bool:
    """True iff peers are visible (semantic intent) AND inject is wired
    (mechanical engagement).

    Class-default-aware via :func:`_resolve_attrib_default`. Class defaults:

    * ``MultiFlowInferencer``: ``visible_flows="self"``,
      ``inject_upstream_artifacts=False`` — simple parallel sampling,
      no peer aggregation; bare MF YAMLs skip the followup default.
    * ``MultiFlowDualInferencer``: ``visible_flows="all"``,
      ``inject_upstream_artifacts=True`` — peer-aware aggregation by
      design; bare MFDual YAMLs apply the followup default automatically.

    Either condition False → followup runtime path doesn't consume
    aggregation framing → defaults would be dead weight, so skip.
    """
    target = parent.get("_target_")
    visible = parent.get("visible_flows")
    if visible is None:
        visible = _resolve_attrib_default(target, "visible_flows", "self")
    if visible == "self":
        return False
    inject = parent.get("inject_upstream_artifacts")
    if inject is None:
        inject = _resolve_attrib_default(target, "inject_upstream_artifacts", False)
    return bool(inject)


FOLLOWUP_AGGREGATION_DEFAULTS = ConditionalTemplateDefaults(
    condition=_aggregation_applicable,
    template_variables=AGGREGATION_DEFAULTS.template_variables,
    template_version=AGGREGATION_DEFAULTS.template_version,
    template_master_version=AGGREGATION_DEFAULTS.template_master_version,
)
"""For per-flow MultiFlow/MFDual ``followup_inferencer`` when (a) peers
are visible (``visible_flows != "self"``) AND (b) injection wiring is
engaged (``inject_upstream_artifacts=True``). Both are class-default-
aware so the YAML doesn't need to spell out flags it's already
implicitly opting into via the parent class choice (e.g. MFDual
already defaults ``visible_flows="all"``)."""


__all__ = [
    "InferencerTemplateDefaults",
    "InferencerTemplateVersionDefaults",
    "ConditionalTemplateDefaults",
    "BREAKDOWN_TEMPLATE_DEFAULTS",
    "AGGREGATION_DEFAULTS",
    "REVIEW_TEMPLATE_DEFAULTS",
    "FOLLOWUP_TEMPLATE_DEFAULTS",
    "FOLLOWUP_AGGREGATION_DEFAULTS",
]
