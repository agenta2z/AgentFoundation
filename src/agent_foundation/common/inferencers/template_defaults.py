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
    FIELD_TEMPLATE_ROOT_SPACE,
    FIELD_TEMPLATE_VARIABLES,
    FIELD_TEMPLATE_VERSION,
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
        if self.template_variables:
            self._merge_dict(node, FIELD_TEMPLATE_VARIABLES, self.template_variables)
        if self.template_extra_feed:
            self._merge_dict(node, FIELD_TEMPLATE_EXTRA_FEED, self.template_extra_feed)

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
    sees empty values and falls back to ``template_version``. This lets the
    migration constants stay terse::

        AGGREGATOR_PREAMBLE_DEFAULTS = InferencerTemplateVersionDefaults(
            template_version="aggregation",
            variable_names=["task_preamble"],
        )

    is equivalent in effect to::

        InferencerTemplateDefaults(
            template_version="aggregation",
            template_variables={"task_preamble": None},
        )

    All apply_to merge semantics are inherited unchanged from the parent.
    """

    def __init__(
        self,
        *,
        template_version: Optional[str] = None,
        variable_names: Optional[list] = None,
        template_root_space: Optional[str] = None,
        template_key: Optional[str] = None,
        template_variables: Optional[dict] = None,
        template_extra_feed: Optional[dict] = None,
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


AGGREGATOR_PREAMBLE_DEFAULTS = InferencerTemplateVersionDefaults(
    template_version=VARIANT_AGGREGATION,
    variable_names=[VAR_TASK_PREAMBLE],
)
"""Universal aggregator minimum — just the ``aggregation`` task_preamble
variant, which renders the wrapper that consumes ``{{ upstream_artifacts }}``
and ``{{ aggregation_guidance }}``.

Used by BTA's ``aggregator_inferencer``. Deliberately NOT the full triplet:
some aggregators (e.g. exec BTA in plan-then-implement) want generic
synthesis instructions sourced from runtime ``aggregation_guidance`` only,
and would conflict with a hardcoded ``task_instructions: aggregation``.
Use :data:`STRUCTURED_AGGREGATION_DEFAULTS` for slots that need the full
structured-aggregation framing (with addenda rendering)."""


STRUCTURED_AGGREGATION_DEFAULTS = InferencerTemplateVersionDefaults(
    template_version=VARIANT_AGGREGATION,
    variable_names=[
        VAR_TASK_PREAMBLE,
        VAR_TASK_INSTRUCTIONS,
        VAR_TASK_RESPONSE_FORMAT,
    ],
)
"""Full structured-aggregation framing — preamble + generic instructions +
response_format that renders structured-output addenda (e.g.,
``include_winner_pick``, ``include_iteration_judgment``) when the
corresponding flag is set in ``template_extra_feed``.

Used by MFDual's ``multi_flow_aggregator_inferencer`` and
:data:`FOLLOWUP_AGGREGATION_DEFAULTS` (per-flow followup). Callers that
want only the universal aggregator preamble use
:data:`AGGREGATOR_PREAMBLE_DEFAULTS` instead."""


REVIEW_TEMPLATE_DEFAULTS = InferencerTemplateDefaults(
    template_key=KEY_REVIEW,
)
"""For any review slot: render the canonical ``review`` template variant."""


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
    template_variables=STRUCTURED_AGGREGATION_DEFAULTS.template_variables,
    template_version=STRUCTURED_AGGREGATION_DEFAULTS.template_version,
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
    "AGGREGATOR_PREAMBLE_DEFAULTS",
    "STRUCTURED_AGGREGATION_DEFAULTS",
    "REVIEW_TEMPLATE_DEFAULTS",
    "FOLLOWUP_AGGREGATION_DEFAULTS",
]
