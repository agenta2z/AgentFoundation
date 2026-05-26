"""Unit tests for InferencerTemplateDefaults primitive + named constants.

Covers:
- scalar fill (template_root_space, template_key) — only fills if absent
- per-key dict merge (template_variables, template_extra_feed) — user wins
- empty-string opt-out — survives merge
- non-dict node — apply_to is a no-op
- ConditionalTemplateDefaults — condition gates application
- _resolve_attrib_default — class-default lookup for conditional gating
- Module-level constants have the expected shape
"""

from __future__ import annotations

import pytest

from agent_foundation.common.inferencers.template_constants import (
    FIELD_TEMPLATE_EXTRA_FEED,
    FIELD_TEMPLATE_KEY,
    FIELD_TEMPLATE_ROOT_SPACE,
    FIELD_TEMPLATE_VARIABLES,
    KEY_REVIEW,
    SPACE_TASK_BREAKDOWN,
    VAR_TASK_INSTRUCTIONS,
    VAR_TASK_PREAMBLE,
    VAR_TASK_RESPONSE_FORMAT,
    VARIANT_AGGREGATION,
)
from agent_foundation.common.inferencers.template_defaults import (
    AGGREGATION_DEFAULTS,
    BREAKDOWN_TEMPLATE_DEFAULTS,
    ConditionalTemplateDefaults,
    FOLLOWUP_AGGREGATION_DEFAULTS,
    FOLLOWUP_TEMPLATE_DEFAULTS,
    InferencerTemplateDefaults,
    InferencerTemplateVersionDefaults,
    REVIEW_TEMPLATE_DEFAULTS,
    _aggregation_applicable,
    _resolve_attrib_default,
)


# ---------------------------------------------------------------------------
# Scalar field fill
# ---------------------------------------------------------------------------


class TestScalarFill:
    def test_fills_template_root_space_when_absent(self):
        defaults = InferencerTemplateDefaults(template_root_space="task_breakdown")
        node: dict = {}
        defaults.apply_to(node)
        assert node == {FIELD_TEMPLATE_ROOT_SPACE: "task_breakdown"}

    def test_does_not_overwrite_existing_template_root_space(self):
        defaults = InferencerTemplateDefaults(template_root_space="task_breakdown")
        node = {FIELD_TEMPLATE_ROOT_SPACE: "implementation"}
        defaults.apply_to(node)
        assert node[FIELD_TEMPLATE_ROOT_SPACE] == "implementation"

    def test_fills_template_key_when_absent(self):
        defaults = InferencerTemplateDefaults(template_key="review")
        node: dict = {}
        defaults.apply_to(node)
        assert node == {FIELD_TEMPLATE_KEY: "review"}

    def test_does_not_overwrite_existing_template_key(self):
        defaults = InferencerTemplateDefaults(template_key="review")
        node = {FIELD_TEMPLATE_KEY: "initial"}
        defaults.apply_to(node)
        assert node[FIELD_TEMPLATE_KEY] == "initial"

    def test_empty_string_template_key_is_treated_as_present_user_value(self):
        # Empty string is a deliberate user value and should NOT be overwritten.
        defaults = InferencerTemplateDefaults(template_key="review")
        node = {FIELD_TEMPLATE_KEY: ""}
        defaults.apply_to(node)
        assert node[FIELD_TEMPLATE_KEY] == ""


# ---------------------------------------------------------------------------
# Dict per-key merge
# ---------------------------------------------------------------------------


class TestDictMerge:
    def test_full_default_when_user_dict_absent(self):
        defaults = InferencerTemplateDefaults(
            template_variables={
                VAR_TASK_PREAMBLE: VARIANT_AGGREGATION,
                VAR_TASK_INSTRUCTIONS: VARIANT_AGGREGATION,
                VAR_TASK_RESPONSE_FORMAT: VARIANT_AGGREGATION,
            },
        )
        node: dict = {}
        defaults.apply_to(node)
        assert node[FIELD_TEMPLATE_VARIABLES] == {
            VAR_TASK_PREAMBLE: VARIANT_AGGREGATION,
            VAR_TASK_INSTRUCTIONS: VARIANT_AGGREGATION,
            VAR_TASK_RESPONSE_FORMAT: VARIANT_AGGREGATION,
        }

    def test_per_key_merge_user_overrides_one_key(self):
        defaults = InferencerTemplateDefaults(
            template_variables={
                VAR_TASK_PREAMBLE: VARIANT_AGGREGATION,
                VAR_TASK_INSTRUCTIONS: VARIANT_AGGREGATION,
                VAR_TASK_RESPONSE_FORMAT: VARIANT_AGGREGATION,
            },
        )
        node = {FIELD_TEMPLATE_VARIABLES: {VAR_TASK_INSTRUCTIONS: "create_role"}}
        defaults.apply_to(node)
        assert node[FIELD_TEMPLATE_VARIABLES] == {
            VAR_TASK_PREAMBLE: VARIANT_AGGREGATION,
            VAR_TASK_INSTRUCTIONS: "create_role",  # user override
            VAR_TASK_RESPONSE_FORMAT: VARIANT_AGGREGATION,
        }

    def test_empty_string_per_key_opt_out_survives_merge(self):
        defaults = InferencerTemplateDefaults(
            template_variables={VAR_TASK_PREAMBLE: VARIANT_AGGREGATION},
        )
        node = {FIELD_TEMPLATE_VARIABLES: {VAR_TASK_PREAMBLE: ""}}
        defaults.apply_to(node)
        assert node[FIELD_TEMPLATE_VARIABLES] == {VAR_TASK_PREAMBLE: ""}

    def test_template_extra_feed_per_key_merge(self):
        defaults = InferencerTemplateDefaults(
            template_extra_feed={"flag_a": True, "flag_b": False},
        )
        node = {FIELD_TEMPLATE_EXTRA_FEED: {"flag_b": True}}
        defaults.apply_to(node)
        assert node[FIELD_TEMPLATE_EXTRA_FEED] == {"flag_a": True, "flag_b": True}

    def test_default_dict_is_deep_copied(self):
        # Mutating the merged result must NOT affect the defaults instance.
        defaults = InferencerTemplateDefaults(
            template_variables={"k": ["a", "b"]},
        )
        node1: dict = {}
        defaults.apply_to(node1)
        node1[FIELD_TEMPLATE_VARIABLES]["k"].append("c")
        node2: dict = {}
        defaults.apply_to(node2)
        assert node2[FIELD_TEMPLATE_VARIABLES]["k"] == ["a", "b"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_apply_to_non_dict_is_noop(self):
        defaults = InferencerTemplateDefaults(template_root_space="x")
        defaults.apply_to(None)  # type: ignore[arg-type]
        defaults.apply_to("not a dict")  # type: ignore[arg-type]
        defaults.apply_to(["list"])  # type: ignore[arg-type]
        # No exception → pass

    def test_empty_defaults_is_noop(self):
        defaults = InferencerTemplateDefaults()
        node = {"existing": "value"}
        defaults.apply_to(node)
        assert node == {"existing": "value"}

    def test_parent_node_arg_is_accepted_but_unused_on_base_class(self):
        defaults = InferencerTemplateDefaults(template_root_space="x")
        node: dict = {}
        defaults.apply_to(node, parent_node={"some": "parent"})
        assert node == {FIELD_TEMPLATE_ROOT_SPACE: "x"}


# ---------------------------------------------------------------------------
# ConditionalTemplateDefaults
# ---------------------------------------------------------------------------


class TestConditionalTemplateDefaults:
    def test_applies_when_condition_true(self):
        defaults = ConditionalTemplateDefaults(
            condition=lambda parent: parent.get("flag") is True,
            template_root_space="x",
        )
        node: dict = {}
        defaults.apply_to(node, parent_node={"flag": True})
        assert node == {FIELD_TEMPLATE_ROOT_SPACE: "x"}

    def test_skips_when_condition_false(self):
        defaults = ConditionalTemplateDefaults(
            condition=lambda parent: parent.get("flag") is True,
            template_root_space="x",
        )
        node: dict = {}
        defaults.apply_to(node, parent_node={"flag": False})
        assert node == {}

    def test_skips_when_parent_node_missing(self):
        defaults = ConditionalTemplateDefaults(
            condition=lambda parent: True,
            template_root_space="x",
        )
        node: dict = {}
        defaults.apply_to(node, parent_node=None)
        assert node == {}


# ---------------------------------------------------------------------------
# _resolve_attrib_default
# ---------------------------------------------------------------------------


class TestResolveAttribDefault:
    def test_resolves_scalar_default_for_multiflow_dual(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (  # noqa: E501
            MultiFlowDualInferencer,
        )

        target = (
            "agent_foundation.common.inferencers.agentic_inferencers."
            "flow_inferencers.multi_flow_dual_inferencer.MultiFlowDualInferencer"
        )
        # MFDual default for visible_flows is "all" (peer-aware aggregation).
        assert _resolve_attrib_default(target, "visible_flows", "self") == "all"
        # MFDual default for inject_upstream_artifacts is True (forwarded to
        # the auto-constructed MultiFlow so wrapper templates can consume the
        # {{ upstream_artifacts }} slot). Differs from MultiFlow's default False.
        assert (
            _resolve_attrib_default(target, "inject_upstream_artifacts", False)
            is True
        )
        del MultiFlowDualInferencer  # silence unused-import linter

    def test_resolves_scalar_default_for_multiflow(self):
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (  # noqa: E501
            MultiFlowInferencer,
        )

        target = (
            "agent_foundation.common.inferencers.agentic_inferencers."
            "flow_inferencers.multi_flow_inferencer.MultiFlowInferencer"
        )
        # MultiFlow default for visible_flows is "self"
        assert _resolve_attrib_default(target, "visible_flows", "ALL") == "self"
        del MultiFlowInferencer

    def test_returns_fallback_for_unknown_field(self):
        target = (
            "agent_foundation.common.inferencers.agentic_inferencers."
            "flow_inferencers.multi_flow_inferencer.MultiFlowInferencer"
        )
        assert _resolve_attrib_default(target, "nonexistent_field", "fb") == "fb"

    def test_returns_fallback_for_unresolvable_target(self):
        assert _resolve_attrib_default("not.a.real.module", "f", "fb") == "fb"
        assert _resolve_attrib_default(None, "f", "fb") == "fb"
        assert _resolve_attrib_default(123, "f", "fb") == "fb"  # type: ignore[arg-type]
        assert _resolve_attrib_default("noModulePath", "f", "fb") == "fb"


# ---------------------------------------------------------------------------
# _aggregation_applicable predicate
# ---------------------------------------------------------------------------


class TestAggregationApplicable:
    MFDUAL_TARGET = (
        "agent_foundation.common.inferencers.agentic_inferencers."
        "flow_inferencers.multi_flow_dual_inferencer.MultiFlowDualInferencer"
    )
    MF_TARGET = (
        "agent_foundation.common.inferencers.agentic_inferencers."
        "flow_inferencers.multi_flow_inferencer.MultiFlowInferencer"
    )

    def test_mfdual_with_inject_true_yaml_applies(self):
        # MFDual: visible_flows defaults "all" via class; inject true via YAML.
        parent = {"_target_": self.MFDUAL_TARGET, "inject_upstream_artifacts": True}
        assert _aggregation_applicable(parent) is True

    def test_mfdual_with_explicit_visible_self_skips(self):
        parent = {
            "_target_": self.MFDUAL_TARGET,
            "visible_flows": "self",
            "inject_upstream_artifacts": True,
        }
        assert _aggregation_applicable(parent) is False

    def test_mfdual_with_inject_false_skips(self):
        # Even though MFDual's visible_flows defaults to "all", inject=false
        # means the runtime path doesn't engage the templates.
        parent = {
            "_target_": self.MFDUAL_TARGET,
            "inject_upstream_artifacts": False,
        }
        assert _aggregation_applicable(parent) is False

    def test_mfdual_default_yaml_applies(self):
        # Bare MFDual (no flags set) → visible="all" + inject=True via class
        # defaults → both conditions met → followup default applies. MFDual
        # is purpose-built for peer-aware aggregation; the class defaults
        # encode that intent.
        parent = {"_target_": self.MFDUAL_TARGET}
        assert _aggregation_applicable(parent) is True

    def test_mf_with_explicit_visible_all_and_inject_true_applies(self):
        parent = {
            "_target_": self.MF_TARGET,
            "visible_flows": "all",
            "inject_upstream_artifacts": True,
        }
        assert _aggregation_applicable(parent) is True

    def test_mf_default_skips_because_visible_self_is_default(self):
        # MultiFlow default visible_flows="self" → no peers visible → skip
        # regardless of inject value.
        parent = {"_target_": self.MF_TARGET, "inject_upstream_artifacts": True}
        assert _aggregation_applicable(parent) is False


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_breakdown_constant_shape(self):
        assert (
            BREAKDOWN_TEMPLATE_DEFAULTS.template_root_space
            == SPACE_TASK_BREAKDOWN
        )
        assert BREAKDOWN_TEMPLATE_DEFAULTS.template_key is None
        assert BREAKDOWN_TEMPLATE_DEFAULTS.template_variables == {}

    def test_aggregation_constant_shape(self):
        assert AGGREGATION_DEFAULTS.template_root_space is None
        assert AGGREGATION_DEFAULTS.template_version == VARIANT_AGGREGATION
        assert AGGREGATION_DEFAULTS.template_master_version == VARIANT_AGGREGATION
        assert AGGREGATION_DEFAULTS.template_variables == {
            VAR_TASK_PREAMBLE: None,
            VAR_TASK_INSTRUCTIONS: None,
            VAR_TASK_RESPONSE_FORMAT: None,
        }

    def test_review_constant_shape(self):
        assert REVIEW_TEMPLATE_DEFAULTS.template_key == KEY_REVIEW
        assert REVIEW_TEMPLATE_DEFAULTS.template_root_space is None

    def test_followup_aggregation_is_conditional(self):
        assert isinstance(FOLLOWUP_AGGREGATION_DEFAULTS, ConditionalTemplateDefaults)
        assert (
            FOLLOWUP_AGGREGATION_DEFAULTS.template_variables
            == AGGREGATION_DEFAULTS.template_variables
        )


# ---------------------------------------------------------------------------
# Fix #9: Review/Followup template defaults
# ---------------------------------------------------------------------------


class TestReviewFollowupDefaults:
    """Fix #9: REVIEW_TEMPLATE_DEFAULTS and FOLLOWUP_TEMPLATE_DEFAULTS
    set the correct template_key; variable auto-discovery is handled by
    the TemplateManager's predefined_variables infrastructure (no explicit
    variable_names or template_version needed)."""

    def test_review_followup_are_plain_defaults(self):
        """Both are plain InferencerTemplateDefaults (not VersionDefaults).
        Variable discovery relies on TemplateManager's predefined_variables,
        not on explicit variable_names declarations."""
        assert isinstance(REVIEW_TEMPLATE_DEFAULTS, InferencerTemplateDefaults)
        assert isinstance(FOLLOWUP_TEMPLATE_DEFAULTS, InferencerTemplateDefaults)
        assert not REVIEW_TEMPLATE_DEFAULTS.template_variables
        assert not FOLLOWUP_TEMPLATE_DEFAULTS.template_variables

    def test_review_default_template_key_is_review(self):
        """REVIEW_TEMPLATE_DEFAULTS has template_key='review'."""
        assert REVIEW_TEMPLATE_DEFAULTS.template_key == KEY_REVIEW

    def test_followup_default_template_key_is_followup(self):
        """FOLLOWUP_TEMPLATE_DEFAULTS has template_key='followup'."""
        from agent_foundation.common.inferencers.template_constants import KEY_FOLLOWUP
        assert FOLLOWUP_TEMPLATE_DEFAULTS.template_key == KEY_FOLLOWUP


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
