"""Integration tests: SLOT_DEFAULTS on the real BTA / Dual / MultiFlow / MFDual.

Where ``test_template_defaults.py`` covers the primitive in isolation and
``RichPythonUtils/.../test_instantiate_slot_defaults.py`` covers the walker
hook with synthetic classes, this file exercises the **real** orchestrator
classes end-to-end:

- A YAML config goes through ``instantiate``, which runs the step-1d hook,
  which finds each orchestrator's ``SLOT_DEFAULTS``, walks each slot path
  (including list-element wildcards), and applies the defaults to the
  child YAML node before construction.
- After instantiation, we assert the post-construction objects have the
  template fields the role default should have planted.
- We also run the same checks against the real
  ``breakdown_multiflow_plan_then_implement.yaml`` so the migration is
  verified by a concrete realistic config (not just synthetic mocks).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest
from attr import attrib, attrs
from omegaconf import OmegaConf

from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.breakdown_then_aggregate_inferencer import (
    BreakdownThenAggregateInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.dual_inferencer import (
    DualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
    MultiFlowDualInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_inferencer import (
    MultiFlowInferencer,
)
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from rich_python_utils.config_utils import instantiate, register_class

# Ensure registered_targets is loaded for alias resolution.
import agent_foundation.common.configs.registered_targets  # noqa: F401
from agent_foundation.common.inferencers.inferencer_workspace import InferencerWorkspace


# ---------------------------------------------------------------------------
# Templated mock — has the four template fields so SLOT_DEFAULTS has somewhere
# to write. Subclassing TemplatedInferencerBase is the cleanest way.
# ---------------------------------------------------------------------------


@attrs
class TemplatedMock(TemplatedInferencerBase):
    """Concrete leaf with template fields. Used as the slot child in tests."""

    _response: str = attrib(default="ok")

    def _infer(self, inference_input, inference_config=None, **kwargs):
        return self._response

    async def _ainfer(self, inference_input, inference_config=None, **kwargs):
        return self._response


# Register the mock as an alias so YAML can reference it by short name.
register_class(
    TemplatedMock,
    "TemplatedMock",
    category="inferencer",
)


# Module path for YAML _target_ strings.
_MOCK_TARGET = (
    "test_slot_defaults_real_orchestrators.TemplatedMock"
)


# ---------------------------------------------------------------------------
# BTA: breakdown_inferencer + aggregator_inferencer defaults
# ---------------------------------------------------------------------------


class TestBTASlotDefaults:
    def test_breakdown_root_space_defaulted(self):
        cfg = OmegaConf.create({
            "_target_": "BTA",
            "breakdown_inferencer": {"_target_": "TemplatedMock"},
        })
        obj = instantiate(cfg)
        assert isinstance(obj, BreakdownThenAggregateInferencer)
        assert obj.breakdown_inferencer.template_root_space == "task_breakdown"

    def test_breakdown_user_root_space_wins(self):
        cfg = OmegaConf.create({
            "_target_": "BTA",
            "breakdown_inferencer": {
                "_target_": "TemplatedMock",
                "template_root_space": "custom_space",
            },
        })
        obj = instantiate(cfg)
        assert obj.breakdown_inferencer.template_root_space == "custom_space"

    def test_aggregator_full_triplet_defaulted(self):
        # BTA's aggregator default is the full structured-aggregation triplet
        # (preamble + instructions + response_format) since Refactor 12+13:
        # version-to-default fallback is safe (Refactor 12) and
        # template_version + None-keyed variables make the YAML clean (Refactor 13).
        cfg = OmegaConf.create({
            "_target_": "BTA",
            "aggregator_inferencer": {"_target_": "TemplatedMock"},
        })
        obj = instantiate(cfg)
        assert obj.aggregator_inferencer.template_version == "aggregation"
        assert obj.aggregator_inferencer.template_variables == {
            "task_preamble": None,
            "task_instructions": None,
            "task_response_format": None,
        }

    def test_aggregator_user_overrides_one_key(self):
        # Caller supplies task_instructions explicitly — per-key merge:
        # preamble + response_format keep None (expanded via template_version),
        # task_instructions takes the explicit value.
        cfg = OmegaConf.create({
            "_target_": "BTA",
            "aggregator_inferencer": {
                "_target_": "TemplatedMock",
                "template_variables": {"task_instructions": "create_role"},
            },
        })
        obj = instantiate(cfg)
        assert obj.aggregator_inferencer.template_version == "aggregation"
        assert obj.aggregator_inferencer.template_variables == {
            "task_preamble": None,
            "task_instructions": "create_role",
            "task_response_format": None,
        }

    def test_aggregator_user_overrides_preamble(self):
        cfg = OmegaConf.create({
            "_target_": "BTA",
            "aggregator_inferencer": {
                "_target_": "TemplatedMock",
                "template_variables": {"task_preamble": "custom"},
            },
        })
        obj = instantiate(cfg)
        # User key wins; default doesn't override; siblings still defaulted.
        assert obj.aggregator_inferencer.template_version == "aggregation"
        assert obj.aggregator_inferencer.template_variables == {
            "task_preamble": "custom",
            "task_instructions": None,
            "task_response_format": None,
        }

    def test_aggregator_per_key_empty_string_opt_out(self):
        # Empty string is a valid opt-out: user explicit task_preamble: ""
        # suppresses it from the rendered feed (an empty-string value
        # bypasses both per-key default fill AND template_version fallback).
        # Edge case for callers who want a fully manual aggregator wrapper.
        cfg = OmegaConf.create({
            "_target_": "BTA",
            "aggregator_inferencer": {
                "_target_": "TemplatedMock",
                "template_variables": {"task_preamble": ""},
            },
        })
        obj = instantiate(cfg)
        assert obj.aggregator_inferencer.template_variables == {
            "task_preamble": "",
            "task_instructions": None,
            "task_response_format": None,
        }

    def test_disable_slot_defaults_skips(self):
        cfg = OmegaConf.create({
            "_target_": "BTA",
            "_disable_slot_defaults_": True,
            "breakdown_inferencer": {"_target_": "TemplatedMock"},
            "aggregator_inferencer": {"_target_": "TemplatedMock"},
        })
        obj = instantiate(cfg)
        # Both defaults skipped → fields stay at attrib defaults.
        assert obj.breakdown_inferencer.template_root_space is None
        assert obj.aggregator_inferencer.template_variables == {}
        assert obj.aggregator_inferencer.template_root_space is None


# ---------------------------------------------------------------------------
# Dual: review_inferencer.template_key default
# ---------------------------------------------------------------------------


class TestDualSlotDefaults:
    def test_review_template_key_defaulted(self):
        cfg = OmegaConf.create({
            "_target_": "Dual",
            "base_inferencer": {"_target_": "TemplatedMock"},
            "review_inferencer": {"_target_": "TemplatedMock"},
        })
        obj = instantiate(cfg)
        assert isinstance(obj, DualInferencer)
        assert obj.review_inferencer.template_key == "review"
        # base_inferencer NOT defaulted (no role default for the base slot)
        assert obj.base_inferencer.template_key == ""

    def test_review_user_key_wins(self):
        cfg = OmegaConf.create({
            "_target_": "Dual",
            "base_inferencer": {"_target_": "TemplatedMock"},
            "review_inferencer": {
                "_target_": "TemplatedMock",
                "template_key": "custom_review",
            },
        })
        obj = instantiate(cfg)
        assert obj.review_inferencer.template_key == "custom_review"


# ---------------------------------------------------------------------------
# Wrapping descent: Dual filling BTA's aggregator slot
# ---------------------------------------------------------------------------


class TestWrappingDescent:
    def test_bta_aggregator_dual_descent(self):
        # BTA's aggregator slot is filled by a Dual. The Dual is template-
        # transparent (declares _TEMPLATE_TRANSPARENT_SLOTS = [base, review,
        # fixer]). Parent BTA's aggregator preamble-only default descends
        # into the Dual's inner CCCs. Dual's own SLOT_DEFAULTS
        # (review.template_key=review) applies when _walk enters the Dual node.
        cfg = OmegaConf.create({
            "_target_": "BTA",
            "aggregator_inferencer": {
                "_target_": "Dual",
                "base_inferencer": {"_target_": "TemplatedMock"},
                "review_inferencer": {"_target_": "TemplatedMock"},
            },
        })
        obj = instantiate(cfg)
        agg = obj.aggregator_inferencer
        assert isinstance(agg, DualInferencer)

        # Both inner CCCs got the BTA aggregator full triplet via descent.
        # Refactor 13: template_version + template_variables with None values.
        for inner in (agg.base_inferencer, agg.review_inferencer):
            assert inner.template_version == "aggregation"
            assert inner.template_variables == {
                "task_preamble": None,
                "task_instructions": None,
                "task_response_format": None,
            }

        # Review additionally got Dual's own template_key=review.
        assert agg.review_inferencer.template_key == "review"
        # Base did NOT get template_key=review (only review_inferencer slot does).
        assert agg.base_inferencer.template_key == ""


# ---------------------------------------------------------------------------
# MultiFlow per-flow followup conditional defaults
# ---------------------------------------------------------------------------


class TestMultiFlowConditionalFollowup:
    def _mf_cfg(self, *, visible_flows=None, inject_upstream_artifacts=None):
        flow_a = {
            "input": "task A",
            "initial_inferencer": {"_target_": "TemplatedMock"},
            "followup_inferencer": {"_target_": "TemplatedMock"},
            "max_dynamic_steps": 1,
        }
        flow_b = {
            "input": "task B",
            "initial_inferencer": {"_target_": "TemplatedMock"},
            "followup_inferencer": {"_target_": "TemplatedMock"},
            "max_dynamic_steps": 1,
        }
        cfg = {
            "_target_": "MultiFlow",
            "flow_configs": [flow_a, flow_b],
        }
        if visible_flows is not None:
            cfg["visible_flows"] = visible_flows
        if inject_upstream_artifacts is not None:
            cfg["inject_upstream_artifacts"] = inject_upstream_artifacts
        return OmegaConf.create(cfg)

    def test_followup_default_skipped_when_visible_self(self):
        # MultiFlow class default visible_flows="self" → skip
        obj = instantiate(self._mf_cfg(inject_upstream_artifacts=True))
        for fc in obj.flow_configs:
            assert fc["followup_inferencer"].template_variables == {}

    def test_followup_default_skipped_when_inject_false(self):
        obj = instantiate(self._mf_cfg(visible_flows="all", inject_upstream_artifacts=False))
        for fc in obj.flow_configs:
            assert fc["followup_inferencer"].template_variables == {}

    def test_followup_default_applied_when_both_conditions_met(self):
        obj = instantiate(self._mf_cfg(visible_flows="all", inject_upstream_artifacts=True))
        for fc in obj.flow_configs:
            assert fc["followup_inferencer"].template_version == "aggregation"
            assert fc["followup_inferencer"].template_variables == {
                "task_preamble": None,
                "task_instructions": None,
                "task_response_format": None,
            }

    def test_followup_user_override_per_flow(self):
        cfg = self._mf_cfg(visible_flows="all", inject_upstream_artifacts=True)
        # Override task_preamble on only flow 0
        cfg.flow_configs[0]["followup_inferencer"]["template_variables"] = {
            "task_preamble": "custom_preamble"
        }
        obj = instantiate(cfg)
        # Flow 0: explicit preamble wins; others remain None (expanded via
        # template_version at render time).
        assert obj.flow_configs[0]["followup_inferencer"].template_version == "aggregation"
        assert obj.flow_configs[0]["followup_inferencer"].template_variables == {
            "task_preamble": "custom_preamble",
            "task_instructions": None,
            "task_response_format": None,
        }
        # Flow 1: full triplet (default), all None values.
        assert obj.flow_configs[1]["followup_inferencer"].template_version == "aggregation"
        assert obj.flow_configs[1]["followup_inferencer"].template_variables == {
            "task_preamble": None,
            "task_instructions": None,
            "task_response_format": None,
        }


# ---------------------------------------------------------------------------
# MFDual: aggregator default + review default + per-flow followup
# ---------------------------------------------------------------------------


class TestMFDualSlotDefaults:
    def _mfdual_cfg(self, *, inject_upstream_artifacts=None, visible_flows=None):
        cfg = {
            "_target_": "MultiFlowDual",
            "flow_configs": [
                {
                    "input": "task A",
                    "initial_inferencer": {"_target_": "TemplatedMock"},
                    "followup_inferencer": {"_target_": "TemplatedMock"},
                    "max_dynamic_steps": 1,
                },
                {
                    "input": "task B",
                    "initial_inferencer": {"_target_": "TemplatedMock"},
                    "followup_inferencer": {"_target_": "TemplatedMock"},
                    "max_dynamic_steps": 1,
                },
            ],
            "review_inferencer": {"_target_": "TemplatedMock"},
            "multi_flow_aggregator_inferencer": {"_target_": "TemplatedMock"},
        }
        if inject_upstream_artifacts is not None:
            cfg["inject_upstream_artifacts"] = inject_upstream_artifacts
        if visible_flows is not None:
            cfg["visible_flows"] = visible_flows
        return OmegaConf.create(cfg)

    def test_review_template_key_defaulted_via_dual_mro(self):
        # MFDual extends Dual → inherits Dual's review_inferencer default.
        obj = instantiate(self._mfdual_cfg())
        assert obj.review_inferencer.template_key == "review"

    def test_multi_flow_aggregator_full_triplet_defaulted(self):
        obj = instantiate(self._mfdual_cfg())
        assert obj.multi_flow_aggregator_inferencer.template_version == "aggregation"
        assert obj.multi_flow_aggregator_inferencer.template_variables == {
            "task_preamble": None,
            "task_instructions": None,
            "task_response_format": None,
        }

    def test_followup_skipped_when_inject_false(self):
        # MFDual default visible_flows="all" (peers visible) but inject=False.
        # Conditional gate fails → no followup default.
        obj = instantiate(self._mfdual_cfg(inject_upstream_artifacts=False))
        for fc in obj.flow_configs:
            assert fc["followup_inferencer"].template_variables == {}

    def test_followup_applied_on_bare_mfdual_via_class_defaults(self):
        # Bare MFDual: visible_flows="all" + inject_upstream_artifacts=True
        # are BOTH MFDual class defaults — followup gets the full triplet
        # automatically. This is the post-Refactor-5 invariant: MFDual is
        # peer-aware-aggregation by design, so the YAML doesn't need to
        # spell out either flag.
        obj = instantiate(self._mfdual_cfg())
        for fc in obj.flow_configs:
            assert fc["followup_inferencer"].template_version == "aggregation"
            assert fc["followup_inferencer"].template_variables == {
                "task_preamble": None,
                "task_instructions": None,
                "task_response_format": None,
            }

    def test_followup_applied_when_inject_true(self):
        # Same as bare default, but with explicit inject=True. Behavior
        # unchanged — explicit value matches the class default.
        obj = instantiate(self._mfdual_cfg(inject_upstream_artifacts=True))
        for fc in obj.flow_configs:
            assert fc["followup_inferencer"].template_version == "aggregation"
            assert fc["followup_inferencer"].template_variables == {
                "task_preamble": None,
                "task_instructions": None,
                "task_response_format": None,
            }

    def test_followup_skipped_when_explicit_visible_self(self):
        # Override visible_flows to "self" → no peers visible → skip.
        obj = instantiate(
            self._mfdual_cfg(inject_upstream_artifacts=True, visible_flows="self")
        )
        for fc in obj.flow_configs:
            assert fc["followup_inferencer"].template_variables == {}

    def test_iteration_judgment_toggle_wires_both_halves(self):
        # `iteration_judgment: true` is a coordinated feature toggle on
        # flow_configs entries — sets `end_condition` to parse_decision_stop
        # AND `followup_inferencer.template_extra_feed["include_iteration_judgment"]`
        # to True. Both halves are inert without the other, so the toggle
        # keeps them in sync.
        from agent_foundation.common.inferencers.flow_parsers import parse_decision_stop
        cfg = OmegaConf.create({
            "_target_": "MultiFlowDual",
            "flow_configs": [
                {
                    "input": "x",
                    "iteration_judgment": True,
                    "max_dynamic_steps": 3,
                    "initial_inferencer": {"_target_": "TemplatedMock"},
                    "followup_inferencer": {"_target_": "TemplatedMock"},
                },
            ],
            "review_inferencer": {"_target_": "TemplatedMock"},
        })
        obj = instantiate(cfg)
        fc = obj.flow_configs[0]
        # Half 1: end_condition auto-set to parse_decision_stop.
        assert fc["end_condition"] is parse_decision_stop
        # Half 2: followup's template_extra_feed has the schema flag.
        assert (
            fc["followup_inferencer"].template_extra_feed.get(
                "include_iteration_judgment"
            )
            is True
        )

    def test_iteration_judgment_user_explicit_overrides_win(self):
        # If user supplies an explicit end_condition, the toggle's setdefault
        # preserves it. Same for the template flag.
        my_end_condition = lambda state, result: True  # noqa: E731
        cfg = OmegaConf.create({
            "_target_": "MultiFlowDual",
            "flow_configs": [
                {
                    "input": "x",
                    "iteration_judgment": True,
                    "max_dynamic_steps": 3,
                    "initial_inferencer": {"_target_": "TemplatedMock"},
                    "followup_inferencer": {
                        "_target_": "TemplatedMock",
                        "template_extra_feed": {"include_iteration_judgment": False},
                    },
                },
            ],
            "review_inferencer": {"_target_": "TemplatedMock"},
        })
        # Inject the lambda end_condition via direct construction (Hydra
        # can't import lambdas) — but that path bypasses _walk. Easier to
        # verify the explicit user template_extra_feed is preserved instead:
        obj = instantiate(cfg)
        fc = obj.flow_configs[0]
        # User's explicit False survives the toggle's setdefault.
        assert fc["followup_inferencer"].template_extra_feed[
            "include_iteration_judgment"
        ] is False

    def test_winner_pick_toggle_wires_aggregator_template_flag(self):
        # `winner_pick: true` on MFDual auto-sets
        # `include_winner_pick: true` in multi_flow_aggregator_inferencer's
        # template_extra_feed. The default winner_parser (parse_winner_tag,
        # set in Refactor 4) extracts the parsed index from the LLM output.
        cfg = OmegaConf.create({
            "_target_": "MultiFlowDual",
            "winner_pick": True,
            "flow_configs": [
                {
                    "input": "x",
                    "initial_inferencer": {"_target_": "TemplatedMock"},
                    "max_dynamic_steps": 1,
                },
            ],
            "review_inferencer": {"_target_": "TemplatedMock"},
            "multi_flow_aggregator_inferencer": {"_target_": "TemplatedMock"},
        })
        obj = instantiate(cfg)
        assert obj.winner_pick is True
        assert obj.multi_flow_aggregator_inferencer.template_extra_feed.get(
            "include_winner_pick"
        ) is True

    def test_winner_pick_user_explicit_false_is_preserved(self):
        # If the user explicitly sets `include_winner_pick: false` on the
        # aggregator, the toggle's setdefault preserves it (deliberate opt-out).
        cfg = OmegaConf.create({
            "_target_": "MultiFlowDual",
            "winner_pick": True,
            "flow_configs": [
                {
                    "input": "x",
                    "initial_inferencer": {"_target_": "TemplatedMock"},
                    "max_dynamic_steps": 1,
                },
            ],
            "review_inferencer": {"_target_": "TemplatedMock"},
            "multi_flow_aggregator_inferencer": {
                "_target_": "TemplatedMock",
                "template_extra_feed": {"include_winner_pick": False},
            },
        })
        obj = instantiate(cfg)
        assert obj.multi_flow_aggregator_inferencer.template_extra_feed[
            "include_winner_pick"
        ] is False

    def test_winner_pick_default_false_no_injection(self):
        # Without winner_pick, no injection — the aggregator's
        # template_extra_feed stays empty (or whatever the user set).
        cfg = OmegaConf.create({
            "_target_": "MultiFlowDual",
            "flow_configs": [
                {
                    "input": "x",
                    "initial_inferencer": {"_target_": "TemplatedMock"},
                    "max_dynamic_steps": 1,
                },
            ],
            "review_inferencer": {"_target_": "TemplatedMock"},
            "multi_flow_aggregator_inferencer": {"_target_": "TemplatedMock"},
        })
        obj = instantiate(cfg)
        assert obj.winner_pick is False
        assert "include_winner_pick" not in (
            obj.multi_flow_aggregator_inferencer.template_extra_feed or {}
        )

    def test_iteration_judgment_false_or_absent_no_op(self):
        # Without iteration_judgment, no auto-wiring happens — flow uses
        # max_dynamic_steps fully, no template flag set.
        cfg = OmegaConf.create({
            "_target_": "MultiFlowDual",
            "flow_configs": [
                {
                    "input": "x",
                    # iteration_judgment absent (= False)
                    "max_dynamic_steps": 3,
                    "initial_inferencer": {"_target_": "TemplatedMock"},
                    "followup_inferencer": {"_target_": "TemplatedMock"},
                },
            ],
            "review_inferencer": {"_target_": "TemplatedMock"},
        })
        obj = instantiate(cfg)
        fc = obj.flow_configs[0]
        # No end_condition added.
        assert fc.get("end_condition") is None
        # No template_extra_feed flag added (other defaults may apply, but
        # NOT include_iteration_judgment).
        assert "include_iteration_judgment" not in (
            fc["followup_inferencer"].template_extra_feed or {}
        )

    def test_input_required_when_propagate_off(self):
        # Without propagate_runtime_input, each flow's "input" must be
        # supplied — MultiFlow's runtime path uses cfg["input"] verbatim.
        cfg = OmegaConf.create({
            "_target_": "MultiFlowDual",
            "flow_configs": [
                {  # missing "input"
                    "initial_inferencer": {"_target_": "TemplatedMock"},
                    "max_dynamic_steps": 1,
                },
            ],
            "review_inferencer": {"_target_": "TemplatedMock"},
        })
        # Hydra wraps the underlying ValueError; match on message text.
        with pytest.raises(Exception, match="missing required key 'input'"):
            instantiate(cfg)

    def test_default_parsers_set_on_class(self):
        # Round-7 dispatch parsers default to graceful fallbacks at the class
        # level — YAMLs no longer need to wire WinnerParser/FinalPlanParser
        # explicitly. The fallbacks return None / passthrough when the
        # expected pattern isn't present, so non-Round-7 use cases are unaffected.
        from agent_foundation.common.inferencers.flow_parsers import (
            parse_finalplan_tag,
            parse_winner_tag,
        )
        obj = instantiate(self._mfdual_cfg())
        assert obj.multi_flow_winner_parser is parse_winner_tag
        assert obj.multi_flow_response_parser is parse_finalplan_tag

    def test_user_can_override_default_parsers(self):
        my_parser = lambda s: 42
        cfg = self._mfdual_cfg()
        # Use a registered alias path to override (simpler than embedding
        # the lambda in the YAML — Hydra needs a target).
        # Construct directly via attrs to verify override semantics.
        from agent_foundation.common.inferencers.agentic_inferencers.flow_inferencers.multi_flow_dual_inferencer import (
            MultiFlowDualInferencer,
        )
        flow_cfg = lambda: {
            "input": "x",
            "initial_inferencer": TemplatedMock(),
            "max_dynamic_steps": 1,
        }
        mfdi = MultiFlowDualInferencer(
            flow_configs=[flow_cfg(), flow_cfg()],
            review_inferencer=TemplatedMock(),
            multi_flow_winner_parser=my_parser,
        )
        assert mfdi.multi_flow_winner_parser is my_parser

    def test_input_optional_when_propagate_on(self):
        # With propagate_runtime_input=True, each cfg["input"] is overwritten
        # at runtime — the YAML can omit it, and the post_init sets a "" sentinel.
        cfg = OmegaConf.create({
            "_target_": "MultiFlowDual",
            "propagate_runtime_input": True,
            "flow_configs": [
                {  # NO "input" key
                    "initial_inferencer": {"_target_": "TemplatedMock"},
                    "max_dynamic_steps": 1,
                },
                {
                    "initial_inferencer": {"_target_": "TemplatedMock"},
                    "max_dynamic_steps": 1,
                },
            ],
            "review_inferencer": {"_target_": "TemplatedMock"},
        })
        obj = instantiate(cfg)
        # Validation passed; placeholder filled with empty string.
        for fc in obj.flow_configs:
            assert fc["input"] == ""


# ---------------------------------------------------------------------------
# Real-world parity: load breakdown_multiflow_plan_then_implement.yaml
# ---------------------------------------------------------------------------


_REAL_YAML = (
    Path(__file__).parents[5]
    / "OpenStartup"
    / "test"
    / "openteam"
    / "resources"
    / "tools"
    / "task"
    / "configs"
    / "breakdown_multiflow_plan_then_implement.yaml"
)


@pytest.mark.skipif(
    not _REAL_YAML.exists(), reason=f"YAML not found at {_REAL_YAML}"
)
class TestRealWorldYAMLParity:
    """Verify the migrated breakdown_multiflow_plan_then_implement.yaml has all
    expected defaults flowing through to the right slots after instantiation.

    The defaults eliminate ~22 lines of YAML boilerplate that previously had
    to be repeated. This test confirms each removed line is faithfully
    reproduced by SLOT_DEFAULTS at instantiation time.
    """

    @pytest.fixture(scope="class")
    def root(self, tmp_path_factory):
        # Need to provide a workspace_root override; the YAML uses ${oc.env:DUAL_WS}.
        ws = tmp_path_factory.mktemp("real_yaml_ws")
        from rich_python_utils.config_utils import load_config

        cfg = load_config(
            str(_REAL_YAML),
            overrides={"workspace": InferencerWorkspace(root=str(ws))},
        )
        return instantiate(cfg)

    def test_outer_dual_review_template_key_defaulted(self, root):
        # Outer Dual → review_inferencer should have template_key=review
        # via Dual.SLOT_DEFAULTS (line removed from YAML).
        assert root.review_inferencer.template_key == "review"
        assert root.review_inferencer.template_root_space == "implementation"

    def test_plan_bta_breakdown_root_space_defaulted(self, root):
        plan_bta = root.base_inferencer.planner_inferencer.base_inferencer
        # template_root_space=task_breakdown defaulted by BTA.SLOT_DEFAULTS
        # (line removed from YAML).
        assert plan_bta.breakdown_inferencer.template_root_space == "task_breakdown"

    def test_plan_bta_aggregator_full_triplet_defaulted(self, root):
        plan_bta = root.base_inferencer.planner_inferencer.base_inferencer
        # Refactor 12+13: BTA's aggregator default is now the full structured-
        # aggregation triplet (preamble + instructions + response_format).
        # All three keys are present with None values; template_version drives
        # expansion at render time. The YAML's previous explicit
        # `template_variables: task_instructions: aggregation` line was dropped
        # in Step 8 since the SLOT_DEFAULTS now supplies it via template_version.
        assert plan_bta.aggregator_inferencer.template_version == "aggregation"
        assert plan_bta.aggregator_inferencer.template_variables == {
            "task_preamble": None,
            "task_instructions": None,
            "task_response_format": None,
        }
        assert plan_bta.aggregator_inferencer.template_root_space == "plan"

    def test_plan_dual_review_template_key_defaulted(self, root):
        plan_dual = root.base_inferencer.planner_inferencer
        # plan Dual review's template_key=review defaulted (line removed).
        assert plan_dual.review_inferencer.template_key == "review"
        assert plan_dual.review_inferencer.template_root_space == "plan"

    def test_per_flow_followup_full_triplet_via_mfdual_conditional(self, root):
        # Worker MFDual has visible_flows=all (default) + inject=true (YAML),
        # so the conditional FOLLOWUP_AGGREGATION_DEFAULTS fires.
        # Refactor 13: keys present with None, template_version drives expansion.
        # Refactor 14: template_root_space=plan inherited from planner_inferencer's
        # subtree-default cascade (`_template_root_space: plan`), through MFDual,
        # through worker_inferencers["__default__"]() factory.
        plan_bta = root.base_inferencer.planner_inferencer.base_inferencer
        worker_inferencers = plan_bta.worker_inferencers["__default__"]
        sample_mfdual = worker_inferencers()
        for i, fc in enumerate(sample_mfdual.flow_configs):
            initial = fc["initial_inferencer"]
            followup = fc["followup_inferencer"]
            # Subtree-default cascade reaches both initial and followup leaves.
            assert initial.template_root_space == "plan", (
                f"flow_configs[{i}].initial_inferencer.template_root_space should "
                f"cascade to 'plan' from planner_inferencer's _template_root_space; "
                f"got {initial.template_root_space!r}"
            )
            assert followup.template_root_space == "plan", (
                f"flow_configs[{i}].followup_inferencer.template_root_space should "
                f"cascade to 'plan'; got {followup.template_root_space!r}"
            )
            assert followup.template_version == "aggregation"
            assert followup.template_variables == {
                "task_preamble": None,
                "task_instructions": None,
                "task_response_format": None,
            }, (
                "Per-flow followup inferencer should have aggregation triplet "
                "defaulted via MFDual.SLOT_DEFAULTS conditional gate "
                "(visible_flows=all, inject_upstream_artifacts=true)"
            )
            # Existing template_extra_feed flag preserved.
            assert followup.template_extra_feed == {"include_iteration_judgment": True}

    def test_multi_flow_aggregator_full_triplet_defaulted(self, root):
        plan_bta = root.base_inferencer.planner_inferencer.base_inferencer
        sample_mfdual = plan_bta.worker_inferencers["__default__"]()
        agg = sample_mfdual.multi_flow_aggregator_inferencer
        # Triplet defaulted via MFDual.SLOT_DEFAULTS (3 lines removed).
        assert agg.template_version == "aggregation"
        assert agg.template_variables == {
            "task_preamble": None,
            "task_instructions": None,
            "task_response_format": None,
        }
        # Existing addendum flag preserved.
        assert agg.template_extra_feed == {"include_winner_pick": True}
        # Refactor 14: template_root_space=plan via planner subtree cascade.
        assert agg.template_root_space == "plan", (
            f"multi_flow_aggregator_inferencer.template_root_space should cascade "
            f"to 'plan' from planner_inferencer's _template_root_space; got "
            f"{agg.template_root_space!r}"
        )

    def test_exec_bta_breakdown_root_space_defaulted(self, root):
        exec_bta = root.base_inferencer.executor_inferencer
        # template_root_space=task_breakdown defaulted (line removed).
        assert exec_bta.breakdown_inferencer.template_root_space == "task_breakdown"

    def test_exec_worker_dual_review_key_defaulted(self, root):
        exec_bta = root.base_inferencer.executor_inferencer
        sample_dual = exec_bta.worker_inferencers["__default__"]()
        # template_key=review defaulted on the worker review (line removed).
        assert sample_dual.review_inferencer.template_key == "review"
        # Refactor 14: template_root_space=implementation cascades from
        # executor_inferencer's `_template_root_space` through worker_inferencers.
        assert sample_dual.base_inferencer.template_root_space == "implementation", (
            f"exec worker base_inferencer.template_root_space should cascade to "
            f"'implementation'; got {sample_dual.base_inferencer.template_root_space!r}"
        )
        assert sample_dual.review_inferencer.template_root_space == "implementation", (
            f"exec worker review_inferencer.template_root_space should cascade to "
            f"'implementation'; got {sample_dual.review_inferencer.template_root_space!r}"
        )

    def test_exec_aggregator_full_triplet_defaulted(self, root):
        exec_bta = root.base_inferencer.executor_inferencer
        agg = exec_bta.aggregator_inferencer
        # Refactor 12+13: BTA's aggregator default is now the full structured-
        # aggregation triplet. For exec BTA, all three variables resolve via
        # the implementation/main/_variables/<name>/aggregation.jinja2 files.
        # task_response_format/aggregation.jinja2 doesn't exist, so Refactor 12's
        # Pass 2 falls back to default if defined or returns None — load_variable
        # returns None and the literal "aggregation" sits unused since the
        # wrapper template doesn't reference {{ task_response_format }}.
        assert agg.template_version == "aggregation"
        assert agg.template_variables == {
            "task_preamble": None,
            "task_instructions": None,
            "task_response_format": None,
        }
        # Refactor 14: template_root_space=implementation cascades from
        # executor_inferencer's `_template_root_space`.
        assert agg.template_root_space == "implementation"

    def test_consensus_config_cascade_reaches_all_duals(self, root):
        # `_consensus_config: max_iterations: 1` at the top of the YAML cascades
        # via the `_-prefix` injection mechanism to every Dual / MFDual
        # descendant — outer Dual, planner Dual, worker MFDual, exec worker
        # Dual all receive max_iterations=1 without any per-node declaration
        # in the YAML body. Each instance is a separate deep-copy.
        outer = root
        plan_dual = root.base_inferencer.planner_inferencer
        plan_bta = plan_dual.base_inferencer
        mfdual = plan_bta.worker_inferencers["__default__"]()
        exec_bta = root.base_inferencer.executor_inferencer
        exec_dual = exec_bta.worker_inferencers["__default__"]()

        for label, dual in [
            ("outer", outer),
            ("planner", plan_dual),
            ("worker MFDual", mfdual),
            ("exec worker", exec_dual),
        ]:
            assert dual.consensus_config.max_iterations == 1, (
                f"{label}.consensus_config.max_iterations should be 1 "
                f"(supplied by _consensus_config cascade); got "
                f"{dual.consensus_config.max_iterations}"
            )

        # Independence: each Dual got its own deep-copied ConsensusConfig.
        instances = [outer.consensus_config, plan_dual.consensus_config,
                     mfdual.consensus_config, exec_dual.consensus_config]
        assert len(set(id(c) for c in instances)) == 4, (
            "each Dual should receive its own deep-copied ConsensusConfig "
            "instance — sharing a single instance would cause action-at-a-"
            "distance bugs if any one Dual mutates its config at runtime"
        )

    def test_fixer_inherits_via_inherits_directive(self, root):
        # The fixer PTI uses _inherits_ to deep-copy the base PTI. The
        # SLOT_DEFAULTS hook fires on the inherited node tree exactly the
        # same way → fixer's plan/exec BTAs also get all defaults.
        # Refactor 14: the `_template_root_space` cascade declarations on
        # planner_inferencer / executor_inferencer are part of the deep-copied
        # subtree, so fixer's leaves also inherit the subtree-default namespace.
        fixer_plan_bta = root.fixer_inferencer.planner_inferencer.base_inferencer
        assert fixer_plan_bta.breakdown_inferencer.template_root_space == "task_breakdown"
        # Mirror of the base plan BTA aggregator: full triplet defaulted via
        # SLOT_DEFAULTS (Refactor 12+13 flip) + template_root_space=plan via
        # Refactor 14 cascade through _inherits_ deep-copy.
        assert fixer_plan_bta.aggregator_inferencer.template_version == "aggregation"
        assert fixer_plan_bta.aggregator_inferencer.template_variables == {
            "task_preamble": None,
            "task_instructions": None,
            "task_response_format": None,
        }
        assert fixer_plan_bta.aggregator_inferencer.template_root_space == "plan"
        # Verify cascade reaches fixer's exec subtree too.
        fixer_exec_bta = root.fixer_inferencer.executor_inferencer
        assert fixer_exec_bta.aggregator_inferencer.template_root_space == "implementation"
        fixer_exec_dual = fixer_exec_bta.worker_inferencers["__default__"]()
        assert fixer_exec_dual.base_inferencer.template_root_space == "implementation"
        assert fixer_exec_dual.review_inferencer.template_root_space == "implementation"

    def test_params_defaults_resolve(self, root):
        # Refactor 15: `_params:` block at YAML root + `${_params.X}`
        # references resolve to the declared defaults at load time.
        plan_bta = root.base_inferencer.planner_inferencer.base_inferencer
        exec_bta = root.base_inferencer.executor_inferencer
        sample_mfdual = plan_bta.worker_inferencers["__default__"]()

        # Default values from `_params:` block in the YAML.
        assert plan_bta.max_breakdown == 2, (
            "plan_bta.max_breakdown should resolve from _params.plan_max_breakdown=2"
        )
        assert exec_bta.max_breakdown == 4, (
            "exec_bta.max_breakdown should resolve from _params.exec_max_breakdown=4"
        )
        for i, fc in enumerate(sample_mfdual.flow_configs):
            assert fc["max_dynamic_steps"] == 3, (
                f"flow[{i}].max_dynamic_steps should resolve from "
                f"_params.flow_max_dynamic_steps=3"
            )
        # _params.consensus_max_iterations cascades via _consensus_config to all Duals.
        assert root.consensus_config.max_iterations == 1
        # `_params` is auto-stripped at root by the `_-prefix` walker step;
        # no constructor receives it.
        assert not hasattr(root, "params"), (
            "_params must be stripped before instantiation (root)"
        )

    def test_params_override_via_load_config(self, tmp_path_factory):
        # Refactor 15: caller can override hyperparams via load_config(overrides=...).
        # The dotted-path override key matches the `_params.<name>` path.
        ws = tmp_path_factory.mktemp("hp_override_ws")
        from rich_python_utils.config_utils import load_config

        cfg = load_config(
            str(_REAL_YAML),
            overrides={
                "workspace": InferencerWorkspace(root=str(ws)),
                "_params.plan_max_breakdown": 7,
                "_params.exec_max_breakdown": 9,
                "_params.flow_max_dynamic_steps": 5,
                "_params.consensus_max_iterations": 3,
            },
        )
        obj = instantiate(cfg)

        plan_bta = obj.base_inferencer.planner_inferencer.base_inferencer
        exec_bta = obj.base_inferencer.executor_inferencer
        sample_mfdual = plan_bta.worker_inferencers["__default__"]()

        assert plan_bta.max_breakdown == 7
        assert exec_bta.max_breakdown == 9
        for fc in sample_mfdual.flow_configs:
            assert fc["max_dynamic_steps"] == 5
        # consensus cascade also picks up the override.
        assert obj.consensus_config.max_iterations == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
