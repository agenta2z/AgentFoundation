"""Property test for SynthesisReport serialization completeness (Property 19).

**Validates: Requirements 7.4**

WHEN the Synthesis_Report is serialized, THE Serialization_System SHALL produce
a human-readable JSON document alongside the ActionGraph.

This test generates random SynthesisReport instances with varying field values
and verifies:
1. to_dict() output contains all 12 required fields
2. to_dict → from_dict round-trip preserves all field values
3. The serialized output is valid JSON
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from science_modeling_tools.automation.meta_agent.models import SynthesisReport


# ---------------------------------------------------------------------------
# Required fields that must appear in every serialized SynthesisReport
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = frozenset({
    "total_steps",
    "deterministic_count",
    "parameterizable_count",
    "agent_node_count",
    "optional_count",
    "user_input_boundary_count",
    "branch_count",
    "loop_count",
    "synthesis_strategy",
    "target_strategy_coverage",
    "template_variables",
    "warnings",
})


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

strategy_names = st.sampled_from(["rule_based", "llm", "hybrid"])

target_strategy_coverage = st.dictionaries(
    keys=st.sampled_from(["xpath", "css", "id", "data-qa", "aria", "agent"]),
    values=st.integers(min_value=0, max_value=100),
    max_size=6,
)

template_variables = st.lists(
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "Nd"), whitelist_characters="_"),
        min_size=1,
        max_size=20,
    ),
    max_size=10,
)

warnings_list = st.lists(
    st.text(min_size=1, max_size=80),
    max_size=5,
)


@st.composite
def synthesis_report(draw):
    """Generate a random SynthesisReport with valid field values."""
    total = draw(st.integers(min_value=0, max_value=200))
    deterministic = draw(st.integers(min_value=0, max_value=total))
    remaining = total - deterministic
    parameterizable = draw(st.integers(min_value=0, max_value=remaining))
    remaining -= parameterizable
    agent_node = draw(st.integers(min_value=0, max_value=remaining))
    remaining -= agent_node
    optional = draw(st.integers(min_value=0, max_value=remaining))

    return SynthesisReport(
        total_steps=total,
        deterministic_count=deterministic,
        parameterizable_count=parameterizable,
        agent_node_count=agent_node,
        optional_count=optional,
        user_input_boundary_count=draw(st.integers(min_value=0, max_value=50)),
        branch_count=draw(st.integers(min_value=0, max_value=20)),
        loop_count=draw(st.integers(min_value=0, max_value=20)),
        synthesis_strategy=draw(strategy_names),
        target_strategy_coverage=draw(target_strategy_coverage),
        template_variables=draw(template_variables),
        warnings=draw(warnings_list),
    )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(report=synthesis_report())
def test_serialized_contains_all_required_fields(report: SynthesisReport):
    """Property 19: to_dict() output contains all 12 required fields.

    **Validates: Requirements 7.4**
    """
    d = report.to_dict()
    assert REQUIRED_FIELDS == set(d.keys()), (
        f"Missing fields: {REQUIRED_FIELDS - set(d.keys())}, "
        f"Extra fields: {set(d.keys()) - REQUIRED_FIELDS}"
    )


@settings(max_examples=100)
@given(report=synthesis_report())
def test_round_trip_preserves_all_fields(report: SynthesisReport):
    """Property 19: to_dict → from_dict round-trip preserves all field values.

    **Validates: Requirements 7.4**
    """
    d = report.to_dict()
    restored = SynthesisReport.from_dict(d)

    assert restored.total_steps == report.total_steps
    assert restored.deterministic_count == report.deterministic_count
    assert restored.parameterizable_count == report.parameterizable_count
    assert restored.agent_node_count == report.agent_node_count
    assert restored.optional_count == report.optional_count
    assert restored.user_input_boundary_count == report.user_input_boundary_count
    assert restored.branch_count == report.branch_count
    assert restored.loop_count == report.loop_count
    assert restored.synthesis_strategy == report.synthesis_strategy
    assert restored.target_strategy_coverage == report.target_strategy_coverage
    assert restored.template_variables == report.template_variables
    assert restored.warnings == report.warnings


@settings(max_examples=100)
@given(report=synthesis_report())
def test_serialized_output_is_valid_json(report: SynthesisReport):
    """Property 19: The serialized output is valid JSON.

    **Validates: Requirements 7.4**
    """
    d = report.to_dict()
    json_str = json.dumps(d)
    parsed = json.loads(json_str)

    assert isinstance(parsed, dict)
    assert REQUIRED_FIELDS == set(parsed.keys())
