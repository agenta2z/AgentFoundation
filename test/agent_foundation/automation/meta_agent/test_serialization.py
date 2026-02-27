"""Unit tests for serialization of meta-agent models and ActionGraph.

Tests cover:
1. SynthesisReport.to_dict() includes all 12 fields
2. SynthesisReport round-trip: to_dict → from_dict produces equivalent report
3. ExecutionTrace round-trip: to_dict → from_dict preserves all fields
4. TraceStep round-trip with all field types (datetime, nested result, etc.)
5. TraceActionResult round-trip
6. ActionGraph JSON round-trip (serialize → deserialize)

Requirements: 7.1, 7.2, 7.3, 7.4
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from agent_foundation.automation.meta_agent.models import (
    ExecutionTrace,
    SynthesisReport,
    TraceActionResult,
    TraceStep,
)
from agent_foundation.automation.schema.action_graph import ActionGraph
from agent_foundation.automation.schema.action_metadata import ActionMetadataRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYNTHESIS_REPORT_FIELDS = [
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
]


def _sample_report(**overrides) -> SynthesisReport:
    defaults = dict(
        total_steps=10,
        deterministic_count=5,
        parameterizable_count=2,
        agent_node_count=1,
        optional_count=1,
        user_input_boundary_count=0,
        branch_count=1,
        loop_count=0,
        synthesis_strategy="hybrid",
        target_strategy_coverage={"xpath": 3, "css": 2},
        template_variables=["search_query", "username"],
        warnings=["Ambiguous branch at step 4"],
    )
    defaults.update(overrides)
    return SynthesisReport(**defaults)


def _sample_result(**overrides) -> TraceActionResult:
    defaults = dict(
        success=True,
        action_skipped=False,
        skip_reason=None,
        value="clicked",
        error=None,
    )
    defaults.update(overrides)
    return TraceActionResult(**defaults)


def _sample_step(**overrides) -> TraceStep:
    defaults = dict(
        action_type="click",
        target="#submit-btn",
        args={"timeout": 5},
        result=_sample_result(),
        timestamp=datetime(2025, 1, 15, 10, 30, 0),
        html_before="<button id='submit-btn'>Submit</button>",
        html_after="<button id='submit-btn' disabled>Submit</button>",
        source_url="https://example.com/form",
        action_group_index=1,
        parallel_index=2,
        reasoning="Clicking the submit button to complete the form",
        metadata={"attempt": 1},
    )
    defaults.update(overrides)
    return TraceStep(**defaults)


def _sample_trace(**overrides) -> ExecutionTrace:
    defaults = dict(
        trace_id="trace-001",
        task_description="Fill out the registration form",
        steps=[_sample_step(), _sample_step(action_type="input_text", target="#email")],
        input_data={"email": "test@example.com"},
        success=True,
        error=None,
        start_time=datetime(2025, 1, 15, 10, 0, 0),
        end_time=datetime(2025, 1, 15, 10, 5, 0),
        session_dir="/tmp/sessions/trace-001",
        turn_count=3,
        metadata={"browser": "chrome"},
    )
    defaults.update(overrides)
    return ExecutionTrace(**defaults)


# ---------------------------------------------------------------------------
# SynthesisReport Tests
# ---------------------------------------------------------------------------


class TestSynthesisReportSerialization:
    """Tests for SynthesisReport JSON serialization."""

    def test_to_dict_includes_all_12_fields(self):
        """SynthesisReport.to_dict() must include all 12 documented fields."""
        report = _sample_report()
        d = report.to_dict()

        for field_name in SYNTHESIS_REPORT_FIELDS:
            assert field_name in d, f"Missing field: {field_name}"
        assert len(d) == len(SYNTHESIS_REPORT_FIELDS)

    def test_to_dict_values_match(self):
        """Serialized values must match the original report attributes."""
        report = _sample_report()
        d = report.to_dict()

        assert d["total_steps"] == 10
        assert d["deterministic_count"] == 5
        assert d["parameterizable_count"] == 2
        assert d["agent_node_count"] == 1
        assert d["optional_count"] == 1
        assert d["user_input_boundary_count"] == 0
        assert d["branch_count"] == 1
        assert d["loop_count"] == 0
        assert d["synthesis_strategy"] == "hybrid"
        assert d["target_strategy_coverage"] == {"xpath": 3, "css": 2}
        assert d["template_variables"] == ["search_query", "username"]
        assert d["warnings"] == ["Ambiguous branch at step 4"]

    def test_round_trip(self):
        """to_dict → from_dict must produce an equivalent SynthesisReport."""
        original = _sample_report()
        restored = SynthesisReport.from_dict(original.to_dict())

        assert restored.total_steps == original.total_steps
        assert restored.deterministic_count == original.deterministic_count
        assert restored.parameterizable_count == original.parameterizable_count
        assert restored.agent_node_count == original.agent_node_count
        assert restored.optional_count == original.optional_count
        assert restored.user_input_boundary_count == original.user_input_boundary_count
        assert restored.branch_count == original.branch_count
        assert restored.loop_count == original.loop_count
        assert restored.synthesis_strategy == original.synthesis_strategy
        assert restored.target_strategy_coverage == original.target_strategy_coverage
        assert restored.template_variables == original.template_variables
        assert restored.warnings == original.warnings

    def test_round_trip_with_defaults(self):
        """Round-trip works for a report using all default values."""
        original = SynthesisReport(
            total_steps=0,
            deterministic_count=0,
            parameterizable_count=0,
            agent_node_count=0,
            optional_count=0,
            user_input_boundary_count=0,
            branch_count=0,
            loop_count=0,
        )
        restored = SynthesisReport.from_dict(original.to_dict())

        assert restored.synthesis_strategy == "rule_based"
        assert restored.target_strategy_coverage == {}
        assert restored.template_variables == []
        assert restored.warnings == []

    def test_to_dict_produces_valid_json(self):
        """to_dict output must be JSON-serializable."""
        report = _sample_report()
        json_str = json.dumps(report.to_dict())
        parsed = json.loads(json_str)
        assert parsed["total_steps"] == 10


# ---------------------------------------------------------------------------
# TraceActionResult Tests
# ---------------------------------------------------------------------------


class TestTraceActionResultSerialization:
    """Tests for TraceActionResult JSON round-trip."""

    def test_round_trip_success(self):
        """Successful result round-trips correctly."""
        original = _sample_result()
        restored = TraceActionResult.from_dict(original.to_dict())

        assert restored.success == original.success
        assert restored.value == original.value
        assert restored.error is None

    def test_round_trip_failure(self):
        """Failed result with error and skip preserves all fields."""
        original = TraceActionResult(
            success=False,
            action_skipped=True,
            skip_reason="Element not found",
            value=None,
            error="TimeoutError: element not visible",
        )
        restored = TraceActionResult.from_dict(original.to_dict())

        assert restored.success is False
        assert restored.action_skipped is True
        assert restored.skip_reason == "Element not found"
        assert restored.value is None
        assert restored.error == "TimeoutError: element not visible"

    def test_round_trip_defaults(self):
        """Default TraceActionResult round-trips correctly."""
        original = TraceActionResult()
        restored = TraceActionResult.from_dict(original.to_dict())

        assert restored.success is True
        assert restored.action_skipped is False
        assert restored.skip_reason is None
        assert restored.value is None
        assert restored.error is None


# ---------------------------------------------------------------------------
# TraceStep Tests
# ---------------------------------------------------------------------------


class TestTraceStepSerialization:
    """Tests for TraceStep JSON round-trip."""

    def test_round_trip_all_fields(self):
        """TraceStep with all fields populated round-trips correctly."""
        original = _sample_step()
        restored = TraceStep.from_dict(original.to_dict())

        assert restored.action_type == original.action_type
        assert restored.target == original.target
        assert restored.args == original.args
        assert restored.result.success == original.result.success
        assert restored.result.value == original.result.value
        assert restored.timestamp == original.timestamp
        assert restored.html_before == original.html_before
        assert restored.html_after == original.html_after
        assert restored.source_url == original.source_url
        assert restored.action_group_index == original.action_group_index
        assert restored.parallel_index == original.parallel_index
        assert restored.reasoning == original.reasoning
        assert restored.metadata == original.metadata

    def test_round_trip_minimal(self):
        """TraceStep with only action_type round-trips correctly."""
        original = TraceStep(action_type="no_op")
        restored = TraceStep.from_dict(original.to_dict())

        assert restored.action_type == "no_op"
        assert restored.target is None
        assert restored.args is None
        assert restored.result is None
        assert restored.timestamp is None
        assert restored.action_group_index == 0
        assert restored.parallel_index == 0

    def test_datetime_serialization(self):
        """Timestamp is serialized as ISO format string and restored."""
        ts = datetime(2025, 6, 15, 14, 30, 45)
        original = TraceStep(action_type="click", timestamp=ts)
        d = original.to_dict()

        assert d["timestamp"] == "2025-06-15T14:30:45"
        restored = TraceStep.from_dict(d)
        assert restored.timestamp == ts

    def test_nested_result_round_trip(self):
        """Nested TraceActionResult is properly serialized and restored."""
        result = TraceActionResult(
            success=False, action_skipped=True, skip_reason="timeout", error="err"
        )
        original = TraceStep(action_type="click", result=result)
        restored = TraceStep.from_dict(original.to_dict())

        assert restored.result is not None
        assert restored.result.success is False
        assert restored.result.action_skipped is True
        assert restored.result.skip_reason == "timeout"
        assert restored.result.error == "err"


# ---------------------------------------------------------------------------
# ExecutionTrace Tests
# ---------------------------------------------------------------------------


class TestExecutionTraceSerialization:
    """Tests for ExecutionTrace JSON round-trip."""

    def test_round_trip_all_fields(self):
        """ExecutionTrace with all fields populated round-trips correctly."""
        original = _sample_trace()
        restored = ExecutionTrace.from_dict(original.to_dict())

        assert restored.trace_id == original.trace_id
        assert restored.task_description == original.task_description
        assert len(restored.steps) == len(original.steps)
        assert restored.input_data == original.input_data
        assert restored.success == original.success
        assert restored.error == original.error
        assert restored.start_time == original.start_time
        assert restored.end_time == original.end_time
        assert restored.session_dir == original.session_dir
        assert restored.turn_count == original.turn_count
        assert restored.metadata == original.metadata

    def test_round_trip_preserves_steps(self):
        """Steps are fully preserved through round-trip."""
        original = _sample_trace()
        restored = ExecutionTrace.from_dict(original.to_dict())

        for orig_step, rest_step in zip(original.steps, restored.steps):
            assert rest_step.action_type == orig_step.action_type
            assert rest_step.target == orig_step.target

    def test_round_trip_failed_trace(self):
        """Failed trace with error preserves error info."""
        original = _sample_trace(success=False, error="Agent crashed", steps=[])
        restored = ExecutionTrace.from_dict(original.to_dict())

        assert restored.success is False
        assert restored.error == "Agent crashed"
        assert restored.steps == []

    def test_round_trip_minimal(self):
        """Minimal trace with only required fields round-trips correctly."""
        original = ExecutionTrace(
            trace_id="min-trace",
            task_description="minimal",
            steps=[],
        )
        restored = ExecutionTrace.from_dict(original.to_dict())

        assert restored.trace_id == "min-trace"
        assert restored.task_description == "minimal"
        assert restored.steps == []
        assert restored.input_data is None
        assert restored.success is True
        assert restored.start_time is None
        assert restored.end_time is None
        assert restored.turn_count == 0

    def test_to_dict_produces_valid_json(self):
        """ExecutionTrace.to_dict() output is JSON-serializable."""
        trace = _sample_trace()
        json_str = json.dumps(trace.to_dict())
        parsed = json.loads(json_str)
        assert parsed["trace_id"] == "trace-001"


# ---------------------------------------------------------------------------
# ActionGraph JSON Round-Trip Tests
# ---------------------------------------------------------------------------


class TestActionGraphJsonRoundTrip:
    """Tests for ActionGraph JSON serialize → deserialize round-trip."""

    @pytest.fixture
    def mock_executor(self):
        return MagicMock()

    def test_simple_graph_round_trip(self, mock_executor):
        """A simple graph with sequential actions round-trips via JSON."""
        graph = ActionGraph(
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        graph.action("click", target="#btn")
        graph.action("input_text", target="#field", args={"text": "hello"})

        json_str = graph.serialize(output_format="json")
        restored = ActionGraph.deserialize(
            json_str,
            output_format="json",
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )

        assert len(restored._nodes) == len(graph._nodes)

    def test_single_action_round_trip(self, mock_executor):
        """A graph with a single action round-trips correctly."""
        graph = ActionGraph(
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        graph.action("visit_url", target="https://example.com")

        json_str = graph.serialize(output_format="json")
        restored = ActionGraph.deserialize(
            json_str,
            output_format="json",
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )

        assert len(restored._nodes) == 1

    def test_json_output_is_valid_json(self, mock_executor):
        """serialize(output_format='json') produces valid JSON."""
        graph = ActionGraph(
            action_executor=mock_executor,
            action_metadata=ActionMetadataRegistry(),
        )
        graph.action("click", target="#btn")

        json_str = graph.serialize(output_format="json")
        parsed = json.loads(json_str)
        assert "nodes" in parsed
        assert "version" in parsed
