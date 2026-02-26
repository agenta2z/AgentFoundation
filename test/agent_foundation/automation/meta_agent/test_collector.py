"""Unit tests for TraceCollector.

Tests cover:
- run_count=0 raises ValueError
- Single run produces one trace
- All-failing runs still return N traces with success=False
- Synthetic data provider integration
- html_before chaining
- Session directory parsing via SessionLogReader

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from science_modeling_tools.automation.meta_agent.collector import TraceCollector
from science_modeling_tools.automation.meta_agent.models import (
    ExecutionTrace,
    TraceStep,
)


# ---------------------------------------------------------------------------
# Mock agent and helpers
# ---------------------------------------------------------------------------

@dataclass
class MockAgentResult:
    """Simulates an agent run result with a session directory."""
    session_dir: Optional[str] = None


class MockAgent:
    """Agent that records calls and returns configurable results."""

    def __init__(
        self,
        session_dir: Optional[str] = None,
        fail_on_indices: Optional[set] = None,
    ):
        self._session_dir = session_dir
        self._fail_on_indices = fail_on_indices or set()
        self.calls: List[Dict[str, Any]] = []
        self._call_count = 0

    def run(self, task_description: str, data: Any = None) -> MockAgentResult:
        idx = self._call_count
        self._call_count += 1
        self.calls.append({"task": task_description, "data": data, "index": idx})

        if idx in self._fail_on_indices:
            raise RuntimeError(f"Agent failed on run {idx}")

        return MockAgentResult(session_dir=self._session_dir)


class MockSyntheticDataProvider:
    """Generates distinct synthetic data sets."""

    def generate(self, count: int) -> List[Dict[str, Any]]:
        return [{"run_index": i, "query": f"query_{i}"} for i in range(count)]


def _create_manifest(session_dir: Path, num_turns: int = 1) -> None:
    """Create a manifest.json for a session directory with the given number of turns."""
    turns = []
    for i in range(1, num_turns + 1):
        turns.append({
            "turn_number": i,
            "start_timestamp": "2024-01-01T10:00:00",
            "log_file": f"turn_{i:03d}",
            "artifacts": [],
            "end_timestamp": "2024-01-01T10:05:00",
        })
    manifest = {
        "session_id": "test_session",
        "creation_timestamp": "2024-01-01T10:00:00",
        "session_type": "TestAgent",
        "status": "completed",
        "session_dir": str(session_dir),
        "session_log_file": "session.jsonl",
        "turns": turns,
    }
    (session_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )


# ---------------------------------------------------------------------------
# ValueError on invalid run_count (Req 1.5)
# ---------------------------------------------------------------------------

class TestRunCountValidation:
    def test_run_count_zero_raises_value_error(self):
        agent = MockAgent()
        collector = TraceCollector(agent)
        with pytest.raises(ValueError, match="run_count must be >= 1"):
            collector.collect("test task", run_count=0)

    def test_run_count_negative_raises_value_error(self):
        agent = MockAgent()
        collector = TraceCollector(agent)
        with pytest.raises(ValueError):
            collector.collect("test task", run_count=-5)


# ---------------------------------------------------------------------------
# Basic collection (Req 1.1)
# ---------------------------------------------------------------------------

class TestBasicCollection:
    def test_single_run_produces_one_trace(self):
        agent = MockAgent()
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=1)
        assert len(traces) == 1
        assert traces[0].task_description == "test task"
        assert traces[0].success is True

    def test_multiple_runs_produce_correct_count(self):
        agent = MockAgent()
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=5)
        assert len(traces) == 5

    def test_agent_called_correct_number_of_times(self):
        agent = MockAgent()
        collector = TraceCollector(agent)
        collector.collect("test task", run_count=3)
        assert len(agent.calls) == 3

    def test_task_description_passed_to_agent(self):
        agent = MockAgent()
        collector = TraceCollector(agent)
        collector.collect("search for flights", run_count=1)
        assert agent.calls[0]["task"] == "search for flights"

    def test_trace_has_unique_trace_id(self):
        agent = MockAgent()
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=3)
        ids = [t.trace_id for t in traces]
        assert len(set(ids)) == 3  # all unique

    def test_trace_has_timestamps(self):
        agent = MockAgent()
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=1)
        assert traces[0].start_time is not None
        assert traces[0].end_time is not None


# ---------------------------------------------------------------------------
# Failure handling (Req 1.4)
# ---------------------------------------------------------------------------

class TestFailureHandling:
    def test_failed_run_produces_trace_with_success_false(self):
        agent = MockAgent(fail_on_indices={0})
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=1)
        assert len(traces) == 1
        assert traces[0].success is False
        assert traces[0].error is not None

    def test_all_failing_runs_still_return_n_traces(self):
        agent = MockAgent(fail_on_indices={0, 1, 2})
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=3)
        assert len(traces) == 3
        assert all(not t.success for t in traces)

    def test_partial_failure_continues_remaining_runs(self):
        agent = MockAgent(fail_on_indices={1})
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=3)
        assert len(traces) == 3
        assert traces[0].success is True
        assert traces[1].success is False
        assert traces[2].success is True

    def test_failed_trace_has_empty_steps(self):
        agent = MockAgent(fail_on_indices={0})
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=1)
        assert traces[0].steps == []


# ---------------------------------------------------------------------------
# Synthetic data provider (Req 1.3)
# ---------------------------------------------------------------------------

class TestSyntheticDataProvider:
    def test_synthetic_data_supplied_to_each_run(self):
        agent = MockAgent()
        provider = MockSyntheticDataProvider()
        collector = TraceCollector(agent, synthetic_data_provider=provider)
        traces = collector.collect("test task", run_count=3)

        # Each call should have received distinct data
        for i, call in enumerate(agent.calls):
            assert call["data"]["run_index"] == i

    def test_explicit_input_data_overrides_provider(self):
        agent = MockAgent()
        provider = MockSyntheticDataProvider()
        collector = TraceCollector(agent, synthetic_data_provider=provider)

        explicit_data = [{"custom": "data_0"}, {"custom": "data_1"}]
        collector.collect("test task", run_count=2, input_data=explicit_data)

        assert agent.calls[0]["data"] == {"custom": "data_0"}
        assert agent.calls[1]["data"] == {"custom": "data_1"}

    def test_no_provider_no_data_passes_none(self):
        agent = MockAgent()
        collector = TraceCollector(agent)
        collector.collect("test task", run_count=1)
        assert agent.calls[0]["data"] is None

    def test_input_data_recorded_in_trace(self):
        agent = MockAgent()
        provider = MockSyntheticDataProvider()
        collector = TraceCollector(agent, synthetic_data_provider=provider)
        traces = collector.collect("test task", run_count=2)
        assert traces[0].input_data == {"run_index": 0, "query": "query_0"}
        assert traces[1].input_data == {"run_index": 1, "query": "query_1"}


# ---------------------------------------------------------------------------
# Session directory parsing (Req 1.2)
# ---------------------------------------------------------------------------

class TestSessionDirectoryParsing:
    def test_parses_jsonl_entries_into_steps(self, tmp_path):
        """Create a minimal session directory and verify parsing."""
        session_dir = tmp_path / "session_001"
        turn_dir = session_dir / "turn_001"
        turn_dir.mkdir(parents=True)

        _create_manifest(session_dir, num_turns=1)

        # Write a flat JSONL file
        jsonl_file = turn_dir / "session.jsonl"
        entries = [
            {"action_type": "click", "target": "#btn", "args": {}, "timestamp": "2024-01-01T10:00:00Z"},
            {"action_type": "input_text", "target": "#input", "args": {"text": "hello"}},
        ]
        with open(jsonl_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        agent = MockAgent(session_dir=str(session_dir))
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=1)

        assert len(traces[0].steps) == 2
        assert traces[0].steps[0].action_type == "click"
        assert traces[0].steps[1].action_type == "input_text"

    def test_counts_turns(self, tmp_path):
        session_dir = tmp_path / "session_001"
        (session_dir / "turn_001").mkdir(parents=True)
        (session_dir / "turn_002").mkdir(parents=True)

        _create_manifest(session_dir, num_turns=2)

        agent = MockAgent(session_dir=str(session_dir))
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=1)
        assert traces[0].turn_count == 2

    def test_no_session_dir_produces_empty_steps(self):
        agent = MockAgent(session_dir=None)
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=1)
        assert traces[0].steps == []

    def test_nonexistent_session_dir_produces_empty_steps(self):
        agent = MockAgent(session_dir="/nonexistent/path")
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=1)
        assert traces[0].steps == []

    def test_missing_manifest_produces_empty_steps(self, tmp_path):
        """Session directory without manifest.json gracefully returns empty steps."""
        session_dir = tmp_path / "session_no_manifest"
        session_dir.mkdir()
        agent = MockAgent(session_dir=str(session_dir))
        collector = TraceCollector(agent)
        traces = collector.collect("test", run_count=1)
        assert traces[0].steps == []


# ---------------------------------------------------------------------------
# HTML artifact loading and chaining
# ---------------------------------------------------------------------------

class TestHtmlArtifacts:
    def test_html_before_chaining(self, tmp_path):
        """When html_before[i] is None, it should be set to html_after[i-1]."""
        session_dir = tmp_path / "session_001"
        turn_dir = session_dir / "turn_001"
        turn_dir.mkdir(parents=True)

        _create_manifest(session_dir, num_turns=1)

        # Write JSONL entries with __parts_file__ reference for html_after on step 0
        jsonl_file = turn_dir / "session.jsonl"
        entries = [
            {
                "action_type": "click",
                "target": "#btn1",
                "body_html_after_last_action": {
                    "__parts_file__": "after_0.html",
                    "__value_type__": "str",
                },
            },
            {"action_type": "click", "target": "#btn2"},
        ]
        with open(jsonl_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        # Create .parts/ with the referenced HTML file
        parts_dir = turn_dir / "session.jsonl.parts"
        parts_dir.mkdir()
        (parts_dir / "after_0.html").write_text("<html>after_0</html>")

        agent = MockAgent(session_dir=str(session_dir))
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=1)

        steps = traces[0].steps
        assert len(steps) == 2
        # Step 0 should have html_after loaded
        assert steps[0].html_after == "<html>after_0</html>"
        # Step 1 should have html_before chained from step 0's html_after
        assert steps[1].html_before == "<html>after_0</html>"

    def test_html_before_not_chained_when_present(self, tmp_path):
        """When html_before[i] is already set, don't overwrite it."""
        session_dir = tmp_path / "session_001"
        turn_dir = session_dir / "turn_001"
        turn_dir.mkdir(parents=True)

        _create_manifest(session_dir, num_turns=1)

        jsonl_file = turn_dir / "session.jsonl"
        entries = [
            {
                "action_type": "click",
                "target": "#btn1",
                "body_html_before_last_action": {
                    "__parts_file__": "before_0.html",
                    "__value_type__": "str",
                },
                "body_html_after_last_action": {
                    "__parts_file__": "after_0.html",
                    "__value_type__": "str",
                },
            },
            {
                "action_type": "click",
                "target": "#btn2",
                "body_html_before_last_action": {
                    "__parts_file__": "before_1.html",
                    "__value_type__": "str",
                },
                "body_html_after_last_action": {
                    "__parts_file__": "after_1.html",
                    "__value_type__": "str",
                },
            },
        ]
        with open(jsonl_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        parts_dir = turn_dir / "session.jsonl.parts"
        parts_dir.mkdir()
        (parts_dir / "before_0.html").write_text("<html>before_0</html>")
        (parts_dir / "after_0.html").write_text("<html>after_0</html>")
        (parts_dir / "before_1.html").write_text("<html>before_1</html>")
        (parts_dir / "after_1.html").write_text("<html>after_1</html>")

        agent = MockAgent(session_dir=str(session_dir))
        collector = TraceCollector(agent)
        traces = collector.collect("test task", run_count=1)

        steps = traces[0].steps
        # Both steps should have their own html_before
        assert steps[0].html_before == "<html>before_0</html>"
        assert steps[1].html_before == "<html>before_1</html>"


# ---------------------------------------------------------------------------
# Agent result extraction
# ---------------------------------------------------------------------------

class TestAgentResultExtraction:
    def test_dict_result_with_session_dir(self):
        """Agent returning a dict with session_dir key."""
        class DictAgent:
            def run(self, task, data=None):
                return {"session_dir": "/tmp/session"}

        collector = TraceCollector(DictAgent())
        traces = collector.collect("test", run_count=1)
        assert traces[0].session_dir == "/tmp/session"

    def test_none_result(self):
        """Agent returning None."""
        class NoneAgent:
            def run(self, task, data=None):
                return None

        collector = TraceCollector(NoneAgent())
        traces = collector.collect("test", run_count=1)
        assert traces[0].session_dir is None
        assert traces[0].steps == []


# ---------------------------------------------------------------------------
# JSONL entry parsing
# ---------------------------------------------------------------------------

class TestJsonlParsing:
    def test_agent_response_entry(self, tmp_path):
        """AgentResponse entries with next_actions are parsed."""
        session_dir = tmp_path / "session"
        turn_dir = session_dir / "turn_001"
        turn_dir.mkdir(parents=True)

        _create_manifest(session_dir, num_turns=1)

        jsonl_file = turn_dir / "session.jsonl"
        entry = {
            "type": "AgentResponse",
            "data": {
                "next_actions": [
                    [{"action_type": "click", "target": "#btn"}]
                ]
            }
        }
        jsonl_file.write_text(json.dumps(entry) + "\n")

        agent = MockAgent(session_dir=str(session_dir))
        collector = TraceCollector(agent)
        traces = collector.collect("test", run_count=1)
        assert len(traces[0].steps) == 1
        assert traces[0].steps[0].action_type == "click"

    def test_action_result_entry(self, tmp_path):
        """AgentActionResults entries are parsed."""
        session_dir = tmp_path / "session"
        turn_dir = session_dir / "turn_001"
        turn_dir.mkdir(parents=True)

        _create_manifest(session_dir, num_turns=1)

        jsonl_file = turn_dir / "session.jsonl"
        entry = {
            "type": "AgentActionResults",
            "data": {
                "action_type": "input_text",
                "target": "#input",
                "result": {"success": True, "value": "typed"},
            }
        }
        jsonl_file.write_text(json.dumps(entry) + "\n")

        agent = MockAgent(session_dir=str(session_dir))
        collector = TraceCollector(agent)
        traces = collector.collect("test", run_count=1)
        assert len(traces[0].steps) == 1
        assert traces[0].steps[0].action_type == "input_text"
        assert traces[0].steps[0].result is not None
        assert traces[0].steps[0].result.success is True

    def test_empty_jsonl_file(self, tmp_path):
        """Empty JSONL file produces no steps."""
        session_dir = tmp_path / "session"
        turn_dir = session_dir / "turn_001"
        turn_dir.mkdir(parents=True)

        _create_manifest(session_dir, num_turns=1)

        (turn_dir / "session.jsonl").write_text("")

        agent = MockAgent(session_dir=str(session_dir))
        collector = TraceCollector(agent)
        traces = collector.collect("test", run_count=1)
        assert traces[0].steps == []
