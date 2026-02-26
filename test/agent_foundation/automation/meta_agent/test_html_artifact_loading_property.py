"""
Property-based test for HTML artifact loading from .parts/ directories.

Feature: meta-agent-workflow, Property 5: HTML artifact loading from .parts/ directories

*For any* trace step whose raw log has associated HTML artifacts in ``.parts/``
directories, the parsed TraceStep's html_before and html_after fields SHALL
contain the loaded HTML content from the corresponding artifact files.

**Validates: Requirements 2.5**
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from hypothesis import given, settings, strategies as st

from science_modeling_tools.automation.meta_agent.collector import TraceCollector
from science_modeling_tools.automation.meta_agent.models import (
    ExecutionTrace,
    TraceStep,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# HTML content: non-empty strings that look like HTML snippets
html_content_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        whitelist_characters="<>/=\"' \n\t",
    ),
    min_size=1,
    max_size=200,
).map(lambda s: f"<html><body>{s}</body></html>")

# Number of steps per trace
num_steps_st = st.integers(min_value=1, max_value=10)

# Action types for JSONL entries
action_type_st = st.sampled_from([
    "click", "input_text", "visit_url", "scroll", "wait",
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockAgentForArtifacts:
    """Agent mock that returns a result pointing to a session directory."""

    def __init__(self, session_dir: str):
        self._session_dir = session_dir

    def run(self, task_description: str, data: Any = None) -> Dict[str, str]:
        return {"session_dir": self._session_dir}


def _create_session_with_artifacts(
    base_dir: Path,
    num_steps: int,
    action_types: List[str],
    html_before_contents: List[Optional[str]],
    html_after_contents: List[Optional[str]],
) -> Path:
    """
    Create a session directory with JSONL entries and .parts/ HTML artifacts
    in SessionLogReader-compatible format.

    Returns the session directory path.
    """
    session_dir = base_dir / "session_001"
    turn_dir = session_dir / "turn_001"
    turn_dir.mkdir(parents=True)

    # Create manifest.json
    manifest = {
        "session_id": "test_session",
        "creation_timestamp": "2024-01-01T10:00:00",
        "session_type": "TestAgent",
        "status": "completed",
        "session_dir": str(session_dir),
        "session_log_file": "session.jsonl",
        "turns": [{
            "turn_number": 1,
            "start_timestamp": "2024-01-01T10:00:00",
            "log_file": "turn_001",
            "artifacts": [],
            "end_timestamp": "2024-01-01T10:05:00",
        }],
    }
    (session_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    # Create .parts/ directory
    parts_dir = turn_dir / "session.jsonl.parts"
    parts_dir.mkdir()

    # Write JSONL entries with __parts_file__ references
    jsonl_file = turn_dir / "session.jsonl"
    with open(jsonl_file, "w") as f:
        for i in range(num_steps):
            entry: Dict[str, Any] = {
                "action_type": action_types[i],
                "target": f"#el{i}",
            }

            if html_before_contents[i] is not None:
                fname = f"before_{i}.html"
                (parts_dir / fname).write_text(
                    html_before_contents[i], encoding="utf-8"
                )
                entry["body_html_before_last_action"] = {
                    "__parts_file__": fname,
                    "__value_type__": "str",
                }

            if html_after_contents[i] is not None:
                fname = f"after_{i}.html"
                (parts_dir / fname).write_text(
                    html_after_contents[i], encoding="utf-8"
                )
                entry["body_html_after_last_action"] = {
                    "__parts_file__": fname,
                    "__value_type__": "str",
                }

            f.write(json.dumps(entry) + "\n")

    return session_dir


# ---------------------------------------------------------------------------
# Property 5: HTML artifact loading from .parts/ directories
# ---------------------------------------------------------------------------


class TestHtmlArtifactLoadingProperty:
    """
    Property 5: HTML artifact loading from .parts/ directories

    *For any* trace step whose raw log has associated HTML artifacts in
    ``.parts/`` directories, the parsed TraceStep's html_before and
    html_after fields SHALL contain the loaded HTML content from the
    corresponding artifact files.

    **Validates: Requirements 2.5**
    """

    @given(
        num_steps=num_steps_st,
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_html_after_loaded_from_parts(
        self, num_steps: int, data: st.DataObject
    ):
        """
        For any trace with N steps where each step has an html_after
        artifact in .parts/, every TraceStep.html_after SHALL contain
        the loaded HTML content.
        """
        action_types = [
            data.draw(action_type_st, label=f"action_type_{i}")
            for i in range(num_steps)
        ]
        html_after_contents = [
            data.draw(html_content_st, label=f"html_after_{i}")
            for i in range(num_steps)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = _create_session_with_artifacts(
                base_dir=Path(tmpdir),
                num_steps=num_steps,
                action_types=action_types,
                html_before_contents=[None] * num_steps,
                html_after_contents=html_after_contents,
            )

            agent = MockAgentForArtifacts(str(session_dir))
            collector = TraceCollector(agent)
            traces = collector.collect("test task", run_count=1)

        steps = traces[0].steps
        assert len(steps) == num_steps

        for i in range(num_steps):
            assert steps[i].html_after == html_after_contents[i], (
                f"Step {i}: html_after should be '{html_after_contents[i]}', "
                f"got '{steps[i].html_after}'"
            )

    @given(
        num_steps=num_steps_st,
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_html_before_loaded_from_parts(
        self, num_steps: int, data: st.DataObject
    ):
        """
        For any trace with N steps where each step has an html_before
        artifact in .parts/, every TraceStep.html_before SHALL contain
        the loaded HTML content.
        """
        action_types = [
            data.draw(action_type_st, label=f"action_type_{i}")
            for i in range(num_steps)
        ]
        html_before_contents = [
            data.draw(html_content_st, label=f"html_before_{i}")
            for i in range(num_steps)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = _create_session_with_artifacts(
                base_dir=Path(tmpdir),
                num_steps=num_steps,
                action_types=action_types,
                html_before_contents=html_before_contents,
                html_after_contents=[None] * num_steps,
            )

            agent = MockAgentForArtifacts(str(session_dir))
            collector = TraceCollector(agent)
            traces = collector.collect("test task", run_count=1)

        steps = traces[0].steps
        assert len(steps) == num_steps

        for i in range(num_steps):
            assert steps[i].html_before == html_before_contents[i], (
                f"Step {i}: html_before should be '{html_before_contents[i]}', "
                f"got '{steps[i].html_before}'"
            )

    @given(
        num_steps=num_steps_st,
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_both_html_before_and_after_loaded(
        self, num_steps: int, data: st.DataObject
    ):
        """
        For any trace with N steps where each step has both html_before
        and html_after artifacts, both fields SHALL contain the correct
        loaded HTML content.
        """
        action_types = [
            data.draw(action_type_st, label=f"action_type_{i}")
            for i in range(num_steps)
        ]
        html_before_contents = [
            data.draw(html_content_st, label=f"html_before_{i}")
            for i in range(num_steps)
        ]
        html_after_contents = [
            data.draw(html_content_st, label=f"html_after_{i}")
            for i in range(num_steps)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = _create_session_with_artifacts(
                base_dir=Path(tmpdir),
                num_steps=num_steps,
                action_types=action_types,
                html_before_contents=html_before_contents,
                html_after_contents=html_after_contents,
            )

            agent = MockAgentForArtifacts(str(session_dir))
            collector = TraceCollector(agent)
            traces = collector.collect("test task", run_count=1)

        steps = traces[0].steps
        assert len(steps) == num_steps

        for i in range(num_steps):
            assert steps[i].html_before == html_before_contents[i], (
                f"Step {i}: html_before mismatch"
            )
            assert steps[i].html_after == html_after_contents[i], (
                f"Step {i}: html_after mismatch"
            )

    @given(
        num_steps=st.integers(min_value=2, max_value=10),
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_html_before_chaining_fills_gaps(
        self, num_steps: int, data: st.DataObject
    ):
        """
        For any trace where html_before is missing for step i but
        html_after is present for step i-1, the chaining logic SHALL
        set html_before[i] = html_after[i-1].
        """
        action_types = [
            data.draw(action_type_st, label=f"action_type_{i}")
            for i in range(num_steps)
        ]
        html_after_contents = [
            data.draw(html_content_st, label=f"html_after_{i}")
            for i in range(num_steps)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # No html_before artifacts — chaining should fill them
            session_dir = _create_session_with_artifacts(
                base_dir=Path(tmpdir),
                num_steps=num_steps,
                action_types=action_types,
                html_before_contents=[None] * num_steps,
                html_after_contents=html_after_contents,
            )

            agent = MockAgentForArtifacts(str(session_dir))
            collector = TraceCollector(agent)
            traces = collector.collect("test task", run_count=1)

        steps = traces[0].steps

        # Step 0 has no html_before (no previous step to chain from)
        assert steps[0].html_before is None, (
            "Step 0 html_before should be None when no artifact and no previous step"
        )

        # Steps 1..N-1 should have html_before chained from previous html_after
        for i in range(1, num_steps):
            assert steps[i].html_before == html_after_contents[i - 1], (
                f"Step {i}: html_before should be chained from step {i-1}'s "
                f"html_after ('{html_after_contents[i-1]}'), "
                f"got '{steps[i].html_before}'"
            )

    @given(
        num_steps=st.integers(min_value=2, max_value=10),
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_html_before_not_overwritten_when_present(
        self, num_steps: int, data: st.DataObject
    ):
        """
        For any trace where html_before[i] is already loaded from an
        artifact, the chaining logic SHALL NOT overwrite it with
        html_after[i-1].
        """
        action_types = [
            data.draw(action_type_st, label=f"action_type_{i}")
            for i in range(num_steps)
        ]
        html_before_contents = [
            data.draw(html_content_st, label=f"html_before_{i}")
            for i in range(num_steps)
        ]
        html_after_contents = [
            data.draw(html_content_st, label=f"html_after_{i}")
            for i in range(num_steps)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = _create_session_with_artifacts(
                base_dir=Path(tmpdir),
                num_steps=num_steps,
                action_types=action_types,
                html_before_contents=html_before_contents,
                html_after_contents=html_after_contents,
            )

            agent = MockAgentForArtifacts(str(session_dir))
            collector = TraceCollector(agent)
            traces = collector.collect("test task", run_count=1)

        steps = traces[0].steps

        for i in range(num_steps):
            assert steps[i].html_before == html_before_contents[i], (
                f"Step {i}: html_before should be the original artifact "
                f"'{html_before_contents[i]}', not overwritten by chaining. "
                f"Got '{steps[i].html_before}'"
            )
