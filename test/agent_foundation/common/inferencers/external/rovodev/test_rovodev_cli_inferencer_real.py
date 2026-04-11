"""Integration tests for RovoDevCliInferencer.

These tests require:
1. acli binary installed and in PATH
2. acli authenticated (run ``acli auth login`` first)
3. Network access to the Rovo Dev proxy

Run with:
    PYTHONPATH=src:../RichPythonUtils/src python3.11 -m pytest \
        test/agent_foundation/common/inferencers/external/rovodev/test_rovodev_cli_inferencer_real.py -vs -m integration
"""

import shutil
from pathlib import Path

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
    RovoDevCliInferencer,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not shutil.which("acli"),
        reason="acli not installed",
    ),
]

DEFAULT_TIMEOUT = 600


def _make_inferencer(tmp_path, **kwargs):
    out_file = str(tmp_path / "rovodev_output.txt")
    defaults = dict(
        working_dir=str(tmp_path),
        output_file=out_file,
        idle_timeout_seconds=DEFAULT_TIMEOUT,
        tool_use_idle_timeout_seconds=DEFAULT_TIMEOUT,
    )
    defaults.update(kwargs)
    return RovoDevCliInferencer(**defaults)


@pytest.fixture
def inferencer(tmp_path):
    return _make_inferencer(tmp_path)


class TestSingleTurn:
    def test_simple_math(self, inferencer):
        result = inferencer("What is 2+2? Reply with ONLY the number, nothing else.")
        assert result.success, f"Expected success, got stderr: {result.stderr}"
        assert "4" in result.output, f"Expected '4' in output, got: {result.output!r}"

    def test_returns_terminal_inferencer_response(self, inferencer):
        result = inferencer("What is 3+3? Reply with ONLY the number, nothing else.")
        assert result.success is True
        assert "6" in result.output, f"Expected '6' in output, got: {result.output!r}"

    @pytest.mark.asyncio
    async def test_async_infer(self, inferencer):
        result = await inferencer.ainfer(
            "What is 5+5? Reply with ONLY the number, nothing else."
        )
        assert result.success is True
        assert "10" in result.output, f"Expected '10' in output, got: {result.output!r}"


class TestOutputFile:
    def test_output_file_captures_clean_text(self, tmp_path):
        inf = _make_inferencer(tmp_path)
        result = inf("What is 7+7? Reply with ONLY the number, nothing else.")
        assert result.success is True
        out = Path(inf.output_file)
        assert out.exists(), f"Output file not created: {inf.output_file}"
        content = out.read_text().strip()
        assert "14" in content, f"Expected '14' in file, got: {content!r}"
        assert "\x1b[" not in content, "Output file should not contain ANSI codes"


class TestSessionManagement:
    def test_new_session_sets_active_session_id(self, inferencer):
        assert inferencer.active_session_id is None
        result = inferencer.new_session("Say hello and nothing else.")
        assert result.success, f"Failed: {result.stderr}"
        assert inferencer.active_session_id is not None
        # Should be a real UUID, not a sentinel
        assert len(inferencer.active_session_id) > 10, f"Expected UUID, got: {inferencer.active_session_id!r}"

    def test_restore_flag_in_command(self, inferencer):
        r1 = inferencer.new_session("Say hi and nothing else.")
        assert r1.success
        cmd = inferencer.construct_command("follow up", resume=True)
        assert "--restore" in cmd

    def test_two_turn_mechanics(self, tmp_path):
        """Verify multi-turn machinery: turn 1 sets session, turn 2 sends --restore."""
        inf = _make_inferencer(tmp_path)

        # Turn 1
        r1 = inf.new_session("Say OK and nothing else.")
        assert r1.success, f"Turn 1 failed: {r1.stderr}"
        assert inf.active_session_id is not None, f"Expected session ID, got None"

        # Turn 2: verify --restore is in command and call succeeds
        r2 = inf("Say DONE and nothing else.")
        assert r2.success, f"Turn 2 failed: {r2.stderr}"
        assert "DONE" in r2.output.upper(), f"Turn 2 output: {r2.output!r}"


class TestRealMultiTurn:
    """Real end-to-end multi-turn context recall test.

    Verifies that ``--restore`` resumes a session and the model retains
    context from previous turns.

    Note:
        The working directory must be a git repo for session workspace
        matching to work. ``tmp_path`` is NOT a git repo, so we init one.
    """

    def test_context_recall_across_turns(self, tmp_path):
        """Turn 1 stores a favorite color, turn 2 recalls it via --restore."""
        import subprocess

        # Initialize a git repo — required for session workspace matching
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-q", "--allow-empty", "-m", "init"],
            cwd=tmp_path, check=True, capture_output=True,
        )

        inf = _make_inferencer(tmp_path)

        # Turn 1: give the model something to remember
        r1 = inf.new_session(
            "My favorite color is BLUE. "
            "Confirm: your favorite color is BLUE."
        )
        assert r1.success, f"Turn 1 failed (rc={r1.return_code}): {r1.stderr}"
        assert "BLUE" in r1.output.upper(), f"Turn 1 missing BLUE: {r1.output!r}"
        assert inf.active_session_id is not None, f"Expected session ID, got None"

        # Turn 2: resume and recall
        r2 = inf(
            "What is my favorite color? "
            "Reply with ONLY the color, nothing else."
        )
        assert r2.success, f"Turn 2 failed (rc={r2.return_code}): {r2.stderr}"
        assert "BLUE" in r2.output.upper(), (
            f"Context recall failed: expected BLUE, got: {r2.output!r}"
        )


    def test_restore_specific_session_by_id(self, tmp_path):
        """Restore a specific session by UUID, not just the most recent."""
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-q", "--allow-empty", "-m", "init"],
            cwd=tmp_path, check=True, capture_output=True,
        )

        out_file = str(tmp_path / "rovodev_output.txt")
        mk = lambda: RovoDevCliInferencer(
            working_dir=str(tmp_path), output_file=out_file,
            idle_timeout_seconds=DEFAULT_TIMEOUT, tool_use_idle_timeout_seconds=DEFAULT_TIMEOUT,
        )

        # Create 3 sessions with different values
        inf_a = mk()
        r_a = inf_a.new_session("My favorite color is BLUE. Confirm: favorite color is BLUE.")
        assert r_a.success, f"Session A failed: {r_a.stderr}"
        sid_a = inf_a.active_session_id
        assert sid_a is not None

        inf_b = mk()
        r_b = inf_b.new_session("My favorite color is GREEN. Confirm: favorite color is GREEN.")
        assert r_b.success, f"Session B failed: {r_b.stderr}"
        sid_b = inf_b.active_session_id
        assert sid_b is not None

        inf_c = mk()
        r_c = inf_c.new_session("My favorite color is RED. Confirm: favorite color is RED.")
        assert r_c.success, f"Session C failed: {r_c.stderr}"
        sid_c = inf_c.active_session_id
        assert sid_c is not None

        # All session IDs must be different
        assert len({sid_a, sid_b, sid_c}) == 3, (
            f"Expected 3 unique IDs, got: {sid_a}, {sid_b}, {sid_c}"
        )

        # Restore Session A (oldest) — if only "most recent" worked, this would return RED
        inf_ra = mk()
        inf_ra.active_session_id = sid_a
        r_ra = inf_ra("What is my favorite color? Reply with ONLY the color.")
        assert r_ra.success, f"Restore A failed: {r_ra.stderr}"
        assert "BLUE" in r_ra.output.upper(), (
            f"Session A should recall BLUE, got: {r_ra.output!r}"
        )

        # Restore Session B (middle)
        inf_rb = mk()
        inf_rb.active_session_id = sid_b
        r_rb = inf_rb("What is my favorite color? Reply with ONLY the color.")
        assert r_rb.success, f"Restore B failed: {r_rb.stderr}"
        assert "GREEN" in r_rb.output.upper(), (
            f"Session B should recall GREEN, got: {r_rb.output!r}"
        )

        # Restore Session C (newest)
        inf_rc = mk()
        inf_rc.active_session_id = sid_c
        r_rc = inf_rc("What is my favorite color? Reply with ONLY the color.")
        assert r_rc.success, f"Restore C failed: {r_rc.stderr}"
        assert "RED" in r_rc.output.upper(), (
            f"Session C should recall RED, got: {r_rc.output!r}"
        )


class TestStreaming:
    def test_infer_streaming_yields_chunks(self, inferencer):
        chunks = list(
            inferencer.infer_streaming("Count from 1 to 3, each on a new line. Nothing else.")
        )
        non_empty = [c for c in chunks if c.strip()]
        assert len(non_empty) > 0, "Expected at least one non-empty chunk"
        full_text = "".join(non_empty)
        assert "1" in full_text, f"Expected '1' in streaming output, got: {full_text!r}"


class TestErrorHandling:
    def test_invalid_acli_path_in_command(self, tmp_path):
        inf = RovoDevCliInferencer(acli_path="/nonexistent/acli", working_dir=str(tmp_path))
        cmd = inf.construct_command("hello")
        assert "/nonexistent/acli" in cmd
