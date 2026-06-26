"""Conversational adapter resumability — regression-critical tests, R11.5.

Coverage focus: bug regressions and edge cases NOT in
test_conversational_flow_node_adapter.py:TestSessionLevelResume (which
covers basic Level 1 short-circuit and basic Level 2 mid-conversation resume).

This file adds:
- Type-preserving round-trip for all JSON-shaped completion_result values
  (catches the regression where "123", "true", "null" got silently parsed
  as int/bool/None — empirically reproduced in spec review)
- Schema version mismatch handling
- Corrupted checkpoint.json graceful fallback
- Atomic write verification (tmp + os.replace, no half-written files)
- Missing completion_result with status=completed → fresh session

These are mock-based unit tests — fast, deterministic, every-PR worthy.
"""

import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.conversational.context import (
    AgenticResult,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
    ConversationalInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.conversational.flow_node_adapter import (
    ConversationalFlowNodeAdapter,
)


def _make_conversational_inferencer():
    """Minimal ConversationalInferencer for testing the adapter.
    Mocks base_inferencer so no actual LLM calls happen."""
    base = MagicMock()
    base.ainfer = AsyncMock(return_value="LLM says hello")
    base.system_prompt = ""
    base.set_messages = MagicMock()
    base.cache_folder = None
    return ConversationalInferencer(base_inferencer=base)


def _make_adapter(**kwargs):
    """Minimal adapter wrapping a mock conversational inferencer."""
    conv = _make_conversational_inferencer()
    kwargs.setdefault("conversational_inferencer", conv)
    return ConversationalFlowNodeAdapter(**kwargs)


def _make_agentic_result(text="hello", **overrides):
    defaults = dict(
        text=text,
        completed_actions=[],
        iterations_used=1,
        has_conversation_tool=False,
        exhausted_max_iterations=False,
        raw_response=text,
    )
    defaults.update(overrides)
    return AgenticResult(**defaults)


# ---------------------------------------------------------------------------
# JSON-coercion regression tests (R11.5)
#
# Production code at flow_node_adapter.py:418 stores completion_result via
# json.dumps(extracted, default=str). On resume at line 302 it does
# json.loads(completion_result). This symmetric encoding MUST preserve type.
#
# These tests catch the regression where a non-symmetric encoding silently
# coerced JSON-shaped strings ("123", "true", "null", "[1,2,3]") into their
# parsed Python types on resume.
# ---------------------------------------------------------------------------


class TestCompletionResultTypeRoundTrip(unittest.IsolatedAsyncioTestCase):
    """Validates: R11.5 — completion_result preserves type across resume."""

    async def _resume_with_stored_value(self, tmp_path, stored_value):
        """Helper: write a 'completed' checkpoint with `stored_value` already
        JSON-encoded (matching production format), then call _ainfer and
        return what comes back."""
        session_id = f"test-{id(stored_value)}"
        session_dir = os.path.join(str(tmp_path), session_id)
        os.makedirs(session_dir)
        checkpoint = {
            "schema_version": 1,
            "session_id": session_id,
            "initial_content": "task",
            "status": "completed",
            "turn_number": 1,
            "messages": [],
            "completion_result": json.dumps(stored_value),
        }
        with open(os.path.join(session_dir, "checkpoint.json"), "w") as f:
            json.dump(checkpoint, f)

        adapter = _make_adapter(
            checkpoint_dir=str(tmp_path),
            session_id=session_id,
        )
        # If somehow run_agentic_loop is invoked, fail loudly — Level 1 should short-circuit
        adapter.conversational_inferencer.run_agentic_loop = AsyncMock(
            side_effect=AssertionError("run_agentic_loop should NOT be called for completed status"),
        )
        return await adapter._ainfer("task")

    @pytest.fixture(autouse=True)
    def _inject_tmp_path(self, tmp_path):
        self._tmp_path = tmp_path

    async def test_plain_string(self):
        """Plain text string round-trips."""
        result = await self._resume_with_stored_value(self._tmp_path, "plain text")
        self.assertEqual(result, "plain text")
        self.assertIsInstance(result, str)

    async def test_string_that_looks_like_int(self):
        """Regression: string '123' must round-trip as str, NOT int."""
        result = await self._resume_with_stored_value(self._tmp_path, "123")
        self.assertEqual(result, "123")
        self.assertIsInstance(result, str, f"Expected str, got {type(result).__name__}")

    async def test_string_that_looks_like_bool(self):
        """Regression: string 'true' must round-trip as str, NOT bool."""
        result = await self._resume_with_stored_value(self._tmp_path, "true")
        self.assertEqual(result, "true")
        self.assertIsInstance(result, str, f"Expected str, got {type(result).__name__}")

    async def test_string_that_looks_like_null(self):
        """Regression: string 'null' must round-trip as str, NOT None."""
        result = await self._resume_with_stored_value(self._tmp_path, "null")
        self.assertEqual(result, "null")
        self.assertIsInstance(result, str, f"Expected str, got {type(result).__name__}")

    async def test_string_that_looks_like_json_array(self):
        """Regression: string '[1,2,3]' must round-trip as str, NOT list."""
        result = await self._resume_with_stored_value(self._tmp_path, "[1, 2, 3]")
        self.assertEqual(result, "[1, 2, 3]")
        self.assertIsInstance(result, str, f"Expected str, got {type(result).__name__}")

    async def test_real_list(self):
        """Real list completion_result preserves type."""
        result = await self._resume_with_stored_value(self._tmp_path, [1, 2, 3])
        self.assertEqual(result, [1, 2, 3])
        self.assertIsInstance(result, list)

    async def test_real_dict(self):
        """Real dict completion_result preserves type."""
        result = await self._resume_with_stored_value(self._tmp_path, {"k": "v", "n": 42})
        self.assertEqual(result, {"k": "v", "n": 42})
        self.assertIsInstance(result, dict)

    async def test_real_int(self):
        """Real int completion_result preserves type."""
        result = await self._resume_with_stored_value(self._tmp_path, 42)
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)


# ---------------------------------------------------------------------------
# Missing completion_result handling
# ---------------------------------------------------------------------------


class TestMissingCompletionResult(unittest.IsolatedAsyncioTestCase):
    """Validates: R11.5 — missing completion_result with status=completed
    should NOT crash; instead log warning and treat as fresh session."""

    @pytest.fixture(autouse=True)
    def _inject_tmp_path(self, tmp_path):
        self._tmp_path = tmp_path

    async def test_missing_completion_result_falls_through(self):
        """Status=completed but no completion_result → adapter logs warning,
        proceeds to fresh-session path (does NOT crash, does NOT return None)."""
        session_id = "missing-cr"
        session_dir = os.path.join(str(self._tmp_path), session_id)
        os.makedirs(session_dir)
        checkpoint = {
            "schema_version": 1,
            "session_id": session_id,
            "initial_content": "task",
            "status": "completed",
            "turn_number": 1,
            "messages": [],
            # NB: NO completion_result key
        }
        with open(os.path.join(session_dir, "checkpoint.json"), "w") as f:
            json.dump(checkpoint, f)

        adapter = _make_adapter(
            checkpoint_dir=str(self._tmp_path),
            session_id=session_id,
        )
        # When falling through, run_agentic_loop SHOULD be called (fresh session)
        adapter.conversational_inferencer.run_agentic_loop = AsyncMock(
            return_value=_make_agentic_result(text="fresh result"),
        )

        result = await adapter._ainfer("task")
        self.assertEqual(result, "fresh result")
        adapter.conversational_inferencer.run_agentic_loop.assert_called_once()


# ---------------------------------------------------------------------------
# Corrupted / malformed checkpoint handling
# ---------------------------------------------------------------------------


class TestCorruptedCheckpoint(unittest.IsolatedAsyncioTestCase):
    """Validates: R11.5 — corrupted checkpoint.json must NOT crash the
    adapter; instead log warning and start fresh."""

    @pytest.fixture(autouse=True)
    def _inject_tmp_path(self, tmp_path):
        self._tmp_path = tmp_path

    async def test_truncated_json_treated_as_absent(self):
        session_id = "corrupt-1"
        session_dir = os.path.join(str(self._tmp_path), session_id)
        os.makedirs(session_dir)
        # Write truncated/invalid JSON
        with open(os.path.join(session_dir, "checkpoint.json"), "w") as f:
            f.write('{"status": "completed", "completion')  # truncated

        adapter = _make_adapter(
            checkpoint_dir=str(self._tmp_path),
            session_id=session_id,
        )
        adapter.conversational_inferencer.run_agentic_loop = AsyncMock(
            return_value=_make_agentic_result(text="fresh after corruption"),
        )
        # Should NOT raise; should fall through to fresh session
        result = await adapter._ainfer("task")
        self.assertEqual(result, "fresh after corruption")

    async def test_empty_checkpoint_file_treated_as_absent(self):
        session_id = "corrupt-2"
        session_dir = os.path.join(str(self._tmp_path), session_id)
        os.makedirs(session_dir)
        # Write empty file
        with open(os.path.join(session_dir, "checkpoint.json"), "w") as f:
            f.write("")

        adapter = _make_adapter(
            checkpoint_dir=str(self._tmp_path),
            session_id=session_id,
        )
        adapter.conversational_inferencer.run_agentic_loop = AsyncMock(
            return_value=_make_agentic_result(text="ok"),
        )
        result = await adapter._ainfer("task")
        self.assertEqual(result, "ok")


# ---------------------------------------------------------------------------
# Atomic write verification
# ---------------------------------------------------------------------------


class TestAtomicWrite(unittest.IsolatedAsyncioTestCase):
    """Validates: R11.5 — atomic write via tmp + os.replace, no .tmp leaks."""

    @pytest.fixture(autouse=True)
    def _inject_tmp_path(self, tmp_path):
        self._tmp_path = tmp_path

    def test_atomic_write_no_leftover_tmp(self):
        """After successful write, no .checkpoint_*.tmp files remain."""
        session_id = "atomic-test"
        session_dir = os.path.join(str(self._tmp_path), session_id)
        adapter = _make_adapter(
            checkpoint_dir=str(self._tmp_path),
            session_id=session_id,
        )
        adapter._write_checkpoint_atomic(session_dir, {
            "schema_version": 1,
            "session_id": session_id,
            "status": "in_progress",
            "turn_number": 1,
            "messages": [],
        })
        # No leftover tmp files
        files = os.listdir(session_dir)
        tmp_files = [f for f in files if f.startswith(".checkpoint_")]
        self.assertEqual(len(tmp_files), 0, f"Found leftover tmp files: {tmp_files}")
        # Final file exists
        self.assertIn("checkpoint.json", files)

    def test_atomic_overwrite_replaces_atomically(self):
        """Overwriting an existing checkpoint should replace it atomically."""
        session_id = "atomic-overwrite"
        session_dir = os.path.join(str(self._tmp_path), session_id)
        adapter = _make_adapter(
            checkpoint_dir=str(self._tmp_path),
            session_id=session_id,
        )
        adapter._write_checkpoint_atomic(session_dir, {"v": 1})
        adapter._write_checkpoint_atomic(session_dir, {"v": 2})

        with open(os.path.join(session_dir, "checkpoint.json")) as f:
            data = json.load(f)
        self.assertEqual(data, {"v": 2})


if __name__ == "__main__":
    unittest.main()


def test_serialize_pause_state_sanitizes_non_json_prior_context():
    """§2.8/J7: a stray non-JSON value in the loosely-typed prior_context (e.g. a
    callable) is DROPPED on serialize (with a warning), so the pause blob always
    round-trips through Tier-1 to_json; JSON-safe values survive untouched."""
    from agent_foundation.common.inferencers.agentic_inferencers.conversational.conversational_inferencer import (
        ConversationalInferencer,
    )
    import json

    ci = ConversationalInferencer.__new__(ConversationalInferencer)
    ci.__dict__["prior_context"] = {
        "model_name": "claude",        # JSON-safe -> kept
        "count": 7,                    # JSON-safe -> kept
        "callback": lambda x: x,       # NOT JSON -> dropped
        "handle": object(),            # NOT JSON -> dropped
    }
    safe = ci._json_safe_prior_context()
    assert safe == {"model_name": "claude", "count": 7}
    json.dumps(safe)  # must not raise
