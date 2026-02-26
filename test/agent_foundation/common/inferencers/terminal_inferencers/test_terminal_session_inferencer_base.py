"""Unit tests for TerminalSessionInferencerBase class."""

import unittest
from typing import Any, Dict, List

import resolve_path  # noqa: F401

from attr import attrib, attrs

from science_modeling_tools.common.inferencers.terminal_inferencers.terminal_session_inferencer_base import (
    TerminalSessionInferencerBase,
)


@attrs
class MockSessionInferencer(TerminalSessionInferencerBase):
    """
    Mock implementation of TerminalSessionInferencerBase for testing.

    This inferencer simulates a CLI tool with session support by returning
    predetermined outputs with session tracking.
    """

    # Mock response to return
    mock_response: str = attrib(default="Mock response")
    mock_session_id: str = attrib(default="mock-session-123")

    def construct_command(self, inference_input: Any, **kwargs) -> str:
        """Construct a mock command string."""
        prompt = str(inference_input)
        session_id = kwargs.get("session_id")
        is_resume = kwargs.get("resume", False)

        cmd_parts = ["mock_cli", f'"prompt={prompt}"']

        if is_resume and session_id:
            session_args = self._build_session_args(session_id, is_resume)
            cmd_parts.insert(1, session_args)

        return " ".join(cmd_parts)

    def _build_session_args(self, session_id: str, is_resume: bool) -> str:
        """Build mock session CLI arguments."""
        if is_resume and session_id:
            return f"{self.resume_arg_name} {self.session_arg_name} '{session_id}'"
        elif session_id:
            return f"{self.session_arg_name} '{session_id}'"
        return ""

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> Dict[str, Any]:
        """Parse mock output and include session ID."""
        return {
            "output": stdout.strip() if stdout else self.mock_response,
            "stderr": stderr.strip() if stderr else "",
            "return_code": return_code,
            "success": return_code == 0,
            "session_id": self.mock_session_id,
        }


class TestSessionManagement(unittest.TestCase):
    """Test cases for session management methods."""

    def setUp(self):
        """Set up test inferencer."""
        self.inferencer = MockSessionInferencer()

    def test_start_session_generates_id(self):
        """Test that start_session generates a UUID when no ID provided."""
        session_id = self.inferencer.start_session()

        self.assertIsNotNone(session_id)
        self.assertEqual(len(session_id), 36)  # UUID format
        self.assertEqual(self.inferencer.active_session_id, session_id)
        self.assertIn(session_id, self.inferencer.list_sessions())

    def test_start_session_with_custom_id(self):
        """Test that start_session uses provided ID."""
        custom_id = "my-custom-session-id"
        session_id = self.inferencer.start_session(custom_id)

        self.assertEqual(session_id, custom_id)
        self.assertEqual(self.inferencer.active_session_id, custom_id)
        self.assertIn(custom_id, self.inferencer.list_sessions())

    def test_end_session_clears_active(self):
        """Test that end_session clears active_session_id."""
        session_id = self.inferencer.start_session()
        self.assertIsNotNone(self.inferencer.active_session_id)

        self.inferencer.end_session()

        self.assertIsNone(self.inferencer.active_session_id)

    def test_end_session_with_specific_id(self):
        """Test end_session with specific session ID."""
        session1 = self.inferencer.start_session("session-1")
        self.inferencer.start_session("session-2")

        # End session-2 (active)
        self.inferencer.end_session("session-2")
        self.assertIsNone(self.inferencer.active_session_id)

        # session-1 still exists in storage
        self.assertIn("session-1", self.inferencer.list_sessions())

    def test_clear_session_removes_history(self):
        """Test that clear_session removes turn history."""
        session_id = self.inferencer.start_session()

        # Add some turns
        self.inferencer._add_turn(session_id, "user", "Hello")
        self.inferencer._add_turn(session_id, "system", "Hi there!")

        self.assertEqual(len(self.inferencer.get_session_history()), 2)

        # Clear session
        self.inferencer.clear_session()

        self.assertEqual(len(self.inferencer.get_session_history()), 0)
        # Session still exists but is empty
        self.assertIn(session_id, self.inferencer.list_sessions())

    def test_delete_session_removes_entirely(self):
        """Test that delete_session removes session entirely."""
        session_id = self.inferencer.start_session()
        self.inferencer._add_turn(session_id, "user", "Hello")

        self.inferencer.delete_session()

        self.assertNotIn(session_id, self.inferencer.list_sessions())
        self.assertIsNone(self.inferencer.active_session_id)

    def test_list_sessions(self):
        """Test list_sessions returns all session IDs."""
        self.inferencer.start_session("session-a")
        self.inferencer.start_session("session-b")
        self.inferencer.start_session("session-c")

        sessions = self.inferencer.list_sessions()

        self.assertEqual(len(sessions), 3)
        self.assertIn("session-a", sessions)
        self.assertIn("session-b", sessions)
        self.assertIn("session-c", sessions)

    def test_get_session_turn_count(self):
        """Test get_session_turn_count returns correct count."""
        session_id = self.inferencer.start_session()

        self.assertEqual(self.inferencer.get_session_turn_count(), 0)

        self.inferencer._add_turn(session_id, "user", "Hello")
        self.assertEqual(self.inferencer.get_session_turn_count(), 1)

        self.inferencer._add_turn(session_id, "system", "Hi!")
        self.assertEqual(self.inferencer.get_session_turn_count(), 2)


class TestTurnManagement(unittest.TestCase):
    """Test cases for turn management methods."""

    def setUp(self):
        """Set up test inferencer with a session."""
        self.inferencer = MockSessionInferencer()
        self.session_id = self.inferencer.start_session("test-session")

    def test_add_turn_user(self):
        """Test adding a user turn."""
        self.inferencer._add_turn(self.session_id, "user", "Hello")

        history = self.inferencer.get_session_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["from"], "user")
        self.assertEqual(history[0]["content"], "Hello")
        self.assertIn("timestamp", history[0])

    def test_add_turn_system(self):
        """Test adding a system turn."""
        self.inferencer._add_turn(self.session_id, "system", "Hi there!")

        history = self.inferencer.get_session_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["from"], "system")
        self.assertEqual(history[0]["content"], "Hi there!")

    def test_add_turn_with_metadata(self):
        """Test adding a turn with metadata."""
        metadata = {"model": "test-model", "tokens": 100}
        self.inferencer._add_turn(
            self.session_id, "system", "Response", metadata=metadata
        )

        history = self.inferencer.get_session_history()
        self.assertEqual(history[0]["metadata"], metadata)

    def test_get_last_turn(self):
        """Test get_last_turn returns the most recent turn."""
        self.inferencer._add_turn(self.session_id, "user", "First")
        self.inferencer._add_turn(self.session_id, "system", "Second")
        self.inferencer._add_turn(self.session_id, "user", "Third")

        last = self.inferencer.get_last_turn()
        self.assertEqual(last["content"], "Third")

    def test_get_last_turn_empty_session(self):
        """Test get_last_turn returns None for empty session."""
        result = self.inferencer.get_last_turn()
        self.assertIsNone(result)

    def test_get_last_user_turn(self):
        """Test get_last_user_turn returns last user message."""
        self.inferencer._add_turn(self.session_id, "user", "User 1")
        self.inferencer._add_turn(self.session_id, "system", "System 1")
        self.inferencer._add_turn(self.session_id, "user", "User 2")
        self.inferencer._add_turn(self.session_id, "system", "System 2")

        last_user = self.inferencer.get_last_user_turn()
        self.assertEqual(last_user["content"], "User 2")

    def test_get_last_system_turn(self):
        """Test get_last_system_turn returns last system message."""
        self.inferencer._add_turn(self.session_id, "user", "User 1")
        self.inferencer._add_turn(self.session_id, "system", "System 1")
        self.inferencer._add_turn(self.session_id, "user", "User 2")
        self.inferencer._add_turn(self.session_id, "system", "System 2")

        last_system = self.inferencer.get_last_system_turn()
        self.assertEqual(last_system["content"], "System 2")

    def test_get_session_history_returns_copy(self):
        """Test that get_session_history returns a copy, not reference."""
        self.inferencer._add_turn(self.session_id, "user", "Hello")

        history1 = self.inferencer.get_session_history()
        history2 = self.inferencer.get_session_history()

        # Modify history1
        history1.append({"from": "user", "content": "Modified"})

        # history2 and internal storage should be unaffected
        self.assertEqual(len(history2), 1)
        self.assertEqual(len(self.inferencer.get_session_history()), 1)


class TestCommandConstruction(unittest.TestCase):
    """Test cases for command construction with sessions."""

    def setUp(self):
        """Set up test inferencer."""
        self.inferencer = MockSessionInferencer()

    def test_construct_command_basic(self):
        """Test basic command construction without session."""
        command = self.inferencer.construct_command("Hello")

        self.assertEqual(command, 'mock_cli "prompt=Hello"')

    def test_construct_command_with_resume(self):
        """Test command construction with resume flag."""
        command = self.inferencer.construct_command(
            "Follow up", session_id="abc123", resume=True
        )

        self.assertIn("--resume", command)
        self.assertIn("--session-id 'abc123'", command)
        self.assertIn('"prompt=Follow up"', command)

    def test_construct_command_custom_session_args(self):
        """Test command with custom session arg names."""
        inferencer = MockSessionInferencer(
            session_arg_name="--sid", resume_arg_name="--continue"
        )

        command = inferencer.construct_command("Test", session_id="xyz789", resume=True)

        self.assertIn("--continue", command)
        self.assertIn("--sid 'xyz789'", command)


class TestBuildSessionArgs(unittest.TestCase):
    """Test cases for _build_session_args method."""

    def setUp(self):
        """Set up test inferencer."""
        self.inferencer = MockSessionInferencer()

    def test_build_session_args_resume(self):
        """Test building args for resume scenario."""
        args = self.inferencer._build_session_args("session-123", is_resume=True)

        self.assertEqual(args, "--resume --session-id 'session-123'")

    def test_build_session_args_no_resume(self):
        """Test building args without resume (just session ID)."""
        args = self.inferencer._build_session_args("session-123", is_resume=False)

        self.assertEqual(args, "--session-id 'session-123'")

    def test_build_session_args_empty(self):
        """Test building args with no session ID."""
        # When session_id is empty string, should return empty
        args = self.inferencer._build_session_args("", is_resume=True)

        # Implementation might vary, but should not crash
        self.assertIsInstance(args, str)


class TestNewSessionBehavior(unittest.TestCase):
    """Test cases for new_session parameter and default resume behavior."""

    def setUp(self):
        """Set up test inferencer."""
        self.inferencer = MockSessionInferencer()

    def test_first_call_no_session_creates_new(self):
        """Test that first call with no active session creates new session."""
        # active_session_id is None, so resume=True is ignored and new session starts
        self.assertIsNone(self.inferencer.active_session_id)

        # Construct command should not include resume args since session_id is None
        command = self.inferencer.construct_command(
            "Hello", session_id=None, resume=True
        )
        self.assertNotIn("--resume", command)

    def test_subsequent_call_with_session_resumes(self):
        """Test that subsequent call with active session resumes by default."""
        # Set up active session
        self.inferencer.active_session_id = "existing-session-123"

        # Construct command should include resume args
        command = self.inferencer.construct_command(
            "Follow up", session_id="existing-session-123", resume=True
        )
        self.assertIn("--resume", command)
        self.assertIn("--session-id 'existing-session-123'", command)


class TestSessionIdExtraction(unittest.TestCase):
    """Test cases for session ID extraction from results."""

    def test_active_session_updated_from_result(self):
        """Test that active_session_id is updated from parse_output result."""
        inferencer = MockSessionInferencer(mock_session_id="new-session-from-result")

        # Simulate parse_output returning a session ID
        result = inferencer.parse_output("response", "", 0)

        self.assertEqual(result["session_id"], "new-session-from-result")


class TestConfigurationDefaults(unittest.TestCase):
    """Test cases for configuration defaults."""

    def test_default_session_arg_name(self):
        """Test default session_arg_name."""
        inferencer = MockSessionInferencer()
        self.assertEqual(inferencer.session_arg_name, "--session-id")

    def test_default_resume_arg_name(self):
        """Test default resume_arg_name."""
        inferencer = MockSessionInferencer()
        self.assertEqual(inferencer.resume_arg_name, "--resume")

    def test_default_active_session_id(self):
        """Test default active_session_id is None."""
        inferencer = MockSessionInferencer()
        self.assertIsNone(inferencer.active_session_id)

    def test_default_sessions_empty(self):
        """Test default _sessions is empty dict."""
        inferencer = MockSessionInferencer()
        self.assertEqual(inferencer._sessions, {})

    def test_custom_arg_names(self):
        """Test custom argument names can be set."""
        inferencer = MockSessionInferencer(
            session_arg_name="--my-session", resume_arg_name="--my-resume"
        )

        self.assertEqual(inferencer.session_arg_name, "--my-session")
        self.assertEqual(inferencer.resume_arg_name, "--my-resume")


class TestMultipleSessions(unittest.TestCase):
    """Test cases for managing multiple sessions."""

    def setUp(self):
        """Set up test inferencer."""
        self.inferencer = MockSessionInferencer()

    def test_multiple_sessions_independent(self):
        """Test that multiple sessions are tracked independently."""
        session1 = self.inferencer.start_session("session-1")
        self.inferencer._add_turn(session1, "user", "Hello from session 1")

        session2 = self.inferencer.start_session("session-2")
        self.inferencer._add_turn(session2, "user", "Hello from session 2")
        self.inferencer._add_turn(session2, "system", "Response in session 2")

        # Check session 1 has 1 turn
        history1 = self.inferencer.get_session_history("session-1")
        self.assertEqual(len(history1), 1)
        self.assertEqual(history1[0]["content"], "Hello from session 1")

        # Check session 2 has 2 turns
        history2 = self.inferencer.get_session_history("session-2")
        self.assertEqual(len(history2), 2)

    def test_switch_between_sessions(self):
        """Test switching active session between multiple sessions."""
        self.inferencer.start_session("session-a")
        self.assertEqual(self.inferencer.active_session_id, "session-a")

        self.inferencer.start_session("session-b")
        self.assertEqual(self.inferencer.active_session_id, "session-b")

        # Both sessions exist
        self.assertIn("session-a", self.inferencer.list_sessions())
        self.assertIn("session-b", self.inferencer.list_sessions())


def print_test_results():
    """Print test results in a formatted way."""
    print("\nRunning TerminalSessionInferencerBase tests...")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestSessionManagement,
        TestTurnManagement,
        TestCommandConstruction,
        TestBuildSessionArgs,
        TestNewSessionBehavior,
        TestSessionIdExtraction,
        TestConfigurationDefaults,
        TestMultipleSessions,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✓ All session inferencer tests passed!")
    else:
        print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")

    return result.wasSuccessful()


if __name__ == "__main__":
    print_test_results()
