"""Tests for DevmateCliInferencer."""

import os
import tempfile
import unittest
from unittest.mock import patch

from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
    DevmateCliInferencer,
)


class DevmateCliInferencerInitTest(unittest.TestCase):
    """Test initialization of DevmateCliInferencer."""

    def test_default_initialization(self):
        """Test inferencer can be created with default values."""
        inferencer = DevmateCliInferencer()

        self.assertEqual(inferencer.model_name, "claude-sonnet-4.5")
        self.assertEqual(inferencer.max_tokens, 32768)
        self.assertTrue(inferencer.no_create_commit)
        self.assertIsNone(inferencer.context_files)
        self.assertFalse(inferencer.headless)
        self.assertFalse(inferencer.dump_output)
        self.assertIsNone(inferencer.timeout_percent)
        self.assertEqual(inferencer.config_name, "freeform")
        self.assertIsNone(inferencer.privacy_type)
        self.assertIsNone(inferencer.extra_cli_args)

    def test_custom_initialization(self):
        """Test inferencer with custom parameters."""
        inferencer = DevmateCliInferencer(
            repo_path="/path/to/repo",
            model_name="claude-3-opus",
            max_tokens=16384,
            no_create_commit=False,
            context_files=["/path/to/file.py"],
            headless=True,
            dump_output=True,
            timeout_percent=80,
            config_name="custom_config",
            privacy_type="PUBLIC",
            extra_cli_args=["--verbose"],
        )

        self.assertEqual(inferencer.repo_path, "/path/to/repo")
        self.assertEqual(inferencer.model_name, "claude-3-opus")
        self.assertEqual(inferencer.max_tokens, 16384)
        self.assertFalse(inferencer.no_create_commit)
        self.assertEqual(inferencer.context_files, ["/path/to/file.py"])
        self.assertTrue(inferencer.headless)
        self.assertTrue(inferencer.dump_output)
        self.assertEqual(inferencer.timeout_percent, 80)
        self.assertEqual(inferencer.config_name, "custom_config")
        self.assertEqual(inferencer.privacy_type, "PUBLIC")
        self.assertEqual(inferencer.extra_cli_args, ["--verbose"])

    def test_default_repo_path(self):
        """Test that repo_path defaults to ~/fbsource."""
        inferencer = DevmateCliInferencer()
        expected_path = os.path.expanduser("~/fbsource")
        self.assertEqual(inferencer.repo_path, expected_path)


class DevmateCliInferencerConstructCommandTest(unittest.TestCase):
    """Test construct_command method."""

    def test_construct_command_basic(self):
        """Test basic command construction."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            no_create_commit=True,
        )

        command = inferencer.construct_command("Hello world")

        self.assertIn("devmate run", command)
        self.assertIn("freeform", command)
        self.assertIn('"prompt=Hello world"', command)
        self.assertIn('"model_name=claude-sonnet-4.5"', command)
        self.assertIn('"max_tokens=32768"', command)
        self.assertIn("--no-create-commit", command)

    def test_construct_command_with_headless(self):
        """Test that --headless flag is added when headless=True."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            headless=True,
            no_create_commit=False,
        )

        command = inferencer.construct_command("Test prompt")

        self.assertIn("--headless", command)
        self.assertNotIn("--no-create-commit", command)

    def test_construct_command_with_dump_output(self):
        """Test that temp file is created and flag added when dump_output=True."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            dump_output=True,
            no_create_commit=False,
        )

        command = inferencer.construct_command("Test prompt")

        self.assertIn("--dump-final-structs-to-file", command)
        self.assertIsNotNone(inferencer._output_file)
        self.assertTrue(inferencer._output_file.endswith(".json"))
        self.assertIn("devmate_output_", inferencer._output_file)

        # Cleanup
        if inferencer._output_file and os.path.exists(inferencer._output_file):
            os.remove(inferencer._output_file)

    def test_construct_command_with_session_resume(self):
        """Test command construction with session resume."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            no_create_commit=False,
        )

        command = inferencer.construct_command(
            "Follow up question",
            session_id="abc123-def456",
            resume=True,
        )

        self.assertIn("--resume", command)
        self.assertIn("--session-id 'abc123-def456'", command)
        # Session args should come before config name
        resume_pos = command.find("--resume")
        freeform_pos = command.find("freeform")
        self.assertLess(resume_pos, freeform_pos)

    def test_construct_command_shell_escaping(self):
        """Test that special characters are properly escaped."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            no_create_commit=False,
        )

        # Test double quotes escaping
        command = inferencer.construct_command('Say "hello"')
        self.assertIn('\\"hello\\"', command)

        # Test dollar sign escaping (variable expansion)
        command = inferencer.construct_command("Print $HOME")
        self.assertIn("\\$HOME", command)

        # Test backtick escaping (command substitution)
        command = inferencer.construct_command("Run `ls`")
        self.assertIn("\\`ls\\`", command)

        # Test backslash escaping (must be escaped first)
        command = inferencer.construct_command("Path: C:\\Users")
        self.assertIn("C:\\\\Users", command)

    def test_construct_command_with_timeout_percent(self):
        """Test that timeout percentage is added."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            timeout_percent=75,
            no_create_commit=False,
        )

        command = inferencer.construct_command("Test")

        self.assertIn("--sandcastle-timeout-percent 75", command)

    def test_construct_command_with_timeout_percent_zero(self):
        """Test that timeout_percent=0 is handled correctly (0 is valid)."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            timeout_percent=0,
            no_create_commit=False,
        )

        command = inferencer.construct_command("Test")

        self.assertIn("--sandcastle-timeout-percent 0", command)

    def test_construct_command_with_privacy_type(self):
        """Test that privacy type is added."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            privacy_type="PRIVATE",
            no_create_commit=False,
        )

        command = inferencer.construct_command("Test")

        self.assertIn("--privacy-type PRIVATE", command)

    def test_construct_command_with_extra_cli_args(self):
        """Test that extra CLI args are appended."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            extra_cli_args=["--verbose", "--debug"],
            no_create_commit=False,
        )

        command = inferencer.construct_command("Test")

        self.assertIn("--verbose", command)
        self.assertIn("--debug", command)

    def test_construct_command_with_context_files(self):
        """Test that context files are added."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            context_files=["/path/to/file1.py", "/path/to/file2.py"],
            no_create_commit=False,
        )

        command = inferencer.construct_command("Test")

        self.assertIn('"context_file=/path/to/file1.py"', command)
        self.assertIn('"context_file=/path/to/file2.py"', command)

    def test_construct_command_custom_config_name(self):
        """Test that custom config name is used."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            config_name="my_custom_config",
            no_create_commit=False,
        )

        command = inferencer.construct_command("Test")

        self.assertIn("my_custom_config", command)
        self.assertNotIn("freeform", command)


class DevmateCliInferencerParseOutputTest(unittest.TestCase):
    """Test parse_output method."""

    def test_parse_output_from_stdout(self):
        """Test parsing output from stdout."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")

        stdout = """Starting Devmate server...
================
Session ID: abc123-def456
Trajectory: https://www.internalfb.com/intern/devai/devmate/inspector/abc123-def456
================
This is the actual response content.
It spans multiple lines.
================
Finished session abc123-def456
"""
        result = inferencer.parse_output(stdout, "", 0)

        self.assertTrue(result["success"])
        self.assertEqual(result["return_code"], 0)
        self.assertEqual(result["session_id"], "abc123-def456")
        self.assertIn("actual response content", result["output"])
        self.assertNotIn("Starting Devmate server", result["output"])
        self.assertNotIn("Finished session", result["output"])

    def test_parse_output_from_dump_file(self):
        """Test parsing output from dump file when dump_output=True."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            dump_output=True,
        )

        # Create a mock dump file
        fd, dump_file = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(
                    '{"final_response": "Response from dump", "session_id": "dump-session-123"}'
                )

            inferencer._output_file = dump_file

            result = inferencer.parse_output("stdout content", "", 0)

            self.assertTrue(result["success"])
            self.assertEqual(result["output"], "Response from dump")
            self.assertEqual(result["session_id"], "dump-session-123")
            self.assertIn("dump_data", result)
            self.assertIn("trajectory_url", result)
            self.assertIn("dump-session-123", result["trajectory_url"])

        finally:
            # Cleanup should have happened in parse_output
            if os.path.exists(dump_file):
                os.remove(dump_file)

    def test_parse_output_fallback_on_dump_error(self):
        """Test that parsing falls back to stdout when dump file has errors."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            dump_output=True,
        )

        # Create an invalid dump file
        fd, dump_file = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("not valid json {{{")

            inferencer._output_file = dump_file

            stdout = "Session ID: fallback-123\nFallback response"
            result = inferencer.parse_output(stdout, "", 0)

            self.assertTrue(result["success"])
            self.assertEqual(result["session_id"], "fallback-123")
            self.assertIn("Fallback response", result["output"])
            self.assertNotIn("dump_data", result)

        finally:
            if os.path.exists(dump_file):
                os.remove(dump_file)

    def test_parse_output_with_error(self):
        """Test parsing output when command fails."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")

        result = inferencer.parse_output("", "Error message", 1)

        self.assertFalse(result["success"])
        self.assertEqual(result["return_code"], 1)
        self.assertEqual(result["error"], "Error message")


class DevmateCliInferencerCleanupTest(unittest.TestCase):
    """Test file cleanup methods."""

    def test_cleanup_output_file(self):
        """Test that temp file is properly cleaned up."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")

        # Create a temp file
        fd, temp_file = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        inferencer._output_file = temp_file

        self.assertTrue(os.path.exists(temp_file))

        inferencer._cleanup_output_file()

        self.assertFalse(os.path.exists(temp_file))
        self.assertIsNone(inferencer._output_file)

    def test_cleanup_output_file_nonexistent(self):
        """Test cleanup when file doesn't exist (should not raise)."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")
        inferencer._output_file = "/nonexistent/path/file.json"

        # Should not raise
        inferencer._cleanup_output_file()
        self.assertIsNone(inferencer._output_file)


class DevmateCliInferencerStreamingTest(unittest.TestCase):
    """Test streaming methods."""

    def test_streaming_disables_dump_output(self):
        """Test that dump_output is disabled during streaming."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            dump_output=True,
        )

        # Mock the parent's _infer_streaming to avoid actual execution
        with patch.object(inferencer, "_infer_streaming") as mock_streaming:
            mock_streaming.return_value = iter(["line1\n", "line2\n"])

            # Consume the generator
            list(inferencer.infer_streaming("test prompt"))

            # dump_output should be temporarily disabled during streaming
            # and restored after
            self.assertTrue(inferencer.dump_output)

    def test_streaming_restores_dump_output_on_error(self):
        """Test that dump_output is restored even if streaming raises."""
        inferencer = DevmateCliInferencer(
            repo_path="/test/repo",
            dump_output=True,
        )

        with patch.object(inferencer, "_infer_streaming") as mock_streaming:
            mock_streaming.side_effect = RuntimeError("Test error")

            # The error should propagate but dump_output should be restored
            with self.assertRaises(RuntimeError):
                list(inferencer.infer_streaming("test prompt"))

            self.assertTrue(inferencer.dump_output)


class DevmateCliInferencerSessionTest(unittest.TestCase):
    """Test session management methods."""

    def test_resume_session_with_active_session(self):
        """Test resuming with active session."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")
        inferencer.active_session_id = "active-session-123"

        with patch.object(inferencer, "infer") as mock_infer:
            mock_infer.return_value = {"success": True}

            inferencer.resume_session("follow up")

            mock_infer.assert_called_once_with(
                "follow up",
                session_id="active-session-123",
                resume=True,
            )

    def test_resume_session_without_session_raises(self):
        """Test that resume_session raises when no session available."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")
        inferencer.active_session_id = None

        with self.assertRaises(ValueError) as context:
            inferencer.resume_session("follow up")

        self.assertIn("No session_id provided", str(context.exception))

    def test_new_session_clears_active_session(self):
        """Test that new_session clears the active session."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")
        inferencer.active_session_id = "old-session-123"

        with patch.object(inferencer, "infer") as mock_infer:
            mock_infer.return_value = {"success": True}

            inferencer.new_session("new prompt")

            self.assertIsNone(inferencer.active_session_id)
            mock_infer.assert_called_once_with("new prompt", resume=False)


class DevmateCliInferencerHelperMethodsTest(unittest.TestCase):
    """Test helper methods."""

    def test_extract_session_id(self):
        """Test session ID extraction from output."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")

        output = "Session ID: abc123-def456-789"
        session_id = inferencer._extract_session_id(output)

        self.assertEqual(session_id, "abc123-def456-789")

    def test_extract_session_id_not_found(self):
        """Test session ID extraction when not present."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")

        output = "No session info here"
        session_id = inferencer._extract_session_id(output)

        self.assertIsNone(session_id)

    def test_extract_trajectory_url(self):
        """Test trajectory URL extraction."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")

        output = "Trajectory: https://www.internalfb.com/intern/devai/devmate/inspector/abc123"
        url = inferencer._extract_trajectory_url(output)

        self.assertEqual(
            url,
            "https://www.internalfb.com/intern/devai/devmate/inspector/abc123",
        )

    def test_is_session_info_line(self):
        """Test session info line detection."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")

        self.assertTrue(inferencer._is_session_info_line("Starting Devmate server..."))
        self.assertTrue(inferencer._is_session_info_line("Session ID: abc123"))
        self.assertTrue(inferencer._is_session_info_line("================"))
        self.assertTrue(inferencer._is_session_info_line("Finished session abc123"))

        self.assertFalse(inferencer._is_session_info_line(""))
        self.assertFalse(inferencer._is_session_info_line("This is content"))

    def test_get_response_text_success(self):
        """Test get_response_text on successful result."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")

        result = {"success": True, "output": "The response text"}
        text = inferencer.get_response_text(result)

        self.assertEqual(text, "The response text")

    def test_get_response_text_failure(self):
        """Test get_response_text on failed result."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")

        result = {"success": False, "error": "Something went wrong"}
        text = inferencer.get_response_text(result)

        self.assertEqual(text, "Something went wrong")


class DevmateCliInferencerDumpDataExtractionTest(unittest.TestCase):
    """Test dump data extraction methods."""

    def test_extract_output_from_dump_with_final_response(self):
        """Test extracting output when final_response is present."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")

        dump_data = {"final_response": "This is the final response", "other": "data"}
        output = inferencer._extract_output_from_dump(dump_data)

        self.assertEqual(output, "This is the final response")

    def test_extract_output_from_dump_without_final_response(self):
        """Test extracting output when final_response is not present."""
        inferencer = DevmateCliInferencer(repo_path="/test/repo")

        dump_data = {"some_key": "some_value", "other": 123}
        output = inferencer._extract_output_from_dump(dump_data)

        # Should fallback to JSON stringification
        self.assertIn("some_key", output)
        self.assertIn("some_value", output)


if __name__ == "__main__":
    unittest.main()
