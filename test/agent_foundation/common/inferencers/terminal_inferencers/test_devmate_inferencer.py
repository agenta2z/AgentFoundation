"""Unit tests for DevmateInferencer command construction and output parsing."""

import unittest

import resolve_path  # noqa: F401

from science_modeling_tools.common.inferencers.terminal_inferencers.devmate.devmate_inferencer import (
    DevmateInferencer,
)


class TestCommandConstruction(unittest.TestCase):
    """Test cases for DevmateInferencer.construct_command()."""

    def setUp(self):
        """Set up default inferencer."""
        self.inferencer = DevmateInferencer()

    def test_basic_prompt(self):
        """Test command construction with a simple prompt."""
        command = self.inferencer.construct_command("Hello")

        # Command should be a shell string with properly quoted arguments:
        # devmate run freeform "prompt=Hello" "model_name=claude-sonnet-4.5" "max_tokens=32768" --no-create-commit
        expected = 'devmate run freeform "prompt=Hello" "model_name=claude-sonnet-4.5" "max_tokens=32768" --no-create-commit'

        self.assertEqual(command, expected)

    def test_returns_string_not_list(self):
        """Test that command is returned as a string (for shell execution)."""
        command = self.inferencer.construct_command("Test")

        self.assertIsInstance(command, str)
        self.assertNotIsInstance(command, list)

    def test_prompt_with_spaces(self):
        """Test command construction with prompt containing spaces."""
        command = self.inferencer.construct_command("Hello World, how are you?")

        self.assertIn('"prompt=Hello World, how are you?"', command)
        self.assertIn("devmate run freeform", command)

    def test_prompt_with_quotes_escaped(self):
        """Test that double quotes in prompt are escaped."""
        command = self.inferencer.construct_command('Say "hello" to me')

        # Double quotes should be escaped
        self.assertIn('"prompt=Say \\"hello\\" to me"', command)

    def test_custom_model_name_override(self):
        """Test command construction with model_name override via kwargs."""
        command = self.inferencer.construct_command(
            "Test prompt", model_name="claude-3-opus"
        )

        self.assertIn('"model_name=claude-3-opus"', command)
        self.assertNotIn('"model_name=claude-sonnet-4.5"', command)

    def test_custom_model_name_in_constructor(self):
        """Test command construction with model_name set in constructor."""
        inferencer = DevmateInferencer(model_name="gpt-4")
        command = inferencer.construct_command("Test prompt")

        self.assertIn('"model_name=gpt-4"', command)

    def test_custom_max_tokens_override(self):
        """Test command construction with max_tokens override via kwargs."""
        command = self.inferencer.construct_command("Test prompt", max_tokens=16384)

        self.assertIn('"max_tokens=16384"', command)
        self.assertNotIn('"max_tokens=32768"', command)

    def test_custom_max_tokens_in_constructor(self):
        """Test command construction with max_tokens set in constructor."""
        inferencer = DevmateInferencer(max_tokens=8192)
        command = inferencer.construct_command("Test prompt")

        self.assertIn('"max_tokens=8192"', command)

    def test_with_context_files(self):
        """Test command construction with context_files."""
        inferencer = DevmateInferencer(
            context_files=["/path/to/file1.py", "/path/to/file2.py"]
        )
        command = inferencer.construct_command("Analyze these files")

        self.assertIn('"context_file=/path/to/file1.py"', command)
        self.assertIn('"context_file=/path/to/file2.py"', command)

    def test_context_files_override_via_kwargs(self):
        """Test command construction with context_files override via kwargs."""
        command = self.inferencer.construct_command(
            "Test prompt", context_files=["/new/file.py"]
        )

        self.assertIn('"context_file=/new/file.py"', command)

    def test_no_create_commit_true(self):
        """Test that --no-create-commit flag is included by default."""
        command = self.inferencer.construct_command("Test prompt")

        self.assertIn("--no-create-commit", command)
        self.assertTrue(command.endswith("--no-create-commit"))

    def test_no_create_commit_false(self):
        """Test command when no_create_commit is False."""
        inferencer = DevmateInferencer(no_create_commit=False)
        command = inferencer.construct_command("Test prompt")

        self.assertNotIn("--no-create-commit", command)

    def test_dict_input_with_prompt_key(self):
        """Test command construction with dict input containing 'prompt' key."""
        command = self.inferencer.construct_command({"prompt": "Dict prompt test"})

        self.assertIn('"prompt=Dict prompt test"', command)

    def test_command_order(self):
        """Test that command parts are in the expected order."""
        command = self.inferencer.construct_command("Test")

        # Verify order by checking positions
        devmate_pos = command.find("devmate run freeform")
        prompt_pos = command.find('"prompt=')
        model_pos = command.find('"model_name=')
        tokens_pos = command.find('"max_tokens=')
        flag_pos = command.find("--no-create-commit")

        self.assertEqual(devmate_pos, 0)  # devmate run freeform at start
        self.assertTrue(prompt_pos < model_pos)
        self.assertTrue(model_pos < tokens_pos)
        self.assertTrue(tokens_pos < flag_pos)  # flag at end

    def test_full_command_matches_bash_pattern(self):
        """Test that generated command matches expected bash pattern."""
        inferencer = DevmateInferencer(
            model_name="claude-sonnet-4.5",
            max_tokens=32768,
            no_create_commit=True,
        )
        command = inferencer.construct_command("Hello, how are you?")

        # Command should match bash pattern exactly:
        # devmate run freeform "prompt=$PROMPT" "model_name=$MODEL" "max_tokens=32768" --no-create-commit
        expected = 'devmate run freeform "prompt=Hello, how are you?" "model_name=claude-sonnet-4.5" "max_tokens=32768" --no-create-commit'

        self.assertEqual(command, expected)


class TestOutputParsing(unittest.TestCase):
    """Test cases for DevmateInferencer.parse_output()."""

    def setUp(self):
        """Set up default inferencer."""
        self.inferencer = DevmateInferencer()

    def test_parse_successful_output(self):
        """Test parsing successful stdout output."""
        stdout = "Hello! I'm doing well. How can I help you today?"
        stderr = ""
        return_code = 0

        result = self.inferencer.parse_output(stdout, stderr, return_code)

        self.assertTrue(result["success"])
        self.assertEqual(result["return_code"], 0)
        self.assertEqual(result["output"], stdout)
        self.assertEqual(result["raw_output"], stdout)
        self.assertEqual(result["stderr"], "")
        self.assertNotIn("error", result)

    def test_parse_output_strips_whitespace(self):
        """Test that output is stripped of leading/trailing whitespace."""
        stdout = "  Response with spaces  \n"
        stderr = "  Some warning  \n"

        result = self.inferencer.parse_output(stdout, stderr, 0)

        self.assertEqual(result["output"], "Response with spaces")
        self.assertEqual(result["stderr"], "Some warning")

    def test_parse_output_with_stderr(self):
        """Test parsing output when stderr is present."""
        stdout = "Some output"
        stderr = "Warning: something happened"

        result = self.inferencer.parse_output(stdout, stderr, 0)

        self.assertTrue(result["success"])
        self.assertEqual(result["stderr"], "Warning: something happened")

    def test_parse_output_with_non_zero_return_code(self):
        """Test parsing output with non-zero return code."""
        stdout = ""
        stderr = "Error: command failed"
        return_code = 1

        result = self.inferencer.parse_output(stdout, stderr, return_code)

        self.assertFalse(result["success"])
        self.assertEqual(result["return_code"], 1)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Error: command failed")

    def test_parse_output_error_without_stderr(self):
        """Test parsing failed output without stderr message."""
        stdout = ""
        stderr = ""
        return_code = 1

        result = self.inferencer.parse_output(stdout, stderr, return_code)

        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("return code 1", result["error"])

    def test_parse_output_empty_success(self):
        """Test parsing empty but successful output."""
        result = self.inferencer.parse_output("", "", 0)

        self.assertTrue(result["success"])
        self.assertEqual(result["output"], "")
        self.assertEqual(result["stderr"], "")

    def test_parse_output_multiline(self):
        """Test parsing multiline output."""
        stdout = "Line 1\nLine 2\nLine 3"

        result = self.inferencer.parse_output(stdout, "", 0)

        self.assertTrue(result["success"])
        self.assertEqual(result["output"], stdout)

    def test_parse_output_extracts_session_id(self):
        """Test that session ID is extracted from DevMate output."""
        stdout = """Starting Devmate server and session...
================================================================================
Session ID: 9db6c3e5-743e-4e19-8afb-3fef49ca17fb
Trajectory: https://www.internalfb.com/intern/devai/devmate/inspector/9db6c3e5-743e-4e19-8afb-3fef49ca17fb
================================================================================
Hello, this is the response."""

        result = self.inferencer.parse_output(stdout, "", 0)

        self.assertEqual(result["session_id"], "9db6c3e5-743e-4e19-8afb-3fef49ca17fb")

    def test_parse_output_extracts_trajectory_url(self):
        """Test that trajectory URL is extracted from DevMate output."""
        stdout = """Session ID: abc123
Trajectory: https://www.internalfb.com/intern/devai/devmate/inspector/abc123
The actual response."""

        result = self.inferencer.parse_output(stdout, "", 0)

        self.assertEqual(
            result["trajectory_url"],
            "https://www.internalfb.com/intern/devai/devmate/inspector/abc123",
        )

    def test_parse_output_cleans_session_headers(self):
        """Test that session headers are removed from output."""
        stdout = """Starting Devmate server and session...
Finished starting Devmate server and session in 6130 ms
Started session 9db6c3e5-743e-4e19-8afb-3fef49ca17fb

================================================================================
Session ID: 9db6c3e5-743e-4e19-8afb-3fef49ca17fb
Trajectory: https://example.com/test
Server logs available at: /path/to/logs
================================================================================

This is the actual response content.

It has multiple lines."""

        result = self.inferencer.parse_output(stdout, "", 0)

        # Output should not contain session info
        self.assertNotIn("Starting Devmate server", result["output"])
        self.assertNotIn("Session ID:", result["output"])
        self.assertNotIn("Trajectory:", result["output"])
        self.assertNotIn("================", result["output"])

        # Output should contain the actual response
        self.assertIn("This is the actual response content", result["output"])
        self.assertIn("It has multiple lines", result["output"])

    def test_parse_output_handles_duplicate_responses(self):
        """Test that duplicate responses separated by --- are deduplicated."""
        stdout = """Python is a programming language.

It is very popular.
---
Python is a programming language.

It is very popular."""

        result = self.inferencer.parse_output(stdout, "", 0)

        # Should only contain one copy of the response
        self.assertEqual(result["output"].count("Python is a programming language"), 1)

    def test_parse_output_preserves_raw_output(self):
        """Test that raw_output preserves the original content."""
        stdout = """Starting Devmate server...
================================================================================
Session ID: test-session
================================================================================
Cleaned response here."""

        result = self.inferencer.parse_output(stdout, "", 0)

        # raw_output should have original content
        self.assertIn("Starting Devmate server", result["raw_output"])
        self.assertIn("Session ID:", result["raw_output"])

        # output should be cleaned
        self.assertNotIn("Starting Devmate server", result["output"])
        self.assertIn("Cleaned response here", result["output"])


class TestGetResponseText(unittest.TestCase):
    """Test cases for DevmateInferencer.get_response_text()."""

    def setUp(self):
        """Set up default inferencer."""
        self.inferencer = DevmateInferencer()

    def test_get_response_text_success(self):
        """Test extracting response text from successful result."""
        result = {
            "success": True,
            "output": "The response text",
            "stderr": "",
            "return_code": 0,
        }

        text = self.inferencer.get_response_text(result)

        self.assertEqual(text, "The response text")

    def test_get_response_text_failure_with_error(self):
        """Test extracting response text from failed result with error."""
        result = {
            "success": False,
            "output": "",
            "stderr": "Some error",
            "return_code": 1,
            "error": "Command failed",
        }

        text = self.inferencer.get_response_text(result)

        self.assertEqual(text, "Command failed")

    def test_get_response_text_failure_without_error(self):
        """Test extracting response text from failed result without error key."""
        result = {
            "success": False,
            "output": "",
            "stderr": "",
            "return_code": 1,
        }

        text = self.inferencer.get_response_text(result)

        self.assertEqual(text, "Unknown error occurred")


class TestInferencerConfiguration(unittest.TestCase):
    """Test cases for DevmateInferencer configuration."""

    def test_default_configuration(self):
        """Test default inferencer configuration values."""
        inferencer = DevmateInferencer()

        self.assertEqual(inferencer.model_name, "claude-sonnet-4.5")
        self.assertEqual(inferencer.max_tokens, 32768)
        self.assertTrue(inferencer.no_create_commit)
        self.assertIsNone(inferencer.context_files)

    def test_custom_configuration(self):
        """Test custom inferencer configuration."""
        inferencer = DevmateInferencer(
            model_name="gpt-4",
            max_tokens=16384,
            no_create_commit=False,
            context_files=["/file1.py", "/file2.py"],
            timeout=600,
        )

        self.assertEqual(inferencer.model_name, "gpt-4")
        self.assertEqual(inferencer.max_tokens, 16384)
        self.assertFalse(inferencer.no_create_commit)
        self.assertEqual(inferencer.context_files, ["/file1.py", "/file2.py"])
        self.assertEqual(inferencer.timeout, 600)

    def test_repo_path_defaults_to_home_fbsource(self):
        """Test that repo_path defaults to ~/fbsource."""
        import os

        inferencer = DevmateInferencer()

        expected_path = os.path.expanduser("~/fbsource")
        self.assertEqual(inferencer.repo_path, expected_path)

    def test_repo_path_can_be_set_custom(self):
        """Test that repo_path can be set to a custom value."""
        inferencer = DevmateInferencer(repo_path="/custom/repo/path")

        self.assertEqual(inferencer.repo_path, "/custom/repo/path")

    def test_working_dir_defaults_to_repo_path(self):
        """Test that working_dir defaults to repo_path."""
        inferencer = DevmateInferencer(repo_path="/my/repo")

        self.assertEqual(inferencer.working_dir, "/my/repo")

    def test_pre_exec_scripts_includes_cd_to_repo(self):
        """Test that pre_exec_scripts automatically includes cd to repo."""
        inferencer = DevmateInferencer(repo_path="/my/repo")

        self.assertIsNotNone(inferencer.pre_exec_scripts)
        self.assertEqual(len(inferencer.pre_exec_scripts), 1)
        self.assertIn('cd "/my/repo"', inferencer.pre_exec_scripts[0])

    def test_custom_pre_exec_scripts_keeps_cd(self):
        """Test that custom pre_exec_scripts still includes cd."""
        inferencer = DevmateInferencer(
            repo_path="/my/repo", pre_exec_scripts=["echo setup"]
        )

        # CD should be prepended to existing scripts
        self.assertEqual(len(inferencer.pre_exec_scripts), 2)
        self.assertIn('cd "/my/repo"', inferencer.pre_exec_scripts[0])
        self.assertEqual(inferencer.pre_exec_scripts[1], "echo setup")

    def test_fail_on_pre_script_error_default_true(self):
        """Test that fail_on_pre_script_error defaults to True."""
        inferencer = DevmateInferencer()

        self.assertTrue(inferencer.fail_on_pre_script_error)


class TestSessionSupport(unittest.TestCase):
    """Test cases for DevmateInferencer session support."""

    def setUp(self):
        """Set up default inferencer."""
        self.inferencer = DevmateInferencer()

    def test_inherits_session_capabilities(self):
        """Test that DevmateInferencer has session management methods."""
        # Check session management methods exist
        self.assertTrue(hasattr(self.inferencer, "start_session"))
        self.assertTrue(hasattr(self.inferencer, "end_session"))
        self.assertTrue(hasattr(self.inferencer, "get_session_history"))
        self.assertTrue(hasattr(self.inferencer, "list_sessions"))

    def test_default_session_arg_names(self):
        """Test default session argument names for DevMate."""
        self.assertEqual(self.inferencer.session_arg_name, "--session-id")
        self.assertEqual(self.inferencer.resume_arg_name, "--resume")

    def test_build_session_args_resume(self):
        """Test _build_session_args for resume scenario."""
        args = self.inferencer._build_session_args("abc-123-def", is_resume=True)

        self.assertEqual(args, "--resume --session-id 'abc-123-def'")

    def test_build_session_args_no_resume(self):
        """Test _build_session_args without resume."""
        args = self.inferencer._build_session_args("abc-123-def", is_resume=False)

        self.assertEqual(args, "--session-id 'abc-123-def'")

    def test_construct_command_with_resume(self):
        """Test command construction with resume flag."""
        command = self.inferencer.construct_command(
            "Follow up question",
            session_id="9db6c3e5-743e-4e19-8afb-3fef49ca17fb",
            resume=True,
        )

        # Command should have resume args before freeform
        self.assertIn("--resume", command)
        self.assertIn("--session-id '9db6c3e5-743e-4e19-8afb-3fef49ca17fb'", command)
        self.assertIn("freeform", command)
        self.assertIn('"prompt=Follow up question"', command)

        # Verify order: resume args should come before freeform
        resume_pos = command.find("--resume")
        freeform_pos = command.find("freeform")
        self.assertTrue(resume_pos < freeform_pos)

    def test_construct_command_without_resume(self):
        """Test command construction without resume."""
        command = self.inferencer.construct_command("New question")

        self.assertNotIn("--resume", command)
        self.assertNotIn("--session-id", command)

    def test_resume_session_method_exists(self):
        """Test that convenience method resume_session exists."""
        self.assertTrue(hasattr(self.inferencer, "resume_session"))

    def test_new_session_method_exists(self):
        """Test that convenience method new_session exists."""
        self.assertTrue(hasattr(self.inferencer, "new_session"))

    def test_resume_session_raises_without_session(self):
        """Test that resume_session raises error without active session."""
        with self.assertRaises(ValueError) as context:
            self.inferencer.resume_session("Follow up")

        self.assertIn("No session_id provided", str(context.exception))


class TestCommandConstructionWithSession(unittest.TestCase):
    """Test cases for command construction with session parameters."""

    def setUp(self):
        """Set up default inferencer."""
        self.inferencer = DevmateInferencer()

    def test_full_resume_command_format(self):
        """Test that resume command matches expected devmate CLI format."""
        command = self.inferencer.construct_command(
            "What was my question?",
            session_id="9db6c3e5-743e-4e19-8afb-3fef49ca17fb",
            resume=True,
            model_name="claude-sonnet-4.5",
            max_tokens=32768,
        )

        # Expected format:
        # devmate run --resume --session-id 'uuid' freeform "prompt=..." "model_name=..." "max_tokens=..." --no-create-commit
        expected_parts = [
            "devmate run",
            "--resume",
            "--session-id '9db6c3e5-743e-4e19-8afb-3fef49ca17fb'",
            "freeform",
            '"prompt=What was my question?"',
            '"model_name=claude-sonnet-4.5"',
            '"max_tokens=32768"',
            "--no-create-commit",
        ]

        for part in expected_parts:
            self.assertIn(part, command)

    def test_command_with_session_id_only(self):
        """Test command with session_id but resume=False."""
        command = self.inferencer.construct_command(
            "Test", session_id="some-session", resume=False
        )

        # Should NOT include session args when resume is False
        self.assertNotIn("--resume", command)
        self.assertNotIn("--session-id", command)


def print_command_examples():
    """Print example commands for manual verification."""
    print("\n" + "=" * 60)
    print("DevMate Command Construction Examples")
    print("=" * 60)

    test_cases = [
        ("Basic prompt", {}, "Hello"),
        (
            "Custom model",
            {"model_name": "claude-3-opus", "max_tokens": 16384},
            "Test",
        ),
        (
            "With context files",
            {"context_files": ["/path/to/file.py"]},
            "Analyze code",
        ),
    ]

    for name, config, prompt in test_cases:
        inferencer = DevmateInferencer(**config)
        command = inferencer.construct_command(prompt)

        print(f"\nTest: {name}")
        print(f"Generated: {command}")
        print("-" * 60)


def print_test_results():
    """Print test results in a formatted way."""
    print("\nRunning DevMate command construction tests...")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestCommandConstruction,
        TestOutputParsing,
        TestGetResponseText,
        TestInferencerConfiguration,
        TestSessionSupport,
        TestCommandConstructionWithSession,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✓ All command construction tests passed!")
    else:
        print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")

    print_command_examples()

    return result.wasSuccessful()


if __name__ == "__main__":
    print_test_results()
