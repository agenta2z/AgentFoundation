"""Unit tests for TerminalInferencerBase class."""

import os
import tempfile
import unittest
from typing import Any, Dict, List

import resolve_path  # noqa: F401

from attr import attrib, attrs

from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_base import (
    TerminalInferencerBase,
)


@attrs
class EchoInferencer(TerminalInferencerBase):
    """
    Test implementation that runs echo command.

    This inferencer simply echoes back the input text using the shell echo command.
    Used for testing stdout capture functionality.
    """

    def construct_command(self, inference_input: Any, **kwargs) -> List[str]:
        """Construct an echo command with the given input."""
        return ["echo", str(inference_input)]

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> Dict[str, Any]:
        """Parse echo output into a result dictionary."""
        return {
            "output": stdout.strip(),
            "stderr": stderr.strip() if stderr else "",
            "return_code": return_code,
            "success": return_code == 0,
        }


@attrs
class FileOutputInferencer(TerminalInferencerBase):
    """
    Test implementation that runs a command and saves output to a file.

    This inferencer is used for testing file output functionality.
    """

    def construct_command(self, inference_input: Any, **kwargs) -> List[str]:
        """Construct command based on input."""
        if isinstance(inference_input, dict):
            cmd = inference_input.get("command", "echo")
            args = inference_input.get("args", [])
            return [cmd] + args
        return ["echo", str(inference_input)]

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> Dict[str, Any]:
        """Parse command output into a result dictionary."""
        return {
            "output": stdout.strip(),
            "stderr": stderr.strip() if stderr else "",
            "return_code": return_code,
            "success": return_code == 0,
        }


@attrs
class SleepInferencer(TerminalInferencerBase):
    """
    Test implementation that runs sleep command.

    Used for testing timeout functionality.
    """

    sleep_seconds: int = attrib(default=5)

    def construct_command(self, inference_input: Any, **kwargs) -> List[str]:
        """Construct a sleep command."""
        seconds = kwargs.get("seconds", self.sleep_seconds)
        return ["sleep", str(seconds)]

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> Dict[str, Any]:
        """Parse sleep output."""
        return {
            "output": stdout.strip() if stdout else "",
            "stderr": stderr.strip() if stderr else "",
            "return_code": return_code,
            "success": return_code == 0,
        }


@attrs
class CatInferencer(TerminalInferencerBase):
    """
    Test implementation that reads file contents using cat.

    Used for testing file reading functionality.
    """

    def construct_command(self, inference_input: Any, **kwargs) -> List[str]:
        """Construct a cat command to read the specified file."""
        return ["cat", str(inference_input)]

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> Dict[str, Any]:
        """Parse cat output."""
        return {
            "output": stdout.strip() if stdout else "",
            "stderr": stderr.strip() if stderr else "",
            "return_code": return_code,
            "success": return_code == 0,
        }


@attrs
class FailingInferencer(TerminalInferencerBase):
    """
    Test implementation that runs a command that fails.

    Used for testing error handling.
    """

    exit_code: int = attrib(default=1)

    def construct_command(self, inference_input: Any, **kwargs) -> List[str]:
        """Construct a failing command."""
        code = kwargs.get("exit_code", self.exit_code)
        return ["bash", "-c", f"exit {code}"]

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> Dict[str, Any]:
        """Parse output from failed command."""
        result = {
            "output": stdout.strip() if stdout else "",
            "stderr": stderr.strip() if stderr else "",
            "return_code": return_code,
            "success": return_code == 0,
        }
        if return_code != 0:
            result["error"] = f"Command failed with exit code {return_code}"
        return result


class TestEchoInferencer(unittest.TestCase):
    """Test cases for stdout output using EchoInferencer."""

    def setUp(self):
        """Set up test inferencer."""
        self.inferencer = EchoInferencer()

    def test_echo_hello_world(self):
        """Test echo 'Hello World' returns expected output."""
        result = self.inferencer.infer("Hello World")

        self.assertTrue(result["success"])
        self.assertEqual(result["return_code"], 0)
        self.assertEqual(result["output"], "Hello World")
        self.assertEqual(result["stderr"], "")

    def test_echo_multiline(self):
        """Test echo with special characters."""
        test_input = "Line 1\nLine 2"
        result = self.inferencer.infer(test_input)

        self.assertTrue(result["success"])
        self.assertIn("Line 1", result["output"])

    def test_echo_empty_string(self):
        """Test echo with empty string."""
        result = self.inferencer.infer("")

        self.assertTrue(result["success"])
        self.assertEqual(result["return_code"], 0)

    def test_echo_special_characters(self):
        """Test echo with special characters."""
        result = self.inferencer.infer("Hello $USER")

        self.assertTrue(result["success"])


class TestFileOutputInferencer(unittest.TestCase):
    """Test cases for file output functionality."""

    def setUp(self):
        """Set up test inferencer with temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.temp_dir, "output.txt")
        self.inferencer = FileOutputInferencer(output_file=self.output_file)

    def tearDown(self):
        """Clean up temp files."""
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_output_saved_to_specified_file(self):
        """Test output is saved to the specified file path."""
        result = self.inferencer.infer("Test output content")

        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(self.output_file))

        with open(self.output_file, "r") as f:
            content = f.read()
        # echo adds a trailing newline, so strip for comparison
        self.assertEqual(content.strip(), "Test output content")

    def test_output_saved_to_temp_file(self):
        """Test output is saved to temp file when no path specified."""
        inferencer = FileOutputInferencer()
        result = inferencer.infer("Temp file content", save_output=True)

        self.assertTrue(result["success"])

    def test_file_content_matches_stdout(self):
        """Test file content matches stdout output."""
        test_content = "File content test"
        result = self.inferencer.infer(test_content)

        self.assertTrue(result["success"])
        self.assertEqual(result["output"], test_content)

        if os.path.exists(self.output_file):
            with open(self.output_file, "r") as f:
                file_content = f.read()
            # echo adds a trailing newline, so strip for comparison
            self.assertEqual(file_content.strip(), test_content)


class TestCatInferencer(unittest.TestCase):
    """Test cases for reading file contents."""

    def setUp(self):
        """Set up test file."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        )
        self.temp_file.write("Test file contents\nLine 2\nLine 3")
        self.temp_file.close()
        self.inferencer = CatInferencer()

    def tearDown(self):
        """Clean up temp file."""
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_cat_reads_file_content(self):
        """Test cat command reads file content correctly."""
        result = self.inferencer.infer(self.temp_file.name)

        self.assertTrue(result["success"])
        self.assertEqual(result["return_code"], 0)
        self.assertIn("Test file contents", result["output"])
        self.assertIn("Line 2", result["output"])
        self.assertIn("Line 3", result["output"])

    def test_cat_nonexistent_file(self):
        """Test cat command with nonexistent file."""
        result = self.inferencer.infer("/nonexistent/file/path.txt")

        self.assertFalse(result["success"])
        self.assertNotEqual(result["return_code"], 0)
        self.assertIn("No such file", result["stderr"])


class TestTimeoutHandling(unittest.TestCase):
    """Test cases for command timeout behavior."""

    def test_command_timeout(self):
        """Test that command times out after specified duration."""
        inferencer = SleepInferencer(timeout=1)
        result = inferencer.infer("sleep", seconds=10)

        self.assertFalse(result["success"])
        self.assertEqual(result["return_code"], -1)

    def test_command_completes_within_timeout(self):
        """Test that command completes if within timeout."""
        inferencer = SleepInferencer(timeout=5)
        result = inferencer.infer("sleep", seconds=1)

        self.assertTrue(result["success"])
        self.assertEqual(result["return_code"], 0)


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling."""

    def test_non_zero_exit_code(self):
        """Test handling of non-zero exit codes."""
        inferencer = FailingInferencer(exit_code=1)
        result = inferencer.infer("test")

        self.assertFalse(result["success"])
        self.assertEqual(result["return_code"], 1)
        self.assertIn("error", result)

    def test_various_exit_codes(self):
        """Test handling of various exit codes."""
        for exit_code in [1, 2, 127, 255]:
            inferencer = FailingInferencer()
            result = inferencer.infer("test", exit_code=exit_code)

            self.assertFalse(result["success"])
            self.assertEqual(result["return_code"], exit_code)

    def test_command_not_found(self):
        """Test handling of command not found error."""
        inferencer = EchoInferencer()
        result = inferencer._execute_command(["nonexistent_command_xyz"])

        self.assertFalse(result["success"])


class TestExecuteCommandHelper(unittest.TestCase):
    """Test _execute_command() helper method directly."""

    def setUp(self):
        """Set up test inferencer."""
        self.inferencer = EchoInferencer()

    def test_execute_command_returns_dict(self):
        """Test that _execute_command returns correct dict structure."""
        result = self.inferencer._execute_command(["echo", "test"])

        self.assertIsInstance(result, dict)
        self.assertIn("stdout", result)
        self.assertIn("stderr", result)
        self.assertIn("return_code", result)
        self.assertIn("success", result)
        self.assertIn("command", result)

    def test_execute_command_stdout(self):
        """Test stdout is captured correctly."""
        result = self.inferencer._execute_command(["echo", "Hello"])

        self.assertEqual(result["stdout"].strip(), "Hello")

    def test_execute_command_stderr(self):
        """Test stderr is captured correctly."""
        result = self.inferencer._execute_command(["bash", "-c", "echo error >&2"])

        self.assertIn("error", result["stderr"])

    def test_execute_command_return_code(self):
        """Test return code is captured correctly."""
        result = self.inferencer._execute_command(["bash", "-c", "exit 42"])

        self.assertEqual(result["return_code"], 42)
        self.assertFalse(result["success"])

    def test_execute_command_success_flag(self):
        """Test success flag is set correctly."""
        success_result = self.inferencer._execute_command(["echo", "ok"])
        self.assertTrue(success_result["success"])

        fail_result = self.inferencer._execute_command(["bash", "-c", "exit 1"])
        self.assertFalse(fail_result["success"])

    def test_execute_command_includes_command_in_result(self):
        """Test that the executed command is included in result."""
        command = ["echo", "test"]
        result = self.inferencer._execute_command(command)

        self.assertEqual(result["command"], command)


class TestWorkingDirectory(unittest.TestCase):
    """Test working directory functionality."""

    def test_default_working_dir(self):
        """Test default working directory is current directory."""
        inferencer = EchoInferencer()
        self.assertEqual(inferencer.working_dir, os.getcwd())

    def test_custom_working_dir(self):
        """Test custom working directory is used."""
        temp_dir = tempfile.mkdtemp()
        try:
            inferencer = EchoInferencer(working_dir=temp_dir)
            self.assertEqual(inferencer.working_dir, temp_dir)

            result = inferencer._execute_command(["pwd"])
            self.assertEqual(result["stdout"].strip(), temp_dir)
        finally:
            os.rmdir(temp_dir)


class TestEnvironmentVariables(unittest.TestCase):
    """Test environment variable functionality."""

    def test_custom_env_vars(self):
        """Test custom environment variables are passed to command."""
        inferencer = EchoInferencer(env_vars={"TEST_VAR": "test_value"})

        result = inferencer._execute_command(["bash", "-c", "echo $TEST_VAR"])

        self.assertEqual(result["stdout"].strip(), "test_value")

    def test_env_vars_dont_persist(self):
        """Test that environment variables don't affect other commands."""
        inferencer = EchoInferencer(env_vars={"UNIQUE_VAR": "unique_value"})
        result = inferencer._execute_command(["bash", "-c", "echo $UNIQUE_VAR"])
        self.assertEqual(result["stdout"].strip(), "unique_value")


class TestSaveOutput(unittest.TestCase):
    """Test output saving functionality."""

    def setUp(self):
        """Set up temp directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_output_to_specified_file(self):
        """Test _save_output saves to specified file."""
        inferencer = EchoInferencer()
        filepath = os.path.join(self.temp_dir, "test_output.txt")

        saved_path = inferencer._save_output("Test content", filepath)

        self.assertEqual(saved_path, filepath)
        with open(filepath, "r") as f:
            self.assertEqual(f.read(), "Test content")

    def test_save_output_creates_temp_file(self):
        """Test _save_output creates temp file when no path specified."""
        inferencer = EchoInferencer()

        saved_path = inferencer._save_output("Temp content")

        self.assertTrue(os.path.exists(saved_path))
        with open(saved_path, "r") as f:
            self.assertEqual(f.read(), "Temp content")

        os.remove(saved_path)


class TestPreExecutionScripts(unittest.TestCase):
    """Test pre-execution script functionality."""

    def setUp(self):
        """Set up temp directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pre_script_runs_before_command(self):
        """Test that pre-execution scripts run before the main command."""
        # Create a file in pre-script, then read it in main command
        test_file = os.path.join(self.temp_dir, "pre_created.txt")
        pre_script = f'echo "pre-created" > "{test_file}"'

        inferencer = CatInferencer(pre_exec_scripts=[pre_script])
        result = inferencer.infer(test_file)

        self.assertTrue(result["success"])
        self.assertIn("pre-created", result["output"])

    def test_pre_script_failure_aborts_command(self):
        """Test that failing pre-script aborts the main command."""
        inferencer = EchoInferencer(
            pre_exec_scripts=["exit 1"], fail_on_pre_script_error=True
        )
        result = inferencer.infer("should not run")

        self.assertFalse(result["success"])
        self.assertIn("Pre-execution script failed", result["stderr"])

    def test_pre_script_failure_continues_when_configured(self):
        """Test that failing pre-script continues when fail_on_pre_script_error=False."""
        inferencer = EchoInferencer(
            pre_exec_scripts=["exit 1"], fail_on_pre_script_error=False
        )
        result = inferencer.infer("Hello World")

        # Main command should still run despite pre-script failure
        self.assertTrue(result["success"])
        self.assertEqual(result["output"], "Hello World")

    def test_multiple_pre_scripts_run_sequentially(self):
        """Test that multiple pre-scripts run in order."""
        test_file = os.path.join(self.temp_dir, "sequential.txt")
        pre_scripts = [
            f'echo "first" > "{test_file}"',
            f'echo "second" >> "{test_file}"',
            f'echo "third" >> "{test_file}"',
        ]

        inferencer = CatInferencer(pre_exec_scripts=pre_scripts)
        result = inferencer.infer(test_file)

        self.assertTrue(result["success"])
        self.assertIn("first", result["output"])
        self.assertIn("second", result["output"])
        self.assertIn("third", result["output"])

    def test_pre_script_cd_changes_directory(self):
        """Test that cd in pre-script works (like devmate use case)."""
        # Create a test file in temp directory
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Hello from temp dir")

        # Pre-script to cd to temp directory
        cd_script = f'cd "{self.temp_dir}" || exit 1'

        # Create inferencer with pre-script
        inferencer = EchoInferencer(pre_exec_scripts=[cd_script])
        result = inferencer.infer("test message")

        self.assertTrue(result["success"])

    def test_pre_scripts_override_via_kwargs(self):
        """Test that pre_exec_scripts can be overridden via kwargs."""
        test_file = os.path.join(self.temp_dir, "override.txt")

        # Constructor has one pre-script
        inferencer = CatInferencer(
            pre_exec_scripts=[f'echo "original" > "{test_file}"']
        )

        # Override via kwargs
        result = inferencer.infer(
            test_file, pre_exec_scripts=[f'echo "overridden" > "{test_file}"']
        )

        self.assertTrue(result["success"])
        self.assertIn("overridden", result["output"])


class TestPostExecutionScripts(unittest.TestCase):
    """Test post-execution script functionality."""

    def setUp(self):
        """Set up temp directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_post_script_runs_after_command(self):
        """Test that post-execution scripts run after the main command."""
        test_file = os.path.join(self.temp_dir, "post_test.txt")
        post_script = f'echo "post-ran" > "{test_file}"'

        inferencer = EchoInferencer(post_exec_scripts=[post_script])
        result = inferencer.infer("Hello")

        self.assertTrue(result["success"])
        # Verify post-script created the file
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, "r") as f:
            self.assertIn("post-ran", f.read())

    def test_post_script_failure_doesnt_fail_by_default(self):
        """Test that failing post-script doesn't fail result by default."""
        inferencer = EchoInferencer(
            post_exec_scripts=["exit 1"], fail_on_post_script_error=False
        )
        result = inferencer.infer("Hello")

        # Main command succeeded, post-script failure doesn't affect result
        self.assertTrue(result["success"])
        self.assertEqual(result["output"], "Hello")

    def test_post_script_failure_fails_when_configured(self):
        """Test that failing post-script fails when fail_on_post_script_error=True."""
        inferencer = EchoInferencer(
            post_exec_scripts=["exit 1"], fail_on_post_script_error=True
        )
        result = inferencer.infer("Hello")

        # Main command succeeded but post-script failed
        self.assertFalse(result["success"])
        self.assertIn("Post-execution script failed", result["stderr"])

    def test_multiple_post_scripts_run_sequentially(self):
        """Test that multiple post-scripts run in order."""
        test_file = os.path.join(self.temp_dir, "post_seq.txt")
        post_scripts = [
            f'echo "post1" > "{test_file}"',
            f'echo "post2" >> "{test_file}"',
        ]

        inferencer = EchoInferencer(post_exec_scripts=post_scripts)
        result = inferencer.infer("Hello")

        self.assertTrue(result["success"])
        with open(test_file, "r") as f:
            content = f.read()
        self.assertIn("post1", content)
        self.assertIn("post2", content)


class TestExecuteScriptHelper(unittest.TestCase):
    """Test _execute_script() and _execute_scripts() helper methods."""

    def setUp(self):
        """Set up test inferencer."""
        self.inferencer = EchoInferencer()

    def test_execute_script_returns_dict(self):
        """Test that _execute_script returns correct dict structure."""
        result = self.inferencer._execute_script("echo test")

        self.assertIsInstance(result, dict)
        self.assertIn("stdout", result)
        self.assertIn("stderr", result)
        self.assertIn("return_code", result)
        self.assertIn("success", result)
        self.assertIn("script", result)

    def test_execute_script_success(self):
        """Test executing a successful script."""
        result = self.inferencer._execute_script("echo hello")

        self.assertTrue(result["success"])
        self.assertEqual(result["return_code"], 0)
        self.assertIn("hello", result["stdout"])

    def test_execute_script_failure(self):
        """Test executing a failing script."""
        result = self.inferencer._execute_script("exit 42")

        self.assertFalse(result["success"])
        self.assertEqual(result["return_code"], 42)

    def test_execute_scripts_all_success(self):
        """Test executing multiple successful scripts."""
        scripts = ["echo one", "echo two", "echo three"]
        result = self.inferencer._execute_scripts(scripts)

        self.assertTrue(result["success"])
        self.assertEqual(len(result["results"]), 3)

    def test_execute_scripts_stops_on_failure(self):
        """Test that _execute_scripts stops on first failure."""
        scripts = ["echo one", "exit 1", "echo three"]
        result = self.inferencer._execute_scripts(scripts)

        self.assertFalse(result["success"])
        self.assertEqual(len(result["results"]), 2)  # Stopped after failure
        self.assertEqual(result["failed_script"], "exit 1")


def print_test_results():
    """Print test results in a formatted way."""
    print("\nRunning TerminalInferencerBase tests...")
    print("=" * 50)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestEchoInferencer,
        TestFileOutputInferencer,
        TestCatInferencer,
        TestTimeoutHandling,
        TestErrorHandling,
        TestExecuteCommandHelper,
        TestWorkingDirectory,
        TestEnvironmentVariables,
        TestSaveOutput,
        TestPreExecutionScripts,
        TestPostExecutionScripts,
        TestExecuteScriptHelper,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✓ All tests passed!")
    else:
        print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")

    return result.wasSuccessful()


if __name__ == "__main__":
    print_test_results()
