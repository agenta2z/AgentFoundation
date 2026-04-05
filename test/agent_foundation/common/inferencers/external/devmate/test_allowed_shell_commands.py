"""Integration tests for the allowed_shell_commands feature.

Tests the end-to-end behavior of ``enable_shell`` and
``allowed_shell_commands`` across the inferencer hierarchy:

- TerminalSessionInferencerBase (validation / precedence warning)
- generate_config_with_allowed_commands (common.py helper)
- DevmateCliInferencer (config gen wiring + two-call sync)
- DevmateSDKInferencer (config gen wiring + precedence)
- ClaudeCodeCliInferencer (Bash filtering / --disallowedTools)
- ClaudeCodeInferencer SDK (Bash removal + empty-list guard)
"""

import logging
import logging.handlers
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_inferencer import (
    ClaudeCodeInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.common import (
    generate_config_with_allowed_commands,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
    DevmateCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_sdk_inferencer import (
    DevmateSDKInferencer,
)


_MOCK_SYNC_CLI = (
    "agent_foundation.common.inferencers."
    "agentic_inferencers.external.devmate.devmate_cli_inferencer."
    "sync_config_to_target"
)
_MOCK_GEN_CLI = (
    "agent_foundation.common.inferencers."
    "agentic_inferencers.external.devmate.devmate_cli_inferencer."
    "generate_config_with_allowed_commands"
)
_MOCK_SYNC_SDK = (
    "agent_foundation.common.inferencers."
    "agentic_inferencers.external.devmate.devmate_sdk_inferencer."
    "sync_config_to_target"
)
_MOCK_GEN_SDK = (
    "agent_foundation.common.inferencers."
    "agentic_inferencers.external.devmate.devmate_sdk_inferencer."
    "generate_config_with_allowed_commands"
)


# =====================================================================
# generate_config_with_allowed_commands  (common.py)
# =====================================================================
class GenerateConfigBasicTest(unittest.TestCase):
    """Core behaviour of the config-generation helper."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _config_name(self, name: str = "freeform_agentic") -> str:
        return os.path.join(self.tmp_dir, "configs", name)

    def test_file_created_in_generated_subdir(self) -> None:
        result = generate_config_with_allowed_commands(
            self._config_name(), ["nvidia-smi"], self.tmp_dir
        )
        self.assertIn("/_generated/", result)
        generated_path = Path(self.tmp_dir) / (result + ".md")
        self.assertTrue(generated_path.exists())

    def test_deterministic_output_regardless_of_order(self) -> None:
        r1 = generate_config_with_allowed_commands(
            self._config_name(), ["nvidia-smi", "nvcc"], self.tmp_dir
        )
        r2 = generate_config_with_allowed_commands(
            self._config_name(), ["nvcc", "nvidia-smi"], self.tmp_dir
        )
        self.assertEqual(r1, r2)

    def test_different_base_configs_produce_different_hashes(self) -> None:
        r1 = generate_config_with_allowed_commands(
            self._config_name("config_a"), ["nvidia-smi"], self.tmp_dir
        )
        r2 = generate_config_with_allowed_commands(
            self._config_name("config_b"), ["nvidia-smi"], self.tmp_dir
        )
        self.assertNotEqual(r1, r2)

    def test_return_value_has_no_md_extension(self) -> None:
        result = generate_config_with_allowed_commands(
            self._config_name(), ["nvidia-smi"], self.tmp_dir
        )
        self.assertFalse(result.endswith(".md"))


class GenerateConfigContentTest(unittest.TestCase):
    """Validate the YAML content of generated config files."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _generate(self, commands: list) -> str:
        config_name = os.path.join(self.tmp_dir, "configs", "freeform_agentic")
        result = generate_config_with_allowed_commands(
            config_name, commands, self.tmp_dir
        )
        return (Path(self.tmp_dir) / (result + ".md")).read_text()

    def test_extends_points_to_parent_dir(self) -> None:
        content = self._generate(["nvidia-smi"])
        self.assertIn("extends: '../freeform_agentic.md'", content)

    def test_contains_allowed_commands_entries(self) -> None:
        content = self._generate(["nvidia-smi", "nvcc"])
        self.assertIn("- executable: 'nvidia-smi'", content)
        self.assertIn("- executable: 'nvcc'", content)

    def test_uses_execute_command_tool_key(self) -> None:
        content = self._generate(["nvidia-smi"])
        self.assertIn("execute_command:", content)
        self.assertIn("mcp_servers:", content)

    def test_has_prompt_template_variable(self) -> None:
        content = self._generate(["nvidia-smi"])
        self.assertIn("${{ prompt:str }}", content)

    def test_single_command_produces_single_entry(self) -> None:
        content = self._generate(["nvidia-smi"])
        self.assertEqual(content.count("- executable:"), 1)


# =====================================================================
# DevmateCliInferencer  --  config generation wiring
# =====================================================================
class DevmateCliAllowedShellCommandsTest(unittest.TestCase):
    """Test allowed_shell_commands wiring in DevmateCliInferencer."""

    @patch(_MOCK_SYNC_CLI)
    @patch(_MOCK_GEN_CLI, return_value="generated/config/path")
    def test_config_gen_called(self, mock_gen, _mock_sync) -> None:
        inferencer = DevmateCliInferencer(
            target_path="/test/repo",
            config_name="freeform",
            allowed_shell_commands=["nvidia-smi"],
        )
        mock_gen.assert_called_once()
        self.assertEqual(inferencer.config_name, "generated/config/path")

    @patch(_MOCK_SYNC_CLI)
    @patch(_MOCK_GEN_CLI)
    def test_config_gen_not_called_when_none(self, mock_gen, _mock_sync) -> None:
        DevmateCliInferencer(target_path="/test/repo", config_name="freeform")
        mock_gen.assert_not_called()

    @patch(_MOCK_SYNC_CLI)
    @patch(_MOCK_GEN_CLI)
    def test_config_gen_skipped_when_enable_shell_false(
        self, mock_gen, _mock_sync
    ) -> None:
        inferencer = DevmateCliInferencer(
            target_path="/test/repo",
            config_name="freeform",
            enable_shell=False,
            allowed_shell_commands=["nvidia-smi"],
        )
        mock_gen.assert_not_called()
        self.assertEqual(inferencer.config_name, "freeform")

    @patch(_MOCK_SYNC_CLI)
    @patch(_MOCK_GEN_CLI, return_value="generated/config")
    def test_two_call_sync_pattern(self, _mock_gen, mock_sync) -> None:
        DevmateCliInferencer(
            target_path="/test/repo",
            config_name="freeform",
            allowed_shell_commands=["nvidia-smi"],
        )
        sync_calls = mock_sync.call_args_list
        # First call syncs base config, second syncs generated config
        self.assertGreaterEqual(len(sync_calls), 2)
        self.assertEqual(sync_calls[0][0][0], "freeform")
        self.assertEqual(sync_calls[1][0][0], "generated/config")

    @patch(_MOCK_SYNC_CLI)
    def test_enable_shell_inherited_from_base(self, _mock_sync) -> None:
        inferencer = DevmateCliInferencer(
            target_path="/test/repo", config_name="freeform"
        )
        self.assertTrue(inferencer.enable_shell)
        # The attr is NOT re-declared on the subclass
        self.assertNotIn("enable_shell", DevmateCliInferencer.__dict__)

    @patch(_MOCK_SYNC_CLI)
    def test_allowed_shell_commands_default_none(self, _mock_sync) -> None:
        inferencer = DevmateCliInferencer(
            target_path="/test/repo", config_name="freeform"
        )
        self.assertIsNone(inferencer.allowed_shell_commands)

    @patch(_MOCK_SYNC_CLI)
    def test_enable_shell_false_in_command_output(self, _mock_sync) -> None:
        inferencer = DevmateCliInferencer(
            target_path="/test/repo",
            config_name="freeform",
            enable_shell=False,
            no_create_commit=False,
        )
        command = inferencer.construct_command("test prompt")
        self.assertIn('"enable_shell=false"', command)


# =====================================================================
# DevmateSDKInferencer  --  config generation wiring
# =====================================================================
class DevmateSDKAllowedShellCommandsTest(unittest.TestCase):
    """Test allowed_shell_commands wiring in DevmateSDKInferencer."""

    @patch(_MOCK_SYNC_SDK)
    @patch(_MOCK_GEN_SDK, return_value="generated/sdk/config")
    def test_config_gen_called(self, mock_gen, _mock_sync) -> None:
        inferencer = DevmateSDKInferencer(
            allowed_shell_commands=["nvidia-smi"],
        )
        mock_gen.assert_called_once()
        self.assertEqual(inferencer.config_file_path, "generated/sdk/config")

    @patch(_MOCK_SYNC_SDK)
    @patch(_MOCK_GEN_SDK)
    def test_config_gen_not_called_when_none(self, mock_gen, _mock_sync) -> None:
        DevmateSDKInferencer()
        mock_gen.assert_not_called()

    @patch(_MOCK_SYNC_SDK)
    @patch(_MOCK_GEN_SDK)
    def test_config_gen_skipped_when_enable_shell_false(
        self, mock_gen, _mock_sync
    ) -> None:
        DevmateSDKInferencer(
            enable_shell=False,
            allowed_shell_commands=["nvidia-smi"],
        )
        mock_gen.assert_not_called()

    def test_allowed_shell_commands_default_none(self) -> None:
        inferencer = DevmateSDKInferencer()
        self.assertIsNone(inferencer.allowed_shell_commands)

    def test_enable_shell_false_precedence_warning(self) -> None:
        with self.assertLogs(level="WARNING") as cm:
            DevmateSDKInferencer(
                enable_shell=False,
                allowed_shell_commands=["nvidia-smi"],
            )
        self.assertTrue(
            any("takes precedence" in msg for msg in cm.output),
            "Expected 'takes precedence' warning, got: %s" % cm.output,
        )


# =====================================================================
# ClaudeCodeCliInferencer  --  enable_shell + allowed_shell_commands
# =====================================================================
class ClaudeCodeCliEnableShellTest(unittest.TestCase):
    """Test enable_shell behaviour in ClaudeCodeCliInferencer."""

    def test_enable_shell_default_true(self) -> None:
        inferencer = ClaudeCodeCliInferencer()
        self.assertTrue(inferencer.enable_shell)

    def test_enable_shell_false_no_allowed_tools_uses_disallowed(self) -> None:
        inferencer = ClaudeCodeCliInferencer(enable_shell=False)
        command = inferencer.construct_command("test")
        self.assertIn('--disallowedTools "Bash"', command)
        self.assertNotIn("--allowedTools", command)

    def test_enable_shell_false_filters_bash_from_allowed_tools(self) -> None:
        inferencer = ClaudeCodeCliInferencer(
            enable_shell=False,
            allowed_tools=["Read", "Write", "Bash"],
        )
        command = inferencer.construct_command("test")
        self.assertIn('--allowedTools "Read,Write"', command)

    def test_enable_shell_false_only_bash_uses_disallowed(self) -> None:
        inferencer = ClaudeCodeCliInferencer(
            enable_shell=False,
            allowed_tools=["Bash"],
        )
        command = inferencer.construct_command("test")
        self.assertIn('--disallowedTools "Bash"', command)
        self.assertNotIn("--allowedTools", command)

    def test_enable_shell_true_preserves_allowed_tools(self) -> None:
        inferencer = ClaudeCodeCliInferencer(
            enable_shell=True,
            allowed_tools=["Read", "Write", "Bash"],
        )
        command = inferencer.construct_command("test")
        self.assertIn('--allowedTools "Read,Write,Bash"', command)

    def test_allowed_shell_commands_default_none(self) -> None:
        inferencer = ClaudeCodeCliInferencer()
        self.assertIsNone(inferencer.allowed_shell_commands)


class ClaudeCodeCliAllowedShellCommandsLoggingTest(unittest.TestCase):
    """Test logging for allowed_shell_commands in ClaudeCodeCliInferencer."""

    def test_logs_info_when_allowed_shell_commands_set(self) -> None:
        with self.assertLogs(level="INFO") as cm:
            ClaudeCodeCliInferencer(allowed_shell_commands=["nvidia-smi"])
        self.assertTrue(
            any(
                "allowed_shell_commands" in msg and "nvidia-smi" in msg
                for msg in cm.output
            ),
            "Expected info log about allowed_shell_commands, got: %s" % cm.output,
        )

    def test_enable_shell_false_with_allowed_shell_commands_warns(self) -> None:
        with self.assertLogs(level="WARNING") as cm:
            ClaudeCodeCliInferencer(
                enable_shell=False,
                allowed_shell_commands=["nvidia-smi"],
            )
        self.assertTrue(
            any("takes precedence" in msg for msg in cm.output),
            "Expected 'takes precedence' warning, got: %s" % cm.output,
        )


# =====================================================================
# ClaudeCodeInferencer SDK  --  enable_shell + allowed_shell_commands
# =====================================================================
class ClaudeCodeSDKEnableShellTest(unittest.TestCase):
    """Test enable_shell behaviour in ClaudeCodeInferencer (SDK)."""

    def test_enable_shell_default_true_keeps_bash(self) -> None:
        inferencer = ClaudeCodeInferencer()
        self.assertTrue(inferencer.enable_shell)
        self.assertIn("Bash", inferencer.allowed_tools)

    def test_enable_shell_false_removes_bash(self) -> None:
        inferencer = ClaudeCodeInferencer(enable_shell=False)
        self.assertNotIn("Bash", inferencer.allowed_tools)
        self.assertEqual(inferencer.allowed_tools, ["Read", "Write"])

    def test_enable_shell_false_custom_tools_filters_bash(self) -> None:
        inferencer = ClaudeCodeInferencer(
            enable_shell=False,
            allowed_tools=["Read", "Bash", "Computer"],
        )
        self.assertEqual(inferencer.allowed_tools, ["Read", "Computer"])

    def test_enable_shell_false_only_bash_raises_value_error(self) -> None:
        """Regression: empty allowed_tools=[] is falsy in SDK transport.

        When enable_shell=False and allowed_tools only contains Bash,
        removing Bash would leave allowed_tools=[] which the SDK treats
        as allow-all. A ValueError is raised instead.
        """
        with self.assertRaises(ValueError) as ctx:
            ClaudeCodeInferencer(
                enable_shell=False,
                allowed_tools=["Bash"],
            )
        self.assertIn("allowed_tools=[]", str(ctx.exception))

    def test_enable_shell_true_preserves_tools(self) -> None:
        custom = ["Read", "Write", "Bash", "Computer"]
        inferencer = ClaudeCodeInferencer(enable_shell=True, allowed_tools=custom)
        self.assertEqual(inferencer.allowed_tools, custom)

    def test_allowed_shell_commands_default_none(self) -> None:
        inferencer = ClaudeCodeInferencer()
        self.assertIsNone(inferencer.allowed_shell_commands)


class ClaudeCodeSDKAllowedShellCommandsLoggingTest(unittest.TestCase):
    """Test logging for allowed_shell_commands in ClaudeCodeInferencer."""

    def test_logs_info_when_allowed_shell_commands_set(self) -> None:
        with self.assertLogs(level="INFO") as cm:
            ClaudeCodeInferencer(allowed_shell_commands=["nvidia-smi"])
        self.assertTrue(
            any(
                "allowed_shell_commands" in msg and "nvidia-smi" in msg
                for msg in cm.output
            ),
            "Expected info log about allowed_shell_commands, got: %s" % cm.output,
        )

    def test_enable_shell_false_with_allowed_shell_commands_warns(self) -> None:
        with self.assertLogs(level="WARNING") as cm:
            ClaudeCodeInferencer(
                enable_shell=False,
                allowed_shell_commands=["nvidia-smi"],
            )
        self.assertTrue(
            any("takes precedence" in msg for msg in cm.output),
            "Expected 'takes precedence' warning, got: %s" % cm.output,
        )


# =====================================================================
# Cross-cutting: TerminalSessionInferencerBase shell config validation
# =====================================================================
class TerminalSessionBaseShellValidationTest(unittest.TestCase):
    """Test the enable_shell / allowed_shell_commands validation
    in TerminalSessionInferencerBase (exercised via DevmateCliInferencer)."""

    @patch(_MOCK_SYNC_CLI)
    def test_enable_shell_false_with_allowed_cmds_warns(self, _mock_sync) -> None:
        with self.assertLogs(level="WARNING") as cm:
            DevmateCliInferencer(
                target_path="/test/repo",
                config_name="freeform",
                enable_shell=False,
                allowed_shell_commands=["nvidia-smi"],
            )
        self.assertTrue(
            any("takes precedence" in msg for msg in cm.output),
            "Expected 'takes precedence' warning, got: %s" % cm.output,
        )

    @patch(_MOCK_SYNC_CLI)
    @patch(_MOCK_GEN_CLI, return_value="generated/config")
    def test_enable_shell_true_with_allowed_cmds_no_precedence_warning(
        self, _mock_gen, _mock_sync
    ) -> None:
        """No 'takes precedence' warning when enable_shell=True."""
        handler = logging.handlers.MemoryHandler(capacity=100)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            DevmateCliInferencer(
                target_path="/test/repo",
                config_name="freeform",
                enable_shell=True,
                allowed_shell_commands=["nvidia-smi"],
            )
            handler.flush()
            records = handler.buffer
            precedence_warnings = [
                r
                for r in records
                if r.levelno >= logging.WARNING and "takes precedence" in r.getMessage()
            ]
            self.assertEqual(
                len(precedence_warnings),
                0,
                "Unexpected 'takes precedence' warning: %s" % precedence_warnings,
            )
        finally:
            root_logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
