"""Unit tests: ClaudeCodeCliInferencer ``effort`` and ``permission_mode`` flags.

Verifies that the inferencer's ``effort`` and ``permission_mode`` attributes
translate into the correct CLI flags in the constructed shell command. These
tests are deterministic, fast, and free — they never invoke the ``claude``
binary. End-to-end behavior against the real subprocess is covered in
``test_effort_and_permission_mode_real.py``.
"""

import unittest

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)


def _make_inferencer(**overrides):
    """Build an inferencer that skips the ``_resolve_claude_command`` subprocess
    probe by passing a non-default ``claude_command``. Tests assert on flag
    fragments rather than the executable prefix, so the stub value is safe."""
    defaults = {"target_path": "/tmp", "claude_command": "claude-stub"}
    defaults.update(overrides)
    return ClaudeCodeCliInferencer(**defaults)


class EffortDefaultTest(unittest.TestCase):
    """Default ``effort`` is ``"max"`` and emits ``--effort max``."""

    def test_default_effort_attribute_is_max(self):
        inf = _make_inferencer()
        self.assertEqual(inf.effort, "max")

    def test_default_command_contains_effort_max(self):
        inf = _make_inferencer()
        cmd = inf.construct_command("hello")
        self.assertIn("--effort max", cmd)


class EffortFlagEmissionTest(unittest.TestCase):
    """Each accepted effort level emits ``--effort <level>``."""

    def test_low(self):
        cmd = _make_inferencer(effort="low").construct_command("hi")
        self.assertIn("--effort low", cmd)

    def test_medium(self):
        cmd = _make_inferencer(effort="medium").construct_command("hi")
        self.assertIn("--effort medium", cmd)

    def test_high(self):
        cmd = _make_inferencer(effort="high").construct_command("hi")
        self.assertIn("--effort high", cmd)

    def test_xhigh(self):
        cmd = _make_inferencer(effort="xhigh").construct_command("hi")
        self.assertIn("--effort xhigh", cmd)

    def test_max(self):
        cmd = _make_inferencer(effort="max").construct_command("hi")
        self.assertIn("--effort max", cmd)

    def test_none_omits_flag_entirely(self):
        cmd = _make_inferencer(effort=None).construct_command("hi")
        self.assertNotIn("--effort", cmd)

    def test_only_one_effort_flag_emitted(self):
        """Sanity: changing effort levels should not double-emit the flag."""
        for level in ("low", "medium", "high", "xhigh", "max"):
            with self.subTest(level=level):
                cmd = _make_inferencer(effort=level).construct_command("hi")
                self.assertEqual(
                    cmd.count("--effort"), 1,
                    f"expected exactly one --effort, got {cmd.count('--effort')}: {cmd}",
                )


class PermissionModeFlagEmissionTest(unittest.TestCase):
    """Each accepted permission mode emits the right CLI flag.

    Note: ``bypassPermissions`` is mapped to the legacy
    ``--dangerously-skip-permissions`` flag; ``default`` emits no flag (CLI
    default behavior). All other modes use ``--permission-mode <mode>``.
    """

    def test_default_attribute_is_bypass_permissions(self):
        self.assertEqual(_make_inferencer().permission_mode, "bypassPermissions")

    def test_bypass_permissions_emits_legacy_flag(self):
        cmd = _make_inferencer(permission_mode="bypassPermissions").construct_command("hi")
        self.assertIn("--dangerously-skip-permissions", cmd)
        self.assertNotIn("--permission-mode bypassPermissions", cmd)

    def test_default_mode_emits_no_flag(self):
        cmd = _make_inferencer(permission_mode="default").construct_command("hi")
        self.assertNotIn("--permission-mode", cmd)
        self.assertNotIn("--dangerously-skip-permissions", cmd)

    def test_plan_mode_emits_permission_mode_plan(self):
        cmd = _make_inferencer(permission_mode="plan").construct_command("hi")
        self.assertIn("--permission-mode plan", cmd)
        self.assertNotIn("--dangerously-skip-permissions", cmd)

    def test_accept_edits_emits_correct_flag(self):
        cmd = _make_inferencer(permission_mode="acceptEdits").construct_command("hi")
        self.assertIn("--permission-mode acceptEdits", cmd)

    def test_auto_emits_correct_flag(self):
        # CLI v2.1.119+ value (not in the SDK's typed Literal but valid for the CLI)
        cmd = _make_inferencer(permission_mode="auto").construct_command("hi")
        self.assertIn("--permission-mode auto", cmd)

    def test_dont_ask_emits_correct_flag(self):
        cmd = _make_inferencer(permission_mode="dontAsk").construct_command("hi")
        self.assertIn("--permission-mode dontAsk", cmd)


class CombinedFlagsTest(unittest.TestCase):
    """Effort and permission-mode flags coexist correctly in one command."""

    def test_plan_plus_xhigh(self):
        cmd = _make_inferencer(
            permission_mode="plan", effort="xhigh"
        ).construct_command("hi")
        self.assertIn("--permission-mode plan", cmd)
        self.assertIn("--effort xhigh", cmd)
        self.assertNotIn("--dangerously-skip-permissions", cmd)

    def test_default_bypass_plus_default_max_effort(self):
        cmd = _make_inferencer().construct_command("hi")
        self.assertIn("--dangerously-skip-permissions", cmd)
        self.assertIn("--effort max", cmd)

    def test_command_well_formed_with_all_options(self):
        """The full command is a syntactically reasonable CLI invocation."""
        cmd = _make_inferencer(
            permission_mode="acceptEdits", effort="high"
        ).construct_command("hello world")
        # Starts with the binary, contains -p, model, effort, permission-mode, and the prompt
        self.assertTrue(cmd.startswith("claude-stub -p"), cmd)
        self.assertIn("--model sonnet", cmd)
        self.assertIn("--effort high", cmd)
        self.assertIn("--permission-mode acceptEdits", cmd)
        # Prompt is the last positional, double-quoted
        self.assertTrue(cmd.endswith('"hello world"'), cmd)


class FlagOrderingTest(unittest.TestCase):
    """``--effort`` is emitted near ``--model`` (after model, before permission)."""

    def test_effort_appears_after_model_and_before_permission(self):
        cmd = _make_inferencer(
            permission_mode="plan", effort="high"
        ).construct_command("hi")
        idx_model = cmd.index("--model")
        idx_effort = cmd.index("--effort")
        idx_perm = cmd.index("--permission-mode")
        self.assertLess(idx_model, idx_effort, "expected --effort after --model")
        self.assertLess(idx_effort, idx_perm, "expected --effort before --permission-mode")


if __name__ == "__main__":
    unittest.main()
