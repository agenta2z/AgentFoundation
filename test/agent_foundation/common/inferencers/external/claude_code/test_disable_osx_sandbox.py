"""Tests for the macOS-sandbox toggle on the Claude Code inferencers.

Background: the Meta ``claude`` launcher wraps each agent in a macOS seatbelt
sandbox. macOS forbids *nesting* seatbelt sandboxes, so when ``claude`` runs
inside an already-sandboxed process its ``sandbox_apply`` fails and the process
exits 71 ("Operation not permitted") with no output. The
``--dangerously-disable-osx-sandbox`` launcher flag skips that nested apply.

These tests cover the shared resolver (env + explicit precedence) and the flag
emission on both the CLI (command string) and SDK (``extra_args``) inferencers.

``CLAUDE_CODE_COMMAND`` is set in the CLI tests purely to short-circuit
``_resolve_claude_command``'s ``claude --version`` subprocess probe (keeping the
tests fast and hermetic); it does not affect the flag logic under test.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code import (
    common,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_cli_inferencer import (
    ClaudeCodeCliInferencer,
)
from agent_foundation.common.inferencers.agentic_inferencers.external.claude_code.claude_code_sdk_inferencer import (
    ClaudeCodeSdkInferencer,
)

ENV = common.ENV_DISABLE_OSX_SANDBOX
FLAG = f"--{common.DANGEROUSLY_DISABLE_OSX_SANDBOX}"


# --------------------------------------------------------------------------- #
# Shared resolver
# --------------------------------------------------------------------------- #

class TestResolver:
    @pytest.mark.parametrize(
        "value,expected",
        [("1", True), ("true", True), ("TRUE", True), (" yes ", True),
         ("on", True), ("0", False), ("false", False), ("", False),
         ("maybe", False)],
    )
    def test_env_flag_enabled(self, value, expected):
        with mock.patch.dict(os.environ, {"X_FLAG": value}, clear=False):
            assert common.env_flag_enabled("X_FLAG") is expected

    def test_env_flag_unset_is_false(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("X_FLAG", None)
            assert common.env_flag_enabled("X_FLAG") is False

    def test_explicit_true_wins_over_env(self):
        with mock.patch.dict(os.environ, {ENV: "0"}, clear=False):
            assert common.resolve_disable_osx_sandbox(True) is True

    def test_explicit_false_wins_over_env(self):
        with mock.patch.dict(os.environ, {ENV: "1"}, clear=False):
            assert common.resolve_disable_osx_sandbox(False) is False

    def test_none_falls_back_to_env_on(self):
        with mock.patch.dict(os.environ, {ENV: "yes"}, clear=False):
            assert common.resolve_disable_osx_sandbox(None) is True

    def test_none_falls_back_to_env_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)
            assert common.resolve_disable_osx_sandbox(None) is False


# --------------------------------------------------------------------------- #
# CLI inferencer — flag emission in the constructed command
# --------------------------------------------------------------------------- #

def _cli_command(**kwargs) -> str:
    # CLAUDE_CODE_COMMAND short-circuits the version probe (fast + hermetic).
    with mock.patch.dict(os.environ, {"CLAUDE_CODE_COMMAND": "claude"}, clear=False):
        inf = ClaudeCodeCliInferencer(target_path="/tmp", **kwargs)
    return inf, inf.construct_command(
        {"prompt": "hi"},
        output_format="stream-json",
        verbose=True,
        include_partial_messages=True,
    )


class TestCliFlag:
    def test_explicit_true_adds_flag_before_subcommand(self):
        inf, cmd = _cli_command(disable_osx_sandbox=True)
        assert inf.disable_osx_sandbox is True
        assert FLAG in cmd
        # Meta launcher option must precede the ``-p`` subcommand.
        assert cmd.index(FLAG) < cmd.index("-p")

    def test_explicit_false_omits_flag(self):
        inf, cmd = _cli_command(disable_osx_sandbox=False)
        assert inf.disable_osx_sandbox is False
        assert FLAG not in cmd

    def test_default_none_with_env_on_adds_flag(self):
        with mock.patch.dict(os.environ, {ENV: "true"}, clear=False):
            inf, cmd = _cli_command()
        assert inf.disable_osx_sandbox is True
        assert FLAG in cmd

    def test_default_none_without_env_omits_flag(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)
            inf, cmd = _cli_command()
        assert inf.disable_osx_sandbox is False
        assert FLAG not in cmd


# --------------------------------------------------------------------------- #
# Regression guard for the pull-introduced __attrs_post_init__ truncation:
# super() + tier resolution must run.
# --------------------------------------------------------------------------- #

class TestPostInitIntact:
    def test_model_tier_resolves(self):
        # Proves __attrs_post_init__ reaches the tier branch (was dead code).
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_COMMAND": "claude"}, clear=False):
            inf = ClaudeCodeCliInferencer(target_path="/tmp", model_tier="lite")
        assert inf.model_name == "haiku"

    def test_super_post_init_ran(self):
        # InferencerBase.__attrs_post_init__ syncs the workspace handle; if
        # super() weren't chained, this attribute machinery wouldn't be set up.
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_COMMAND": "claude"}, clear=False):
            inf = ClaudeCodeCliInferencer(target_path="/tmp")
        # effective_cwd is provided by the base and depends on base post-init
        # state; accessing it must not raise and must reflect target_path.
        assert str(inf.effective_cwd) == "/tmp"


# --------------------------------------------------------------------------- #
# SDK inferencer — flag routed through extra_args
# --------------------------------------------------------------------------- #

class TestSdkFlag:
    def test_explicit_true_routes_to_extra_args(self):
        inf = ClaudeCodeSdkInferencer(target_path="/tmp", disable_osx_sandbox=True)
        assert inf.disable_osx_sandbox is True
        _, extra_args = inf._build_permission_effort_kwargs()
        # value-less boolean flag → None
        assert extra_args.get(common.DANGEROUSLY_DISABLE_OSX_SANDBOX, "MISSING") is None

    def test_explicit_false_omits(self):
        inf = ClaudeCodeSdkInferencer(target_path="/tmp", disable_osx_sandbox=False)
        _, extra_args = inf._build_permission_effort_kwargs()
        assert common.DANGEROUSLY_DISABLE_OSX_SANDBOX not in extra_args

    def test_default_none_with_env_on(self):
        with mock.patch.dict(os.environ, {ENV: "1"}, clear=False):
            inf = ClaudeCodeSdkInferencer(target_path="/tmp")
        assert inf.disable_osx_sandbox is True
        _, extra_args = inf._build_permission_effort_kwargs()
        assert common.DANGEROUSLY_DISABLE_OSX_SANDBOX in extra_args

    def test_default_none_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)
            inf = ClaudeCodeSdkInferencer(target_path="/tmp")
        assert inf.disable_osx_sandbox is False
        _, extra_args = inf._build_permission_effort_kwargs()
        assert common.DANGEROUSLY_DISABLE_OSX_SANDBOX not in extra_args
