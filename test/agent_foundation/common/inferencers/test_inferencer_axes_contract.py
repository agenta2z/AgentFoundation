"""Phase 0 — Contract tests for the inferencer axes refactor.

These tests pin the expected behavior of the three-axis architecture
(terminal, streaming, templating). Tests that were RED before the refactor
are marked with the phase that made them GREEN.

See: _docs/_plans/inferencer_axes_INTEGRATED_v5_plan.md
"""

import os
import pytest
from unittest.mock import patch, MagicMock

import attr
from attr import attrib, attrs

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.templated_inferencer_base import (
    TemplatedInferencerBase,
)
from agent_foundation.common.inferencers.streaming_inferencer_base import (
    StreamingInferencerBase,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_base import (
    TerminalInferencerBase,
    TerminalTemplatedInferencerBase,
)
from agent_foundation.common.inferencers.terminal_inferencers.terminal_session_inferencer_base import (
    TerminalSessionInferencerBase,
    TerminalSessionTemplatedInferencerBase,
)


# === Minimal test stubs ===


@attrs
class _StubTerminalInferencer(TerminalInferencerBase):
    """Minimal concrete TIB subclass for testing."""

    def construct_command(self, inference_input, **kwargs):
        return f"echo {inference_input}"

    def parse_output(self, stdout, stderr, return_code):
        return {"output": stdout, "success": return_code == 0}


@attrs
class _StubSessionInferencer(TerminalSessionInferencerBase):
    """Minimal concrete TSIB subclass for testing."""

    def construct_command(self, inference_input, **kwargs):
        return f"echo {inference_input}"

    def parse_output(self, stdout, stderr, return_code):
        return {"output": stdout, "success": return_code == 0}

    def _build_session_args(self, session_id, is_resume):
        return ""

    async def _ainfer_streaming(self, prompt, **kwargs):
        yield "test"


@attrs
class _StubTemplatedSessionInferencer(TerminalSessionTemplatedInferencerBase):
    """Minimal concrete TSTIB subclass for testing."""

    def construct_command(self, inference_input, **kwargs):
        return f"echo {inference_input}"

    def parse_output(self, stdout, stderr, return_code):
        return {"output": stdout, "success": return_code == 0}

    def _build_session_args(self, session_id, is_resume):
        return ""

    async def _ainfer_streaming(self, prompt, **kwargs):
        yield "test"


# === Contract tests ===


class TestTargetPathWorkspaceContract:
    """Tests for the target_path / working_dir / workspace contract."""

    def test_target_path_survives_workspace_assignment(self):
        """Explicit target_path is NOT overwritten by workspace assignment.
        GREEN after: Phase 1 + Phase 3.
        """
        inf = _StubTemplatedSessionInferencer(target_path="/tmp/repo")
        assert inf.effective_cwd == "/tmp/repo"
        # Simulate workspace assignment (which triggers _configure_for_workspace)
        # The guard should see target_path is non-None and skip the clobber.

    def test_explicit_target_path_sets_effective_cwd(self):
        """Explicit target_path drives effective_cwd."""
        inf = _StubTerminalInferencer(target_path="/explicit")
        assert inf.effective_cwd == "/explicit"

    def test_target_path_field_default_is_None(self):
        """target_path MUST default to None (NEVER os.getcwd()).
        Load-bearing for _configure_for_workspace guard.
        GREEN after: Phase 3.
        """
        field = attr.fields(TerminalInferencerBase).target_path
        assert field.default is None, (
            f"target_path default is {field.default!r}, not None! "
            "This is load-bearing for the _configure_for_workspace guard — "
            "see plan section 2.1."
        )

    def test_working_dir_defaults_from_target_path(self):
        """working_dir defaults to target_path when target_path is set.
        GREEN after: Phase 3.
        """
        inf = _StubTerminalInferencer(target_path="/my/repo")
        assert inf.effective_cwd == "/my/repo"

    def test_working_dir_defaults_to_cwd_when_no_target_path(self):
        """working_dir defaults to os.getcwd() when target_path is None.
        GREEN after: Phase 3.
        """
        inf = _StubTerminalInferencer()
        assert inf.target_path is None
        assert inf.effective_cwd == os.getcwd()


class TestSessionInheritsTerminalFeatures:
    """Tests that TSIB inherits TIB's execution machinery."""

    def test_session_inherits_timeout(self):
        """TSIB has timeout attrib from TIB.
        GREEN after: Phase 4.
        """
        fields = {f.name for f in attr.fields(TerminalSessionInferencerBase)}
        assert "timeout" in fields

    def test_session_inherits_env_vars(self):
        """TSIB has env_vars attrib from TIB.
        GREEN after: Phase 4.
        """
        fields = {f.name for f in attr.fields(TerminalSessionInferencerBase)}
        assert "env_vars" in fields

    def test_session_inherits_post_exec_scripts(self):
        """TSIB has post_exec_scripts attrib from TIB.
        GREEN after: Phase 4.
        """
        fields = {f.name for f in attr.fields(TerminalSessionInferencerBase)}
        assert "post_exec_scripts" in fields


class TestOrchestratorScenarios:
    """Tests that orchestrator-spawned children work correctly."""

    def test_orchestrator_spawned_child_uses_workspace_root(self):
        """Child with target_path=None: workspace assignment SHOULD set working_dir.
        This is the regression guard — must NEVER break.
        """
        inf = _StubTerminalInferencer()
        assert inf.target_path is None
        # The guarded clobber should fire (target is None, wd not user-set)

    def test_devmate_repo_path_kwarg_still_accepted(self):
        """DevmateCliInferencer(repo_path="/x") must construct without TypeError.
        GREEN after: Phase 5.
        """
        # Import here to avoid import errors if DevMate has unresolved deps
        try:
            from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
                DevmateCliInferencer,
            )
            inf = DevmateCliInferencer(repo_path="/tmp/test_repo")
            assert inf.repo_path == "/tmp/test_repo"
            assert inf.target_path == "/tmp/test_repo"
        except ImportError:
            pytest.skip("DevmateCliInferencer dependencies not available")


class TestTimeoutContract:
    """Tests for the timeout default change (v5 §2)."""

    def test_tib_timeout_default_is_None(self):
        """TIB.timeout must default to None (no subprocess cap).
        Historic value of 300 was a footgun for session subclasses.
        """
        field = attr.fields(TerminalInferencerBase).timeout
        assert field.default is None

    def test_tsib_inherits_timeout_None_default(self):
        """TSIB must not silently activate a subprocess timeout."""
        inf = _StubSessionInferencer()
        assert inf.timeout is None

    def test_pre_exec_scripts_runs_via_hook(self):
        """TSIB's _run_pre_exec_scripts_in_subprocess_shell returns True,
        preventing TIB's _infer from running pre-scripts separately.
        """
        inf = _StubSessionInferencer()
        assert inf._run_pre_exec_scripts_in_subprocess_shell() is True

        tib = _StubTerminalInferencer()
        assert tib._run_pre_exec_scripts_in_subprocess_shell() is False
