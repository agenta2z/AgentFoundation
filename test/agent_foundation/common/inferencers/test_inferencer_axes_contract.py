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

    def test_devmate_repo_path_kwarg_removed(self):
        """DevmateCliInferencer(repo_path="/x") must raise TypeError now —
        repo_path was fully removed in the 3-path consolidation; callers
        must pass target_path instead.
        """
        try:
            from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
                DevmateCliInferencer,
            )
        except ImportError:
            pytest.skip("DevmateCliInferencer dependencies not available")
        with pytest.raises(TypeError, match="repo_path"):
            DevmateCliInferencer(repo_path="/tmp/test_repo")


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


# =============================================================================
# === 3-path consolidation contract (Step 11 of recursive-kindling-widget) ===
# =============================================================================


class TestThreePathModelPromotion:
    """target_path + effective_cwd promoted from TIB to InferencerBase."""

    def test_target_path_is_on_inferencer_base(self):
        """target_path attrib lives on IB so SDK-family inferencers
        (which inherit StreamingIB, not TIB) can use the canonical name.
        """
        field_names = {f.name for f in attr.fields(InferencerBase)}
        assert "target_path" in field_names
        assert attr.fields(InferencerBase).target_path.default is None

    def test_effective_cwd_is_on_inferencer_base(self):
        """effective_cwd @property lives on IB so any inferencer can
        derive a subprocess cwd / SDK root_folder uniformly.
        """
        assert hasattr(InferencerBase, "effective_cwd")
        assert isinstance(InferencerBase.effective_cwd, property)

    def test_effective_cwd_fallback_chain(self):
        """Priority: target_path > workspace.root > os.getcwd()."""
        from agent_foundation.common.inferencers.inferencer_workspace import (
            InferencerWorkspace,
        )

        inf = _StubTerminalInferencer(target_path="/a")
        assert inf.effective_cwd == "/a"
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            inf = _StubTerminalInferencer(workspace=InferencerWorkspace(root=tmpdir))
            assert inf.effective_cwd == tmpdir
        inf = _StubTerminalInferencer()
        assert inf.effective_cwd == os.getcwd()

    def test_sdk_family_inherits_target_path(self):
        """SDK-family inferencers (no TIB inheritance) get target_path
        via IB promotion.
        """
        for module_path, cls_name in (
            (
                "agent_foundation.common.inferencers.agentic_inferencers."
                "external.claude_code.claude_code_sdk_inferencer",
                "ClaudeCodeSdkInferencer",
            ),
            (
                "agent_foundation.common.inferencers.agentic_inferencers."
                "external.devmate.devmate_sdk_inferencer",
                "DevmateSDKInferencer",
            ),
            (
                "agent_foundation.common.inferencers.agentic_inferencers."
                "external.rovodev.rovodev_serve_inferencer",
                "RovoDevServeInferencer",
            ),
        ):
            try:
                import importlib
                mod = importlib.import_module(module_path)
                cls = getattr(mod, cls_name)
            except ImportError:
                pytest.skip(f"{cls_name} dependencies not available")
                continue
            field_names = {f.name for f in attr.fields(cls)}
            assert "target_path" in field_names, (
                f"{cls_name} missing inherited target_path"
            )


class TestNoOrphanPathAttribs:
    """Verify the orphan path attribs were truly removed everywhere."""

    def test_no_orphan_path_attribs_on_external_classes(self):
        """No external inferencer should have any of the legacy
        per-class path attribs.
        """
        modules_and_classes = [
            (
                "agent_foundation.common.inferencers.agentic_inferencers."
                "external.claude_code.claude_code_sdk_inferencer",
                "ClaudeCodeSdkInferencer",
            ),
            (
                "agent_foundation.common.inferencers.agentic_inferencers."
                "external.devmate.devmate_cli_inferencer",
                "DevmateCliInferencer",
            ),
            (
                "agent_foundation.common.inferencers.agentic_inferencers."
                "external.devmate.devmate_sdk_inferencer",
                "DevmateSDKInferencer",
            ),
            (
                "agent_foundation.common.inferencers.agentic_inferencers."
                "external.rovodev.rovodev_serve_inferencer",
                "RovoDevServeInferencer",
            ),
            (
                "agent_foundation.common.inferencers.agentic_inferencers."
                "tool_inferencers.tool_as_inferencer",
                "ToolAsInferencer",
            ),
        ]
        orphan_names = (
            "repo_path", "root_folder", "working_dir", "cwd",
            "repo", "root_path", "target_dir", "repo_dir",
            "repository_path", "workdir",
        )
        import importlib
        for module_path, cls_name in modules_and_classes:
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, cls_name)
            except ImportError:
                continue
            field_names = {f.name for f in attr.fields(cls)}
            for orphan in orphan_names:
                assert orphan not in field_names, (
                    f"{cls_name} has orphan path attrib {orphan!r} — "
                    "use target_path (inherited from InferencerBase) instead"
                )


class TestSourcePathConsolidation:
    """Devmate classes drop the source_path shadow + override
    _detect_source_root for the .hg/-based detection.
    """

    def test_source_path_not_shadowed_on_devmate_classes(self):
        """DevmateCli + DevmateSDK should NOT redeclare source_path.

        attrs creates per-class ``Attribute`` wrappers even for inherited
        fields, so identity (``is``) is unreliable. Use the ``inherited``
        flag when available, fall back to default-match heuristic.
        """
        try:
            from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
                DevmateCliInferencer,
            )
            from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_sdk_inferencer import (
                DevmateSDKInferencer,
            )
        except ImportError:
            pytest.skip("Devmate dependencies not available")
        ib_default = attr.fields(InferencerBase).source_path.default
        for cls in (DevmateCliInferencer, DevmateSDKInferencer):
            cls_field = attr.fields(cls).source_path
            if hasattr(cls_field, "inherited"):
                assert cls_field.inherited, (
                    f"{cls.__name__} re-declares source_path — should "
                    "inherit IB's attrib and override _detect_source_root instead"
                )
            else:
                assert cls_field.default == ib_default, (
                    f"{cls.__name__}.source_path default differs from IB"
                )

    def test_devmate_detect_source_root_returns_fbsource_root(self):
        """DevmateCli._detect_source_root walks up for .hg/ marker.

        Only enforced when the test runs in a Sapling checkout; skipped
        otherwise.
        """
        try:
            from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
                DevmateCliInferencer,
            )
        except ImportError:
            pytest.skip("DevmateCliInferencer dependencies not available")
        detected = DevmateCliInferencer._detect_source_root()
        if detected is None or not os.path.exists(os.path.join(detected, ".hg")):
            pytest.skip(
                "No .hg/ marker reachable from the inferencer source "
                "file (likely a non-Sapling env or pip-installed test)"
            )

    def test_get_source_repo_root_is_removed(self):
        """The buggy parents[8]-hardcoded helper is gone; replaced by
        cls-aware _detect_fbsource_root_for.
        """
        try:
            from agent_foundation.common.inferencers.agentic_inferencers.external.devmate import (
                common as devmate_common,
            )
        except ImportError:
            pytest.skip("Devmate common module not available")
        assert not hasattr(devmate_common, "get_source_repo_root")
        assert hasattr(devmate_common, "_detect_fbsource_root_for")


class TestHasLocalAccessLeakFixed:
    """has_local_access correctly set on each leaf — local agents True,
    server-side agents False.
    """

    def test_local_agents_advertise_local_access(self):
        """All inferencers whose agents write local files should have
        has_local_access=True (was leaking False on KiroCli, DevmateSDK,
        RovoDevServe pre-fix).
        """
        cases = [
            (
                "agent_foundation.common.inferencers.agentic_inferencers."
                "external.kiro.kiro_cli_inferencer",
                "KiroCliInferencer",
            ),
            (
                "agent_foundation.common.inferencers.agentic_inferencers."
                "external.devmate.devmate_sdk_inferencer",
                "DevmateSDKInferencer",
            ),
            (
                "agent_foundation.common.inferencers.agentic_inferencers."
                "external.rovodev.rovodev_serve_inferencer",
                "RovoDevServeInferencer",
            ),
            (
                "agent_foundation.common.inferencers.agentic_inferencers."
                "external.claude_code.claude_code_sdk_inferencer",
                "ClaudeCodeSdkInferencer",
            ),
            (
                "agent_foundation.common.inferencers.agentic_inferencers."
                "external.devmate.devmate_cli_inferencer",
                "DevmateCliInferencer",
            ),
        ]
        import importlib
        for module_path, cls_name in cases:
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, cls_name)
            except ImportError:
                continue
            field = attr.fields(cls).has_local_access
            assert field.default is True, (
                f"{cls_name}.has_local_access defaults to {field.default!r}; "
                "expected True (the agent writes local files)"
            )

    def test_inferencer_base_default_is_false(self):
        """IB's default stays False — server-side inferencers (Metamate)
        inherit it intentionally.
        """
        field = attr.fields(InferencerBase).has_local_access
        assert field.default is False


class TestLegacyClaudeCodeInferencerDeleted:
    """The strict-subset legacy class is fully gone."""

    def test_legacy_class_import_fails(self):
        with pytest.raises(ImportError):
            import importlib
            importlib.import_module(
                "agent_foundation.common.inferencers.agentic_inferencers."
                "external.claude_code.claude_code_inferencer"
            )


class TestDevmateCliCleanups:
    """DevmateCli post-removal contracts."""

    def test_repo_path_attrib_removed(self):
        try:
            from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
                DevmateCliInferencer,
            )
        except ImportError:
            pytest.skip("DevmateCliInferencer dependencies not available")
        field_names = {f.name for f in attr.fields(DevmateCliInferencer)}
        assert "repo_path" not in field_names
        assert "target_path" in field_names

    def test_target_path_default_is_fbsource(self):
        """DevmateCli post_init still defaults target_path to ~/fbsource
        (preserves user-visible behavior from the old repo_path default).

        Mocks sync_config_to_target to prevent test failure on environments
        without a real ~/fbsource Sapling checkout.
        """
        try:
            from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
                DevmateCliInferencer,
            )
        except ImportError:
            pytest.skip("DevmateCliInferencer dependencies not available")
        with patch(
            "agent_foundation.common.inferencers.agentic_inferencers"
            ".external.devmate.devmate_cli_inferencer.sync_config_to_target"
        ):
            inf = DevmateCliInferencer()
            assert inf.target_path == os.path.expanduser("~/fbsource")

    def test_no_cd_script_in_pre_exec_scripts(self):
        """The redundant `cd $target_path` pre-exec script is gone —
        the framework's cwd=effective_cwd argument already handles it.
        """
        try:
            from agent_foundation.common.inferencers.agentic_inferencers.external.devmate.devmate_cli_inferencer import (
                DevmateCliInferencer,
            )
        except ImportError:
            pytest.skip("DevmateCliInferencer dependencies not available")
        with patch(
            "agent_foundation.common.inferencers.agentic_inferencers"
            ".external.devmate.devmate_cli_inferencer.sync_config_to_target"
        ):
            inf = DevmateCliInferencer(target_path="/tmp/foo")
        scripts = inf.pre_exec_scripts or []
        for script in scripts:
            assert not script.startswith('cd "/tmp/foo"'), (
                f"cd_script present in pre_exec_scripts ({script!r}); "
                "should be deleted — framework subprocess cwd is already "
                "effective_cwd"
            )

