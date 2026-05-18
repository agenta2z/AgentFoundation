"""Regression tests for RovoDevCliInferencer._compose_config_override_for_cli().

Verifies that effective_allowed_paths from the base class are properly merged
into the acli --config-override JSON under toolPermissions.allowedExternalPaths,
without clobbering the user's existing agent.modelId override or any paths
they explicitly placed in the override.
"""

import json
import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rich_python_utils.path_utils import AllowedPath, PathAccess

from agent_foundation.common.inferencers.agentic_inferencers.external.rovodev.rovodev_cli_inferencer import (
    RovoDevCliInferencer,
)


def _ws_mock(root: str) -> MagicMock:
    ws = MagicMock()
    ws.root = root
    return ws


def _extract_config_override(command: str) -> dict:
    """Pull the --config-override JSON object out of a constructed command string."""
    parts = shlex.split(command)
    for i, p in enumerate(parts):
        if p == "--config-override":
            return json.loads(parts[i + 1])
    raise AssertionError(f"--config-override not found in command: {command}")


def _has_config_override(command: str) -> bool:
    return "--config-override" in shlex.split(command)


@pytest.fixture(autouse=True)
def _stub_acli_path():
    """Avoid requiring acli to be installed for these unit tests."""
    with patch("shutil.which", return_value="/usr/local/bin/acli"):
        yield


class TestComposeConfigOverrideDefaults:
    """The default config_override (modelId) is preserved when no extra paths are present."""

    def test_default_modelid_preserved_when_no_paths(self):
        # No workspace, no additional_allowed_paths → composed override equals
        # the default model-only override.
        inf = RovoDevCliInferencer()
        composed = inf._compose_config_override_for_cli()
        assert composed is not None
        assert json.loads(composed) == json.loads(inf.config_override)

    def test_none_override_and_no_paths_returns_none(self):
        inf = RovoDevCliInferencer(config_override=None)
        assert inf._compose_config_override_for_cli() is None


class TestComposeConfigOverrideMergesPaths:
    """When effective_allowed_paths is non-empty, paths are merged into
    toolPermissions.allowedExternalPaths AND the modelId override is preserved."""

    def test_additional_allowed_paths_merged(self, tmp_path):
        external = tmp_path / "external"
        external.mkdir()

        inf = RovoDevCliInferencer(
            additional_allowed_paths=[AllowedPath(str(external))]
        )
        composed = inf._compose_config_override_for_cli()
        assert composed is not None
        parsed = json.loads(composed)

        # modelId preserved
        assert parsed["agent"]["modelId"] == "anthropic:claude-opus-4-7"
        # path added
        assert str(external.resolve()) in parsed["toolPermissions"]["allowedExternalPaths"]

    def test_workspace_root_auto_injection_lands_in_override(self, tmp_path):
        codebase = tmp_path / "codebase"
        codebase.mkdir()
        task_tree = tmp_path / "task_tree"
        task_tree.mkdir()

        inf = RovoDevCliInferencer(target_path=str(codebase))
        inf._workspace = _ws_mock(str(task_tree))

        composed = inf._compose_config_override_for_cli()
        parsed = json.loads(composed)
        assert parsed["agent"]["modelId"] == "anthropic:claude-opus-4-7"
        assert parsed["toolPermissions"]["allowedExternalPaths"] == [
            str(task_tree.resolve())
        ]

    def test_user_override_paths_are_preserved_and_unioned(self, tmp_path):
        external1 = tmp_path / "ext1"
        external1.mkdir()
        external2 = tmp_path / "ext2"
        external2.mkdir()
        task_tree = tmp_path / "task_tree"
        task_tree.mkdir()
        codebase = tmp_path / "codebase"
        codebase.mkdir()

        # User pre-baked external1 into config_override; auto-include should add
        # task_tree and additional_allowed_paths should add external2. Order:
        # existing-first, then new.
        user_override = json.dumps(
            {
                "agent": {"modelId": "anthropic:claude-sonnet-4.6"},
                "toolPermissions": {"allowedExternalPaths": [str(external1.resolve())]},
            }
        )
        inf = RovoDevCliInferencer(
            target_path=str(codebase),
            config_override=user_override,
            additional_allowed_paths=[AllowedPath(str(external2))],
        )
        inf._workspace = _ws_mock(str(task_tree))

        parsed = json.loads(inf._compose_config_override_for_cli())
        # The user's modelId must NOT be reverted to the default.
        assert parsed["agent"]["modelId"] == "anthropic:claude-sonnet-4.6"
        # All three paths present, existing-first ordering preserved.
        assert parsed["toolPermissions"]["allowedExternalPaths"] == [
            str(external1.resolve()),
            str(external2.resolve()),
            str(task_tree.resolve()),
        ]

    def test_no_duplicate_paths(self, tmp_path):
        # If the user already has workspace.root in their override AND base
        # auto-includes it → must not duplicate.
        task_tree = tmp_path / "task_tree"
        task_tree.mkdir()
        codebase = tmp_path / "codebase"
        codebase.mkdir()

        user_override = json.dumps(
            {
                "agent": {"modelId": "anthropic:claude-opus-4-7"},
                "toolPermissions": {"allowedExternalPaths": [str(task_tree.resolve())]},
            }
        )
        inf = RovoDevCliInferencer(
            target_path=str(codebase),
            config_override=user_override,
        )
        inf._workspace = _ws_mock(str(task_tree))

        parsed = json.loads(inf._compose_config_override_for_cli())
        assert parsed["toolPermissions"]["allowedExternalPaths"] == [
            str(task_tree.resolve())
        ]


class TestComposeConfigOverrideMalformedInputs:
    """Resilient against bogus user-supplied overrides."""

    def test_malformed_json_override_falls_back_gracefully(self, tmp_path):
        # User shoved invalid JSON in config_override; we should still produce
        # something usable (a fresh dict containing only our paths) rather than
        # crash.
        external = tmp_path / "ext"
        external.mkdir()

        inf = RovoDevCliInferencer(
            config_override="this is not json",
            additional_allowed_paths=[AllowedPath(str(external))],
        )
        composed = inf._compose_config_override_for_cli()
        assert composed is not None
        parsed = json.loads(composed)
        # modelId gone (we fell back to {}), but our path is present.
        assert "agent" not in parsed
        assert parsed["toolPermissions"]["allowedExternalPaths"] == [
            str(external.resolve())
        ]

    def test_non_object_json_override_treated_as_empty(self, tmp_path):
        # config_override is valid JSON but a list — not a dict. Same fallback.
        external = tmp_path / "ext"
        external.mkdir()

        inf = RovoDevCliInferencer(
            config_override='["a", "b"]',
            additional_allowed_paths=[AllowedPath(str(external))],
        )
        parsed = json.loads(inf._compose_config_override_for_cli())
        assert parsed["toolPermissions"]["allowedExternalPaths"] == [
            str(external.resolve())
        ]

    def test_non_dict_tool_permissions_replaced(self, tmp_path):
        # User's config_override has toolPermissions as a string — invalid shape.
        # We should replace it with our well-formed dict, NOT crash.
        external = tmp_path / "ext"
        external.mkdir()

        inf = RovoDevCliInferencer(
            config_override=json.dumps({"toolPermissions": "broken"}),
            additional_allowed_paths=[AllowedPath(str(external))],
        )
        parsed = json.loads(inf._compose_config_override_for_cli())
        assert isinstance(parsed["toolPermissions"], dict)
        assert parsed["toolPermissions"]["allowedExternalPaths"] == [
            str(external.resolve())
        ]


class TestConstructCommandIntegration:
    """End-to-end: paths flow through to the actual constructed shell command."""

    def test_construct_command_legacy_mode_includes_paths(self, tmp_path):
        task_tree = tmp_path / "task_tree"
        task_tree.mkdir()
        codebase = tmp_path / "codebase"
        codebase.mkdir()

        inf = RovoDevCliInferencer(
            target_path=str(codebase),
            enable_legacy=True,
            raw_output_to_file=False,  # avoid temp file noise in the test
        )
        inf._workspace = _ws_mock(str(task_tree))

        command = inf.construct_command("hello")
        assert _has_config_override(command), command
        parsed = _extract_config_override(command)
        assert str(task_tree.resolve()) in parsed["toolPermissions"]["allowedExternalPaths"]

    def test_construct_command_non_legacy_mode_includes_paths(self, tmp_path):
        task_tree = tmp_path / "task_tree"
        task_tree.mkdir()
        codebase = tmp_path / "codebase"
        codebase.mkdir()

        inf = RovoDevCliInferencer(
            target_path=str(codebase),
            enable_legacy=False,
            raw_output_to_file=False,
        )
        inf._workspace = _ws_mock(str(task_tree))

        command = inf.construct_command("hello")
        assert _has_config_override(command), command
        parsed = _extract_config_override(command)
        assert str(task_tree.resolve()) in parsed["toolPermissions"]["allowedExternalPaths"]

    def test_no_workspace_and_no_paths_still_emits_modelid_override(self):
        # Regression guard: removing the unconditional --config-override emission
        # must not break the default modelId-only behavior.
        inf = RovoDevCliInferencer(enable_legacy=True, raw_output_to_file=False)
        command = inf.construct_command("hello")
        assert _has_config_override(command)
        parsed = _extract_config_override(command)
        assert parsed["agent"]["modelId"] == "anthropic:claude-opus-4-7"

    def test_explicit_none_override_and_no_paths_emits_no_flag(self):
        inf = RovoDevCliInferencer(
            enable_legacy=True,
            config_override=None,
            raw_output_to_file=False,
        )
        command = inf.construct_command("hello")
        assert not _has_config_override(command), command
