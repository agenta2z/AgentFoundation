"""Regression tests for ``InferencerBase.effective_allowed_paths`` and its
auto-inclusion of ``workspace.root``.

Background: added 2026-05-17 to address the worker_1 fix-step failure mode in
task-7ae9058e (the fix step couldn't read the prior artifact because it sat
in a sibling subtree outside acli's workspace whitelist).

The base inferencer (``InferencerBase``, NOT ``TerminalInferencerBase``)
surfaces a backend-agnostic ``additional_allowed_paths: List[AllowedPath]``
slot plus an ``effective_allowed_paths`` property that auto-includes
``workspace.root`` whenever a workspace is set. Subclasses (e.g.,
``RovoDevCliInferencer``) translate this into their native flag.

The field lives at ``InferencerBase`` so orchestrators (LWI, BTA, Dual, MFI)
and non-terminal inferencers (API, SDK) can also carry it. The base impl
does NOT compare against the subprocess cwd — that's a backend-specific
optimisation. Redundant-with-cwd cases are harmless; subclasses can dedupe
at translation time if they care.
"""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from attr import attrs

from rich_python_utils.path_utils import AllowedPath, PathAccess

from agent_foundation.common.inferencers.inferencer_base import InferencerBase
from agent_foundation.common.inferencers.terminal_inferencers.terminal_inferencer_base import (
    TerminalInferencerBase,
)


@attrs
class _ConcreteInferencer(TerminalInferencerBase):
    """Minimal concrete subclass for testing base class behavior."""

    def construct_command(self, inference_input: Any, **kwargs: Any) -> List[str]:
        return ["echo", str(inference_input)]

    def parse_output(self, stdout: str, stderr: str, return_code: int) -> Dict[str, Any]:
        return {"output": stdout, "stderr": stderr, "return_code": return_code, "success": return_code == 0}


@attrs
class _MinimalNonTerminalInferencer(InferencerBase):
    """Minimal direct-InferencerBase subclass for testing that the field
    is at the right level (orchestrators / API inferencers can use it too).
    """

    def _infer(self, inference_input: Any, inference_config: Any = None, **kwargs: Any) -> Any:
        return None  # not actually executed in these tests

    async def _ainfer(self, inference_input: Any, inference_config: Any = None, **kwargs: Any) -> Any:
        return None  # not actually executed in these tests


def _ws_mock(root: str) -> MagicMock:
    """Build a minimal mock workspace exposing only the ``root`` attribute."""
    ws = MagicMock()
    ws.root = root
    return ws


class TestEffectiveAllowedPathsField:
    """The new ``additional_allowed_paths`` field defaults to an empty list and
    accepts AllowedPath instances."""

    def test_default_is_empty(self):
        inf = _ConcreteInferencer()
        assert inf.additional_allowed_paths == []

    def test_accepts_allowed_path_instances(self):
        inf = _ConcreteInferencer(
            additional_allowed_paths=[
                AllowedPath("/etc/hosts", access=PathAccess.READ),
                AllowedPath("/tmp/scratch"),
            ]
        )
        assert len(inf.additional_allowed_paths) == 2
        assert inf.additional_allowed_paths[0].path == "/etc/hosts"
        assert inf.additional_allowed_paths[0].access == PathAccess.READ
        assert inf.additional_allowed_paths[1].access == PathAccess.ALL


class TestEffectiveAllowedPathsAutoInclude:
    """``effective_allowed_paths`` auto-injects workspace.root whenever a
    workspace is set (no cwd comparison at the base level)."""

    def test_no_workspace_no_auto_include(self, tmp_path):
        # No workspace set; only user-provided entries should appear.
        inf = _ConcreteInferencer(
            target_path=str(tmp_path),
            additional_allowed_paths=[AllowedPath(str(tmp_path / "external"))],
        )
        paths = [ap.path for ap in inf.effective_allowed_paths]
        assert paths == [str((tmp_path / "external").resolve())]

    def test_auto_include_when_target_path_differs_from_workspace_root(self, tmp_path):
        # Simulates the D-1 setup: target_path = user's codebase,
        # workspace.root = orchestrator's task tree (distinct paths).
        codebase = tmp_path / "codebase"
        codebase.mkdir()
        task_tree = tmp_path / "task_tree"
        task_tree.mkdir()

        inf = _ConcreteInferencer(target_path=str(codebase))
        inf._workspace = _ws_mock(str(task_tree))

        paths = [ap.path for ap in inf.effective_allowed_paths]
        assert paths == [str(task_tree.resolve())]
        # And it carries PathAccess.ALL.
        assert inf.effective_allowed_paths[0].access == PathAccess.ALL

    def test_workspace_root_auto_included_even_when_equal_to_cwd(self, tmp_path):
        # Base-level property does NOT compare against cwd — it just includes
        # workspace.root when set. Backend-side dedup (e.g., RovoDev's
        # _compose_config_override_for_cli) handles redundancy if it matters.
        ws_dir = tmp_path / "ws"
        ws_dir.mkdir()

        inf = _ConcreteInferencer()  # target_path None → cwd falls through to workspace.root
        inf._workspace = _ws_mock(str(ws_dir))

        assert inf.effective_cwd == str(ws_dir)
        paths = [ap.path for ap in inf.effective_allowed_paths]
        assert paths == [str(ws_dir.resolve())]

    def test_workspace_root_auto_included_even_when_equal_to_target_path(self, tmp_path):
        # Same dir set as both target_path and workspace.root → workspace.root
        # is still auto-included (no comparison; harmless redundancy).
        same = tmp_path / "same"
        same.mkdir()

        inf = _ConcreteInferencer(target_path=str(same))
        inf._workspace = _ws_mock(str(same))

        paths = [ap.path for ap in inf.effective_allowed_paths]
        assert paths == [str(same.resolve())]

    def test_user_provided_entry_for_workspace_root_not_duplicated(self, tmp_path):
        # If the caller already passed workspace.root explicitly, auto-include
        # must not duplicate it (even with a different access level).
        codebase = tmp_path / "codebase"
        codebase.mkdir()
        task_tree = tmp_path / "task_tree"
        task_tree.mkdir()

        inf = _ConcreteInferencer(
            target_path=str(codebase),
            additional_allowed_paths=[
                AllowedPath(str(task_tree), access=PathAccess.READ),
            ],
        )
        inf._workspace = _ws_mock(str(task_tree))

        paths = inf.effective_allowed_paths
        assert len(paths) == 1, f"expected 1 entry, got {[(p.path, p.access) for p in paths]}"
        # The user's entry wins (READ), auto-include is skipped.
        assert paths[0].access == PathAccess.READ


class TestEffectiveAllowedPathsDedup:
    """Deduplication is by resolved absolute path."""

    def test_dedupe_within_user_entries(self, tmp_path):
        # Two entries pointing at the same resolved path → dedupe to one.
        target = tmp_path / "target"
        target.mkdir()
        inf = _ConcreteInferencer(
            additional_allowed_paths=[
                AllowedPath(str(target)),
                AllowedPath(str(target) + "/."),  # same resolved path, different spelling
                AllowedPath(str(target)),  # exact duplicate
            ]
        )
        resolved = inf.effective_allowed_paths
        assert len(resolved) == 1
        assert resolved[0].path == str(target.resolve())

    def test_invalid_path_is_dropped_silently(self, tmp_path):
        # Embedded null byte → Path.resolve raises → entry must be dropped,
        # not propagated. Other valid entries must still appear.
        valid = tmp_path / "valid"
        valid.mkdir()
        inf = _ConcreteInferencer(
            additional_allowed_paths=[
                AllowedPath("\x00invalid"),  # null byte; resolve will raise
                AllowedPath(str(valid)),
            ]
        )
        paths = [ap.path for ap in inf.effective_allowed_paths]
        # The invalid one is gone; the valid one survives.
        assert paths == [str(valid.resolve())]

    def test_empty_path_string_dropped(self, tmp_path):
        valid = tmp_path / "valid"
        valid.mkdir()
        inf = _ConcreteInferencer(
            additional_allowed_paths=[
                AllowedPath(""),
                AllowedPath(str(valid)),
            ]
        )
        paths = [ap.path for ap in inf.effective_allowed_paths]
        assert paths == [str(valid.resolve())]


class TestEffectiveAllowedPathsOrdering:
    """Order: user entries first (preserved), then auto-included workspace.root."""

    def test_user_entries_precede_auto_workspace(self, tmp_path):
        codebase = tmp_path / "codebase"
        codebase.mkdir()
        task_tree = tmp_path / "task_tree"
        task_tree.mkdir()
        extra1 = tmp_path / "extra1"
        extra1.mkdir()
        extra2 = tmp_path / "extra2"
        extra2.mkdir()

        inf = _ConcreteInferencer(
            target_path=str(codebase),
            additional_allowed_paths=[
                AllowedPath(str(extra1)),
                AllowedPath(str(extra2), access=PathAccess.READ),
            ],
        )
        inf._workspace = _ws_mock(str(task_tree))

        paths = [ap.path for ap in inf.effective_allowed_paths]
        assert paths == [
            str(extra1.resolve()),
            str(extra2.resolve()),
            str(task_tree.resolve()),
        ]


class TestBackwardsCompat:
    """Concrete inferencers without explicit additional_allowed_paths must
    keep the same behavior they had before this change."""

    def test_default_constructor_still_works(self):
        inf = _ConcreteInferencer()
        assert inf.additional_allowed_paths == []
        assert inf.effective_allowed_paths == []

    def test_target_path_only_still_works(self, tmp_path):
        inf = _ConcreteInferencer(target_path=str(tmp_path))
        assert inf.effective_cwd == str(tmp_path)
        assert inf.effective_allowed_paths == []


class TestFieldLivesAtInferencerBase:
    """The field/property are on ``InferencerBase`` — NOT only on
    ``TerminalInferencerBase``. This matters for orchestrators (LWI, BTA,
    Dual, MFI) and non-terminal inferencers (API, SDK), which can carry the
    field and (in orchestrators' case) propagate it to children.

    If a future refactor accidentally demotes the field back to
    ``TerminalInferencerBase``, this test class will catch it.
    """

    def test_field_present_on_minimal_inferencer_base_subclass(self):
        inf = _MinimalNonTerminalInferencer()
        assert inf.additional_allowed_paths == []
        assert inf.effective_allowed_paths == []

    def test_can_set_additional_allowed_paths_on_non_terminal(self, tmp_path):
        # An orchestrator / API-only / SDK-only inferencer can carry paths.
        external = tmp_path / "external"
        external.mkdir()
        inf = _MinimalNonTerminalInferencer(
            additional_allowed_paths=[AllowedPath(str(external))]
        )
        paths = [ap.path for ap in inf.effective_allowed_paths]
        assert paths == [str(external.resolve())]

    def test_workspace_root_auto_included_on_non_terminal(self, tmp_path):
        # The auto-include logic does NOT depend on having an effective_cwd
        # (terminal-specific concept) — it just needs ``_workspace``.
        task_tree = tmp_path / "task_tree"
        task_tree.mkdir()
        inf = _MinimalNonTerminalInferencer()
        inf._workspace = _ws_mock(str(task_tree))

        paths = [ap.path for ap in inf.effective_allowed_paths]
        assert paths == [str(task_tree.resolve())]
        assert inf.effective_allowed_paths[0].access == PathAccess.ALL

    def test_attribute_is_defined_on_inferencer_base_itself(self):
        # Catch a future regression where the field is moved/duplicated on a
        # subclass: the attrib MUST appear in InferencerBase.__attrs_attrs__.
        names = {a.name for a in InferencerBase.__attrs_attrs__}
        assert "additional_allowed_paths" in names, (
            "additional_allowed_paths must live on InferencerBase so that "
            "orchestrators and non-terminal inferencers can carry it. "
            f"Found attribs: {sorted(names)}"
        )
