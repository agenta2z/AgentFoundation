"""Unit tests for the factored path-completion helper (Commit 6).

Focus: the hardened containment check that replaces string ``startswith`` with
``Path.resolve().relative_to(...)`` — exercised directly against the helper,
independent of any web framework.
"""
from __future__ import annotations

import pytest

from agent_foundation.common.workspace.path_completion import (
    complete_path,
    PathContainmentError,
    PrefixNotADirectory,
)


@pytest.fixture
def tree(tmp_path):
    """root/ with two dirs, one file, one dotfile."""
    root = tmp_path / "root"
    (root / "alpha" / "nested").mkdir(parents=True)
    (root / "beta").mkdir()
    (root / "notes.txt").write_text("hi", encoding="utf-8")
    (root / ".hidden").write_text("x", encoding="utf-8")
    return root


def test_basic_dir_suggestions(tree):
    out = complete_path(str(tree), dirs_only=True)
    names = {s["name"] for s in out["suggestions"]}
    assert names == {"alpha/", "beta/"}
    assert all(s["is_dir"] for s in out["suggestions"])
    assert all(s["path"].endswith("/") for s in out["suggestions"])


def test_dirs_only_false_returns_files(tree):
    out = complete_path(str(tree), dirs_only=False)
    names = {s["name"] for s in out["suggestions"]}
    assert "notes.txt" in names
    assert "alpha/" in names
    # Dotfiles always hidden.
    assert ".hidden" not in names


def test_parent_traversal_rejected(tree):
    with pytest.raises(PathContainmentError):
        complete_path(str(tree), partial="../")


def test_sibling_prefix_attack_rejected(tmp_path):
    """partial pointing at a sibling dir whose name shares the prefix string.

    base=/tmp/root, partial="../root2" resolves to /tmp/root2 which is NOT
    under /tmp/root. A string startswith(base) check would wrongly accept it.
    """
    (tmp_path / "root").mkdir()
    (tmp_path / "root2").mkdir()
    with pytest.raises(PathContainmentError):
        complete_path(str(tmp_path / "root"), partial="../root2")


def test_missing_prefix_raises(tmp_path):
    with pytest.raises(PrefixNotADirectory):
        complete_path(str(tmp_path / "nope"))


def test_limit_is_capped(tmp_path):
    root = tmp_path / "many"
    root.mkdir()
    for i in range(10):
        (root / f"d{i:02d}").mkdir()
    out = complete_path(str(root), dirs_only=True, limit=3)
    assert len(out["suggestions"]) == 3


def test_partial_fragment_filters(tree):
    out = complete_path(str(tree), partial="al")
    names = {s["name"] for s in out["suggestions"]}
    assert names == {"alpha/"}
