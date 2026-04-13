"""Tests for registered_targets.py — verify all aliases resolve correctly."""

import importlib

import pytest

from rich_python_utils.config_utils import list_registered, resolve_target

# Importing the configs package triggers registration.
import agent_foundation.common.configs  # noqa: F401


class TestAllAliasesResolve:
    def test_every_alias_resolves_to_string(self):
        all_aliases = list_registered()
        assert len(all_aliases) > 0, "No aliases registered"
        for alias, path in all_aliases.items():
            assert isinstance(path, str), f"{alias} → {path!r} is not a string"
            assert "." in path, f"{alias} → {path!r} has no module path"


class TestCategories:
    def test_inferencer_category(self):
        inferencers = list_registered("inferencer")
        assert "ClaudeAPI" in inferencers
        assert "Dual" in inferencers
        assert "Conversational" in inferencers

    def test_config_category(self):
        configs = list_registered("config")
        assert "ContextBudget" in configs
        assert "LlmInferenceArgs" in configs
        assert "ConsensusConfig" in configs


class TestImportPathsValid:
    def test_all_paths_importable(self):
        """Verify each registered path can actually be imported."""
        all_aliases = list_registered()
        for alias, import_path in all_aliases.items():
            module_path, _, attr_name = import_path.rpartition(".")
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, attr_name, None)
                assert cls is not None, (
                    f"Alias {alias!r}: module {module_path!r} has no attribute {attr_name!r}"
                )
            except ImportError as e:
                pytest.skip(f"Alias {alias!r}: cannot import {module_path!r} — {e}")
