"""Tests for factory convenience functions."""

import pytest

# Importing configs triggers alias registration.
from agent_foundation.common.configs import load_inferencer


class TestLoadInferencer:
    def test_by_name(self):
        obj = load_inferencer("claude_api")
        assert type(obj).__name__ == "ClaudeApiInferencer"
        assert obj.model_id == "claude-sonnet-4-20250514"

    def test_with_overrides(self):
        obj = load_inferencer("claude_api", overrides={"model_id": "other-model"})
        assert obj.model_id == "other-model"

    def test_not_found(self):
        with pytest.raises(FileNotFoundError, match="No inferencer config"):
            load_inferencer("nonexistent_config")
