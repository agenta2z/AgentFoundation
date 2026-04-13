"""Tests for loading and instantiating YAML inferencer configs."""

from pathlib import Path

import pytest

from rich_python_utils.config_utils import instantiate, load_config

# Importing configs triggers alias registration.
import agent_foundation.common.configs  # noqa: F401

_YAML_DIR = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "agent_foundation"
    / "common"
    / "configs"
    / "yaml"
)


@pytest.fixture
def load_yaml():
    def _load(relative_path: str, overrides=None):
        return load_config(str(_YAML_DIR / relative_path), overrides=overrides)
    return _load


class TestClaudeApiYaml:
    def test_load_and_instantiate(self, load_yaml):
        cfg = load_yaml("inferencers/claude_api.yaml")
        obj = instantiate(cfg)
        assert type(obj).__name__ == "ClaudeApiInferencer"
        assert obj.model_id == "claude-sonnet-4-20250514"
        assert obj.max_retry == 3


class TestConversationalYaml:
    def test_nested_instantiation(self, load_yaml):
        cfg = load_yaml("inferencers/conversational.yaml")
        obj = instantiate(cfg)
        assert type(obj).__name__ == "ConversationalInferencer"
        assert type(obj.base_inferencer).__name__ == "ClaudeApiInferencer"
        assert type(obj.context_budget).__name__ == "ContextBudget"
        assert obj.context_budget.prior_context_max == 2000


class TestDualYaml:
    def test_shorthand_expansion(self, load_yaml):
        cfg = load_yaml("inferencers/dual.yaml")
        obj = instantiate(cfg)
        assert type(obj).__name__ == "DualInferencer"
        # String shorthand "ClaudeAPI" was expanded to full objects
        assert type(obj.base_inferencer).__name__ == "ClaudeApiInferencer"
        assert type(obj.review_inferencer).__name__ == "ClaudeApiInferencer"


class TestDefaultInferenceArgs:
    def test_load(self, load_yaml):
        cfg = load_yaml("inference_args/default.yaml")
        obj = instantiate(cfg)
        assert type(obj).__name__ == "CommonLlmInferenceArgs"
        assert obj.temperature == 1.0
        assert obj.max_tokens == 2000
