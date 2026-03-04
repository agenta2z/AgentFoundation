"""
Unit tests for KnowledgeConsolidator.

Tests consolidation modes (ENABLED, DISABLED, DISABLED_FOR_SHORT_KNOWLEDGE),
user_profile exclusion, LLM failure graceful fallback, and prompt formatting.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import pytest

from agent_foundation.knowledge.retrieval.knowledge_consolidator import (
    KnowledgeConsolidator,
)
from agent_foundation.knowledge.retrieval.models.enums import ConsolidationMode


# ── Helpers ──────────────────────────────────────────────────────────────────


def _sample_groups():
    """Sample formatted_groups dict as returned by KnowledgeProvider formatting."""
    return {
        "user_profile": "FirstName: Alice\nLocation: Seattle",
        "context": "[fact] Organic eggs at Safeway cost $5.99/dozen.",
        "instructions": "[procedure] 1) Log in. 2) Add items. 3) Checkout.",
    }


def _make_mock_llm(return_value="Consolidated output here."):
    """Create a mock LLM function that records calls."""
    mock = MagicMock(return_value=return_value)
    return mock


# ── Tests ────────────────────────────────────────────────────────────────────


class TestDisabledMode:
    """DISABLED mode should never call LLM and return input unchanged."""

    def test_disabled_mode_returns_input_unchanged(self):
        llm = _make_mock_llm()
        consolidator = KnowledgeConsolidator(
            llm_fn=llm, mode=ConsolidationMode.DISABLED
        )
        groups = _sample_groups()

        result = consolidator.consolidate("test query", groups)

        assert result is groups  # Same object, not a copy
        llm.assert_not_called()


class TestEnabledMode:
    """ENABLED mode should call LLM and add consolidated_knowledge key."""

    def test_enabled_mode_calls_llm_and_adds_key(self):
        llm = _make_mock_llm("Deduplicated knowledge output.")
        consolidator = KnowledgeConsolidator(
            llm_fn=llm, mode=ConsolidationMode.ENABLED
        )
        groups = _sample_groups()

        result = consolidator.consolidate("what are egg prices?", groups)

        # LLM was called
        llm.assert_called_once()
        # consolidated_knowledge key added
        assert KnowledgeConsolidator.CONSOLIDATED_KEY in result
        assert result[KnowledgeConsolidator.CONSOLIDATED_KEY] == "Deduplicated knowledge output."

    def test_original_keys_always_preserved(self):
        llm = _make_mock_llm("Consolidated.")
        consolidator = KnowledgeConsolidator(
            llm_fn=llm, mode=ConsolidationMode.ENABLED
        )
        groups = _sample_groups()

        result = consolidator.consolidate("query", groups)

        # All original keys still present
        assert "user_profile" in result
        assert "context" in result
        assert "instructions" in result
        # Plus the new key
        assert KnowledgeConsolidator.CONSOLIDATED_KEY in result

    def test_enabled_mode_empty_knowledge_skips(self):
        llm = _make_mock_llm()
        consolidator = KnowledgeConsolidator(
            llm_fn=llm, mode=ConsolidationMode.ENABLED
        )
        groups = {"user_profile": "Alice", "context": "", "instructions": "   "}

        result = consolidator.consolidate("query", groups)

        # Only user_profile has content, but it's excluded from consolidation.
        # No content keys → skip
        assert result is groups
        llm.assert_not_called()


class TestDisabledForShortKnowledge:
    """DISABLED_FOR_SHORT_KNOWLEDGE mode should skip when below threshold."""

    def test_short_below_threshold_skips(self):
        llm = _make_mock_llm()
        consolidator = KnowledgeConsolidator(
            llm_fn=llm,
            mode=ConsolidationMode.DISABLED_FOR_SHORT_KNOWLEDGE,
            short_knowledge_threshold=9999,  # Very high threshold
        )
        groups = _sample_groups()

        result = consolidator.consolidate("query", groups)

        assert result is groups
        llm.assert_not_called()

    def test_short_above_threshold_consolidates(self):
        llm = _make_mock_llm("Consolidated.")
        consolidator = KnowledgeConsolidator(
            llm_fn=llm,
            mode=ConsolidationMode.DISABLED_FOR_SHORT_KNOWLEDGE,
            short_knowledge_threshold=1,  # Very low threshold
        )
        groups = _sample_groups()

        result = consolidator.consolidate("query", groups)

        llm.assert_called_once()
        assert KnowledgeConsolidator.CONSOLIDATED_KEY in result


class TestLLMFailure:
    """LLM failures should return input dict unchanged."""

    def test_llm_failure_returns_input_unchanged(self):
        llm = MagicMock(side_effect=RuntimeError("API error"))
        consolidator = KnowledgeConsolidator(
            llm_fn=llm, mode=ConsolidationMode.ENABLED
        )
        groups = _sample_groups()

        result = consolidator.consolidate("query", groups)

        # Returns input unchanged on failure
        assert result is groups
        assert KnowledgeConsolidator.CONSOLIDATED_KEY not in result

    def test_llm_empty_response_returns_input_unchanged(self):
        llm = _make_mock_llm("   ")  # Whitespace-only
        consolidator = KnowledgeConsolidator(
            llm_fn=llm, mode=ConsolidationMode.ENABLED
        )
        groups = _sample_groups()

        result = consolidator.consolidate("query", groups)

        assert result is groups
        assert KnowledgeConsolidator.CONSOLIDATED_KEY not in result


class TestUserProfileExclusion:
    """user_profile should not be included in the content sent to LLM."""

    def test_user_profile_excluded_from_consolidation_input(self):
        captured_prompts = []

        def capturing_llm(prompt):
            captured_prompts.append(prompt)
            return "Consolidated."

        consolidator = KnowledgeConsolidator(
            llm_fn=capturing_llm, mode=ConsolidationMode.ENABLED
        )
        groups = {
            "user_profile": "SECRET_USER_DATA",
            "context": "Some context knowledge.",
        }

        consolidator.consolidate("query", groups)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # user_profile content should NOT be in the prompt
        assert "SECRET_USER_DATA" not in prompt
        # context content SHOULD be in the prompt
        assert "Some context knowledge." in prompt


class TestPromptFormatting:
    """Verify the prompt sent to LLM contains query and knowledge."""

    def test_prompt_contains_query_and_knowledge(self):
        captured_prompts = []

        def capturing_llm(prompt):
            captured_prompts.append(prompt)
            return "Consolidated."

        consolidator = KnowledgeConsolidator(
            llm_fn=capturing_llm, mode=ConsolidationMode.ENABLED
        )
        groups = {
            "context": "Eggs cost $5.99",
            "instructions": "Step 1: Add to cart",
        }

        consolidator.consolidate("what are egg prices?", groups)

        prompt = captured_prompts[0]
        assert "what are egg prices?" in prompt
        assert "Eggs cost $5.99" in prompt
        assert "Step 1: Add to cart" in prompt


class TestTokenThreshold:
    """Verify threshold uses count_tokens (len // 4), not len."""

    def test_threshold_uses_token_count(self):
        """A string of 800 chars = 200 tokens. Threshold of 201 should skip."""
        llm = _make_mock_llm()
        consolidator = KnowledgeConsolidator(
            llm_fn=llm,
            mode=ConsolidationMode.DISABLED_FOR_SHORT_KNOWLEDGE,
            short_knowledge_threshold=201,  # Just above 200 tokens
        )
        # Create content that is exactly 800 chars = 200 tokens
        groups = {"context": "x" * 800}

        result = consolidator.consolidate("query", groups)

        # 800 chars / 4 = 200 tokens < 201 threshold → should skip
        assert result is groups
        llm.assert_not_called()

    def test_threshold_at_boundary_consolidates(self):
        """A string of 804 chars = 201 tokens. Threshold of 201 should consolidate."""
        llm = _make_mock_llm("Consolidated.")
        consolidator = KnowledgeConsolidator(
            llm_fn=llm,
            mode=ConsolidationMode.DISABLED_FOR_SHORT_KNOWLEDGE,
            short_knowledge_threshold=201,
        )
        # Create content that is 804 chars = 201 tokens
        groups = {"context": "x" * 804}

        result = consolidator.consolidate("query", groups)

        # 804 chars / 4 = 201 tokens >= 201 threshold → should consolidate
        llm.assert_called_once()
        assert KnowledgeConsolidator.CONSOLIDATED_KEY in result
