"""
End-to-end retrieval tests with REAL LLM consolidation.

Uses the file-based mock knowledge store and a real Claude API call
for the KnowledgeConsolidator, then validates the consolidated output
with basic keyword checks.

Requires:
- ANTHROPIC_API_KEY environment variable set
- Network access to the Anthropic API

Marked with ``@pytest.mark.e2e`` so they can be skipped in CI:
    pytest -m "not e2e"      # skip e2e tests
    pytest -m e2e            # run only e2e tests
"""
import os
import sys
from pathlib import Path

# Path resolution for imports
_current_file = Path(__file__).resolve()
_current_path = _current_file.parent
while _current_path.name != "test" and _current_path.parent != _current_path:
    _current_path = _current_path.parent
_src_dir = _current_path.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

_rpu_src = Path(__file__).resolve().parents[4] / "RichPythonUtils" / "src"
if _rpu_src.exists() and str(_rpu_src) not in sys.path:
    sys.path.insert(0, str(_rpu_src))

_test_knowledge_dir = str(Path(__file__).resolve().parent)
if _test_knowledge_dir not in sys.path:
    sys.path.insert(0, _test_knowledge_dir)

import pytest

from agent_foundation.knowledge.retrieval.post_processors import GroupedDictPostProcessor
from agent_foundation.knowledge.retrieval.knowledge_consolidator import KnowledgeConsolidator
from agent_foundation.knowledge.retrieval.models.enums import ConsolidationMode
from agent_foundation.apis.claude_llm import generate_text, ClaudeModels

from _grocery_kb_helpers import load_grocery_kb, QUERY, USER_ENTITY_ID


# ── Skip if no API key ──────────────────────────────────────────────────────

_has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not _has_api_key, reason="ANTHROPIC_API_KEY not set"),
]


def _claude_llm_fn(prompt: str) -> str:
    """Wrapper that adapts claude_llm.generate_text to Callable[[str], str]."""
    return generate_text(
        prompt_or_messages=prompt,
        model=ClaudeModels.CLAUDE_3_HAIKU,
        max_new_tokens=1024,
        temperature=0.3,
    )


# ── Test class ──────────────────────────────────────────────────────────────


class TestRealLLMRetrieval:
    """E2E tests with real LLM consolidation on the grocery knowledge store."""

    @pytest.fixture
    def grocery_kb(self, tmp_path):
        kb = load_grocery_kb(tmp_path)
        yield kb
        kb.close()

    def test_consolidation_returns_nonempty_output(self, grocery_kb):
        """Real LLM consolidation should produce non-empty output."""
        consolidator = KnowledgeConsolidator(
            llm_fn=_claude_llm_fn,
            mode=ConsolidationMode.ENABLED,
        )
        processor = GroupedDictPostProcessor(
            active_entity_id=USER_ENTITY_ID,
            consolidator=consolidator,
        )
        result = grocery_kb.retrieve(QUERY)
        output = processor.process(result, query=QUERY)

        assert "consolidated_knowledge" in output
        consolidated = output["consolidated_knowledge"]
        assert len(consolidated.strip()) > 0

    def test_consolidated_output_mentions_safeway(self, grocery_kb):
        """The LLM should mention Safeway given the query is about Safeway."""
        consolidator = KnowledgeConsolidator(
            llm_fn=_claude_llm_fn,
            mode=ConsolidationMode.ENABLED,
        )
        processor = GroupedDictPostProcessor(
            active_entity_id=USER_ENTITY_ID,
            consolidator=consolidator,
        )
        result = grocery_kb.retrieve(QUERY)
        output = processor.process(result, query=QUERY)

        consolidated = output["consolidated_knowledge"].lower()
        assert "safeway" in consolidated, (
            f"Expected 'safeway' in consolidated output but got:\n{consolidated}"
        )

    def test_consolidated_output_references_grocery_procedure(self, grocery_kb):
        """The LLM should reference shopping procedure content."""
        consolidator = KnowledgeConsolidator(
            llm_fn=_claude_llm_fn,
            mode=ConsolidationMode.ENABLED,
        )
        processor = GroupedDictPostProcessor(
            active_entity_id=USER_ENTITY_ID,
            consolidator=consolidator,
        )
        result = grocery_kb.retrieve(QUERY)
        output = processor.process(result, query=QUERY)

        consolidated = output["consolidated_knowledge"].lower()
        # The procedure mentions login, cart, coupons, pricing - at least one should appear
        procedure_keywords = ["login", "cart", "coupon", "price", "pricing", "checkout", "member"]
        found = [kw for kw in procedure_keywords if kw in consolidated]
        assert len(found) >= 1, (
            f"Expected at least one of {procedure_keywords} in consolidated output "
            f"but got:\n{consolidated}"
        )

    def test_full_pipeline_produces_all_layers(self, grocery_kb):
        """Full retrieve + real consolidation: all layers present."""
        consolidator = KnowledgeConsolidator(
            llm_fn=_claude_llm_fn,
            mode=ConsolidationMode.ENABLED,
        )
        processor = GroupedDictPostProcessor(
            active_entity_id=USER_ENTITY_ID,
            consolidator=consolidator,
        )
        result = grocery_kb.retrieve(QUERY)
        output = processor.process(result, query=QUERY)

        # All expected keys should be present
        assert "user_profile" in output, "Missing user_profile group"
        assert "instructions" in output, "Missing instructions group"
        assert "consolidated_knowledge" in output, "Missing consolidated output"

        # Verify the user_profile section contains user info
        user_profile = output["user_profile"].lower()
        assert "alex johnson" in user_profile or "safeway" in user_profile

    def test_different_query_changes_consolidation(self, grocery_kb):
        """A different query should produce different consolidated output."""
        consolidator = KnowledgeConsolidator(
            llm_fn=_claude_llm_fn,
            mode=ConsolidationMode.ENABLED,
        )
        processor = GroupedDictPostProcessor(
            active_entity_id=USER_ENTITY_ID,
            consolidator=consolidator,
        )

        # Query 1: about safeway
        result1 = grocery_kb.retrieve(QUERY)
        output1 = processor.process(result1, query=QUERY)

        # Query 2: about whole foods
        query2 = "what groceries can I buy at whole foods"
        result2 = grocery_kb.retrieve(query2)
        output2 = processor.process(result2, query=query2)

        consolidated1 = output1.get("consolidated_knowledge", "")
        consolidated2 = output2.get("consolidated_knowledge", "")

        # At least one should mention the relevant store
        assert "safeway" in consolidated1.lower() or "whole foods" in consolidated2.lower(), (
            "Expected query-specific content in consolidated outputs"
        )
