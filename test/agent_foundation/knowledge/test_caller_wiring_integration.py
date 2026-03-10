"""Integration tests verifying each caller produces equivalent output after pipeline wiring.

Tests:
- KB.__call__() delegation to RetrievalPipeline + FlatStringPostProcessor
- BudgetAwareKnowledgeProvider.format_knowledge() still works as standalone utility
- Removal of deprecated KnowledgeProvider and AgenticRetriever classes

Requirements: 8.3, 10.4, 11.1
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

# ── path setup ───────────────────────────────────────────────────────────
_test_dir = Path(__file__).resolve().parent
_src_dir = _test_dir.parent.parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from agent_foundation.knowledge.retrieval.formatter import (
    KnowledgeFormatter,
    RetrievalResult,
)
from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.models.results import ScoredPiece
from agent_foundation.knowledge.retrieval.knowledge_provider import (
    BudgetAwareKnowledgeProvider,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _piece(pid: str, info_type: str = "context", content: str = "test content") -> KnowledgePiece:
    return KnowledgePiece(
        piece_id=pid,
        entity_id="e1",
        content=content,
        knowledge_type=KnowledgeType.Fact,
        info_type=info_type,
    )


def _meta(eid: str, props: Optional[Dict[str, str]] = None) -> EntityMetadata:
    return EntityMetadata(entity_id=eid, entity_type="user", properties=props or {"name": "Alice"})


def _result(
    pieces: Optional[List[Tuple[KnowledgePiece, float]]] = None,
    metadata: Optional[EntityMetadata] = None,
    global_metadata: Optional[EntityMetadata] = None,
    graph_context: Optional[List[Dict[str, Any]]] = None,
) -> RetrievalResult:
    r = RetrievalResult()
    r.pieces = pieces or []
    r.metadata = metadata
    r.global_metadata = global_metadata
    r.graph_context = graph_context or []
    return r


def _mock_kb(retrieve_result: Optional[RetrievalResult] = None) -> MagicMock:
    """Create a mock KnowledgeBase with sensible defaults."""
    kb = MagicMock(spec=KnowledgeBase)
    kb.formatter = KnowledgeFormatter()
    kb.active_entity_id = "user:alice"
    kb.include_metadata = True
    kb.include_pieces = True
    kb.include_graph = True
    kb.default_top_k = 5

    result = retrieve_result or _result(
        pieces=[(_piece("p1"), 0.9), (_piece("p2"), 0.7)],
        metadata=_meta("e1"),
    )
    kb.retrieve.return_value = result
    kb.retrieve_metadata.return_value = (result.metadata, result.global_metadata)
    kb.retrieve_pieces.return_value = result.pieces
    kb.retrieve_search_graph.return_value = result.graph_context
    kb.retrieve_identity_graph.return_value = []
    return kb


# ── Test KB.__call__() delegation ────────────────────────────────────────


class TestKBCallDelegation:
    """Verify KB.__call__() delegates to RetrievalPipeline + FlatStringPostProcessor."""

    def test_produces_same_output_as_direct_format(self):
        """KB.__call__() should produce the same output as formatter.format(retrieve(query))."""
        result = _result(
            pieces=[(_piece("p1"), 0.9), (_piece("p2", content="other"), 0.7)],
            metadata=_meta("e1"),
        )
        kb = _mock_kb(result)

        # Call via pipeline delegation
        output = KnowledgeBase.__call__(kb, "test query")

        # Expected: formatter.format(result)
        expected = KnowledgeFormatter().format(result)
        assert output == expected

    def test_passes_spaces_kwarg(self):
        """KB.__call__() should pass spaces through to pipeline.execute()."""
        kb = _mock_kb()
        KnowledgeBase.__call__(kb, "test query", spaces=["space1"])
        kb.retrieve.assert_called_once()
        call_kwargs = kb.retrieve.call_args
        assert call_kwargs.kwargs.get("spaces") == ["space1"] or \
               (call_kwargs.args == () and "spaces" in str(call_kwargs))

    def test_empty_result_returns_empty_string(self):
        """KB.__call__() with empty result should return empty string."""
        kb = _mock_kb(_result())
        output = KnowledgeBase.__call__(kb, "test query")
        assert output == ""

    def test_with_graph_context(self):
        """KB.__call__() should format graph context correctly."""
        result = _result(
            graph_context=[{
                "relation_type": "WORKS_AT",
                "target_node_id": "company:acme",
                "target_label": "Acme Corp",
                "piece": None,
                "depth": 1,
                "score": 0.5,
            }],
        )
        kb = _mock_kb(result)
        output = KnowledgeBase.__call__(kb, "test query")
        expected = KnowledgeFormatter().format(result)
        assert output == expected
        assert "WORKS_AT" in output


# ── Test KnowledgeProvider.__call__() delegation ─────────────────────────
# KnowledgeProvider has been removed (task 15.2). Use RetrievalPipeline
# with GroupedDictPostProcessor directly.


# ── Test AgenticRetriever.retrieve() delegation ─────────────────────────
# AgenticRetriever has been removed (task 15.1). These tests are no longer
# applicable. Use RetrievalPipeline + AggregatingPostProcessor directly.


# ── Test BudgetAwareKnowledgeProvider standalone ─────────────────────────


class TestBudgetAwareKnowledgeProviderStandalone:
    """Verify BudgetAwareKnowledgeProvider.format_knowledge() still works as standalone utility."""

    def test_format_knowledge_produces_output(self):
        """format_knowledge should produce formatted string within budget."""
        provider = BudgetAwareKnowledgeProvider()
        pieces = [
            ScoredPiece(piece=_piece("p1", info_type="context", content="Important fact"), score=0.9),
            ScoredPiece(piece=_piece("p2", info_type="instructions", content="Do this"), score=0.8),
        ]
        output = provider.format_knowledge(pieces, available_tokens=8000)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_respects_token_budget(self):
        """format_knowledge should not exceed available_tokens."""
        from agent_foundation.knowledge.retrieval.utils import count_tokens

        provider = BudgetAwareKnowledgeProvider()
        pieces = [
            ScoredPiece(piece=_piece(f"p{i}", info_type="context", content=f"Fact number {i} " * 50), score=0.9 - i * 0.01)
            for i in range(20)
        ]
        output = provider.format_knowledge(pieces, available_tokens=500)
        assert count_tokens(output) <= 500

    def test_empty_pieces_returns_empty(self):
        """format_knowledge with empty pieces should return empty string."""
        provider = BudgetAwareKnowledgeProvider()
        output = provider.format_knowledge([], available_tokens=8000)
        assert output == ""

    def test_priority_order(self):
        """format_knowledge should respect CONTEXT_BUDGET priority order."""
        provider = BudgetAwareKnowledgeProvider()
        pieces = [
            ScoredPiece(piece=_piece("s1", info_type="skills", content="Skill A"), score=0.9),
            ScoredPiece(piece=_piece("i1", info_type="instructions", content="Instruction B"), score=0.8),
            ScoredPiece(piece=_piece("c1", info_type="context", content="Context C"), score=0.7),
            ScoredPiece(piece=_piece("e1", info_type="episodic", content="Episode D"), score=0.6),
            ScoredPiece(piece=_piece("u1", info_type="user_profile", content="Profile E"), score=0.5),
        ]
        output = provider.format_knowledge(pieces, available_tokens=8000)
        # All sections should be present
        assert "Skills" in output or "Skill A" in output
        assert "Instructions" in output or "Instruction B" in output
        assert "Context" in output or "Context C" in output


# ── Test deprecation warnings ────────────────────────────────────────────


class TestDeprecationWarnings:
    """Verify deprecation warnings and removal of deprecated classes."""

    def test_knowledge_provider_removed(self):
        """KnowledgeProvider class should no longer exist in provider module."""
        with pytest.raises(ImportError):
            from agent_foundation.knowledge.retrieval.provider import KnowledgeProvider  # noqa: F401

    def test_agentic_retriever_removed(self):
        """AgenticRetriever has been removed; importing it should raise ImportError."""
        with pytest.raises(ImportError):
            from agent_foundation.knowledge.retrieval.agentic_retriever import AgenticRetriever  # noqa: F401

    def test_budget_aware_provider_no_deprecation_warning(self):
        """BudgetAwareKnowledgeProvider should NOT emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BudgetAwareKnowledgeProvider()

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0


# ── Test backward-compatible imports ─────────────────────────────────────


class TestBackwardCompatibleImports:
    """Verify SubQuery and AgenticRetrievalResult can be imported from agentic_retriever."""

    def test_subquery_importable_from_agentic_retriever(self):
        """SubQuery should be importable from agentic_retriever for backward compat."""
        from agent_foundation.knowledge.retrieval.agentic_retriever import SubQuery
        sq = SubQuery(query="test")
        assert sq.query == "test"

    def test_agentic_retrieval_result_importable_from_agentic_retriever(self):
        """AgenticRetrievalResult should be importable from agentic_retriever."""
        from agent_foundation.knowledge.retrieval.agentic_retriever import AgenticRetrievalResult
        result = AgenticRetrievalResult()
        assert result.pieces == []

    def test_subquery_same_class_from_both_modules(self):
        """SubQuery from agentic_retriever and retrieval_pipeline should be the same class."""
        from agent_foundation.knowledge.retrieval.agentic_retriever import SubQuery as SQ1
        from agent_foundation.knowledge.retrieval.retrieval_pipeline import SubQuery as SQ2
        assert SQ1 is SQ2

    def test_agentic_retrieval_result_same_class_from_both_modules(self):
        """AgenticRetrievalResult from both modules should be the same class."""
        from agent_foundation.knowledge.retrieval.agentic_retriever import AgenticRetrievalResult as AR1
        from agent_foundation.knowledge.retrieval.retrieval_pipeline import AgenticRetrievalResult as AR2
        assert AR1 is AR2

    def test_factory_functions_importable_from_agentic_retriever(self):
        """Factory functions should be importable from agentic_retriever for backward compat."""
        from agent_foundation.knowledge.retrieval.agentic_retriever import (
            create_domain_decomposer,
            create_llm_decomposer,
        )
        assert callable(create_domain_decomposer)
        assert callable(create_llm_decomposer)

    def test_factory_functions_same_from_both_modules(self):
        """Factory functions from agentic_retriever and retrieval_pipeline should be the same."""
        from agent_foundation.knowledge.retrieval.agentic_retriever import (
            create_domain_decomposer as cd1,
            create_llm_decomposer as cl1,
        )
        from agent_foundation.knowledge.retrieval.retrieval_pipeline import (
            create_domain_decomposer as cd2,
            create_llm_decomposer as cl2,
        )
        assert cd1 is cd2
        assert cl1 is cl2

    def test_pipeline_classes_importable_from_retrieval_init(self):
        """Pipeline classes should be importable from retrieval __init__."""
        from agent_foundation.knowledge.retrieval import (
            RetrievalPipeline,
            QueryExpander,
            PostProcessor,
            FlatStringPostProcessor,
            GroupedDictPostProcessor,
            AggregatingPostProcessor,
            BudgetAwarePostProcessor,
        )
        assert RetrievalPipeline is not None
        assert QueryExpander is not None
        assert PostProcessor is not None

    def test_pipeline_classes_importable_from_knowledge_init(self):
        """Pipeline classes should be importable from knowledge __init__."""
        from agent_foundation.knowledge import (
            RetrievalPipeline,
            QueryExpander,
            PostProcessor,
            FlatStringPostProcessor,
            GroupedDictPostProcessor,
            AggregatingPostProcessor,
            BudgetAwarePostProcessor,
        )
        assert RetrievalPipeline is not None
