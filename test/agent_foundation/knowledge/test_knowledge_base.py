"""
Unit tests for KnowledgeBase orchestrator.

Tests CRUD operations, context manager, sensitive content rejection,
bulk_load, merge logic, graph knowledge extraction, and retrieval.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 6.1, 6.2, 6.3, 6.4, 6.5,
              8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.2, 9.4
"""
import json
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

import pytest

from agent_foundation.knowledge.retrieval.knowledge_base import KnowledgeBase
from agent_foundation.knowledge.retrieval.formatter import RetrievalResult
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.models.entity_metadata import EntityMetadata
from rich_python_utils.service_utils.graph_service.graph_node import (
    GraphNode,
    GraphEdge,
)
from agent_foundation.knowledge.retrieval.stores.metadata.keyvalue_adapter import (
    KeyValueMetadataStore,
)
from agent_foundation.knowledge.retrieval.stores.pieces.retrieval_adapter import (
    RetrievalKnowledgePieceStore,
)
from agent_foundation.knowledge.retrieval.stores.graph.graph_adapter import (
    GraphServiceEntityGraphStore,
)
from rich_python_utils.service_utils.keyvalue_service.memory_keyvalue_service import (
    MemoryKeyValueService,
)
from rich_python_utils.service_utils.retrieval_service.memory_retrieval_service import (
    MemoryRetrievalService,
)
from rich_python_utils.service_utils.graph_service.memory_graph_service import (
    MemoryGraphService,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def stores():
    """Create adapter-backed stores using in-memory services."""
    metadata_store = KeyValueMetadataStore(kv_service=MemoryKeyValueService())
    piece_store = RetrievalKnowledgePieceStore(retrieval_service=MemoryRetrievalService())
    graph_store = GraphServiceEntityGraphStore(graph_service=MemoryGraphService())
    return metadata_store, piece_store, graph_store


@pytest.fixture
def kb(stores):
    """Create a KnowledgeBase with adapter-backed stores."""
    metadata_store, piece_store, graph_store = stores
    return KnowledgeBase(
        metadata_store=metadata_store,
        piece_store=piece_store,
        graph_store=graph_store,
        active_entity_id="user:xinli",
    )


@pytest.fixture
def populated_kb(kb, stores):
    """Create a KnowledgeBase with some pre-loaded data."""
    metadata_store, piece_store, graph_store = stores

    # Add metadata
    user_meta = EntityMetadata(
        entity_id="user:xinli",
        entity_type="user",
        properties={"name": "Xinli", "location": "Seattle"},
    )
    metadata_store.save_metadata(user_meta)

    global_meta = EntityMetadata(
        entity_id="global",
        entity_type="global",
        properties={"app_version": "1.0"},
    )
    metadata_store.save_metadata(global_meta)

    # Add knowledge pieces (entity-scoped)
    piece1 = KnowledgePiece(
        content="Prefers organic eggs",
        piece_id="piece-1",
        knowledge_type=KnowledgeType.Preference,
        tags=["grocery", "eggs"],
        entity_id="user:xinli",
    )
    piece_store.add(piece1)

    piece2 = KnowledgePiece(
        content="Costco membership is Executive tier",
        piece_id="piece-2",
        knowledge_type=KnowledgeType.Fact,
        tags=["costco", "membership"],
        entity_id="user:xinli",
    )
    piece_store.add(piece2)

    # Add global piece
    global_piece = KnowledgePiece(
        content="Egg prices vary by season",
        piece_id="piece-global",
        knowledge_type=KnowledgeType.Fact,
        tags=["eggs", "pricing"],
        entity_id=None,
    )
    piece_store.add(global_piece)

    # Add graph nodes and relations using GraphNode/GraphEdge
    user_node = GraphNode(
        node_id="user:xinli", node_type="user", label="Xinli"
    )
    store_node = GraphNode(
        node_id="store:costco", node_type="store", label="Costco"
    )
    graph_store.add_node(user_node)
    graph_store.add_node(store_node)

    relation = GraphEdge(
        source_id="user:xinli",
        target_id="store:costco",
        edge_type="SHOPS_AT",
        properties={"membership_tier": "Executive"},
    )
    graph_store.add_relation(relation)

    return kb


# ── CRUD Tests ───────────────────────────────────────────────────────────────


class TestAddPiece:
    """Tests for KnowledgeBase.add_piece."""

    def test_add_piece_returns_id(self, kb):
        piece = KnowledgePiece(content="Test knowledge", piece_id="test-1")
        result = kb.add_piece(piece)
        assert result == "test-1"

    def test_add_piece_persists(self, kb, stores):
        _, piece_store, _ = stores
        piece = KnowledgePiece(content="Persisted knowledge", piece_id="test-2")
        kb.add_piece(piece)
        retrieved = piece_store.get_by_id("test-2")
        assert retrieved is not None
        assert retrieved.content == "Persisted knowledge"

    def test_add_piece_rejects_empty_content(self, kb):
        piece = KnowledgePiece.__new__(KnowledgePiece)
        piece.content = ""
        piece.piece_id = "empty-1"
        with pytest.raises(ValueError, match="non-empty"):
            kb.add_piece(piece)

    def test_add_piece_rejects_whitespace_content(self, kb):
        piece = KnowledgePiece.__new__(KnowledgePiece)
        piece.content = "   \t\n  "
        piece.piece_id = "ws-1"
        with pytest.raises(ValueError, match="non-empty"):
            kb.add_piece(piece)


class TestUpdatePiece:
    """Tests for KnowledgeBase.update_piece."""

    def test_update_existing_piece(self, kb):
        piece = KnowledgePiece(content="Original", piece_id="upd-1")
        kb.add_piece(piece)

        piece.content = "Updated content"
        result = kb.update_piece(piece)
        assert result is True

    def test_update_refreshes_timestamp(self, kb, stores):
        _, piece_store, _ = stores
        piece = KnowledgePiece(content="Original", piece_id="upd-2")
        kb.add_piece(piece)
        original_updated = piece.updated_at

        piece.content = "Updated content"
        kb.update_piece(piece)

        retrieved = piece_store.get_by_id("upd-2")
        assert retrieved.content == "Updated content"
        assert retrieved.updated_at >= original_updated

    def test_update_nonexistent_returns_false(self, kb):
        piece = KnowledgePiece(content="Ghost", piece_id="nonexistent")
        result = kb.update_piece(piece)
        assert result is False

    def test_update_rejects_sensitive_content(self, kb):
        piece = KnowledgePiece(content="Safe content", piece_id="upd-3")
        kb.add_piece(piece)

        piece.content = "api_key= sk-12345"
        with pytest.raises(ValueError, match="sensitive"):
            kb.update_piece(piece)


class TestRemovePiece:
    """Tests for KnowledgeBase.remove_piece."""

    def test_remove_existing_piece(self, kb):
        piece = KnowledgePiece(content="To be removed", piece_id="rm-1")
        kb.add_piece(piece)
        result = kb.remove_piece("rm-1")
        assert result is True

    def test_remove_nonexistent_returns_false(self, kb):
        result = kb.remove_piece("nonexistent")
        assert result is False

    def test_remove_then_get_returns_none(self, kb, stores):
        _, piece_store, _ = stores
        piece = KnowledgePiece(content="Temporary", piece_id="rm-2")
        kb.add_piece(piece)
        kb.remove_piece("rm-2")
        assert piece_store.get_by_id("rm-2") is None


# ── Setter method tests ──────────────────────────────────────────────────────


class TestSetterMethods:
    """Tests for KnowledgeBase setter methods (set_hybrid_retriever, etc.)."""

    def test_set_hybrid_retriever(self, kb):
        from agent_foundation.knowledge.retrieval.hybrid_search import (
            HybridRetriever,
            HybridSearchConfig,
        )

        def dummy_vector(query, entity_id=None, tags=None, top_k=10):
            return []

        def dummy_keyword(query, entity_id=None, tags=None, top_k=10):
            return []

        retriever = HybridRetriever(
            vector_search_fn=dummy_vector,
            keyword_search_fn=dummy_keyword,
        )
        kb.set_hybrid_retriever(retriever)
        assert kb._hybrid_retriever is retriever

    def test_set_temporal_decay(self, kb):
        from agent_foundation.knowledge.retrieval.temporal_decay import TemporalDecayConfig

        config = TemporalDecayConfig(half_life_days=14.0)
        kb.set_temporal_decay(config)
        assert kb._temporal_decay_config is config
        assert kb._temporal_decay_config.half_life_days == 14.0

    def test_set_mmr_config(self, kb):
        from agent_foundation.knowledge.retrieval.mmr_reranking import MMRConfig

        config = MMRConfig(lambda_param=0.5)
        kb.set_mmr_config(config)
        assert kb._mmr_config is config
        assert kb._mmr_config.lambda_param == 0.5

    def test_initial_state_has_no_enhancements(self, kb):
        assert kb._hybrid_retriever is None
        assert kb._temporal_decay_config is None
        assert kb._mmr_config is None


# ── Domain-aware retrieval tests ─────────────────────────────────────────────


class TestRetrieveWithDomainFilters:
    """Tests for KnowledgeBase.retrieve with domain/tags/min_results params."""

    def test_retrieve_accepts_new_params(self, populated_kb):
        """retrieve() accepts domain, secondary_domains, tags, min_results without error."""
        result = populated_kb.retrieve(
            "eggs",
            domain="general",
            secondary_domains=["data_engineering"],
            tags=["eggs"],
            min_results=1,
        )
        assert result is not None

    def test_retrieve_without_new_params_unchanged(self, populated_kb):
        """retrieve() without new params behaves identically to before."""
        result = populated_kb.retrieve("eggs")
        assert result is not None
        # Should still return pieces (standard path)
        assert isinstance(result.pieces, list)

    def test_retrieve_with_tags_filter(self, populated_kb):
        """retrieve() with tags filter uses the fallback path."""
        result = populated_kb.retrieve("eggs", tags=["eggs"])
        assert result is not None

    def test_retrieve_with_domain_filter(self, populated_kb):
        """retrieve() with domain filter uses the fallback path."""
        result = populated_kb.retrieve("eggs", domain="general")
        assert result is not None


# ── Fallback strategy tests ──────────────────────────────────────────────────


class TestRetrievePiecesWithFallback:
    """Tests for KnowledgeBase._retrieve_pieces_with_fallback."""

    def test_fallback_returns_list(self, populated_kb):
        """_retrieve_pieces_with_fallback returns a list of (piece, score) tuples."""
        result = populated_kb._retrieve_pieces_with_fallback(
            query="eggs",
            entity_id="user:xinli",
            top_k=5,
            tags=["eggs"],
        )
        assert isinstance(result, list)

    def test_fallback_tier4_pure_semantic(self, populated_kb):
        """When domain doesn't match anything, falls back to pure semantic."""
        result = populated_kb._retrieve_pieces_with_fallback(
            query="eggs",
            entity_id="user:xinli",
            top_k=5,
            domain="nonexistent_domain",
            min_results=1,
        )
        # Should still return results via tier 4 fallback
        assert isinstance(result, list)

    def test_fallback_with_secondary_domains(self, populated_kb):
        """Tier 2 expands to secondary_domains."""
        result = populated_kb._retrieve_pieces_with_fallback(
            query="eggs",
            entity_id="user:xinli",
            top_k=5,
            domain="nonexistent_domain",
            secondary_domains=["general"],
            min_results=1,
        )
        assert isinstance(result, list)

    def test_fallback_respects_top_k(self, populated_kb):
        """Fallback never returns more than top_k results."""
        result = populated_kb._retrieve_pieces_with_fallback(
            query="eggs",
            entity_id="user:xinli",
            top_k=1,
            min_results=0,
        )
        assert len(result) <= 1


# ── Enhanced retrieval path tests ────────────────────────────────────────────


class TestEnhancedRetrievalPath:
    """Tests for the enhanced retrieval path with HybridRetriever."""

    def test_hybrid_retriever_path_used(self, kb):
        """When hybrid retriever is set, it is used for retrieval."""
        from agent_foundation.knowledge.retrieval.hybrid_search import HybridRetriever
        from agent_foundation.knowledge.retrieval.models.results import ScoredPiece

        call_log = []

        def mock_vector(query, entity_id=None, tags=None, top_k=10):
            call_log.append(("vector", query))
            piece = KnowledgePiece(content="hybrid result", piece_id="hyb-1")
            return [(piece, 0.9)]

        def mock_keyword(query, entity_id=None, tags=None, top_k=10):
            call_log.append(("keyword", query))
            return []

        retriever = HybridRetriever(
            vector_search_fn=mock_vector,
            keyword_search_fn=mock_keyword,
        )
        kb.set_hybrid_retriever(retriever)

        result = kb.retrieve("test query")
        # The hybrid retriever should have been called
        assert len(call_log) > 0
        assert any(t[0] == "vector" for t in call_log)

    def test_hybrid_with_temporal_decay(self, kb):
        """Enhanced path applies temporal decay when configured."""
        from agent_foundation.knowledge.retrieval.hybrid_search import HybridRetriever
        from agent_foundation.knowledge.retrieval.temporal_decay import TemporalDecayConfig

        piece = KnowledgePiece(
            content="decayed result",
            piece_id="decay-1",
            info_type="context",
        )

        def mock_vector(query, entity_id=None, tags=None, top_k=10):
            return [(piece, 0.9)]

        def mock_keyword(query, entity_id=None, tags=None, top_k=10):
            return []

        retriever = HybridRetriever(
            vector_search_fn=mock_vector,
            keyword_search_fn=mock_keyword,
        )
        kb.set_hybrid_retriever(retriever)
        kb.set_temporal_decay(TemporalDecayConfig(enabled=True, half_life_days=30.0))

        result = kb.retrieve("test query")
        assert result.pieces is not None

    def test_hybrid_with_mmr(self, kb):
        """Enhanced path applies MMR when configured."""
        from agent_foundation.knowledge.retrieval.hybrid_search import HybridRetriever
        from agent_foundation.knowledge.retrieval.mmr_reranking import MMRConfig

        piece = KnowledgePiece(
            content="mmr result",
            piece_id="mmr-1",
            embedding=[1.0, 0.0, 0.0],
        )

        def mock_vector(query, entity_id=None, tags=None, top_k=10):
            return [(piece, 0.9)]

        def mock_keyword(query, entity_id=None, tags=None, top_k=10):
            return []

        retriever = HybridRetriever(
            vector_search_fn=mock_vector,
            keyword_search_fn=mock_keyword,
        )
        kb.set_hybrid_retriever(retriever)
        kb.set_mmr_config(MMRConfig(enabled=True, lambda_param=0.7))

        result = kb.retrieve("test query")
        assert result.pieces is not None

    def test_graph_retrieval_preserved_with_hybrid(self, populated_kb):
        """graph_retrieval_ignore_pieces_already_retrieved still works with hybrid path."""
        from agent_foundation.knowledge.retrieval.hybrid_search import HybridRetriever

        piece = KnowledgePiece(
            content="hybrid graph test",
            piece_id="hg-1",
        )

        def mock_vector(query, entity_id=None, tags=None, top_k=10):
            return [(piece, 0.9)]

        def mock_keyword(query, entity_id=None, tags=None, top_k=10):
            return []

        retriever = HybridRetriever(
            vector_search_fn=mock_vector,
            keyword_search_fn=mock_keyword,
        )
        populated_kb.set_hybrid_retriever(retriever)
        populated_kb.graph_retrieval_ignore_pieces_already_retrieved = True

        result = populated_kb.retrieve("test query")
        # Should not error — graph dedup logic still works
        assert result is not None


# ── Preservation tests ───────────────────────────────────────────────────────


class TestPreservation:
    """Tests that AgentFoundation-specific features are preserved."""

    def test_graph_retrieval_ignore_attribute_exists(self, kb):
        """graph_retrieval_ignore_pieces_already_retrieved attribute is preserved."""
        assert hasattr(kb, "graph_retrieval_ignore_pieces_already_retrieved")

    def test_should_skip_graph_piece_method_exists(self, kb):
        """_should_skip_graph_piece helper method is preserved."""
        assert hasattr(kb, "_should_skip_graph_piece")
        assert callable(kb._should_skip_graph_piece)

    def test_extract_graph_knowledge_has_already_retrieved_param(self, kb):
        """_extract_graph_knowledge retains already_retrieved_piece_ids parameter."""
        import inspect
        sig = inspect.signature(kb._extract_graph_knowledge)
        assert "already_retrieved_piece_ids" in sig.parameters
