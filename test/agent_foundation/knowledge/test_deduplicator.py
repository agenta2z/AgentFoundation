"""Unit tests for the three-tier deduplicator module."""

import json
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from agent_foundation.knowledge.ingestion.deduplicator import (
    DedupConfig,
    ThreeTierDeduplicator,
)
from agent_foundation.knowledge.retrieval.models.enums import DedupAction
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.models.results import DedupResult
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore


class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory store for testing."""

    def __init__(self, pieces: Optional[List[KnowledgePiece]] = None):
        self._pieces: dict[str, KnowledgePiece] = {}
        for p in (pieces or []):
            self._pieces[p.piece_id] = p

    def add(self, piece: KnowledgePiece) -> str:
        self._pieces[piece.piece_id] = piece
        return piece.piece_id

    def get_by_id(self, piece_id: str) -> Optional[KnowledgePiece]:
        return self._pieces.get(piece_id)

    def update(self, piece: KnowledgePiece) -> bool:
        if piece.piece_id in self._pieces:
            self._pieces[piece.piece_id] = piece
            return True
        return False

    def remove(self, piece_id: str) -> bool:
        return self._pieces.pop(piece_id, None) is not None

    def search(
        self, query, entity_id=None, knowledge_type=None, tags=None, top_k=5
    ) -> List[Tuple[KnowledgePiece, float]]:
        """Return all pieces with a fixed score for testing."""
        results = []
        for p in self._pieces.values():
            if entity_id is not None and p.entity_id != entity_id:
                continue
            results.append((p, self._score))
        return results[:top_k]

    def list_all(self, entity_id=None, knowledge_type=None) -> List[KnowledgePiece]:
        results = []
        for p in self._pieces.values():
            if entity_id is not None and p.entity_id != entity_id:
                continue
            if knowledge_type is not None and p.knowledge_type != knowledge_type:
                continue
            results.append(p)
        return results

    _score: float = 0.5  # default search score


def _dummy_embedding_fn(text: str) -> List[float]:
    return [0.1, 0.2, 0.3]


class TestDedupConfig:
    """Tests for the DedupConfig dataclass."""

    def test_defaults(self):
        config = DedupConfig()
        assert config.auto_dedup_threshold == 0.98
        assert config.llm_judge_threshold == 0.85
        assert config.enable_tier1 is True
        assert config.enable_tier2 is True
        assert config.enable_tier3 is True

    def test_custom_values(self):
        config = DedupConfig(
            auto_dedup_threshold=0.95,
            llm_judge_threshold=0.80,
            enable_tier1=False,
        )
        assert config.auto_dedup_threshold == 0.95
        assert config.llm_judge_threshold == 0.80
        assert config.enable_tier1 is False


class TestTier1HashCheck:
    """Tests for Tier 1 content hash deduplication (Requirement 12.2)."""

    def test_exact_hash_match_returns_no_op(self):
        existing = KnowledgePiece(content="Hello world", entity_id="e1")
        store = InMemoryPieceStore([existing])
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn)

        new_piece = KnowledgePiece(content="Hello world", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        assert result.action == DedupAction.NO_OP
        assert result.existing_piece_id == existing.piece_id
        assert "hash" in result.reason.lower()

    def test_no_hash_match_continues(self):
        existing = KnowledgePiece(content="Different content", entity_id="e1")
        store = InMemoryPieceStore([existing])
        # Set search score below llm_judge_threshold so tier 2 returns ADD
        store._score = 0.5
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn)

        new_piece = KnowledgePiece(content="Unique content", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        # Should not be NO_OP from tier 1
        assert result.action == DedupAction.ADD

    def test_tier1_disabled_skips_hash_check(self):
        existing = KnowledgePiece(content="Hello world", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.5  # Below llm_judge_threshold
        config = DedupConfig(enable_tier1=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, config=config)

        new_piece = KnowledgePiece(content="Hello world", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        # With tier1 disabled, even exact content goes to tier2
        # Score 0.5 < 0.85 so should be ADD
        assert result.action == DedupAction.ADD

    def test_computes_hash_if_missing(self):
        store = InMemoryPieceStore()
        store._score = 0.5
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn)

        piece = KnowledgePiece(content="Test content")
        # content_hash should be auto-computed, but let's ensure tier1 handles None
        piece.content_hash = None
        result = dedup.deduplicate(piece)

        # After dedup, the piece should have a content_hash set
        assert piece.content_hash is not None


class TestTier2EmbeddingSimilarity:
    """Tests for Tier 2 embedding similarity (Requirements 12.3, 12.4, 12.6)."""

    def test_high_similarity_returns_no_op(self):
        """Score > auto_dedup_threshold → NO_OP (Requirement 12.4)."""
        existing = KnowledgePiece(content="Some content", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.99  # Above auto_dedup_threshold
        config = DedupConfig(enable_tier1=False)  # Skip tier1 to test tier2
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, config=config)

        new_piece = KnowledgePiece(content="Different text", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        assert result.action == DedupAction.NO_OP
        assert result.existing_piece_id == existing.piece_id
        assert result.similarity_score == 0.99

    def test_low_similarity_returns_add(self):
        """Score < llm_judge_threshold → ADD (Requirement 12.6)."""
        existing = KnowledgePiece(content="Some content", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.5  # Below llm_judge_threshold
        config = DedupConfig(enable_tier1=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, config=config)

        new_piece = KnowledgePiece(content="Very different", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        assert result.action == DedupAction.ADD
        assert result.similarity_score == 0.5

    def test_no_similar_pieces_returns_add(self):
        """Empty store → ADD."""
        store = InMemoryPieceStore()
        config = DedupConfig(enable_tier1=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, config=config)

        new_piece = KnowledgePiece(content="New content")
        result = dedup.deduplicate(new_piece)

        assert result.action == DedupAction.ADD

    def test_tier2_disabled_skips_embedding_check(self):
        existing = KnowledgePiece(content="Some content", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.99
        config = DedupConfig(enable_tier1=False, enable_tier2=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, config=config)

        new_piece = KnowledgePiece(content="Different text", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        # Both tiers disabled → falls through to ADD
        assert result.action == DedupAction.ADD
        assert result.reason == "No duplicates found"

    def test_computes_embedding_if_missing(self):
        store = InMemoryPieceStore()
        config = DedupConfig(enable_tier1=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, config=config)

        piece = KnowledgePiece(content="Test content")
        assert piece.embedding is None
        dedup.deduplicate(piece)
        assert piece.embedding is not None


class TestTier3LLMJudge:
    """Tests for Tier 3 LLM judge (Requirements 12.5, 12.7)."""

    def test_borderline_similarity_invokes_llm(self):
        """Score between thresholds → Tier 3 invoked (Requirement 12.5)."""
        existing = KnowledgePiece(content="Some content", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.90  # Between 0.85 and 0.98

        llm_response = json.dumps({
            "action": "MERGE",
            "reasoning": "Complementary info",
            "contradiction_detected": False,
        })
        llm_fn = MagicMock(return_value=llm_response)
        config = DedupConfig(enable_tier1=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, llm_fn=llm_fn, config=config)

        new_piece = KnowledgePiece(content="Related content", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        assert result.action == DedupAction.MERGE
        assert result.existing_piece_id == existing.piece_id
        assert result.similarity_score == 0.90
        llm_fn.assert_called_once()

    def test_llm_failure_defaults_to_add(self):
        """LLM failure → ADD with warning (Requirement 12.7)."""
        existing = KnowledgePiece(content="Some content", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.90

        llm_fn = MagicMock(side_effect=RuntimeError("LLM unavailable"))
        config = DedupConfig(enable_tier1=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, llm_fn=llm_fn, config=config)

        new_piece = KnowledgePiece(content="Related content", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        assert result.action == DedupAction.ADD
        assert "error" in result.reason.lower()
        assert result.similarity_score == 0.90

    def test_llm_returns_no_op(self):
        existing = KnowledgePiece(content="Some content", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.90

        llm_response = json.dumps({
            "action": "NO_OP",
            "reasoning": "Near duplicate",
            "contradiction_detected": False,
        })
        llm_fn = MagicMock(return_value=llm_response)
        config = DedupConfig(enable_tier1=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, llm_fn=llm_fn, config=config)

        new_piece = KnowledgePiece(content="Related content", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        assert result.action == DedupAction.NO_OP
        assert result.existing_piece_id == existing.piece_id

    def test_llm_returns_add_no_existing_piece_id(self):
        existing = KnowledgePiece(content="Some content", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.90

        llm_response = json.dumps({
            "action": "ADD",
            "reasoning": "Different topics",
            "contradiction_detected": False,
        })
        llm_fn = MagicMock(return_value=llm_response)
        config = DedupConfig(enable_tier1=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, llm_fn=llm_fn, config=config)

        new_piece = KnowledgePiece(content="Related content", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        assert result.action == DedupAction.ADD
        assert result.existing_piece_id is None

    def test_llm_invalid_action_defaults_to_add(self):
        existing = KnowledgePiece(content="Some content", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.90

        llm_response = json.dumps({
            "action": "INVALID_ACTION",
            "reasoning": "Bad response",
        })
        llm_fn = MagicMock(return_value=llm_response)
        config = DedupConfig(enable_tier1=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, llm_fn=llm_fn, config=config)

        new_piece = KnowledgePiece(content="Related content", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        assert result.action == DedupAction.ADD

    def test_tier3_disabled_returns_add_for_borderline(self):
        """Tier 3 disabled → borderline cases default to ADD."""
        existing = KnowledgePiece(content="Some content", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.90

        config = DedupConfig(enable_tier1=False, enable_tier3=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, config=config)

        new_piece = KnowledgePiece(content="Related content", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        assert result.action == DedupAction.ADD
        assert result.similarity_score == 0.90

    def test_no_llm_fn_defaults_to_add(self):
        """No llm_fn provided → Tier 3 returns ADD."""
        existing = KnowledgePiece(content="Some content", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.90

        config = DedupConfig(enable_tier1=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, config=config)

        new_piece = KnowledgePiece(content="Related content", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        assert result.action == DedupAction.ADD

    def test_contradiction_detected(self):
        existing = KnowledgePiece(content="Some content", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.90

        llm_response = json.dumps({
            "action": "ADD",
            "reasoning": "Contradictory info",
            "contradiction_detected": True,
        })
        llm_fn = MagicMock(return_value=llm_response)
        config = DedupConfig(enable_tier1=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, llm_fn=llm_fn, config=config)

        new_piece = KnowledgePiece(content="Contradicting content", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        assert result.contradiction_detected is True


class TestDedupResultFields:
    """Tests for DedupResult field completeness (Requirement 12.8)."""

    def test_result_has_all_fields(self):
        result = DedupResult(
            action=DedupAction.ADD,
            reason="test",
            existing_piece_id="p1",
            similarity_score=0.95,
            contradiction_detected=True,
        )
        assert result.action == DedupAction.ADD
        assert result.reason == "test"
        assert result.existing_piece_id == "p1"
        assert result.similarity_score == 0.95
        assert result.contradiction_detected is True

    def test_result_defaults(self):
        result = DedupResult(action=DedupAction.ADD)
        assert result.reason == ""
        assert result.existing_piece_id is None
        assert result.similarity_score == 0.0
        assert result.contradiction_detected is False


class TestAllTiersDisabled:
    """Tests for edge case where all tiers are disabled."""

    def test_all_disabled_returns_add(self):
        store = InMemoryPieceStore()
        config = DedupConfig(enable_tier1=False, enable_tier2=False, enable_tier3=False)
        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn, config=config)

        piece = KnowledgePiece(content="Any content")
        result = dedup.deduplicate(piece)

        assert result.action == DedupAction.ADD
        assert result.reason == "No duplicates found"


class TestTierInteraction:
    """Tests for tier interaction and flow."""

    def test_tier1_match_short_circuits_tier2(self):
        """Tier 1 match should prevent Tier 2 from running."""
        existing = KnowledgePiece(content="Hello world", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.99  # Would trigger NO_OP in tier2

        embedding_fn = MagicMock(return_value=[0.1, 0.2, 0.3])
        dedup = ThreeTierDeduplicator(store, embedding_fn)

        new_piece = KnowledgePiece(content="Hello world", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        assert result.action == DedupAction.NO_OP
        assert "hash" in result.reason.lower()
        # Embedding function should NOT have been called
        embedding_fn.assert_not_called()

    def test_full_pipeline_tier1_miss_tier2_add(self):
        """Tier 1 miss → Tier 2 low score → ADD."""
        existing = KnowledgePiece(content="Existing content", entity_id="e1")
        store = InMemoryPieceStore([existing])
        store._score = 0.3

        dedup = ThreeTierDeduplicator(store, _dummy_embedding_fn)

        new_piece = KnowledgePiece(content="Completely different", entity_id="e1")
        result = dedup.deduplicate(new_piece)

        assert result.action == DedupAction.ADD
