"""Tests for MergeStrategyManager."""

import json
from typing import List, Optional, Tuple

from agent_foundation.knowledge.ingestion.merge_strategy import (
    MergeStrategyConfig,
    MergeStrategyManager,
)
from agent_foundation.knowledge.retrieval.models.enums import (
    MergeAction,
    MergeStrategy,
    MergeType,
    SuggestionStatus,
)
from agent_foundation.knowledge.retrieval.models.knowledge_piece import (
    KnowledgePiece,
    KnowledgeType,
)
from agent_foundation.knowledge.retrieval.models.results import (
    MergeCandidate,
    MergeResult,
)
from agent_foundation.knowledge.retrieval.stores.pieces.base import KnowledgePieceStore


class InMemoryPieceStore(KnowledgePieceStore):
    """Minimal in-memory store for testing."""

    def __init__(self, pieces: Optional[List[KnowledgePiece]] = None):
        self._pieces = {p.piece_id: p for p in (pieces or [])}

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
        self,
        query: str,
        entity_id=None,
        knowledge_type=None,
        tags=None,
        top_k: int = 5,
    ) -> List[Tuple[KnowledgePiece, float]]:
        return []

    def list_all(self, entity_id=None, knowledge_type=None) -> List[KnowledgePiece]:
        return list(self._pieces.values())


def _make_piece(**kwargs) -> KnowledgePiece:
    """Helper to create a KnowledgePiece with sensible defaults."""
    defaults = {"content": "test content"}
    defaults.update(kwargs)
    return KnowledgePiece(**defaults)


def _make_candidate(piece_id: str = "existing-1", similarity: float = 0.9) -> MergeCandidate:
    return MergeCandidate(
        piece_id=piece_id,
        similarity=similarity,
        merge_type=MergeType.OVERLAPPING,
        reason="Similar content",
    )


def _fake_llm_fn(prompt: str) -> str:
    """Returns a valid merge JSON response."""
    return json.dumps({
        "merged_content": "merged result",
        "merged_domain": "general",
        "merged_tags": ["tag1", "tag2"],
        "merge_notes": "Combined both pieces",
    })


# ── MergeStrategyConfig tests ──


class TestMergeStrategyConfig:
    def test_defaults(self):
        config = MergeStrategyConfig()
        assert config.default_by_type[KnowledgeType.Fact] == MergeStrategy.AUTO_MERGE_ON_INGEST
        assert config.default_by_type[KnowledgeType.Procedure] == MergeStrategy.MANUAL_ONLY
        assert config.default_by_type[KnowledgeType.Instruction] == MergeStrategy.SUGGESTION_ON_INGEST
        assert config.default_by_type[KnowledgeType.Episodic] == MergeStrategy.POST_INGESTION_AUTO
        assert config.default_by_type[KnowledgeType.Example] == MergeStrategy.SUGGESTION_ON_INGEST
        assert config.allow_override is True
        assert config.suggestion_expiry_days == 30

    def test_custom_config(self):
        config = MergeStrategyConfig(
            default_by_type={KnowledgeType.Fact: MergeStrategy.MANUAL_ONLY},
            allow_override=False,
            suggestion_expiry_days=7,
        )
        assert config.default_by_type[KnowledgeType.Fact] == MergeStrategy.MANUAL_ONLY
        assert config.allow_override is False
        assert config.suggestion_expiry_days == 7


# ── get_strategy tests ──


class TestGetStrategy:
    def test_returns_default_for_type(self):
        store = InMemoryPieceStore()
        manager = MergeStrategyManager(piece_store=store)
        piece = _make_piece(knowledge_type=KnowledgeType.Fact)
        assert manager.get_strategy(piece) == MergeStrategy.AUTO_MERGE_ON_INGEST

    def test_returns_override_when_set(self):
        store = InMemoryPieceStore()
        manager = MergeStrategyManager(piece_store=store)
        piece = _make_piece(
            knowledge_type=KnowledgeType.Fact,
            merge_strategy=MergeStrategy.MANUAL_ONLY.value,
        )
        assert manager.get_strategy(piece) == MergeStrategy.MANUAL_ONLY

    def test_ignores_override_when_disabled(self):
        config = MergeStrategyConfig(allow_override=False)
        store = InMemoryPieceStore()
        manager = MergeStrategyManager(piece_store=store, config=config)
        piece = _make_piece(
            knowledge_type=KnowledgeType.Fact,
            merge_strategy=MergeStrategy.MANUAL_ONLY.value,
        )
        assert manager.get_strategy(piece) == MergeStrategy.AUTO_MERGE_ON_INGEST

    def test_fallback_for_unknown_type(self):
        """Types not in the default map fall back to AUTO_MERGE_ON_INGEST."""
        config = MergeStrategyConfig(default_by_type={})
        store = InMemoryPieceStore()
        manager = MergeStrategyManager(piece_store=store, config=config)
        piece = _make_piece(knowledge_type=KnowledgeType.Fact)
        assert manager.get_strategy(piece) == MergeStrategy.AUTO_MERGE_ON_INGEST


# ── auto-merge tests ──


class TestAutoMerge:
    def test_no_candidates_returns_no_candidates(self):
        store = InMemoryPieceStore()
        manager = MergeStrategyManager(piece_store=store, llm_fn=_fake_llm_fn)
        piece = _make_piece(knowledge_type=KnowledgeType.Fact)
        result = manager.apply_strategy(piece, [])
        assert result.action == MergeAction.NO_CANDIDATES

    def test_auto_merge_with_llm(self):
        existing = _make_piece(content="existing content", piece_id="existing-1")
        store = InMemoryPieceStore([existing])
        manager = MergeStrategyManager(piece_store=store, llm_fn=_fake_llm_fn)

        new_piece = _make_piece(content="new content", knowledge_type=KnowledgeType.Fact)
        candidate = _make_candidate(piece_id="existing-1")
        result = manager.apply_strategy(new_piece, [candidate])

        assert result.action == MergeAction.MERGED
        assert result.merged_with == "existing-1"
        # The existing piece should be deactivated
        updated_existing = store.get_by_id("existing-1")
        assert updated_existing.is_active is False

    def test_auto_merge_no_llm_falls_back_to_suggestion(self):
        """Requirement 13.6: no LLM → fall back to suggestion."""
        existing = _make_piece(content="existing", piece_id="existing-1")
        store = InMemoryPieceStore([existing])
        manager = MergeStrategyManager(piece_store=store, llm_fn=None)

        piece = _make_piece(knowledge_type=KnowledgeType.Fact)
        candidate = _make_candidate(piece_id="existing-1")
        result = manager.apply_strategy(piece, [candidate])

        assert result.action == MergeAction.PENDING_REVIEW
        assert piece.pending_merge_suggestion == "existing-1"

    def test_auto_merge_candidate_not_found(self):
        store = InMemoryPieceStore()  # empty store
        manager = MergeStrategyManager(piece_store=store, llm_fn=_fake_llm_fn)

        piece = _make_piece(knowledge_type=KnowledgeType.Fact)
        candidate = _make_candidate(piece_id="nonexistent")
        result = manager.apply_strategy(piece, [candidate])

        assert result.action == MergeAction.ERROR
        assert "nonexistent" in result.error

    def test_auto_merge_llm_failure_returns_error(self):
        existing = _make_piece(content="existing", piece_id="existing-1")
        store = InMemoryPieceStore([existing])

        def bad_llm(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        manager = MergeStrategyManager(piece_store=store, llm_fn=bad_llm)
        piece = _make_piece(knowledge_type=KnowledgeType.Fact)
        candidate = _make_candidate(piece_id="existing-1")
        result = manager.apply_strategy(piece, [candidate])

        assert result.action == MergeAction.ERROR
        assert "LLM unavailable" in result.error

    def test_merged_piece_has_correct_fields(self):
        existing = _make_piece(
            content="existing content",
            piece_id="existing-1",
            knowledge_type=KnowledgeType.Instruction,
            info_type="context",
            entity_id="entity-1",
            tags=["old-tag"],
            version=2,
        )
        store = InMemoryPieceStore([existing])
        manager = MergeStrategyManager(piece_store=store, llm_fn=_fake_llm_fn)

        new_piece = _make_piece(
            content="new content",
            knowledge_type=KnowledgeType.Instruction,
            merge_strategy=MergeStrategy.AUTO_MERGE_ON_INGEST.value,
            tags=["new-tag"],
        )
        candidate = _make_candidate(piece_id="existing-1")
        result = manager.apply_strategy(new_piece, [candidate])

        assert result.action == MergeAction.MERGED
        merged = store.get_by_id(result.piece_id)
        assert merged is not None
        assert merged.content == "merged result"
        assert merged.supersedes == "existing-1"
        assert merged.version == 3  # existing was version 2
        assert merged.knowledge_type == KnowledgeType.Instruction
        assert merged.entity_id == "entity-1"


# ── suggestion-on-ingest tests ──


class TestSuggestionOnIngest:
    def test_sets_pending_fields(self):
        store = InMemoryPieceStore()
        manager = MergeStrategyManager(piece_store=store)

        piece = _make_piece(
            knowledge_type=KnowledgeType.Instruction,
        )
        candidate = _make_candidate(piece_id="candidate-1")
        result = manager.apply_strategy(piece, [candidate])

        assert result.action == MergeAction.PENDING_REVIEW
        assert piece.pending_merge_suggestion == "candidate-1"
        assert piece.merge_suggestion_reason == "Similar content"
        assert piece.suggestion_status == SuggestionStatus.PENDING.value

    def test_no_candidates_returns_no_candidates(self):
        store = InMemoryPieceStore()
        manager = MergeStrategyManager(piece_store=store)

        piece = _make_piece(knowledge_type=KnowledgeType.Instruction)
        result = manager.apply_strategy(piece, [])
        assert result.action == MergeAction.NO_CANDIDATES


# ── post-ingestion strategy tests ──


class TestPostIngestionStrategies:
    def test_post_ingestion_auto_defers(self):
        store = InMemoryPieceStore()
        manager = MergeStrategyManager(piece_store=store)

        piece = _make_piece(knowledge_type=KnowledgeType.Episodic)
        result = manager.apply_strategy(piece, [_make_candidate()])

        assert result.action == MergeAction.DEFERRED
        assert piece.merge_processed is False

    def test_post_ingestion_suggestion_defers(self):
        store = InMemoryPieceStore()
        config = MergeStrategyConfig(
            default_by_type={KnowledgeType.Fact: MergeStrategy.POST_INGESTION_SUGGESTION}
        )
        manager = MergeStrategyManager(piece_store=store, config=config)

        piece = _make_piece(knowledge_type=KnowledgeType.Fact)
        result = manager.apply_strategy(piece, [_make_candidate()])

        assert result.action == MergeAction.DEFERRED
        assert piece.merge_processed is False


# ── manual-only tests ──


class TestManualOnly:
    def test_returns_no_auto_merge(self):
        store = InMemoryPieceStore()
        manager = MergeStrategyManager(piece_store=store)

        piece = _make_piece(knowledge_type=KnowledgeType.Procedure)
        result = manager.apply_strategy(piece, [_make_candidate()])

        assert result.action == MergeAction.NO_AUTO_MERGE
        assert result.piece_id == piece.piece_id
